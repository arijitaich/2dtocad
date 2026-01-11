"""
COMPLETE JEWELRY MANUFACTURING PIPELINE

AI Models Used:
- Step 1 (Image Selection): gemini-3-pro-preview
- Step 2 (Component Detection): gemini-3-pro-preview
- Step 3 (Image Generation): gemini-3-pro-image-preview
- Step 4 (3D Generation): Hunyuan3D-2 via Huggingface Space
- Step 5 (Tessellation): Pure Python + NumPy

Full workflow:
1. INPUT: Product images from client
2. IMAGE QUALITY SELECTION (Gemini 3 Pro Preview): Select best image for 3D
3. COMPONENT DETECTION (Gemini 3 Pro Preview): Detect ring, gemstone, prongs, etc.
4. IMAGE GENERATION (Gemini 3 Pro Image Preview): Clean reference image per component
5. 3D GENERATION (Hunyuan3D-2): High-quality 3D model per component
6. SURFACE TESSELLATION: Wrap each 3D with parametric square grids
7. CAD EXPORT: Save as editable Rhino .3dm file
8. CAD DESIGNER: Input client dimensions → Ready for mold manufacturing

Usage:
    python manufacturing_pipeline.py <image_path> [--square-size 0.5] [--use-meshy]
"""

from google import genai
from google.genai import types
import base64
import json
import os
import sys
import time
import requests
from pathlib import Path
from dotenv import load_dotenv
from PIL import Image

# Load environment variables
load_dotenv()

# API Keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MESHY_API_KEY = os.getenv("MESHY_API_KEY")

if not GOOGLE_API_KEY:
    print("ERROR: GOOGLE_API_KEY not found. Please set in .env file")
    sys.exit(1)

# Initialize Gemini client
client = genai.Client(api_key=GOOGLE_API_KEY, http_options={'api_version': 'v1alpha'})

# Hunyuan3D-2 pipelines (lazy loaded)
hunyuan_pipeline = None


# ============================================================================
# PROMPT DEFINITIONS
# ============================================================================

RESPONSE_FORMAT = """
{
    "product_name": "string",
    "product_category": "string",
    "overall_description": "string",
    "components": [
        {
            "component_id": "integer",
            "component_name": "string",
            "component_type": "string (structural/decorative/functional)",
            "image_generation_prompt": {
                "main_prompt": "string - detailed 100-200 word description for image generation",
                "style_modifiers": "string - Product photography, studio lighting, 3/4 view, gray background",
                "negative_prompt": "string - things to avoid",
                "camera_angle": "string",
                "lighting_description": "string"
            },
            "material_suggestion": "string",
            "shape_description": "string",
            "dimensions_estimate": {
                "relative_size": "string (small/medium/large)",
                "shape_type": "string (cylindrical/rectangular/spherical/irregular)"
            },
            "position": "string",
            "connections": ["list of component_ids"],
            "manufacturing_notes": "string"
        }
    ],
    "assembly_order": ["list of component_ids"],
    "complexity_rating": "string (simple/moderate/complex)",
    "additional_notes": "string"
}
"""

MASTER_PROMPT = f"""You are an expert product designer and manufacturing engineer. Analyze the provided image and detect all component structures required to manufacture this product.

Your task:
1. Identify the product in the image
2. Break down into individual manufacturable components
3. For each component, create a detailed image generation prompt for 3D reconstruction
4. Components should be suitable for mold manufacturing

IMPORTANT IMAGE PROMPT GUIDELINES:
- Each component must be described as a SINGLE isolated object
- Include specific material details (metallic sheen, matte finish, etc.)
- Describe exact shape, proportions, surface details
- Optimized for Hunyuan3D-2 3D reconstruction

Respond ONLY with valid JSON in this format:
{RESPONSE_FORMAT}"""


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_mime_type(file_path: str) -> str:
    """Determine MIME type from file extension."""
    ext = Path(file_path).suffix.lower()
    return {
        '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
        '.png': 'image/png', '.gif': 'image/gif',
        '.webp': 'image/webp', '.bmp': 'image/bmp'
    }.get(ext, 'image/jpeg')


def load_image_as_base64(image_path: str) -> str:
    """Load image file to base64 string."""
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


# ============================================================================
# STEP 1: COMPONENT DETECTION (Gemini 3 Pro Preview)
# ============================================================================

def detect_components(image_path: str) -> dict:
    """Analyze image with Gemini 3 Pro Preview to detect manufacturing components."""
    print(f"\n📸 Analyzing image with Gemini 3 Pro Preview: {image_path}")
    
    image_data = load_image_as_base64(image_path)
    mime_type = get_mime_type(image_path)
    
    response = client.models.generate_content(
        model="gemini-3-pro-preview",  # Gemini 3 Pro Preview for detection
        contents=[
            types.Content(parts=[
                types.Part(text=MASTER_PROMPT),
                types.Part(
                    inline_data=types.Blob(
                        mime_type=mime_type,
                        data=base64.b64decode(image_data)
                    ),
                    media_resolution={"level": "media_resolution_high"}
                )
            ])
        ]
    )
    
    response_text = response.text
    
    # Parse JSON
    if "```json" in response_text:
        response_text = response_text.split("```json")[1].split("```")[0]
    elif "```" in response_text:
        response_text = response_text.split("```")[1].split("```")[0]
    
    # Add delay to avoid Gemini rate limiting
    print("   ⏳ Waiting 3s to avoid rate limiting...")
    time.sleep(3)
    
    try:
        return json.loads(response_text.strip())
    except json.JSONDecodeError as e:
        print(f"⚠️ JSON parse error: {e}")
        return {"raw_response": response_text, "parse_error": str(e)}


# ============================================================================
# STEP 1: IMAGE QUALITY SELECTION (Pick best from provided reference images)
# Uses: Gemini 3 Pro Preview
# ============================================================================

IMAGE_QUALITY_PROMPT = """You are an expert at evaluating images for 3D reconstruction quality.

Analyze this image and rate its suitability for image-to-3D conversion (like Hunyuan3D-2).

EVALUATION CRITERIA:
1. **Object Clarity** (0-25): Is the main object clearly visible and in focus?
2. **Background Simplicity** (0-25): Is the background clean/neutral (gray, white, simple)?
3. **Lighting Quality** (0-20): Good lighting with visible depth and shadows?
4. **View Angle** (0-15): Good 3/4 view showing depth, not flat front-on?
5. **Object Isolation** (0-15): Single object, well-centered, no clutter?

Respond ONLY with valid JSON:
{
    "object_clarity": <0-25>,
    "background_simplicity": <0-25>,
    "lighting_quality": <0-20>,
    "view_angle": <0-15>,
    "object_isolation": <0-15>,
    "total_score": <0-100>,
    "issues": ["list any problems"],
    "recommendation": "brief explanation"
}"""


def evaluate_image_for_3d(image_path: str) -> dict:
    """Evaluate a single image's suitability for 3D reconstruction using Gemini 3 Pro Preview."""
    print(f"    📊 Evaluating with Gemini 3 Pro Preview: {os.path.basename(image_path)}")
    
    try:
        image_data = load_image_as_base64(image_path)
        mime_type = get_mime_type(image_path)
        
        response = client.models.generate_content(
            model="gemini-3-pro-preview",  # Gemini 3 Pro Preview for quality evaluation
            contents=[
                types.Content(parts=[
                    types.Part(text=IMAGE_QUALITY_PROMPT),
                    types.Part(
                        inline_data=types.Blob(
                            mime_type=mime_type,
                            data=base64.b64decode(image_data)
                        )
                    )
                ])
            ]
        )
        
        response_text = response.text
        
        # Parse JSON
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0]
        
        result = json.loads(response_text.strip())
        result["image_path"] = image_path
        result["filename"] = os.path.basename(image_path)
        
        print(f"      Score: {result.get('total_score', 0)}/100")
        
        # Add delay to avoid Gemini rate limiting
        time.sleep(2)
        
        return result
        
    except Exception as e:
        print(f"      ✗ Error evaluating: {e}")
        return {
            "image_path": image_path,
            "filename": os.path.basename(image_path),
            "total_score": 0,
            "error": str(e)
        }


def select_best_image(image_folder: str) -> dict:
    """
    Evaluate all images in a folder and select the best one for 3D conversion.
    
    Returns dict with:
        - best_image_path: path to the selected image
        - all_evaluations: list of all image evaluations
        - selection_reason: why this image was chosen
    """
    print(f"\n🔍 STEP 2: IMAGE QUALITY SELECTION")
    print(f"   Folder: {image_folder}")
    
    # Find all images in folder
    image_extensions = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}
    images = []
    
    for f in os.listdir(image_folder):
        ext = Path(f).suffix.lower()
        if ext in image_extensions:
            images.append(os.path.join(image_folder, f))
    
    if not images:
        print("   ✗ No images found in folder!")
        return None
    
    print(f"   Found {len(images)} candidate images")
    print(f"   Analyzing each for 3D reconstruction quality...\n")
    
    # Evaluate each image
    evaluations = []
    for img_path in images:
        eval_result = evaluate_image_for_3d(img_path)
        evaluations.append(eval_result)
    
    # Sort by score (highest first)
    evaluations.sort(key=lambda x: x.get("total_score", 0), reverse=True)
    
    best = evaluations[0]
    
    print(f"\n   🏆 BEST IMAGE SELECTED:")
    print(f"      File: {best['filename']}")
    print(f"      Score: {best.get('total_score', 0)}/100")
    print(f"      Reason: {best.get('recommendation', 'Highest overall score')}")
    
    if len(evaluations) > 1:
        print(f"\n   📋 All Rankings:")
        for i, ev in enumerate(evaluations, 1):
            print(f"      {i}. {ev['filename']}: {ev.get('total_score', 0)}/100")
    
    return {
        "best_image_path": best["image_path"],
        "best_score": best.get("total_score", 0),
        "all_evaluations": evaluations,
        "selection_reason": best.get("recommendation", "Highest overall score")
    }


def select_best_from_list(image_paths: list) -> dict:
    """
    Evaluate a list of specific image paths and select the best one.
    
    Args:
        image_paths: List of full paths to images to evaluate
        
    Returns dict with best image info
    """
    print(f"\n🔍 STEP 2: IMAGE QUALITY SELECTION")
    print(f"   Evaluating {len(image_paths)} provided images...")
    
    if not image_paths:
        print("   ✗ No images provided!")
        return None
    
    # Evaluate each image
    evaluations = []
    for img_path in image_paths:
        if os.path.exists(img_path):
            eval_result = evaluate_image_for_3d(img_path)
            evaluations.append(eval_result)
        else:
            print(f"    ⚠️ File not found: {img_path}")
    
    if not evaluations:
        print("   ✗ No valid images to evaluate!")
        return None
    
    # Sort by score (highest first)
    evaluations.sort(key=lambda x: x.get("total_score", 0), reverse=True)
    
    best = evaluations[0]
    
    print(f"\n   🏆 BEST IMAGE SELECTED:")
    print(f"      File: {best['filename']}")
    print(f"      Score: {best.get('total_score', 0)}/100")
    print(f"      Reason: {best.get('recommendation', 'Highest overall score')}")
    
    if len(evaluations) > 1:
        print(f"\n   📋 All Rankings:")
        for i, ev in enumerate(evaluations, 1):
            status = "✓ SELECTED" if i == 1 else ""
            print(f"      {i}. {ev['filename']}: {ev.get('total_score', 0)}/100 {status}")
    
    return {
        "best_image_path": best["image_path"],
        "best_score": best.get("total_score", 0),
        "all_evaluations": evaluations,
        "selection_reason": best.get("recommendation", "Highest overall score")
    }


# ============================================================================
# STEP 3: COMPONENT IMAGE GENERATION (Clean images optimized for 3D)
# Uses: Gemini 3 Pro Image Preview with new SDK format
# ============================================================================

import mimetypes

def save_binary_file(file_name: str, data: bytes) -> str:
    """Save binary data to file."""
    with open(file_name, "wb") as f:
        f.write(data)
    return file_name


def generate_component_image(component: dict, output_dir: str, reference_image: str, max_retries: int = 3) -> str:
    """
    Generate a clean reference image for a component using Gemini 3 Pro Image Preview.
    Uses the NEW google-genai SDK format with streaming and inline_data extraction.
    
    Args:
        component: Component dict with name, prompts, etc.
        output_dir: Where to save generated image
        reference_image: Original best-selected image as reference
        max_retries: Number of retry attempts
        
    Returns:
        Path to generated image or None
    """
    name = component.get("component_name", "component")
    comp_id = component.get("component_id", 0)
    prompt_data = component.get("image_generation_prompt", {})
    
    main_prompt = prompt_data.get("main_prompt", f"A {name}")
    style = prompt_data.get("style_modifiers", "Product photography, studio lighting, gray background")
    material = component.get("material_suggestion", "")
    shape = component.get("shape_description", "")
    
    # Create prompts of varying complexity for retry attempts
    prompts = [
        # Attempt 1: Full detailed prompt
        f"""Generate a clean, isolated image of this specific jewelry component: {name}

COMPONENT DETAILS:
{main_prompt}
Material: {material}
Shape: {shape}

CRITICAL REQUIREMENTS FOR 3D RECONSTRUCTION:
- Single isolated object ONLY - just the {name}, nothing else
- Perfectly centered in frame
- Neutral mid-gray solid background (RGB: 128, 128, 128)
- Professional studio lighting with soft shadows
- 3/4 view angle showing depth and dimension
- High resolution, extremely sharp details
- NO text, watermarks, logos, or extra objects
- Photorealistic product photography style

{style}

Use the attached reference image to understand the exact style. Generate ONLY the {name} component isolated.""",

        # Attempt 2: Simplified prompt
        f"""Generate a product photo of a {name} for jewelry.
- Isolated object on gray background
- Studio lighting, 3/4 view angle
- Photorealistic, high detail
- Material: {material if material else 'gold metal'}
Based on the reference image style.""",

        # Attempt 3: Minimal prompt
        f"""Create a professional product photo of: {name}
Gray background, studio lighting, centered, isolated object.
Jewelry photography style."""
    ]
    
    print(f"  🎨 Generating image with Gemini 3 Pro Image Preview for: {name}")
    
    safe_name = "".join(c if c.isalnum() else "_" for c in name)
    
    for attempt in range(min(max_retries, len(prompts))):
        try:
            current_prompt = prompts[attempt]
            
            if attempt > 0:
                print(f"    ↻ Retry {attempt + 1}/{max_retries} with simplified prompt...")
                time.sleep(2)  # Brief pause between retries
            
            # Build contents with reference image using new SDK format
            parts = []
            
            # Add the reference image first (inline base64)
            if reference_image and os.path.exists(reference_image):
                ref_data = load_image_as_base64(reference_image)
                ref_mime = get_mime_type(reference_image)
                parts.append(
                    types.Part.from_bytes(
                        mime_type=ref_mime,
                        data=base64.b64decode(ref_data)
                    )
                )
                parts.append(types.Part.from_text(text=f"Reference image above. {current_prompt}"))
            else:
                parts.append(types.Part.from_text(text=current_prompt))
            
            contents = [
                types.Content(
                    role="user",
                    parts=parts
                )
            ]
            
            # Configure image generation with new SDK format
            generate_content_config = types.GenerateContentConfig(
                response_modalities=["IMAGE", "TEXT"],
                image_config=types.ImageConfig(
                    image_size="1K",  # 1024x1024 output
                ),
            )
            
            # Generate with Gemini 3 Pro Image Preview using streaming
            filepath = None
            file_index = 0
            
            for chunk in client.models.generate_content_stream(
                model="gemini-3-pro-image-preview",  # Gemini 3 Pro Image Preview
                contents=contents,
                config=generate_content_config,
            ):
                # Check for valid candidate content
                if (
                    chunk.candidates is None
                    or chunk.candidates[0].content is None
                    or chunk.candidates[0].content.parts is None
                ):
                    continue
                
                # Check for image data in the response
                part = chunk.candidates[0].content.parts[0]
                if part.inline_data and part.inline_data.data:
                    inline_data = part.inline_data
                    data_buffer = inline_data.data
                    file_extension = mimetypes.guess_extension(inline_data.mime_type) or '.png'
                    
                    filepath = os.path.join(output_dir, f"component_{comp_id}_{safe_name}{file_extension}")
                    save_binary_file(filepath, data_buffer)
                    print(f"    ✓ Saved: {filepath}")
                    
                    # Add delay after successful image generation
                    print(f"    ⏳ Waiting 3s before next generation...")
                    time.sleep(3)
                    
                    return filepath
                else:
                    # Text response (might be explanation)
                    if hasattr(chunk, 'text') and chunk.text:
                        print(f"    📝 Model response: {chunk.text[:100]}...")
            
            # If we got here without returning, no image was generated
            print(f"    ⚠️ No image in response for attempt {attempt + 1}")
            
        except Exception as e:
            print(f"    ⚠️ Attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                import traceback
                traceback.print_exc()
    
    print(f"    ✗ No image generated for {name} after {max_retries} attempts")
    return None


# ============================================================================
# STEP 4A: 3D GENERATION (HUNYUAN3D-2 via Huggingface Space API)
# ============================================================================

# Huggingface Space client (lazy loaded)
hunyuan_client = None


def init_hunyuan3d_api():
    """Initialize Hunyuan3D-2 via Huggingface Spaces API."""
    global hunyuan_client
    
    if hunyuan_client is not None:
        return True
    
    print("\n🔧 Connecting to Hunyuan3D-2 Huggingface Space...")
    
    try:
        from gradio_client import Client
        
        # Check for Huggingface token for more ZeroGPU quota
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        
        if hf_token:
            print("  🔑 Using Huggingface token for authentication")
            hunyuan_client = Client("tencent/Hunyuan3D-2", token=hf_token)
        else:
            print("  ℹ️ No HF_TOKEN found - using anonymous access (limited quota)")
            print("    💡 Set HF_TOKEN env var for more ZeroGPU quota")
            hunyuan_client = Client("tencent/Hunyuan3D-2")
        
        print("  ✓ Connected to tencent/Hunyuan3D-2 Space")
        print("  ✓ Using cloud GPU for high-quality generation")
        return True
        
    except Exception as e:
        print(f"  ✗ Failed to connect to Hunyuan3D-2 Space: {e}")
        print("  💡 Install gradio_client: pip install gradio_client")
        return False


def generate_3d_hunyuan(image_path: str, component_name: str, output_dir: str) -> dict:
    """
    Generate 3D model using Hunyuan3D-2 via Huggingface Space API.
    
    This uses the official tencent/Hunyuan3D-2 Space which produces
    much better results than local inference.
    """
    global hunyuan_client
    
    if hunyuan_client is None:
        if not init_hunyuan3d_api():
            return None
    
    print(f"  🎲 Generating 3D with Hunyuan3D-2 API: {component_name}")
    
    start = time.time()
    
    try:
        from gradio_client import handle_file
        import shutil
        
        # Call the shape_generation endpoint
        # Parameters: image, steps, guidance_scale, seed, octree_resolution, etc.
        result = hunyuan_client.predict(
            caption=None,                    # Optional text caption
            image=handle_file(image_path),   # Input image
            mv_image_front=None,             # Optional multi-view images
            mv_image_back=None,
            mv_image_left=None,
            mv_image_right=None,
            steps=30,                        # Generation steps (more = better quality)
            guidance_scale=5.0,              # CFG scale
            seed=1234,                       # Random seed
            octree_resolution=256,           # Resolution (higher = more detail)
            check_box_rembg=True,            # Remove background
            num_chunks=8000,                 # Mesh chunks
            randomize_seed=True,             # Randomize seed for variety
            api_name="/shape_generation"
        )
        
        # Result tuple: (file_dict, html_output, mesh_stats, seed)
        # file_dict format: {'value': '/path/to/mesh.glb', '__type__': 'update'}
        file_result = result[0] if isinstance(result, (list, tuple)) else result
        
        # Handle dict result format from Gradio
        if isinstance(file_result, dict):
            model_file = file_result.get('value') or file_result.get('path') or file_result.get('name')
        else:
            model_file = file_result
        
        safe_name = "".join(c if c.isalnum() else "_" for c in component_name)
        glb_path = os.path.join(output_dir, f"{safe_name}_3d.glb")
        obj_path = os.path.join(output_dir, f"{safe_name}_3d.obj")
        
        print(f"    📦 Source file: {model_file}")
        
        if model_file and isinstance(model_file, str) and os.path.exists(model_file):
            # Copy the result to our output directory
            shutil.copy(model_file, glb_path)
            print(f"    ✓ Downloaded: {glb_path}")
            
            # Convert to OBJ using trimesh
            try:
                import trimesh
                mesh = trimesh.load(glb_path)
                mesh.export(obj_path)
                print(f"    ✓ Converted to OBJ: {obj_path}")
            except Exception as e:
                print(f"    ⚠️ Could not convert to OBJ: {e}")
                obj_path = None
        else:
            print(f"    ✗ No model file returned")
            return None
        
        elapsed = time.time() - start
        print(f"    ✓ Generated in {elapsed:.1f}s")
        
        # Log mesh stats if available
        if isinstance(result, (list, tuple)) and len(result) > 2:
            mesh_stats = result[2]
            if isinstance(mesh_stats, dict):
                faces = mesh_stats.get('number_of_faces', 'N/A')
                verts = mesh_stats.get('number_of_vertices', 'N/A')
                gen_time = mesh_stats.get('time', {}).get('total', 'N/A')
                print(f"    📊 Mesh: {verts} vertices, {faces} faces")
                if gen_time != 'N/A':
                    print(f"    ⏱️ Cloud generation time: {gen_time:.1f}s")
        
        return {
            "component_name": component_name,
            "glb_path": glb_path if os.path.exists(glb_path) else None,
            "obj_path": obj_path if obj_path and os.path.exists(obj_path) else None,
            "generation_time": elapsed
        }
        
    except Exception as e:
        print(f"    ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================================
# STEP 3B: 3D GENERATION (MESHY - FALLBACK)
# ============================================================================

def generate_3d_meshy(image_path: str, component_name: str) -> dict:
    """Generate 3D model using Meshy API (fallback)."""
    if not MESHY_API_KEY:
        print("  ✗ MESHY_API_KEY not configured")
        return None
    
    print(f"  🎲 Generating 3D with Meshy: {component_name}")
    
    with open(image_path, 'rb') as f:
        image_b64 = base64.b64encode(f.read()).decode('utf-8')
    
    mime_type = get_mime_type(image_path)
    data_uri = f"data:{mime_type};base64,{image_b64}"
    
    headers = {
        "Authorization": f"Bearer {MESHY_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "image_url": data_uri,
        "ai_model": "latest",
        "enable_pbr": True,
        "should_remesh": True,
        "topology": "quad",
        "target_polycount": 30000
    }
    
    try:
        response = requests.post(
            "https://api.meshy.ai/openapi/v1/image-to-3d",
            headers=headers, json=payload, timeout=60
        )
        
        if response.status_code in [200, 201, 202]:
            result = response.json()
            task_id = result.get("result")
            print(f"    ✓ Task created: {task_id}")
            return {"task_id": task_id, "component_name": component_name}
        else:
            print(f"    ✗ API error: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"    ✗ Error: {e}")
        return None


def wait_for_meshy_tasks(tasks: list, output_dir: str, max_wait: int = 30) -> list:
    """Wait for Meshy tasks and download results."""
    if not tasks:
        return []
    
    print("\n⏳ Waiting for Meshy 3D generation...")
    
    completed = []
    pending = tasks.copy()
    start = time.time()
    
    while pending and (time.time() - start) < max_wait * 60:
        for task in pending[:]:
            headers = {"Authorization": f"Bearer {MESHY_API_KEY}"}
            
            try:
                resp = requests.get(
                    f"https://api.meshy.ai/openapi/v1/image-to-3d/{task['task_id']}",
                    headers=headers, timeout=30
                )
                
                if resp.status_code == 200:
                    status = resp.json()
                    progress = status.get("progress", 0)
                    state = status.get("status", "UNKNOWN")
                    
                    print(f"  {task['component_name']}: {state} ({progress}%)")
                    
                    if state == "SUCCEEDED":
                        glb_url = status.get("model_urls", {}).get("glb")
                        if glb_url:
                            safe_name = "".join(c if c.isalnum() else "_" for c in task['component_name'])
                            glb_path = os.path.join(output_dir, f"{safe_name}_3d.glb")
                            
                            glb_resp = requests.get(glb_url, timeout=120)
                            with open(glb_path, 'wb') as f:
                                f.write(glb_resp.content)
                            
                            task["glb_path"] = glb_path
                            print(f"    ✓ Downloaded: {glb_path}")
                        
                        completed.append(task)
                        pending.remove(task)
                    
                    elif state in ["FAILED", "EXPIRED"]:
                        pending.remove(task)
                        
            except Exception as e:
                print(f"  Error checking {task['component_name']}: {e}")
        
        if pending:
            time.sleep(10)
    
    return completed


# ============================================================================
# STEP 5: SURFACE TESSELLATION (Connected Grid for CAD)
# ============================================================================

def tessellate_mesh(mesh_path: str, output_dir: str, square_size: float = 0.5, 
                    num_regions: int = 6, method: str = "matrix",
                    shell_thickness: float = 0.5, subdivisions: int = 2) -> dict:
    """
    Wrap 3D mesh with parametric grid skin.
    
    Methods:
    - 'matrix': Complete quad mesh wrapping the ENTIRE surface (like The Matrix)
                Creates a continuous connected skin of squares covering everything
    - 'connected': Shared vertices between adjacent quads with region grouping
    
    Matrix skin creates:
    - Complete surface coverage (every part of the 3D model is wrapped)
    - Connected quad faces (all squares share edges with neighbors)
    - Hollow shell export (inner + outer skin with thickness)
    - Wireframe version (edges as tubes)
    
    This allows CAD designers to:
    - See the entire 3D shape as an editable mesh of connected squares
    - Remove the solid and keep just the skin/shell
    - Edit for mold manufacturing
    """
    print(f"  📐 Tessellating mesh: {Path(mesh_path).name}")
    print(f"     Method: {method}")
    
    if method == "matrix":
        print(f"     Shell thickness: {shell_thickness}")
        print(f"     Subdivisions: {subdivisions}")
        print("     → Creating COMPLETE connected quad skin wrapping ENTIRE surface")
    else:
        print(f"     Square size: {square_size}")
        print(f"     Method: connected (shared vertices, region grouping)")
    
    try:
        from mesh_to_cad_grid import process_mesh
        
        outputs = process_mesh(
            mesh_path=mesh_path,
            output_dir=output_dir,
            square_size=square_size,
            method=method,
            num_regions=num_regions,
            export_formats=['3dm', 'obj', 'json', 'gh'],
            shell_thickness=shell_thickness,
            subdivisions=subdivisions
        )
        
        if method == "matrix":
            print(f"    ✓ MATRIX SKIN created - entire surface wrapped with connected squares!")
            print(f"    ✓ Shell thickness: {shell_thickness}")
            if outputs:
                if 'matrix_skin' in outputs:
                    print(f"    → Matrix shell: {outputs['matrix_skin']}")
                if 'wireframe_skin' in outputs:
                    print(f"    → Wireframe: {outputs['wireframe_skin']}")
                if 'quad_skin' in outputs:
                    print(f"    → Quad skin: {outputs['quad_skin']}")
        else:
            print(f"    ✓ Connected grid created with {square_size} unit squares")
            print(f"    ✓ Surface divided into {num_regions} editable regions")
        return outputs
        
    except ImportError:
        print("    ⚠️ mesh_to_cad_grid not available, skipping tessellation")
        return None
    except Exception as e:
        print(f"    ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_pipeline(image_folder: str, output_dir: str = None, 
                 square_size: float = 0.5, num_regions: int = 6,
                 use_meshy: bool = False):
    """
    Run the complete manufacturing pipeline.
    
    Args:
        image_folder: Path to folder containing reference images (e.g., design_samples/)
        output_dir: Output directory (auto-generated if None)
        square_size: Size of grid squares for tessellation
        num_regions: Number of surface regions for CAD grouping
        use_meshy: Use Meshy API instead of Hunyuan3D-2
    """
    print("\n" + "="*70)
    print("🏭 JEWELRY MANUFACTURING PIPELINE")
    print("="*70)
    
    # Setup output directory
    if output_dir is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "manufacturing_output",
            timestamp
        )
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n📁 Output: {output_dir}")
    print(f"📂 Input folder: {image_folder}")
    
    # ========== STEP 1: Image Quality Selection ==========
    print("\n" + "-"*70)
    print("STEP 1: IMAGE QUALITY SELECTION (Gemini 3 Pro Preview)")
    print("-"*70)
    
    selection_result = select_best_image(image_folder)
    
    if not selection_result:
        print("❌ No suitable images found!")
        return None
    
    best_image_path = selection_result["best_image_path"]
    
    # Save selection analysis
    selection_path = os.path.join(output_dir, "image_selection.json")
    with open(selection_path, 'w', encoding='utf-8') as f:
        json.dump(selection_result, f, indent=2)
    print(f"\n✓ Selection analysis saved to: {selection_path}")
    
    # ========== STEP 2: Component Detection (Optional - for metadata) ==========
    print("\n" + "-"*70)
    print("STEP 2: COMPONENT DETECTION (Gemini 3 Pro Preview)")
    print("-"*70)
    
    components_result = detect_components(best_image_path)
    
    # Save analysis
    analysis_path = os.path.join(output_dir, "component_analysis.json")
    with open(analysis_path, 'w', encoding='utf-8') as f:
        json.dump(components_result, f, indent=2)
    
    product_name = components_result.get("product_name", "product")
    components = components_result.get("components", [])
    
    print(f"\n✓ Product identified: {product_name}")
    print(f"✓ Detected {len(components)} components:")
    for c in components:
        print(f"  • {c.get('component_id')}: {c.get('component_name')} ({c.get('component_type')})")
    
    # ========== STEP 3: Component Image Generation ==========
    print("\n" + "-"*70)
    print("STEP 3: COMPONENT IMAGE GENERATION (Gemini 3 Pro Image Preview)")
    print("-"*70)
    print(f"   Reference image: {os.path.basename(best_image_path)}")
    print(f"   Generating clean images for {len(components)} components...")
    
    generated_images = []
    images_dir = os.path.join(output_dir, "component_images")
    os.makedirs(images_dir, exist_ok=True)
    
    for component in components:
        img_path = generate_component_image(component, images_dir, best_image_path)
        if img_path:
            generated_images.append({
                "component": component,
                "image_path": img_path
            })
    
    print(f"\n✓ Generated {len(generated_images)} component images")
    
    if not generated_images:
        print("❌ No component images generated!")
        return None
    
    # ========== STEP 4: 3D Generation ==========
    print("\n" + "-"*70)
    print(f"STEP 4: 3D GENERATION ({'Meshy' if use_meshy else 'Hunyuan3D-2'})")
    print("-"*70)
    print(f"   Converting {len(generated_images)} component images to 3D...")
    
    generated_3d = []
    models_dir = os.path.join(output_dir, "3d_models")
    os.makedirs(models_dir, exist_ok=True)
    
    if use_meshy:
        # Use Meshy API
        meshy_tasks = []
        for item in generated_images:
            task = generate_3d_meshy(
                item["image_path"], 
                item["component"]["component_name"]
            )
            if task:
                meshy_tasks.append(task)
        
        if meshy_tasks:
            completed = wait_for_meshy_tasks(meshy_tasks, models_dir)
            generated_3d = completed
    else:
        # Use Hunyuan3D-2
        for item in generated_images:
            result = generate_3d_hunyuan(
                item["image_path"],
                item["component"]["component_name"],
                models_dir
            )
            if result:
                generated_3d.append(result)
    
    print(f"\n✓ Generated {len(generated_3d)} 3D models")
    
    if not generated_3d:
        print("❌ No 3D models generated!")
        return None
    
    # ========== STEP 5: Surface Tessellation ==========
    print("\n" + "-"*70)
    print("STEP 5: SURFACE TESSELLATION (Hollow Shell Grid)")
    print("-"*70)
    print(f"  Creating MATRIX SKIN - complete quad mesh wrapping ENTIRE surface!")
    print(f"  Like The Matrix - continuous connected grid covering everything!")
    print(f"  Shell thickness: 0.5")
    print(f"  This creates a hollow skin you can edit in CAD for mold manufacturing!")
    
    tessellated = []
    for model in generated_3d:
        mesh_path = model.get("glb_path") or model.get("obj_path")
        if mesh_path and os.path.exists(mesh_path):
            grid_output_dir = os.path.join(output_dir, "grids", Path(mesh_path).stem)
            os.makedirs(grid_output_dir, exist_ok=True)
            result = tessellate_mesh(
                mesh_path, grid_output_dir, 
                square_size=square_size, 
                num_regions=num_regions,
                method="matrix",  # Use Matrix skin for complete surface coverage!
                shell_thickness=0.5,
                subdivisions=2
            )
            if result:
                tessellated.append({
                    "component_name": model["component_name"],
                    "grid_outputs": result,
                    # Matrix skin outputs
                    "matrix_skin": result.get("matrix_skin", None),
                    "wireframe_skin": result.get("wireframe_skin", None),
                    "quad_skin": result.get("quad_skin", None)
                })
    
    print(f"\n✓ Created Matrix skins for {len(tessellated)} meshes")
    
    # ========== SUMMARY ==========
    print("\n" + "="*70)
    print("✅ PIPELINE COMPLETE")
    print("="*70)
    print(f"\n📁 Output directory: {output_dir}")
    print(f"\n📊 Results:")
    print(f"   Images evaluated: {len(selection_result['all_evaluations'])}")
    print(f"   Best image: {os.path.basename(best_image_path)} ({selection_result['best_score']}/100)")
    print(f"   Components detected: {len(components)}")
    print(f"   Component images generated: {len(generated_images)}")
    print(f"   3D models created: {len(generated_3d)}")
    print(f"   Meshes tessellated: {len(tessellated)}")
    
    if tessellated:
        print(f"\n📐 HOLLOW SHELL CAD FILES (Carved-out grid - no solid inside):")
        for t in tessellated:
            shell_path = t.get("hollow_shell")
            if shell_path:
                print(f"   🎯 {t['component_name']}: {shell_path}")
            outputs = t.get("grid_outputs", {})
            if outputs:
                if outputs.get("3dm"):
                    print(f"      Full grid: {outputs['3dm']}")
    
    print(f"\n🔧 FOR CAD DESIGNERS - The hollow shell is your design!")
    print(f"   The carved-out square grid wrapping the surface IS the CAD design")
    print(f"   • No need to manually tessellate the surface")
    print(f"   • Adjust frame thickness in Rhino")
    print(f"   • Modify individual square frames as needed")
    print(f"   • Export directly to manufacturing (CNC/laser/3D print)")
    
    print(f"\n🔧 Next steps for CAD designer:")
    print(f"   1. Open .3dm files in Rhino")
    print(f"   2. Adjust square dimensions using Grasshopper script")
    print(f"   3. Input client specifications")
    print(f"   4. Export for mold manufacturing")
    
    # Save pipeline summary
    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "input_folder": image_folder,
        "output_directory": output_dir,
        "square_size": square_size,
        "3d_engine": "meshy" if use_meshy else "hunyuan3d-2",
        "image_selection": {
            "best_image": best_image_path,
            "best_score": selection_result["best_score"],
            "total_evaluated": len(selection_result["all_evaluations"]),
            "reason": selection_result["selection_reason"]
        },
        "product_name": product_name,
        "components_detected": len(components),
        "component_images": [item["image_path"] for item in generated_images],
        "generated_3d": generated_3d,
        "tessellated": tessellated
    }
    
    summary_path = os.path.join(output_dir, "pipeline_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, default=str)
    
    return summary


def run_from_existing(existing_output_dir: str, square_size: float = 0.01, 
                      num_regions: int = 6, use_meshy: bool = False):
    """
    Run from an existing output folder, skipping already-completed steps.
    
    - Skips Gemini calls (Steps 1-3) - uses existing component images
    - Skips 3D generation (Step 4) if 3D models already exist
    - Always runs tessellation (Step 5) with current settings
    
    Args:
        existing_output_dir: Path to existing output folder (e.g., manufacturing_output/20260111_071245)
        square_size: Size of grid squares for tessellation
        num_regions: Number of surface regions for CAD grouping
        use_meshy: Use Meshy API instead of Hunyuan3D-2
    """
    print("\n" + "="*70)
    print("🏭 JEWELRY MANUFACTURING PIPELINE (FROM EXISTING)")
    print("="*70)
    print(f"\n📁 Using existing output: {existing_output_dir}")
    
    # Check for existing 3D models first
    models_dir = os.path.join(existing_output_dir, "3d_models")
    existing_3d_models = []
    
    if os.path.isdir(models_dir):
        for f in os.listdir(models_dir):
            if f.endswith('.glb') or f.endswith('.obj'):
                model_path = os.path.join(models_dir, f)
                # Extract component name from filename
                name = Path(f).stem
                if name.endswith('_3d'):
                    name = name[:-3]  # Remove _3d suffix
                name = name.replace("_", " ")
                
                # Prefer GLB over OBJ for same model
                if f.endswith('.glb'):
                    existing_3d_models.append({
                        "component_name": name,
                        "glb_path": model_path,
                        "obj_path": model_path.replace('.glb', '.obj') if os.path.exists(model_path.replace('.glb', '.obj')) else None
                    })
    
    # Remove duplicates (keep GLB version)
    seen_names = set()
    unique_models = []
    for m in existing_3d_models:
        if m['component_name'] not in seen_names:
            seen_names.add(m['component_name'])
            unique_models.append(m)
    existing_3d_models = unique_models
    
    # If we have existing 3D models, skip to tessellation
    if existing_3d_models:
        print("   ⏩ Skipping Steps 1-4 (using existing 3D models)")
        print(f"\n✓ Found {len(existing_3d_models)} existing 3D models:")
        for model in existing_3d_models:
            print(f"   • {model['component_name']}: {os.path.basename(model['glb_path'] or model.get('obj_path', 'N/A'))}")
        
        generated_3d = existing_3d_models
    else:
        # Need to generate 3D models from component images
        print("   ⏩ Skipping Gemini Steps 1-3 (using existing component images)")
        
        # Find existing component images
        images_dir = os.path.join(existing_output_dir, "component_images")
        
        if not os.path.isdir(images_dir):
            print(f"❌ Component images folder not found: {images_dir}")
            return None
        
        # Get all images
        image_extensions = {'.png', '.jpg', '.jpeg', '.webp'}
        component_images = []
        
        for f in os.listdir(images_dir):
            ext = Path(f).suffix.lower()
            if ext in image_extensions:
                img_path = os.path.join(images_dir, f)
                # Extract component name from filename
                name = Path(f).stem
                if name.startswith("component_"):
                    # Parse "component_1_Gold_Ring_Mounting" -> "Gold Ring Mounting"
                    parts = name.split("_", 2)
                    if len(parts) >= 3:
                        name = parts[2].replace("_", " ")
                
                component_images.append({
                    "component": {"component_name": name, "component_id": len(component_images) + 1},
                    "image_path": img_path
                })
        
        if not component_images:
            print(f"❌ No component images found in {images_dir}")
            return None
        
        print(f"\n✓ Found {len(component_images)} existing component images:")
        for item in component_images:
            print(f"   • {item['component']['component_name']}: {os.path.basename(item['image_path'])}")
        
        # ========== STEP 4: 3D Generation ==========
        print("\n" + "-"*70)
        print(f"STEP 4: 3D GENERATION ({'Meshy' if use_meshy else 'Hunyuan3D-2'})")
        print("-"*70)
        print(f"   Converting {len(component_images)} component images to 3D...")
        
        generated_3d = []
        os.makedirs(models_dir, exist_ok=True)
        
        if use_meshy:
            meshy_tasks = []
            for item in component_images:
                task = generate_3d_meshy(
                    item["image_path"], 
                    item["component"]["component_name"]
                )
                if task:
                    meshy_tasks.append(task)
            
            if meshy_tasks:
                completed = wait_for_meshy_tasks(meshy_tasks, models_dir)
                generated_3d = completed
        else:
            for item in component_images:
                result = generate_3d_hunyuan(
                    item["image_path"],
                    item["component"]["component_name"],
                    models_dir
                )
                if result:
                    generated_3d.append(result)
        
        print(f"\n✓ Generated {len(generated_3d)} 3D models")
        
        if not generated_3d:
            print("❌ No 3D models generated!")
            return None
    
    # ========== STEP 5: Surface Tessellation ==========
    print("\n" + "-"*70)
    print("STEP 5: SURFACE TESSELLATION (Matrix Skin)")
    print("-"*70)
    print(f"  Creating MATRIX SKIN - complete quad mesh wrapping ENTIRE surface!")
    print(f"  Like The Matrix - continuous connected grid covering everything!")
    print(f"  Shell thickness: 0.5")
    print(f"  This creates a hollow skin you can edit in CAD for mold manufacturing!")
    
    tessellated = []
    for model in generated_3d:
        mesh_path = model.get("glb_path") or model.get("obj_path")
        if mesh_path and os.path.exists(mesh_path):
            grid_output_dir = os.path.join(existing_output_dir, "grids", Path(mesh_path).stem)
            os.makedirs(grid_output_dir, exist_ok=True)
            result = tessellate_mesh(
                mesh_path, grid_output_dir, 
                square_size=square_size, 
                num_regions=num_regions,
                method="matrix",  # Use Matrix skin for complete surface coverage!
                shell_thickness=0.5,
                subdivisions=2
            )
            if result:
                tessellated.append({
                    "component_name": model["component_name"],
                    "grid_outputs": result,
                    # Matrix skin outputs
                    "matrix_skin": result.get("matrix_skin", None),
                    "wireframe_skin": result.get("wireframe_skin", None),
                    "quad_skin": result.get("quad_skin", None)
                })
    
    print(f"\n✓ Created Matrix skins for {len(tessellated)} meshes")
    
    # ========== SUMMARY ==========
    print("\n" + "="*70)
    print("✅ PIPELINE COMPLETE (FROM EXISTING)")
    print("="*70)
    print(f"\n📁 Output directory: {existing_output_dir}")
    print(f"\n📊 Results:")
    print(f"   3D models used: {len(generated_3d)}")
    print(f"   Meshes tessellated: {len(tessellated)}")
    
    if tessellated:
        print(f"\n📐 HOLLOW SHELL CAD FILES (Carved-out grid - no solid inside):")
        for t in tessellated:
            shell_path = t.get("hollow_shell")
            if shell_path:
                print(f"   🎯 {t['component_name']}: {shell_path}")
            outputs = t.get("grid_outputs", {})
            if outputs:
                if outputs.get("obj"):
                    print(f"      Grid OBJ: {outputs['obj']}")
    
    print(f"\n🔧 FOR CAD DESIGNERS - The hollow shell is your design!")
    print(f"   The carved-out square grid wrapping the surface IS the CAD design")
    print(f"   • Open the _hollow_shell.obj files in your CAD software")
    print(f"   • Each square is a frame (hollow, not solid)")
    print(f"   • Adjust frame thickness as needed")
    print(f"   • Export directly to manufacturing")
    
    return {
        "output_dir": existing_output_dir,
        "generated_3d": generated_3d,
        "tessellated": tessellated
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Complete Jewelry Manufacturing Pipeline",
        epilog="""
Examples:
  python manufacturing_pipeline.py ./design_samples/
  python manufacturing_pipeline.py ./design_samples/ --square-size 0.3
  python manufacturing_pipeline.py ./design_samples/ --num-regions 8
  python manufacturing_pipeline.py ./design_samples/ --use-meshy
  
  # Skip Gemini, use existing component images:
  python manufacturing_pipeline.py --from-existing ./manufacturing_output/20260111_071245
        """
    )
    parser.add_argument("image_folder", nargs="?", help="Path to folder containing reference images")
    parser.add_argument("-o", "--output", help="Output directory")
    parser.add_argument("-s", "--square-size", type=float, default=0.01,
                        help="Grid square size - smaller = more squares (default: 0.01 for millions)")
    parser.add_argument("-r", "--num-regions", type=int, default=6,
                        help="Number of surface regions for CAD grouping (default: 6)")
    parser.add_argument("--high-density", action="store_true",
                        help="Use very high density grid (millions of tiny squares)")
    parser.add_argument("--use-meshy", action="store_true",
                        help="Use Meshy API instead of Hunyuan3D-2")
    parser.add_argument("--from-existing", metavar="DIR",
                        help="Skip Gemini steps, use existing component images from this output folder")
    
    args = parser.parse_args()
    
    # Use tiny squares if high-density mode
    square_size = args.square_size
    if args.high_density:
        square_size = 0.005  # Very tiny squares = millions of frames
        print("🔥 HIGH DENSITY MODE: Creating millions of tiny square frames!")
    
    # Check if running from existing folder
    if args.from_existing:
        if not os.path.isdir(args.from_existing):
            print(f"ERROR: Folder not found: {args.from_existing}")
            sys.exit(1)
        
        run_from_existing(
            existing_output_dir=os.path.abspath(args.from_existing),
            square_size=square_size,
            num_regions=args.num_regions,
            use_meshy=args.use_meshy
        )
    else:
        # Regular full pipeline
        if not args.image_folder:
            print("ERROR: Please provide image_folder or use --from-existing")
            parser.print_help()
            sys.exit(1)
        
        if not os.path.isdir(args.image_folder):
            print(f"ERROR: Folder not found: {args.image_folder}")
            sys.exit(1)
        
        run_pipeline(
            image_folder=os.path.abspath(args.image_folder),
            output_dir=args.output,
            square_size=square_size,
            num_regions=args.num_regions,
            use_meshy=args.use_meshy
        )


if __name__ == "__main__":
    main()
