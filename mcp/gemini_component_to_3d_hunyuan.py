"""
Gemini Component Structure Detector with Hunyuan3D-2 3D Generation Pipeline

This script:
1. Analyzes product images using Gemini 3 Pro Preview to detect components
2. Generates detailed image prompts for each component
3. Creates reference images using Gemini image generation
4. Uses LOCAL Hunyuan3D-2 to create 3D models (instead of Meshy API)
"""

from google import genai
from google.genai import types
import base64
import json
import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv
from PIL import Image
import torch

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    print("ERROR: GOOGLE_API_KEY not found in environment variables.")
    print("Please set GOOGLE_API_KEY in your .env file or as an environment variable.")
    sys.exit(1)

# The media_resolution parameter is currently only available in the v1alpha API version.
client = genai.Client(api_key=GOOGLE_API_KEY, http_options={'api_version': 'v1alpha'})

# Hunyuan3D-2 pipeline (lazy loaded)
hunyuan_shape_pipeline = None
hunyuan_paint_pipeline = None

# Define the expected JSON response format with image generation prompts
RESPONSE_FORMAT = """
{
    "product_name": "string - Name or description of the product",
    "product_category": "string - Category of the product (e.g., jewelry, furniture, electronics, etc.)",
    "overall_description": "string - Brief description of the entire product",
    "components": [
        {
            "component_id": "integer - Unique identifier for the component",
            "component_name": "string - Name of the component",
            "component_type": "string - Type/category of component (e.g., structural, decorative, functional)",
            "image_generation_prompt": {
                "main_prompt": "string - A detailed prompt (100-200 words) for generating a high-quality reference image of this component. The prompt should describe: the exact shape, material appearance, surface finish, color, reflectivity, and any distinctive features. Optimized for 3D reconstruction.",
                "style_modifiers": "string - Style instructions: 'Product photography, studio lighting, 3/4 view angle, neutral mid-gray solid background (#808080), clean professional render, high resolution, sharp details, no shadows on background'",
                "negative_prompt": "string - Things to avoid: 'transparent background, white background, black background, multiple objects, text, watermarks, blurry, low quality, distorted proportions'",
                "camera_angle": "string - Recommended camera angle for best 3D reconstruction (e.g., '3/4 front view', 'isometric view', 'front-facing with slight tilt')",
                "lighting_description": "string - Lighting setup description for the render"
            },
            "material_suggestion": "string - Suggested material for this component",
            "shape_description": "string - Geometric description of the component shape",
            "dimensions_estimate": {
                "relative_size": "string - small/medium/large relative to the product",
                "shape_type": "string - e.g., cylindrical, rectangular, spherical, irregular"
            },
            "position": "string - Where this component is located in the product",
            "connections": ["list of component_ids this part connects to"],
            "manufacturing_notes": "string - Notes on how this might be manufactured or assembled"
        }
    ],
    "assembly_order": ["list of component_ids in suggested assembly sequence"],
    "complexity_rating": "string - simple/moderate/complex",
    "additional_notes": "string - Any other relevant observations"
}
"""

# Master prompt for component detection
MASTER_PROMPT = f"""You are an expert product designer and manufacturing engineer. Analyze the provided image and detect all the component structures that would be required to build this product.

Your task is to:
1. Identify the product shown in the image
2. Break down the product into its individual components
3. For each component, create a detailed image generation prompt that would produce a high-quality reference image suitable for 3D model reconstruction
4. The image prompts should describe components as if they are being photographed for a product catalog

IMPORTANT IMAGE PROMPT GUIDELINES:
- Each component's image prompt must describe a SINGLE isolated object on a neutral mid-gray solid background
- Include specific details about material appearance (metallic sheen, matte finish, transparency, etc.)
- Describe the exact shape, proportions, and any surface details or textures
- The prompt should result in an image showing a 3/4 view angle with clean studio lighting
- This is optimized for Hunyuan3D-2 reconstruction which works best with clean backgrounds

IMPORTANT: You MUST respond with a valid JSON object following this exact format:
{RESPONSE_FORMAT}

Analyze the image carefully and provide a comprehensive breakdown of all visible and implied structural components. Be specific about shapes that could be created in 3D modeling software like Rhino/Grasshopper.

Respond ONLY with the JSON object, no additional text before or after."""


def get_mime_type(file_path: str) -> str:
    """Determine the MIME type based on file extension."""
    extension = Path(file_path).suffix.lower()
    mime_types = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.webp': 'image/webp',
        '.bmp': 'image/bmp'
    }
    return mime_types.get(extension, 'image/jpeg')


def load_image_as_base64(image_path: str) -> str:
    """Load an image file and convert it to base64."""
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


# Prompt for selecting the best reference image for 3D conversion
BEST_IMAGE_SELECTION_PROMPT = """You are an expert in image-to-3D conversion and 3D reconstruction. I am providing you with multiple reference images of the same product. 

Your task is to analyze each image and determine which ONE image is BEST suited for image-to-3D conversion based on the following criteria:

EVALUATION CRITERIA (in order of importance):
1. **Camera Angle**: 3/4 view or isometric view is ideal - shows multiple faces of the object. Front-only or side-only views are less ideal.
2. **Background**: Clean, solid, neutral backgrounds are best. Busy or complex backgrounds are problematic.
3. **Lighting**: Even, professional studio lighting with soft shadows. Harsh shadows or overexposure are bad.
4. **Object Clarity**: Sharp focus, clear edges, no motion blur. The object's shape should be clearly defined.
5. **Object Isolation**: Single object in frame with clear separation from background is ideal.
6. **Material Visibility**: Materials, textures, and surface details should be clearly visible.
7. **Completeness**: The entire object should be visible without cropping important parts.
8. **Resolution/Quality**: Higher quality images with more detail are preferred.

Analyze each provided image and respond with a JSON object in this EXACT format:
{
    "analysis": [
        {
            "image_index": 1,
            "camera_angle_score": 1-10,
            "background_score": 1-10,
            "lighting_score": 1-10,
            "clarity_score": 1-10,
            "isolation_score": 1-10,
            "material_visibility_score": 1-10,
            "completeness_score": 1-10,
            "quality_score": 1-10,
            "total_score": "sum of all scores",
            "notes": "brief explanation of strengths and weaknesses"
        }
    ],
    "best_image_index": "integer - the 1-based index of the best image for 3D conversion",
    "reasoning": "string - detailed explanation of why this image was selected as the best"
}

Respond ONLY with the JSON object, no additional text."""


def select_best_reference_image(image_paths: list) -> dict:
    """
    Analyze multiple reference images and select the best one for image-to-3D conversion.
    Returns a dict with the best image path and analysis results.
    """
    if not image_paths:
        return None
    
    if len(image_paths) == 1:
        return {
            "best_image_path": image_paths[0],
            "best_image_index": 1,
            "analysis": None,
            "reasoning": "Only one image available"
        }
    
    print(f"\nAnalyzing {len(image_paths)} images to find the best reference for 3D conversion...")
    
    # Load all images
    images = []
    for img_path in image_paths:
        try:
            img = Image.open(img_path)
            images.append(img)
            print(f"  Loaded: {Path(img_path).name}")
        except Exception as e:
            print(f"  Warning: Could not load {img_path}: {e}")
    
    if not images:
        return None
    
    # Build contents with all images
    contents = [BEST_IMAGE_SELECTION_PROMPT]
    for i, img in enumerate(images, 1):
        contents.append(f"Image {i}:")
        contents.append(img)
    
    try:
        # Send to Gemini for analysis
        response = client.models.generate_content(
            model="gemini-3-pro-preview",
            contents=contents
        )
        
        response_text = response.text
        
        # Parse JSON response
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0]
        
        result = json.loads(response_text.strip())
        
        best_index = int(result.get("best_image_index", 1)) - 1  # Convert to 0-based
        if 0 <= best_index < len(image_paths):
            best_path = image_paths[best_index]
        else:
            best_path = image_paths[0]
            best_index = 0
        
        print(f"\n  ✓ Best image selected: {Path(best_path).name}")
        print(f"    Reasoning: {result.get('reasoning', 'N/A')[:100]}...")
        
        return {
            "best_image_path": best_path,
            "best_image_index": best_index + 1,  # Return 1-based for display
            "analysis": result.get("analysis"),
            "reasoning": result.get("reasoning"),
            "all_images": image_paths
        }
        
    except json.JSONDecodeError as e:
        print(f"  Warning: Could not parse analysis response: {e}")
        return {
            "best_image_path": image_paths[0],
            "best_image_index": 1,
            "analysis": None,
            "reasoning": "Analysis failed, defaulting to first image"
        }
    except Exception as e:
        print(f"  Error during image analysis: {e}")
        return {
            "best_image_path": image_paths[0],
            "best_image_index": 1,
            "analysis": None,
            "reasoning": f"Error: {e}, defaulting to first image"
        }


def detect_components(image_path: str) -> dict:
    """
    Analyze an image using Gemini 3 Pro Preview to detect component structures.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    image_data = load_image_as_base64(image_path)
    mime_type = get_mime_type(image_path)
    
    print(f"Analyzing image: {image_path}")
    print(f"MIME type: {mime_type}")
    print("Sending request to Gemini 3 Pro Preview for component detection...")
    
    response = client.models.generate_content(
        model="gemini-3-pro-preview",
        contents=[
            types.Content(
                parts=[
                    types.Part(text=MASTER_PROMPT),
                    types.Part(
                        inline_data=types.Blob(
                            mime_type=mime_type,
                            data=base64.b64decode(image_data),
                        ),
                        media_resolution={"level": "media_resolution_high"}
                    )
                ]
            )
        ]
    )
    
    response_text = response.text
    
    try:
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0]
        
        result = json.loads(response_text.strip())
        return result
    except json.JSONDecodeError as e:
        print(f"Warning: Could not parse JSON response: {e}")
        print("Raw response:")
        print(response_text)
        return {"raw_response": response_text, "parse_error": str(e)}


def generate_component_image(component: dict, output_dir: str, best_reference: dict = None) -> str:
    """
    Generate a reference image for a component using Gemini 3 Pro Image Preview.
    Uses the best reference image (determined by prior analysis) for optimal context.
    Returns the path to the saved image.
    """
    component_name = component.get("component_name", "component")
    component_id = component.get("component_id", 0)
    image_prompt_data = component.get("image_generation_prompt", {})
    
    # Build the full prompt for image generation
    main_prompt = image_prompt_data.get("main_prompt", f"A {component_name}")
    style_modifiers = image_prompt_data.get("style_modifiers", 
        "Product photography, studio lighting, 3/4 view angle, neutral mid-gray solid background (#808080), clean professional render, high resolution, sharp details")
    
    # Build prompt with reference image context if provided
    if best_reference and best_reference.get("best_image_path"):
        full_prompt = f"""I am providing a reference image of the complete product. This reference image was selected as the BEST image for 3D reconstruction based on its camera angle, lighting, clarity, and background.

Based on this reference image, generate an image of this SPECIFIC COMPONENT only:

COMPONENT TO GENERATE: {component_name}

DETAILED COMPONENT DESCRIPTION:
{main_prompt}

INSTRUCTIONS:
1. Study the reference image carefully to understand the exact material, color, finish, and style
2. Extract ONLY the "{component_name}" component from the product
3. Generate a NEW image showing ONLY this single component isolated by itself
4. Match the exact material appearance, color tones, surface finish, and quality from the reference
5. The generated image must be optimized for image-to-3D conversion

CRITICAL REQUIREMENTS FOR 3D RECONSTRUCTION:
- Single isolated object only (just the {component_name}), centered in frame
- Neutral mid-gray solid background (RGB: 128, 128, 128) - NOT white, NOT black, NOT transparent
- Professional studio lighting with soft shadows
- 3/4 view angle (slightly rotated to show depth and form)
- High resolution with sharp, clean details
- Match the exact material appearance, color, and finish from the reference image
- No text, watermarks, logos, or additional objects
- Clean edges with good contrast against the gray background
- Photorealistic product photography style

{style_modifiers}"""
    else:
        full_prompt = f"""{main_prompt}

CRITICAL REQUIREMENTS FOR 3D RECONSTRUCTION:
- Single isolated object only, centered in frame
- Neutral mid-gray solid background (RGB: 128, 128, 128) - NOT white, NOT black, NOT transparent
- Professional studio lighting with soft shadows
- 3/4 view angle (slightly rotated to show depth and form)
- High resolution with sharp, clean details
- No text, watermarks, logos, or additional objects
- Clean edges with good contrast against the gray background
- Photorealistic product photography style

{style_modifiers}"""
    
    print(f"\nGenerating image for component {component_id}: {component_name}")
    if best_reference and best_reference.get("best_image_path"):
        print(f"  Using best reference: {Path(best_reference['best_image_path']).name}")
    print(f"  Prompt preview: {main_prompt[:60]}...")
    
    try:
        # Build contents with the best reference image
        contents = [full_prompt]
        
        if best_reference and best_reference.get("best_image_path"):
            try:
                ref_image = Image.open(best_reference["best_image_path"])
                contents.append(ref_image)
            except Exception as e:
                print(f"  Warning: Could not load reference image: {e}")
        
        # Use Gemini 3 Pro Image Preview for image generation
        response = client.models.generate_content(
            model="gemini-3-pro-image-preview",
            contents=contents,
            config=types.GenerateContentConfig(
                response_modalities=['TEXT', 'IMAGE'],
                image_config=types.ImageConfig(
                    aspect_ratio="1:1",
                    image_size="2K"
                )
            )
        )
        
        # Process the response parts
        for part in response.candidates[0].content.parts:
            # Check for image data using the as_image() method
            try:
                image = part.as_image()
                if image:
                    # Create safe filename
                    safe_name = "".join(c if c.isalnum() else "_" for c in component_name)
                    filename = f"component_{component_id}_{safe_name}.png"
                    filepath = os.path.join(output_dir, filename)
                    
                    # Save the image using PIL
                    image.save(filepath)
                    
                    print(f"  ✓ Image saved: {filepath}")
                    return filepath
            except (AttributeError, TypeError):
                # Not an image part, check for inline_data
                if hasattr(part, 'inline_data') and part.inline_data:
                    mime_type = part.inline_data.mime_type
                    if 'image' in mime_type:
                        ext = '.png' if 'png' in mime_type else '.jpg'
                        safe_name = "".join(c if c.isalnum() else "_" for c in component_name)
                        filename = f"component_{component_id}_{safe_name}{ext}"
                        filepath = os.path.join(output_dir, filename)
                        
                        with open(filepath, 'wb') as f:
                            f.write(part.inline_data.data)
                        
                        print(f"  ✓ Image saved: {filepath}")
                        return filepath
        
        print(f"  ✗ No image generated for {component_name}")
        return None
        
    except Exception as e:
        print(f"  ✗ Error generating image: {e}")
        return None


def init_hunyuan3d(model_path: str = "tencent/Hunyuan3D-2", use_turbo: bool = True, generate_texture: bool = True):
    """
    Initialize Hunyuan3D-2 pipelines.
    
    Args:
        model_path: HuggingFace model path
        use_turbo: Use the faster turbo version
        generate_texture: Whether to also load the texture generation pipeline
    """
    global hunyuan_shape_pipeline, hunyuan_paint_pipeline
    
    print("\n" + "="*60)
    print("Initializing Hunyuan3D-2 (this may take a few minutes on first run)...")
    print("="*60)
    
    # Check GPU
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  GPU: {gpu_name}")
        print(f"  VRAM: {gpu_mem:.1f} GB")
    else:
        device = "cpu"
        print("  WARNING: CUDA not available, using CPU (will be very slow)")
    
    try:
        from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
        
        # Select the appropriate subfolder based on turbo mode
        if use_turbo:
            subfolder = "hunyuan3d-dit-v2-0-turbo"
            print(f"  Loading Hunyuan3D-2 Turbo (faster)...")
        else:
            subfolder = "hunyuan3d-dit-v2-0"
            print(f"  Loading Hunyuan3D-2 Standard...")
        
        hunyuan_shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            model_path,
            subfolder=subfolder
        )
        print(f"  ✓ Shape generation model loaded")
        
        if generate_texture and gpu_mem >= 12:  # Only load texture if enough VRAM
            from hy3dgen.texgen import Hunyuan3DPaintPipeline
            hunyuan_paint_pipeline = Hunyuan3DPaintPipeline.from_pretrained(model_path)
            print(f"  ✓ Texture generation model loaded")
        elif generate_texture:
            print(f"  ⚠ Skipping texture model (requires 16GB+ VRAM)")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error initializing Hunyuan3D-2: {e}")
        return False


def create_hunyuan_3d(image_path: str, component_name: str, output_dir: str, generate_texture: bool = True) -> dict:
    """
    Generate a 3D model from an image using Hunyuan3D-2.
    
    Args:
        image_path: Path to the input image
        component_name: Name of the component
        output_dir: Directory to save the output
        generate_texture: Whether to generate textures
    
    Returns:
        Dict with paths to generated files
    """
    global hunyuan_shape_pipeline, hunyuan_paint_pipeline
    
    if hunyuan_shape_pipeline is None:
        print("  ✗ Hunyuan3D-2 not initialized. Call init_hunyuan3d() first.")
        return None
    
    print(f"\nGenerating 3D model for: {component_name}")
    print(f"  Input image: {image_path}")
    
    start_time = time.time()
    
    try:
        # Generate shape (bare mesh)
        print("  Generating 3D shape...")
        mesh = hunyuan_shape_pipeline(image=image_path)[0]
        
        shape_time = time.time() - start_time
        print(f"  ✓ Shape generated in {shape_time:.1f}s")
        
        # Generate texture if enabled and pipeline available
        if generate_texture and hunyuan_paint_pipeline is not None:
            print("  Generating texture...")
            texture_start = time.time()
            mesh = hunyuan_paint_pipeline(mesh, image=image_path)
            texture_time = time.time() - texture_start
            print(f"  ✓ Texture generated in {texture_time:.1f}s")
        
        # Save the mesh
        safe_name = "".join(c if c.isalnum() else "_" for c in component_name)
        
        # Save as GLB (with texture if available)
        glb_path = os.path.join(output_dir, f"{safe_name}_3d.glb")
        mesh.export(glb_path)
        print(f"  ✓ GLB saved: {glb_path}")
        
        # Also save as OBJ for compatibility
        obj_path = os.path.join(output_dir, f"{safe_name}_3d.obj")
        mesh.export(obj_path)
        print(f"  ✓ OBJ saved: {obj_path}")
        
        total_time = time.time() - start_time
        print(f"  Total time: {total_time:.1f}s")
        
        return {
            "component_name": component_name,
            "glb_path": glb_path,
            "obj_path": obj_path,
            "generation_time": total_time
        }
        
    except Exception as e:
        print(f"  ✗ Error generating 3D model: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main function to run the pipeline."""
    # Check command line arguments
    if len(sys.argv) < 2:
        print("Usage: python gemini_component_to_3d_hunyuan.py <image_path> [additional_image_paths...]")
        print("\nExample:")
        print("  python gemini_component_to_3d_hunyuan.py product.png")
        print("  python gemini_component_to_3d_hunyuan.py ref1.png ref2.png ref3.png")
        sys.exit(1)
    
    # Collect all image paths
    image_paths = []
    for arg in sys.argv[1:]:
        if os.path.exists(arg):
            image_paths.append(os.path.abspath(arg))
        else:
            print(f"Warning: Image not found: {arg}")
    
    if not image_paths:
        print("ERROR: No valid image paths provided")
        sys.exit(1)
    
    # Create output directory with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generated_components_hunyuan", timestamp)
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    
    # ========== STEP 0: Select Best Reference Image ==========
    print("\n" + "="*60)
    print("STEP 0: Selecting Best Reference Image")
    print("="*60)
    
    best_reference = select_best_reference_image(image_paths)
    
    if best_reference:
        # Save the selection analysis
        selection_file = os.path.join(output_dir, "best_image_selection.json")
        with open(selection_file, 'w', encoding='utf-8') as f:
            json.dump(best_reference, f, indent=2, default=str)
        print(f"\nSelection analysis saved to: {selection_file}")
    
    # Use the best image (or first if selection failed) for component detection
    primary_image = best_reference["best_image_path"] if best_reference else image_paths[0]
    
    # ========== STEP 1: Detect Components ==========
    print("\n" + "="*60)
    print("STEP 1: Detecting Components with Gemini 3 Pro Preview")
    print("="*60)
    
    result = detect_components(primary_image)
    
    # Save the component analysis
    output_file = os.path.join(output_dir, "component_analysis.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    print(f"\nComponent analysis saved to: {output_file}")
    
    # Check for parse errors
    if "parse_error" in result:
        print("\nERROR: Failed to parse component analysis. Check the output file for raw response.")
        sys.exit(1)
    
    # Print summary
    print(f"\n{'='*60}")
    print("COMPONENT ANALYSIS SUMMARY")
    print(f"{'='*60}")
    print(f"Product: {result.get('product_name', 'Unknown')}")
    print(f"Category: {result.get('product_category', 'Unknown')}")
    print(f"Complexity: {result.get('complexity_rating', 'Unknown')}")
    print(f"Components found: {len(result.get('components', []))}")
    
    components = result.get("components", [])
    if components:
        print("\nComponents:")
        for comp in components:
            print(f"  {comp.get('component_id')}: {comp.get('component_name')} ({comp.get('component_type')})")
    
    # ========== STEP 2: Generate Component Images ==========
    print("\n" + "="*60)
    print("STEP 2: Generating Component Images with Gemini")
    print("="*60)
    
    generated_images = []
    for component in components:
        image_path = generate_component_image(component, output_dir, best_reference)
        if image_path:
            generated_images.append({
                "component": component,
                "image_path": image_path
            })
    
    print(f"\nGenerated {len(generated_images)} component images")
    
    # ========== STEP 3: Generate 3D Models with Hunyuan3D-2 ==========
    print("\n" + "="*60)
    print("STEP 3: Generating 3D Models with Hunyuan3D-2")
    print("="*60)
    
    if not generated_images:
        print("No images generated, skipping 3D generation")
        return
    
    # Initialize Hunyuan3D-2
    # Note: For 12GB VRAM, we'll generate shapes only (texture requires 16GB+)
    if not init_hunyuan3d(use_turbo=True, generate_texture=False):
        print("Failed to initialize Hunyuan3D-2. Exiting.")
        return
    
    # Generate 3D models for each component
    generated_3d = []
    for item in generated_images:
        component = item["component"]
        image_path = item["image_path"]
        
        result_3d = create_hunyuan_3d(
            image_path=image_path,
            component_name=component.get("component_name", "component"),
            output_dir=output_dir,
            generate_texture=False  # Skip texture for 12GB VRAM
        )
        
        if result_3d:
            generated_3d.append(result_3d)
    
    # ========== Summary ==========
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    print(f"Output directory: {output_dir}")
    print(f"Components detected: {len(components)}")
    print(f"Images generated: {len(generated_images)}")
    print(f"3D models created: {len(generated_3d)}")
    
    if generated_3d:
        print("\nGenerated 3D models:")
        for model in generated_3d:
            print(f"  - {model['component_name']}: {model['glb_path']}")
            print(f"    Generation time: {model['generation_time']:.1f}s")
    
    # Save summary
    summary = {
        "timestamp": timestamp,
        "input_images": image_paths,
        "best_reference_image": best_reference.get("best_image_path") if best_reference else None,
        "output_directory": output_dir,
        "components_detected": len(components),
        "images_generated": len(generated_images),
        "models_created": len(generated_3d),
        "generated_3d_models": generated_3d
    }
    
    summary_file = os.path.join(output_dir, "pipeline_summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSummary saved to: {summary_file}")


if __name__ == "__main__":
    main()
