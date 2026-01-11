"""
Gemini Component Structure Detector with 3D Generation Pipeline

This script:
1. Analyzes product images using Gemini 3 Pro Preview to detect components
2. Generates detailed image prompts for each component
3. Creates reference images using Gemini image generation
4. Sends images to Meshy 6 to create 3D models
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

# Load environment variables from .env file
load_dotenv()

# Get API keys from environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MESHY_API_KEY = os.getenv("MESHY_API_KEY")

if not GOOGLE_API_KEY:
    print("ERROR: GOOGLE_API_KEY not found in environment variables.")
    print("Please set GOOGLE_API_KEY in your .env file or as an environment variable.")
    sys.exit(1)

if not MESHY_API_KEY:
    print("WARNING: MESHY_API_KEY not found. 3D generation will be skipped.")
    print("Please set MESHY_API_KEY in your .env file to enable 3D model generation.")

# The media_resolution parameter is currently only available in the v1alpha API version.
client = genai.Client(api_key=GOOGLE_API_KEY, http_options={'api_version': 'v1alpha'})

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
- This is optimized for Meshy 3D reconstruction which works best with mid-gray backgrounds

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
        # Fall back to first image
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


def create_meshy_3d_task(image_path: str, component_name: str) -> dict:
    """
    Send an image to Meshy API to create a 3D model using Meshy 6.
    Returns the task information.
    """
    if not MESHY_API_KEY:
        print("  ✗ MESHY_API_KEY not configured, skipping 3D generation")
        return None
    
    print(f"\nCreating Meshy 3D task for: {component_name}")
    
    # Load image and convert to base64 data URI
    with open(image_path, 'rb') as f:
        image_data = base64.b64encode(f.read()).decode('utf-8')
    
    mime_type = get_mime_type(image_path)
    data_uri = f"data:{mime_type};base64,{image_data}"
    
    # Create the Meshy task using Meshy 6 (latest)
    headers = {
        "Authorization": f"Bearer {MESHY_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "image_url": data_uri,
        "ai_model": "latest",  # Meshy 6 Preview
        "enable_pbr": True,
        "should_remesh": True,
        "should_texture": True,
        "save_pre_remeshed_model": True,
        "topology": "quad",
        "target_polycount": 30000
    }
    
    try:
        response = requests.post(
            "https://api.meshy.ai/openapi/v1/image-to-3d",
            headers=headers,
            json=payload,
            timeout=60
        )
        
        if response.status_code in [200, 201, 202]:
            result = response.json()
            task_id = result.get("result")
            print(f"  ✓ Meshy task created: {task_id}")
            return {"task_id": task_id, "component_name": component_name}
        else:
            print(f"  ✗ Meshy API error: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        print(f"  ✗ Error creating Meshy task: {e}")
        return None


def check_meshy_task_status(task_id: str) -> dict:
    """
    Check the status of a Meshy 3D generation task.
    """
    if not MESHY_API_KEY:
        return None
    
    headers = {
        "Authorization": f"Bearer {MESHY_API_KEY}"
    }
    
    try:
        response = requests.get(
            f"https://api.meshy.ai/openapi/v1/image-to-3d/{task_id}",
            headers=headers,
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error checking task status: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"Error checking task status: {e}")
        return None


def wait_for_meshy_tasks(tasks: list, output_dir: str, max_wait_minutes: int = 30) -> list:
    """
    Wait for all Meshy tasks to complete and download the results.
    """
    if not tasks:
        return []
    
    print(f"\n{'='*60}")
    print("Waiting for Meshy 3D generation to complete...")
    print(f"{'='*60}")
    
    completed_tasks = []
    pending_tasks = tasks.copy()
    start_time = time.time()
    max_wait_seconds = max_wait_minutes * 60
    
    while pending_tasks and (time.time() - start_time) < max_wait_seconds:
        for task in pending_tasks[:]:
            task_id = task["task_id"]
            status = check_meshy_task_status(task_id)
            
            if status:
                progress = status.get("progress", 0)
                task_status = status.get("status", "UNKNOWN")
                
                print(f"  {task['component_name']}: {task_status} ({progress}%)")
                
                if task_status == "SUCCEEDED":
                    task["result"] = status
                    task["model_urls"] = status.get("model_urls", {})
                    completed_tasks.append(task)
                    pending_tasks.remove(task)
                    
                    # Download the GLB file
                    glb_url = status.get("model_urls", {}).get("glb")
                    if glb_url:
                        try:
                            safe_name = "".join(c if c.isalnum() else "_" for c in task['component_name'])
                            glb_path = os.path.join(output_dir, f"{safe_name}_3d.glb")
                            
                            glb_response = requests.get(glb_url, timeout=120)
                            with open(glb_path, 'wb') as f:
                                f.write(glb_response.content)
                            
                            task["local_glb_path"] = glb_path
                            print(f"    ✓ Downloaded: {glb_path}")
                        except Exception as e:
                            print(f"    ✗ Error downloading GLB: {e}")
                    
                elif task_status in ["FAILED", "CANCELED"]:
                    task["error"] = status.get("task_error", {}).get("message", "Unknown error")
                    completed_tasks.append(task)
                    pending_tasks.remove(task)
                    print(f"    ✗ Task failed: {task.get('error')}")
        
        if pending_tasks:
            time.sleep(10)  # Wait 10 seconds before checking again
    
    if pending_tasks:
        print(f"\nWarning: {len(pending_tasks)} tasks did not complete within {max_wait_minutes} minutes")
        for task in pending_tasks:
            task["error"] = "Timeout"
            completed_tasks.append(task)
    
    return completed_tasks


def list_available_images(folder_path: str) -> list:
    """List all image files in the specified folder."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp'}
    images = []
    
    if os.path.exists(folder_path):
        for file in os.listdir(folder_path):
            if Path(file).suffix.lower() in image_extensions:
                images.append(os.path.join(folder_path, file))
    
    return images


def main():
    """Main function to run the full pipeline."""
    script_dir = Path(__file__).parent
    default_image_folder = script_dir / "design_samples"
    
    # Check command line arguments
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        available_images = list_available_images(str(default_image_folder))
        
        if not available_images:
            print(f"No images found in {default_image_folder}")
            print("Usage: python gemini_component_to_3d.py <image_path>")
            sys.exit(1)
        
        print("\nAvailable images in design_samples folder:")
        print("-" * 50)
        for i, img in enumerate(available_images, 1):
            print(f"{i}. {Path(img).name}")
        print("-" * 50)
        
        try:
            choice = input("\nEnter the number of the image to analyze (or press Enter for first image): ").strip()
            if choice == "":
                choice = 1
            else:
                choice = int(choice)
            
            if 1 <= choice <= len(available_images):
                image_path = available_images[choice - 1]
            else:
                print("Invalid choice. Using first image.")
                image_path = available_images[0]
        except ValueError:
            print("Invalid input. Using first image.")
            image_path = available_images[0]
    
    # Create output directory for this run
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = script_dir / "generated_components" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    
    try:
        # Gather all reference images from design_samples folder
        reference_images = list_available_images(str(default_image_folder))
        
        # Step 0: Select the best reference image for 3D conversion
        print("\n" + "=" * 60)
        print("STEP 0: SELECTING BEST REFERENCE IMAGE FOR 3D CONVERSION")
        print("=" * 60)
        
        best_reference = select_best_reference_image(reference_images)
        
        if best_reference:
            # Save the best image selection analysis
            selection_file = output_dir / "best_image_selection.json"
            with open(selection_file, 'w') as f:
                json.dump(best_reference, f, indent=2, default=str)
            print(f"\nBest image selection saved to: {selection_file}")
        
        # Step 1: Detect components using the best reference image
        print("\n" + "=" * 60)
        print("STEP 1: COMPONENT DETECTION")
        print("=" * 60)
        
        # Use the best reference image for component detection
        analysis_image = best_reference["best_image_path"] if best_reference else image_path
        result = detect_components(analysis_image)
        
        # Save component analysis
        analysis_file = output_dir / "component_analysis.json"
        with open(analysis_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nComponent analysis saved to: {analysis_file}")
        
        # Check if we got valid components
        components = result.get("components", [])
        if not components:
            print("No components detected. Exiting.")
            return
        
        print(f"\nDetected {len(components)} components:")
        for comp in components:
            print(f"  - {comp.get('component_id')}: {comp.get('component_name')}")
        
        # Step 2: Generate images for each component using best reference
        print("\n" + "=" * 60)
        print("STEP 2: IMAGE GENERATION (with best reference image)")
        print("=" * 60)
        
        if best_reference:
            print(f"Using best reference: {Path(best_reference['best_image_path']).name}")
        
        generated_images = []
        for component in components:
            image_file = generate_component_image(component, str(output_dir), best_reference)
            if image_file:
                generated_images.append({
                    "component": component,
                    "image_path": image_file
                })
        
        print(f"\nGenerated {len(generated_images)} images")
        
        # Step 3: Send to Meshy for 3D generation
        print("\n" + "=" * 60)
        print("STEP 3: 3D MODEL GENERATION (MESHY 6)")
        print("=" * 60)
        
        if not MESHY_API_KEY:
            print("Skipping 3D generation - MESHY_API_KEY not configured")
        else:
            meshy_tasks = []
            for img_data in generated_images:
                task = create_meshy_3d_task(
                    img_data["image_path"],
                    img_data["component"]["component_name"]
                )
                if task:
                    task["component"] = img_data["component"]
                    meshy_tasks.append(task)
            
            if meshy_tasks:
                # Wait for tasks to complete
                completed = wait_for_meshy_tasks(meshy_tasks, str(output_dir))
                
                # Update result with 3D model info
                for task in completed:
                    comp_id = task["component"]["component_id"]
                    for comp in result["components"]:
                        if comp["component_id"] == comp_id:
                            comp["meshy_task_id"] = task.get("task_id")
                            comp["model_urls"] = task.get("model_urls", {})
                            comp["local_glb_path"] = task.get("local_glb_path")
                            if task.get("error"):
                                comp["meshy_error"] = task["error"]
                
                # Save updated results
                with open(analysis_file, 'w') as f:
                    json.dump(result, f, indent=2)
        
        # Final summary
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETE")
        print("=" * 60)
        print(f"\nOutput directory: {output_dir}")
        print(f"Component analysis: {analysis_file}")
        print(f"Generated images: {len(generated_images)}")
        
        if MESHY_API_KEY and meshy_tasks:
            successful_3d = sum(1 for t in completed if not t.get("error"))
            print(f"3D models generated: {successful_3d}/{len(meshy_tasks)}")
        
        print("\nDone!")
        return result
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error during processing: {e}")
        raise


if __name__ == "__main__":
    main()
