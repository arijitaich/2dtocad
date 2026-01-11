"""
Gemini Component Structure Detector

This script analyzes product images using Google's Gemini 3 Pro Preview model
to detect and identify the component structures required to build the product.
"""

from google import genai
from google.genai import types
import base64
import json
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variable
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    print("ERROR: GOOGLE_API_KEY not found in environment variables.")
    print("Please set GOOGLE_API_KEY in your .env file or as an environment variable.")
    sys.exit(1)

# The media_resolution parameter is currently only available in the v1alpha API version.
client = genai.Client(api_key=GOOGLE_API_KEY, http_options={'api_version': 'v1alpha'})

# Define the expected JSON response format
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
            "search_keywords": ["list of 3-5 specific keywords to search for reference images of this component online (e.g., 'gold ring band', 'diamond solitaire setting', 'prong mount jewelry')"],
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
3. Describe each component's shape, material, and position
4. Suggest an assembly order for manufacturing

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


def detect_components(image_path: str) -> dict:
    """
    Analyze an image using Gemini 3 Pro Preview to detect component structures.
    
    Args:
        image_path: Path to the image file to analyze
        
    Returns:
        Dictionary containing the detected components and their details
    """
    # Validate image path
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Load and encode the image
    image_data = load_image_as_base64(image_path)
    mime_type = get_mime_type(image_path)
    
    print(f"Analyzing image: {image_path}")
    print(f"MIME type: {mime_type}")
    print("Sending request to Gemini 3 Pro Preview...")
    
    # Send request to Gemini
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
    
    # Parse the response
    response_text = response.text
    
    # Try to extract JSON from the response
    try:
        # Clean up the response if it contains markdown code blocks
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
    """Main function to run the component detector."""
    # Default image folder
    script_dir = Path(__file__).parent
    default_image_folder = script_dir / "design_samples"
    
    # Check command line arguments
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # List available images and let user choose
        available_images = list_available_images(str(default_image_folder))
        
        if not available_images:
            print(f"No images found in {default_image_folder}")
            print("Usage: python gemini_component_detector.py <image_path>")
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
    
    # Detect components
    try:
        result = detect_components(image_path)
        
        # Output the result
        print("\n" + "=" * 60)
        print("COMPONENT DETECTION RESULTS")
        print("=" * 60)
        print(json.dumps(result, indent=2))
        
        # Save results to file
        output_file = Path(image_path).stem + "_components.json"
        output_path = script_dir / output_file
        
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")
        
        return result
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error during analysis: {e}")
        raise


if __name__ == "__main__":
    main()
