"""
OpenAI GPT Integration for Rhino API

This script demonstrates how to connect GPT to your Rhino API using function calling.
"""

import openai
import requests
import json
from typing import Optional

# Configuration
RHINO_API_URL = "https://jina-emotional-isabella.ngrok-free.dev"  # Your ngrok URL
OPENAI_API_KEY = "your-openai-api-key-here"  # Add your OpenAI API key

# Initialize OpenAI client
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Define functions that GPT can call
FUNCTIONS = [
    {
        "name": "create_sphere",
        "description": "Create a sphere in Rhino 3D at specified coordinates with given radius",
        "parameters": {
            "type": "object",
            "properties": {
                "x": {"type": "number", "description": "X coordinate"},
                "y": {"type": "number", "description": "Y coordinate"},
                "z": {"type": "number", "description": "Z coordinate"},
                "radius": {"type": "number", "description": "Sphere radius"}
            },
            "required": ["radius"]
        }
    },
    {
        "name": "create_box",
        "description": "Create a box in Rhino 3D",
        "parameters": {
            "type": "object",
            "properties": {
                "x": {"type": "number", "description": "X coordinate of corner"},
                "y": {"type": "number", "description": "Y coordinate of corner"},
                "z": {"type": "number", "description": "Z coordinate of corner"},
                "width": {"type": "number", "description": "Box width"},
                "height": {"type": "number", "description": "Box height"},
                "depth": {"type": "number", "description": "Box depth"}
            },
            "required": ["width", "height", "depth"]
        }
    },
    {
        "name": "execute_rhino_code",
        "description": "Execute Python code in Rhino using rhinoscriptsyntax",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python code to execute in Rhino"}
            },
            "required": ["code"]
        }
    },
    {
        "name": "get_screenshot",
        "description": "Get a screenshot of the current Rhino viewport",
        "parameters": {
            "type": "object",
            "properties": {}
        }
    }
]

def call_rhino_api(endpoint: str, method: str = "POST", data: dict = None, params: dict = None):
    """Call the Rhino API"""
    url = f"{RHINO_API_URL}{endpoint}"
    try:
        if method == "POST":
            response = requests.post(url, json=data, params=params, timeout=30)
        else:
            response = requests.get(url, params=params, timeout=30)
        return response.json()
    except Exception as e:
        return {"success": False, "error": str(e)}

def create_sphere(x: float = 0, y: float = 0, z: float = 0, radius: float = 1):
    """Create a sphere in Rhino"""
    return call_rhino_api("/rhino/create/sphere", params={"x": x, "y": y, "z": z, "radius": radius})

def create_box(x: float = 0, y: float = 0, z: float = 0, 
               width: float = 1, height: float = 1, depth: float = 1):
    """Create a box in Rhino"""
    return call_rhino_api("/rhino/create/box", params={
        "x": x, "y": y, "z": z, 
        "width": width, "height": height, "depth": depth
    })

def execute_rhino_code(code: str):
    """Execute Python code in Rhino"""
    return call_rhino_api("/rhino/command", params={"code": code})

def get_screenshot():
    """Get Rhino viewport screenshot"""
    return call_rhino_api("/rhino/screenshot", method="GET")

# Function dispatcher
FUNCTION_MAP = {
    "create_sphere": create_sphere,
    "create_box": create_box,
    "execute_rhino_code": execute_rhino_code,
    "get_screenshot": get_screenshot
}

def chat_with_rhino(user_message: str, conversation_history: list = None):
    """
    Chat with GPT and let it control Rhino
    """
    if conversation_history is None:
        conversation_history = []
    
    # Add system message
    if not conversation_history:
        conversation_history.append({
            "role": "system",
            "content": "You are a helpful assistant that can control Rhino 3D. You can create geometry, execute Python code, and take screenshots. Always confirm what you've done."
        })
    
    # Add user message
    conversation_history.append({"role": "user", "content": user_message})
    
    # Call GPT with function calling
    response = client.chat.completions.create(
        model="gpt-4o",  # or "gpt-3.5-turbo"
        messages=conversation_history,
        functions=FUNCTIONS,
        function_call="auto"
    )
    
    message = response.choices[0].message
    
    # Check if GPT wants to call a function
    if message.function_call:
        function_name = message.function_call.name
        function_args = json.loads(message.function_call.arguments)
        
        print(f"\n🔧 GPT is calling: {function_name}")
        print(f"   Arguments: {function_args}")
        
        # Execute the function
        function_to_call = FUNCTION_MAP[function_name]
        function_result = function_to_call(**function_args)
        
        print(f"   Result: {function_result}\n")
        
        # Add function call and result to conversation
        conversation_history.append({
            "role": "assistant",
            "content": None,
            "function_call": {
                "name": function_name,
                "arguments": message.function_call.arguments
            }
        })
        conversation_history.append({
            "role": "function",
            "name": function_name,
            "content": json.dumps(function_result)
        })
        
        # Get GPT's response about the function result
        second_response = client.chat.completions.create(
            model="gpt-4o",
            messages=conversation_history
        )
        
        final_message = second_response.choices[0].message.content
        conversation_history.append({"role": "assistant", "content": final_message})
        
        return final_message, conversation_history
    
    else:
        # Regular text response
        conversation_history.append({"role": "assistant", "content": message.content})
        return message.content, conversation_history

# Example usage
if __name__ == "__main__":
    print("🦏 Rhino + GPT Interactive Session")
    print("=" * 60)
    print("Talk to GPT and it will control Rhino for you!")
    print("Type 'quit' to exit\n")
    
    conversation = None
    
    while True:
        user_input = input("You: ")
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
        
        try:
            response, conversation = chat_with_rhino(user_input, conversation)
            print(f"\nGPT: {response}\n")
        except Exception as e:
            print(f"Error: {e}\n")
