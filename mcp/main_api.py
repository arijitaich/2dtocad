#!/usr/bin/env python3
"""
RhinoMCP - Simple HTTP API Server

A standalone REST API for controlling Rhino 3D without MCP dependencies.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import json
from typing import Optional, Dict, Any
import logging
import socket
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RhinoHTTPAPI")

# Create FastAPI app
app = FastAPI(
    title="Rhino 3D API",
    description="HTTP API for controlling Rhino 3D",
    version="1.0.0"
)

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rhino connection manager
class RhinoConnection:
    def __init__(self, host='localhost', port=9876):
        self.host = host
        self.port = port
        self.socket = None
        
    def connect(self):
        """Connect to Rhino script"""
        try:
            if self.socket:
                self.socket.close()
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(5)
            self.socket.connect((self.host, self.port))
            logger.info(f"Connected to Rhino at {self.host}:{self.port}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Rhino: {e}")
            self.socket = None
            return False
    
    def is_connected(self):
        """Check if connected"""
        return self.socket is not None
    
    def send_command(self, command_dict):
        """Send command to Rhino"""
        try:
            if not self.is_connected():
                if not self.connect():
                    raise Exception("Not connected to Rhino")
            
            # Send command
            message = json.dumps(command_dict) + '\n'
            self.socket.sendall(message.encode('utf-8'))
            
            # Receive response
            response_data = b''
            while True:
                chunk = self.socket.recv(4096)
                if not chunk:
                    break
                response_data += chunk
                if b'\n' in response_data:
                    break
            
            response = json.loads(response_data.decode('utf-8'))
            return response
            
        except Exception as e:
            logger.error(f"Error sending command: {e}")
            self.socket = None
            raise
    
    def disconnect(self):
        """Disconnect from Rhino"""
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
            self.socket = None

# Global connection
rhino_conn = RhinoConnection()

# Models
class CommandRequest(BaseModel):
    command: str
    parameters: Dict[str, Any] = {}

class CommandResponse(BaseModel):
    success: bool
    result: Any = None
    error: Optional[str] = None

@app.on_event("startup")
async def startup_event():
    """Try to connect on startup"""
    logger.info("Starting Rhino HTTP API...")
    time.sleep(1)  # Give Rhino client time to start
    rhino_conn.connect()

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    rhino_conn.disconnect()

@app.get("/")
async def root():
    """API information"""
    return {
        "name": "Rhino 3D HTTP API",
        "version": "1.0.0",
        "status": "online",
        "rhino_connected": rhino_conn.is_connected(),
        "docs": "/docs"
    }

@app.get("/health")
async def health():
    """Health check"""
    return {
        "status": "healthy",
        "rhino_connected": rhino_conn.is_connected()
    }

@app.post("/reconnect")
async def reconnect():
    """Reconnect to Rhino"""
    if rhino_conn.connect():
        return {"success": True, "message": "Connected to Rhino"}
    return {"success": False, "message": "Failed to connect to Rhino"}

@app.post("/execute")
async def execute_command(request: CommandRequest) -> CommandResponse:
    """Execute a command in Rhino"""
    try:
        if not rhino_conn.is_connected():
            rhino_conn.connect()
        
        result = rhino_conn.send_command({
            "command": request.command,
            **request.parameters
        })
        
        return CommandResponse(
            success=result.get("success", True),
            result=result
        )
        
    except Exception as e:
        logger.error(f"Error executing command: {e}")
        return CommandResponse(
            success=False,
            error=str(e)
        )

@app.post("/rhino/command")
async def rhino_command(code: str):
    """Execute Python code in Rhino"""
    try:
        result = rhino_conn.send_command({
            "command": "execute",
            "code": code
        })
        return {"success": True, "result": result}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/rhino/screenshot")
async def screenshot():
    """Get viewport screenshot"""
    try:
        result = rhino_conn.send_command({"command": "screenshot"})
        return {
            "success": True,
            "image": result.get("screenshot"),
            "format": "base64"
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/rhino/create/sphere")
async def create_sphere(x: float = 0, y: float = 0, z: float = 0, radius: float = 1):
    """Create a sphere in Rhino"""
    code = f"rs.AddSphere([{x}, {y}, {z}], {radius})"
    return await rhino_command(code)

@app.post("/rhino/create/box")
async def create_box(
    x: float = 0, y: float = 0, z: float = 0,
    width: float = 1, height: float = 1, depth: float = 1
):
    """Create a box in Rhino"""
    code = f"""
import rhinoscriptsyntax as rs
corner = [{x}, {y}, {z}]
opposite = [{x + width}, {y + height}, {z + depth}]
rs.AddBox([corner, opposite])
"""
    return await rhino_command(code)

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🦏 Rhino 3D HTTP API Server")
    print("="*70)
    print("\n📍 Server: http://0.0.0.0:8000")
    print("📚 Docs:   http://localhost:8000/docs")
    print("❤️  Health: http://localhost:8000/health")
    print("\n⚠️  Make sure to run rhino_mcp_client.py in Rhino first!")
    print("="*70 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
