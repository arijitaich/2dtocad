#!/usr/bin/env python3
"""
RhinoMCP - FastAPI HTTP Server

Alternative HTTP server implementation using FastAPI directly.
This creates a proper REST API wrapper around the MCP tools.
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import json
import asyncio
from typing import Optional, Dict, Any
import logging

# Import the app to get access to tools
from rhino_mcp.server import app as mcp_app
from rhino_mcp.rhino_tools import get_rhino_connection

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RhinoMCPHTTP")

# Create FastAPI app
api = FastAPI(
    title="Rhino MCP API",
    description="HTTP API for controlling Rhino 3D via MCP",
    version="1.0.0"
)

# Add CORS middleware
api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class ToolRequest(BaseModel):
    tool_name: str
    parameters: Dict[str, Any] = {}

class ToolResponse(BaseModel):
    success: bool
    result: Any = None
    error: Optional[str] = None

@api.get("/")
async def root():
    """Root endpoint - API info"""
    return {
        "name": "Rhino MCP API",
        "version": "1.0.0",
        "status": "online",
        "endpoints": {
            "tools": "/tools",
            "execute": "/execute",
            "health": "/health"
        }
    }

@api.get("/health")
async def health():
    """Health check endpoint"""
    rhino_conn = get_rhino_connection()
    return {
        "status": "healthy",
        "rhino_connected": rhino_conn.is_connected() if rhino_conn else False
    }

@api.get("/tools")
async def list_tools():
    """List all available MCP tools"""
    # Get tools from the MCP app
    tools_list = []
    
    # Extract tool information from mcp_app
    if hasattr(mcp_app, '_tools'):
        for tool_name, tool_func in mcp_app._tools.items():
            tools_list.append({
                "name": tool_name,
                "description": getattr(tool_func, '__doc__', 'No description available')
            })
    
    return {
        "tools": tools_list,
        "count": len(tools_list)
    }

@api.post("/execute")
async def execute_tool(request: ToolRequest) -> ToolResponse:
    """Execute a tool with given parameters"""
    try:
        logger.info(f"Executing tool: {request.tool_name} with params: {request.parameters}")
        
        # Get the tool function from mcp_app
        if not hasattr(mcp_app, '_tools') or request.tool_name not in mcp_app._tools:
            raise HTTPException(status_code=404, detail=f"Tool '{request.tool_name}' not found")
        
        tool_func = mcp_app._tools[request.tool_name]
        
        # Execute the tool
        result = await tool_func(**request.parameters)
        
        return ToolResponse(
            success=True,
            result=result
        )
        
    except Exception as e:
        logger.error(f"Error executing tool: {str(e)}")
        return ToolResponse(
            success=False,
            error=str(e)
        )

# Rhino-specific convenience endpoints
@api.post("/rhino/execute")
async def execute_rhino_command(command: str):
    """Execute a raw Rhino command"""
    try:
        rhino_conn = get_rhino_connection()
        if not rhino_conn or not rhino_conn.is_connected():
            raise HTTPException(status_code=503, detail="Rhino not connected")
        
        result = rhino_conn.send_command({"command": "execute", "code": command})
        return {"success": True, "result": result}
    except Exception as e:
        return {"success": False, "error": str(e)}

@api.get("/rhino/screenshot")
async def get_screenshot():
    """Get a screenshot from Rhino viewport"""
    try:
        rhino_conn = get_rhino_connection()
        if not rhino_conn or not rhino_conn.is_connected():
            raise HTTPException(status_code=503, detail="Rhino not connected")
        
        result = rhino_conn.send_command({"command": "screenshot"})
        return {"success": True, "image_base64": result.get("screenshot")}
    except Exception as e:
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🦏 Rhino MCP HTTP API Server")
    print("="*60)
    print("\nServer starting on: http://0.0.0.0:8000")
    print("\nEndpoints:")
    print("  📋 API Info:     http://localhost:8000")
    print("  🔧 List Tools:   http://localhost:8000/tools")
    print("  ⚡ Execute Tool: http://localhost:8000/execute")
    print("  ❤️  Health:       http://localhost:8000/health")
    print("\nDocs:")
    print("  📚 Swagger UI:   http://localhost:8000/docs")
    print("  📖 ReDoc:        http://localhost:8000/redoc")
    print("\n" + "="*60 + "\n")
    
    uvicorn.run(api, host="0.0.0.0", port=8000, log_level="info")
