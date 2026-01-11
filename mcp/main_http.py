#!/usr/bin/env python3
"""
RhinoMCP - HTTP/SSE Server

This script runs the RhinoMCP server over HTTP using Server-Sent Events (SSE).
This allows remote access and integration with other LLM clients.
"""

import os

# Set environment variables for SSE transport configuration
os.environ['MCP_TRANSPORT'] = 'sse'
os.environ['MCP_HOST'] = '0.0.0.0'
os.environ['MCP_PORT'] = '8000'

from rhino_mcp.server import app

if __name__ == "__main__":
    print("Starting RhinoMCP HTTP server...")
    print("Server will be available at: http://0.0.0.0:8000")
    print("SSE endpoint: http://0.0.0.0:8000/sse")
    print("\nYou can connect to this server from:")
    print("  - Local machine: http://localhost:8000/sse")
    print("  - Network: http://<your-ip>:8000/sse")
    
    # Run with SSE transport
    app.run(transport='sse')
