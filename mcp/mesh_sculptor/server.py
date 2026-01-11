"""
Mesh Sculptor Web Server

FastAPI backend for the 3D mesh sculpting tool.
Handles file uploads, mesh processing, and exports.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import trimesh
import numpy as np
import json
import os
import shutil
from pathlib import Path
from datetime import datetime
import uvicorn

app = FastAPI(title="Mesh Sculptor", description="3D Mesh Sculpting Tool for CAD Designers")

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories
BASE_DIR = Path(__file__).parent
UPLOADS_DIR = BASE_DIR / "uploads"
EXPORTS_DIR = BASE_DIR / "exports"
STATIC_DIR = BASE_DIR / "static"

UPLOADS_DIR.mkdir(exist_ok=True)
EXPORTS_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def root():
    """Serve the main sculpting interface."""
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.post("/api/upload")
async def upload_mesh(file: UploadFile = File(...)):
    """Upload a mesh file (OBJ, STL, GLB, PLY)."""
    
    # Validate file extension
    allowed_extensions = {'.obj', '.stl', '.glb', '.gltf', '.ply'}
    ext = Path(file.filename).suffix.lower()
    
    if ext not in allowed_extensions:
        raise HTTPException(400, f"Invalid file type. Allowed: {allowed_extensions}")
    
    # Save uploaded file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = "".join(c if c.isalnum() or c in '._-' else '_' for c in file.filename)
    upload_path = UPLOADS_DIR / f"{timestamp}_{safe_name}"
    
    with open(upload_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    # Load and analyze mesh
    try:
        mesh = trimesh.load(str(upload_path), force='mesh')
        
        # Get mesh info
        bounds = mesh.bounds
        dimensions = bounds[1] - bounds[0]
        center = mesh.centroid
        
        # Convert to JSON-serializable format for Three.js
        vertices = mesh.vertices.tolist()
        faces = mesh.faces.tolist()
        
        # Calculate vertex normals for smooth shading
        vertex_normals = mesh.vertex_normals.tolist()
        
        return {
            "success": True,
            "filename": file.filename,
            "stored_as": upload_path.name,
            "mesh_data": {
                "vertices": vertices,
                "faces": faces,
                "normals": vertex_normals,
            },
            "info": {
                "vertex_count": len(mesh.vertices),
                "face_count": len(mesh.faces),
                "dimensions": {
                    "width": float(dimensions[0]),
                    "height": float(dimensions[1]),
                    "depth": float(dimensions[2])
                },
                "center": center.tolist(),
                "bounds_min": bounds[0].tolist(),
                "bounds_max": bounds[1].tolist(),
                "is_watertight": mesh.is_watertight
            }
        }
        
    except Exception as e:
        raise HTTPException(500, f"Failed to process mesh: {str(e)}")


@app.post("/api/export")
async def export_mesh(data: dict):
    """Export edited mesh to file."""
    
    try:
        vertices = np.array(data["vertices"])
        faces = np.array(data["faces"])
        format_type = data.get("format", "obj")
        filename = data.get("filename", "sculpted_mesh")
        
        # Create mesh
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        mesh.fix_normals()
        
        # Generate export filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = "".join(c if c.isalnum() or c in '._-' else '_' for c in filename)
        
        if format_type == "obj":
            export_path = EXPORTS_DIR / f"{safe_name}_{timestamp}.obj"
        elif format_type == "stl":
            export_path = EXPORTS_DIR / f"{safe_name}_{timestamp}.stl"
        elif format_type == "ply":
            export_path = EXPORTS_DIR / f"{safe_name}_{timestamp}.ply"
        else:
            export_path = EXPORTS_DIR / f"{safe_name}_{timestamp}.obj"
        
        mesh.export(str(export_path))
        
        # Calculate final stats
        dimensions = mesh.bounds[1] - mesh.bounds[0]
        
        return {
            "success": True,
            "export_path": str(export_path),
            "filename": export_path.name,
            "download_url": f"/api/download/{export_path.name}",
            "info": {
                "vertex_count": len(mesh.vertices),
                "face_count": len(mesh.faces),
                "dimensions": {
                    "width": float(dimensions[0]),
                    "height": float(dimensions[1]),
                    "depth": float(dimensions[2])
                },
                "is_watertight": mesh.is_watertight
            }
        }
        
    except Exception as e:
        raise HTTPException(500, f"Export failed: {str(e)}")


@app.get("/api/download/{filename}")
async def download_file(filename: str):
    """Download exported file."""
    file_path = EXPORTS_DIR / filename
    
    if not file_path.exists():
        raise HTTPException(404, "File not found")
    
    return FileResponse(
        str(file_path),
        media_type="application/octet-stream",
        filename=filename
    )


@app.get("/api/load-sample")
async def load_sample():
    """Load the sample matrix skin mesh."""
    
    # Look for the scaled matrix skin
    sample_paths = [
        Path(__file__).parent.parent / "manufacturing_output/20260111_071245/grids/Gold_Ring_Mounting__Shank_and_Setting__3d_matrix_skin_scaled_18.0x16.7x2.3mm.obj",
        Path(__file__).parent.parent / "manufacturing_output/20260111_071245/grids/Gold_Ring_Mounting__Shank_and_Setting__3d_matrix_skin.obj",
    ]
    
    sample_path = None
    for p in sample_paths:
        if p.exists():
            sample_path = p
            break
    
    if not sample_path:
        raise HTTPException(404, "Sample mesh not found")
    
    try:
        mesh = trimesh.load(str(sample_path), force='mesh')
        
        bounds = mesh.bounds
        dimensions = bounds[1] - bounds[0]
        center = mesh.centroid
        
        return {
            "success": True,
            "filename": sample_path.name,
            "mesh_data": {
                "vertices": mesh.vertices.tolist(),
                "faces": mesh.faces.tolist(),
                "normals": mesh.vertex_normals.tolist(),
            },
            "info": {
                "vertex_count": len(mesh.vertices),
                "face_count": len(mesh.faces),
                "dimensions": {
                    "width": float(dimensions[0]),
                    "height": float(dimensions[1]),
                    "depth": float(dimensions[2])
                },
                "center": center.tolist(),
                "bounds_min": bounds[0].tolist(),
                "bounds_max": bounds[1].tolist(),
                "is_watertight": mesh.is_watertight
            }
        }
        
    except Exception as e:
        raise HTTPException(500, f"Failed to load sample: {str(e)}")


@app.get("/api/list-files")
async def list_files():
    """List available mesh files."""
    
    uploads = []
    exports = []
    
    for f in UPLOADS_DIR.iterdir():
        if f.is_file():
            uploads.append({
                "name": f.name,
                "size": f.stat().st_size,
                "modified": datetime.fromtimestamp(f.stat().st_mtime).isoformat()
            })
    
    for f in EXPORTS_DIR.iterdir():
        if f.is_file():
            exports.append({
                "name": f.name,
                "size": f.stat().st_size,
                "modified": datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
                "download_url": f"/api/download/{f.name}"
            })
    
    return {
        "uploads": uploads,
        "exports": exports
    }


if __name__ == "__main__":
    print("\n" + "="*60)
    print("🎨 MESH SCULPTOR - 3D Mesh Editing Tool")
    print("="*60)
    print("\n  Starting server at http://localhost:8000")
    print("  Open your browser to start sculpting!\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
