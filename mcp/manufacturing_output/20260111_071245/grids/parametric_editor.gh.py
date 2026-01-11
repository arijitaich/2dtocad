"""
MATRIX SKIN PARAMETRIC EDITOR - Grasshopper Python Script
==========================================================

Generic parametric controls for editing ANY 3D model's matrix skin:
- Adjust height of top/bottom regions
- Scale specific areas (width, depth)
- Modify thickness of sections
- Transform selected regions

HOW TO USE IN GRASSHOPPER:
==========================

1. SETUP:
   - Open Rhino 7/8
   - Open Grasshopper (type "Grasshopper" in command line)
   - File > Import the matrix_skin.obj file into Rhino first
   
2. CREATE THE DEFINITION:
   - Add a "Python 3" component (or GhPython)
   - Copy this entire script into the Python component
   - Right-click the component and add these inputs:
     * mesh (Mesh) - connect to the imported mesh
     * top_height_adjust (float) - Number Slider: -1.0 to 1.0
     * top_scale (float) - Number Slider: 0.5 to 2.0
     * middle_thickness (float) - Number Slider: 0.5 to 2.0
     * bottom_height_adjust (float) - Number Slider: -1.0 to 1.0
     * top_threshold (float) - Number Slider: 0.5 to 0.95 (defines "top" region)
     * bottom_threshold (float) - Number Slider: 0.05 to 0.5 (defines "bottom" region)
   
3. OUTPUTS:
   - edited_mesh: The modified mesh
   - top_points: Control points for top region (for visualization)
   - region_info: Text info about detected regions

"""

import Rhino.Geometry as rg
import rhinoscriptsyntax as rs
import math
from System.Collections.Generic import List

# ============================================================
# REGION DETECTION (Generic - works with any 3D model)
# ============================================================

def detect_regions_by_height(mesh, top_threshold=0.7, bottom_threshold=0.3):
    """
    Detect regions based on normalized height (Z-axis).
    Works with any 3D model.
    
    Args:
        mesh: The input mesh
        top_threshold: Normalized Z above this = top region (0-1)
        bottom_threshold: Normalized Z below this = bottom region (0-1)
    
    Returns dict with vertex indices for each region.
    """
    regions = {
        'top': [],       # Upper vertices
        'middle': [],    # Middle section
        'bottom': []     # Lower vertices
    }
    
    if not mesh:
        return regions
    
    vertices = mesh.Vertices
    
    # Get bounding box to normalize positions
    bbox = mesh.GetBoundingBox(True)
    min_z = bbox.Min.Z
    max_z = bbox.Max.Z
    height = max_z - min_z
    
    if height <= 0:
        return regions
    
    # Classify vertices by normalized Z position
    for i in range(vertices.Count):
        v = vertices[i]
        z_norm = (v.Z - min_z) / height  # 0 = bottom, 1 = top
        
        if z_norm >= top_threshold:
            regions['top'].append(i)
        elif z_norm <= bottom_threshold:
            regions['bottom'].append(i)
        else:
            regions['middle'].append(i)
    
    return regions


def detect_regions_by_distance(mesh, center_threshold=0.3):
    """
    Detect regions based on distance from center (for radial objects).
    
    Args:
        mesh: The input mesh
        center_threshold: Normalized distance - below this = inner region
    
    Returns dict with vertex indices for each region.
    """
    regions = {
        'inner': [],     # Close to center axis
        'outer': []      # Far from center axis
    }
    
    if not mesh:
        return regions
    
    vertices = mesh.Vertices
    bbox = mesh.GetBoundingBox(True)
    
    center_x = (bbox.Min.X + bbox.Max.X) / 2
    center_y = (bbox.Min.Y + bbox.Max.Y) / 2
    max_dist = math.sqrt((bbox.Max.X - center_x)**2 + (bbox.Max.Y - center_y)**2)
    
    if max_dist <= 0:
        return regions
    
    for i in range(vertices.Count):
        v = vertices[i]
        dist = math.sqrt((v.X - center_x)**2 + (v.Y - center_y)**2)
        dist_norm = dist / max_dist
        
        if dist_norm <= center_threshold:
            regions['inner'].append(i)
        else:
            regions['outer'].append(i)
    
    return regions


# ============================================================
# TRANSFORMATION FUNCTIONS
# ============================================================

def adjust_region_height(mesh, vertex_indices, height_adjust):
    """Move selected vertices up or down (Z-axis)."""
    if not mesh or not vertex_indices:
        return mesh
    
    new_mesh = mesh.Duplicate()
    vertices = mesh.Vertices
    new_vertices = new_mesh.Vertices
    
    for i in vertex_indices:
        pt = vertices[i]
        new_vertices.SetVertex(i, pt.X, pt.Y, pt.Z + height_adjust)
    
    new_mesh.Normals.ComputeNormals()
    return new_mesh


def scale_region_xy(mesh, vertex_indices, scale_factor):
    """Scale selected vertices in XY plane (from center)."""
    if not mesh or not vertex_indices:
        return mesh
    
    bbox = mesh.GetBoundingBox(True)
    center_x = (bbox.Min.X + bbox.Max.X) / 2
    center_y = (bbox.Min.Y + bbox.Max.Y) / 2
    
    new_mesh = mesh.Duplicate()
    vertices = mesh.Vertices
    new_vertices = new_mesh.Vertices
    
    for i in vertex_indices:
        pt = vertices[i]
        dx = pt.X - center_x
        dy = pt.Y - center_y
        new_x = center_x + dx * scale_factor
        new_y = center_y + dy * scale_factor
        new_vertices.SetVertex(i, new_x, new_y, pt.Z)
    
    new_mesh.Normals.ComputeNormals()
    return new_mesh


def scale_region_uniform(mesh, vertex_indices, scale_factor):
    """Scale selected vertices uniformly from their centroid."""
    if not mesh or not vertex_indices:
        return mesh
    
    vertices = mesh.Vertices
    
    # Calculate centroid of selected vertices
    sum_x, sum_y, sum_z = 0, 0, 0
    for i in vertex_indices:
        pt = vertices[i]
        sum_x += pt.X
        sum_y += pt.Y
        sum_z += pt.Z
    
    n = len(vertex_indices)
    centroid_x = sum_x / n
    centroid_y = sum_y / n
    centroid_z = sum_z / n
    
    new_mesh = mesh.Duplicate()
    new_vertices = new_mesh.Vertices
    
    for i in vertex_indices:
        pt = vertices[i]
        new_x = centroid_x + (pt.X - centroid_x) * scale_factor
        new_y = centroid_y + (pt.Y - centroid_y) * scale_factor
        new_z = centroid_z + (pt.Z - centroid_z) * scale_factor
        new_vertices.SetVertex(i, new_x, new_y, new_z)
    
    new_mesh.Normals.ComputeNormals()
    return new_mesh


def get_region_points(mesh, vertex_indices, sample_count=100):
    """Get sample points from a region for visualization."""
    if not mesh or not vertex_indices:
        return []
    
    vertices = mesh.Vertices
    points = []
    
    # Sample evenly from the region
    step = max(1, len(vertex_indices) // sample_count)
    for idx in range(0, len(vertex_indices), step):
        i = vertex_indices[idx]
        pt = vertices[i]
        points.append(rg.Point3d(pt.X, pt.Y, pt.Z))
    
    return points


# ============================================================
# MAIN EXECUTION
# ============================================================

# Default values if inputs not connected
if 'top_height_adjust' not in dir():
    top_height_adjust = 0.0
if 'top_scale' not in dir():
    top_scale = 1.0
if 'middle_thickness' not in dir():
    middle_thickness = 1.0
if 'bottom_height_adjust' not in dir():
    bottom_height_adjust = 0.0
if 'top_threshold' not in dir():
    top_threshold = 0.7
if 'bottom_threshold' not in dir():
    bottom_threshold = 0.3

# Process mesh if available
if 'mesh' in dir() and mesh:
    # Detect regions by height
    regions = detect_regions_by_height(mesh, top_threshold, bottom_threshold)
    
    # Apply adjustments
    edited_mesh = mesh.Duplicate()
    
    # Adjust top region
    if top_height_adjust != 0:
        edited_mesh = adjust_region_height(edited_mesh, regions['top'], top_height_adjust)
    
    if top_scale != 1.0:
        edited_mesh = scale_region_xy(edited_mesh, regions['top'], top_scale)
    
    # Adjust middle region (thickness)
    if middle_thickness != 1.0:
        edited_mesh = scale_region_xy(edited_mesh, regions['middle'], middle_thickness)
    
    # Adjust bottom region
    if bottom_height_adjust != 0:
        edited_mesh = adjust_region_height(edited_mesh, regions['bottom'], bottom_height_adjust)
    
    # Get visualization points
    top_points = get_region_points(edited_mesh, regions['top'])
    top_points = List[rg.Point3d](top_points)
    
    # Region info
    region_info = f"""Regions detected:
  Top (Z > {top_threshold}): {len(regions['top'])} vertices
  Middle: {len(regions['middle'])} vertices
  Bottom (Z < {bottom_threshold}): {len(regions['bottom'])} vertices
  
Adjustments applied:
  Top height: {top_height_adjust:+.3f}
  Top scale: {top_scale:.2f}x
  Middle thickness: {middle_thickness:.2f}x
  Bottom height: {bottom_height_adjust:+.3f}"""
    
    print(region_info)
else:
    edited_mesh = None
    top_points = []
    region_info = "No mesh connected. Connect a mesh to the 'mesh' input."
    print(region_info)

# Outputs (connect these to Grasshopper outputs)
# a = edited_mesh
# b = top_points
# c = region_info
