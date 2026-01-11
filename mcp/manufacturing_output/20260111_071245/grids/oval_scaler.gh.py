"""
OVAL SCALER - Single Grasshopper Python Component
==================================================

This script scales a mesh non-uniformly to create ovals from circles.

SETUP IN GRASSHOPPER (Rhino 8):
===============================

1. Double-click canvas → type "Python" → select "Python 3 Script"

2. Copy ALL of this code into the Python component (double-click to edit)

3. Right-click the Python component → "+" to add these INPUTS:
   - mesh      (Type Hint: Mesh)
   - scale_x   (Type Hint: float)  
   - scale_y   (Type Hint: float)
   - scale_z   (Type Hint: float)

4. Right-click → "+" to add this OUTPUT:
   - oval_mesh

5. Connect:
   - [Pipeline] → mesh input
   - [Number Slider 0.5-2.0] → scale_x
   - [Number Slider 0.5-2.0] → scale_y  
   - [Number Slider 0.5-2.0] → scale_z (or just set to 1.0)

6. Slide the sliders to make ovals!

"""

import Rhino.Geometry as rg

# Default values if not connected
if 'scale_x' not in dir() or scale_x is None:
    scale_x = 1.0
if 'scale_y' not in dir() or scale_y is None:
    scale_y = 1.0
if 'scale_z' not in dir() or scale_z is None:
    scale_z = 1.0

# Process the mesh
if 'mesh' in dir() and mesh is not None:
    # Duplicate the mesh so we don't modify the original
    oval_mesh = mesh.Duplicate()
    
    # Get the center of the mesh for scaling around center
    bbox = oval_mesh.GetBoundingBox(True)
    center_x = (bbox.Min.X + bbox.Max.X) / 2
    center_y = (bbox.Min.Y + bbox.Max.Y) / 2
    center_z = (bbox.Min.Z + bbox.Max.Z) / 2
    
    # Scale each vertex
    vertices = oval_mesh.Vertices
    for i in range(vertices.Count):
        pt = vertices[i]
        
        # Scale relative to center
        new_x = center_x + (pt.X - center_x) * scale_x
        new_y = center_y + (pt.Y - center_y) * scale_y
        new_z = center_z + (pt.Z - center_z) * scale_z
        
        vertices.SetVertex(i, new_x, new_y, new_z)
    
    # Recompute normals
    oval_mesh.Normals.ComputeNormals()
    
    # Output info
    info = f"Scaled: X={scale_x:.2f}, Y={scale_y:.2f}, Z={scale_z:.2f}"
    print(info)
else:
    oval_mesh = None
    print("No mesh connected! Connect a mesh to the 'mesh' input.")
