"""
Mesh Solidifier - Convert surface mesh to watertight solid shell

This module takes a surface mesh (like the matrix skin) and converts it
to a proper watertight/manifold mesh that CAD software can use.
"""

import numpy as np
import trimesh
from pathlib import Path
from typing import Dict, Optional


def solidify_mesh(input_path: str, 
                  thickness: float = 0.8,
                  direction: str = "inward",
                  output_path: str = None) -> Dict:
    """
    Convert a surface mesh to a watertight solid shell.
    
    Takes any mesh (triangles or quads) and extrudes it by thickness
    to create a proper solid that CAD software can use.
    
    Args:
        input_path: Path to input mesh (.obj, .stl, .glb, etc.)
        thickness: Shell wall thickness in mesh units
        direction: "inward", "outward", or "both"
        output_path: Output file path (default: adds _solid suffix)
        
    Returns:
        Dictionary with results and output path
    """
    print(f"\n{'='*60}")
    print("MESH SOLIDIFIER - Creating Watertight Shell")
    print(f"{'='*60}")
    print(f"  Input: {input_path}")
    print(f"  Shell thickness: {thickness}")
    print(f"  Direction: {direction}")
    
    # Load the mesh
    mesh = trimesh.load(input_path, force='mesh')
    vertices = mesh.vertices
    faces = mesh.faces
    num_verts = len(vertices)
    num_faces = len(faces)
    
    print(f"\n  Loaded mesh:")
    print(f"    Vertices: {num_verts:,}")
    print(f"    Faces: {num_faces:,}")
    print(f"    Watertight (before): {mesh.is_watertight}")
    
    # Get vertex normals for offset direction
    print(f"\n  Step 1: Computing vertex normals...")
    vertex_normals = mesh.vertex_normals
    
    # Create offset vertices
    print(f"  Step 2: Creating offset surface...")
    if direction == "inward":
        outer_verts = vertices.copy()
        inner_verts = vertices - vertex_normals * thickness
    elif direction == "outward":
        outer_verts = vertices + vertex_normals * thickness
        inner_verts = vertices.copy()
    else:  # both
        half = thickness / 2
        outer_verts = vertices + vertex_normals * half
        inner_verts = vertices - vertex_normals * half
    
    # Build solid mesh with both surfaces
    print(f"  Step 3: Building solid shell geometry...")
    
    # Combined vertices: [outer_surface, inner_surface]
    solid_vertices = np.vstack([outer_verts, inner_verts])
    
    solid_faces = []
    
    # Outer surface faces (original orientation)
    for f in faces:
        solid_faces.append([f[0], f[1], f[2]])
    
    # Inner surface faces (reversed winding for inward normal)
    for f in faces:
        i0, i1, i2 = f[0] + num_verts, f[1] + num_verts, f[2] + num_verts
        solid_faces.append([i0, i2, i1])  # Reversed winding
    
    # Find boundary edges to seal
    print(f"  Step 4: Finding boundary edges to seal...")
    
    # Count edge occurrences to find boundaries
    edge_count = {}
    edge_face_normal = {}
    
    for face_idx, f in enumerate(faces):
        for i in range(3):
            v1, v2 = f[i], f[(i + 1) % 3]
            edge = tuple(sorted([v1, v2]))
            edge_count[edge] = edge_count.get(edge, 0) + 1
            # Store which direction this edge goes in this face
            if edge not in edge_face_normal:
                edge_face_normal[edge] = (v1, v2)  # Store original order
    
    # Boundary edges appear only once
    boundary_edges = [e for e, count in edge_count.items() if count == 1]
    print(f"    Found {len(boundary_edges):,} boundary edges")
    
    # Seal each boundary edge with two triangles (quad between outer and inner)
    for edge in boundary_edges:
        v1_out, v2_out = edge
        v1_in, v2_in = v1_out + num_verts, v2_out + num_verts
        # Create quad to connect outer edge to inner edge (as 2 triangles)
        solid_faces.append([v1_out, v2_out, v2_in])
        solid_faces.append([v1_out, v2_in, v1_in])
    
    # Create final solid mesh
    print(f"  Step 5: Creating watertight mesh...")
    
    solid_mesh = trimesh.Trimesh(
        vertices=solid_vertices,
        faces=solid_faces
    )
    
    # Clean up the mesh aggressively to make it watertight
    print(f"  Step 6: Repairing mesh for watertight manifold...")
    solid_mesh.merge_vertices()
    solid_mesh.update_faces(solid_mesh.unique_faces())  # Remove duplicate faces
    solid_mesh.update_faces(solid_mesh.nondegenerate_faces())  # Remove degenerate faces
    solid_mesh.fix_normals()
    
    # Try to fill any remaining holes
    if not solid_mesh.is_watertight:
        print(f"    Filling holes...")
        solid_mesh.fill_holes()
    
    # Check results
    is_watertight = solid_mesh.is_watertight
    volume = solid_mesh.volume if is_watertight else None
    
    print(f"\n  ✓ SOLID SHELL CREATED!")
    print(f"    Vertices: {len(solid_mesh.vertices):,}")
    print(f"    Faces: {len(solid_mesh.faces):,}")
    print(f"    Watertight: {'✓ YES' if is_watertight else '✗ NO'}")
    if volume:
        print(f"    Volume: {volume:.4f} cubic units")
    print(f"    Shell thickness: {thickness}")
    
    # Determine output path
    if output_path is None:
        input_p = Path(input_path)
        output_path = str(input_p.parent / f"{input_p.stem}_solid{input_p.suffix}")
    
    # Export
    solid_mesh.export(output_path)
    print(f"\n  ✓ Exported: {output_path}")
    
    # Also export STL if not already STL
    if not output_path.lower().endswith('.stl'):
        stl_path = str(Path(output_path).with_suffix('.stl'))
        solid_mesh.export(stl_path)
        print(f"  ✓ Exported STL: {stl_path}")
    else:
        stl_path = output_path
    
    return {
        'output_path': output_path,
        'stl_path': stl_path,
        'vertices': len(solid_mesh.vertices),
        'faces': len(solid_mesh.faces),
        'watertight': is_watertight,
        'volume': volume,
        'thickness': thickness,
        'boundary_edges_sealed': len(boundary_edges),
        'mesh': solid_mesh  # Return mesh object for further processing
    }


def solidify_and_scale(input_path: str,
                       thickness: float = 0.8,
                       target_width: float = None,
                       target_height: float = None,
                       target_depth: float = None,
                       material: str = "gold",
                       cost_per_gram: float = None,
                       output_dir: str = None) -> Dict:
    """
    Solidify a mesh and scale to target dimensions with cost calculation.
    
    This is the main function for manufacturing preparation:
    1. Solidify the surface mesh into watertight solid
    2. Scale to target dimensions
    3. Calculate material volume and cost
    
    Args:
        input_path: Path to surface mesh
        thickness: Shell wall thickness
        target_width: Target width in mm (X dimension)
        target_height: Target height in mm (Y dimension)  
        target_depth: Target depth in mm (Z dimension)
        material: Material name (gold, silver, platinum, brass, etc.)
        cost_per_gram: Cost per gram (uses defaults if not provided)
        output_dir: Output directory (default: same as input)
        
    Returns:
        Dictionary with all manufacturing specs
    """
    # Material defaults
    MATERIALS = {
        'gold': {'density': 19.3, 'default_cost': 65.0},
        'silver': {'density': 10.5, 'default_cost': 0.85},
        'platinum': {'density': 21.45, 'default_cost': 32.0},
        'palladium': {'density': 12.0, 'default_cost': 45.0},
        'brass': {'density': 8.5, 'default_cost': 0.05},
        'bronze': {'density': 8.9, 'default_cost': 0.08},
        'copper': {'density': 8.96, 'default_cost': 0.01},
        'steel': {'density': 7.85, 'default_cost': 0.005},
        'titanium': {'density': 4.5, 'default_cost': 0.15},
        'resin': {'density': 1.2, 'default_cost': 0.10},
        'wax': {'density': 0.9, 'default_cost': 0.05},
    }
    
    mat_info = MATERIALS.get(material.lower(), {'density': 10.0, 'default_cost': 1.0})
    density = mat_info['density']
    if cost_per_gram is None:
        cost_per_gram = mat_info['default_cost']
    
    print(f"\n{'='*60}")
    print("SOLIDIFY AND SCALE FOR MANUFACTURING")
    print(f"{'='*60}")
    
    # Load the input mesh first
    input_mesh = trimesh.load(input_path, force='mesh')
    
    # Step 1: Scale FIRST (before solidifying)
    # This way the solidification thickness is in the final units
    uniform_scale = 1.0
    if target_width or target_height or target_depth:
        print(f"\n  Step 1: Scaling to target dimensions...")
        
        # Get current dimensions
        current_dims = input_mesh.bounds[1] - input_mesh.bounds[0]
        
        # Calculate scale factors
        scale_factors = [1.0, 1.0, 1.0]
        specified = []
        
        if target_width is not None:
            scale_factors[0] = target_width / current_dims[0]
            specified.append(scale_factors[0])
        if target_height is not None:
            scale_factors[1] = target_height / current_dims[1]
            specified.append(scale_factors[1])
        if target_depth is not None:
            scale_factors[2] = target_depth / current_dims[2]
            specified.append(scale_factors[2])
        
        # Use uniform scale (average of specified)
        if specified:
            uniform_scale = np.mean(specified)
            scale_factors = [uniform_scale, uniform_scale, uniform_scale]
        
        # Apply scaling
        input_mesh.vertices = input_mesh.vertices * scale_factors
        
        new_dims = input_mesh.bounds[1] - input_mesh.bounds[0]
        print(f"    Scale factor: {uniform_scale:.4f}")
        print(f"    New dimensions: {new_dims[0]:.2f} x {new_dims[1]:.2f} x {new_dims[2]:.2f} mm")
        
        # Save the scaled mesh to a temp file
        scaled_path = str(Path(input_path).parent / f"_temp_scaled.obj")
        input_mesh.export(scaled_path)
        input_to_solidify = scaled_path
    else:
        input_to_solidify = input_path
    
    # Step 2: Solidify with the desired thickness (now in final mm units)
    print(f"\n  Step 2: Solidifying with {thickness}mm wall thickness...")
    solid_result = solidify_mesh(input_to_solidify, thickness=thickness, direction="inward")
    solid_mesh = solid_result['mesh']
    
    # Clean up temp file
    if input_to_solidify != input_path:
        try:
            Path(scaled_path).unlink()
        except:
            pass
    
    # Step 3: Calculate volume and cost
    final_dims = solid_mesh.bounds[1] - solid_mesh.bounds[0]
    
    # Shell thickness stays the same - it's the physical wall thickness we want
    # (the mesh was built with the original thickness, then scaled)
    final_thickness = thickness  # Keep original thickness
    
    if solid_mesh.is_watertight:
        volume_mm3 = abs(solid_mesh.volume)
        print(f"\n  ✓ Mesh is watertight - using exact volume")
    else:
        # For shell meshes, the volume IS the thickness * surface area
        # But we need to use the single-sided surface area (outer surface only)
        # The solid mesh has both inner and outer surfaces, so divide by 2
        single_surface_area = solid_mesh.area / 2.0
        volume_mm3 = single_surface_area * final_thickness
        print(f"\n  ⚠ Mesh not perfectly watertight - using shell approximation")
        print(f"    Surface area (single side): {single_surface_area:.2f} mm²")
        print(f"    Shell thickness: {final_thickness:.2f} mm")
    
    volume_cm3 = volume_mm3 / 1000.0  # mm³ to cm³
    weight_grams = volume_cm3 * density
    total_cost = weight_grams * cost_per_gram
    
    # Determine output path
    if output_dir is None:
        output_dir = str(Path(input_path).parent)
    
    # Create filename with dimensions
    base_name = Path(input_path).stem.replace('_matrix_skin', '')
    dim_str = f"{final_dims[0]:.1f}x{final_dims[1]:.1f}x{final_dims[2]:.1f}mm"
    output_path = str(Path(output_dir) / f"{base_name}_solid_{dim_str}.obj")
    stl_path = str(Path(output_dir) / f"{base_name}_solid_{dim_str}.stl")
    
    # Export
    solid_mesh.export(output_path)
    solid_mesh.export(stl_path)
    
    # Print summary
    print(f"\n{'='*60}")
    print("📋 MANUFACTURING SPECIFICATION SUMMARY")
    print(f"{'='*60}")
    print(f"   ✓ WATERTIGHT SOLID MESH" if solid_mesh.is_watertight else "   ⚠ Shell approximation used")
    print(f"\n📐 DIMENSIONS:")
    print(f"   Width (X):     {final_dims[0]:.2f} mm")
    print(f"   Height (Y):    {final_dims[1]:.2f} mm")
    print(f"   Depth (Z):     {final_dims[2]:.2f} mm")
    print(f"   Shell thickness: {thickness:.2f} mm")
    print(f"\n📦 VOLUME:")
    print(f"   Volume: {volume_mm3:.2f} mm³")
    print(f"   Volume: {volume_cm3:.4f} cm³")
    print(f"\n⚖️  MATERIAL ({material.upper()}):")
    print(f"   Density:    {density} g/cm³")
    print(f"   Weight:     {weight_grams:.2f} grams")
    print(f"   Cost/gram:  ${cost_per_gram:.2f}")
    print(f"\n💰 ESTIMATED COST: ${total_cost:.2f}")
    print(f"\n📁 OUTPUT FILES:")
    print(f"   OBJ: {output_path}")
    print(f"   STL: {stl_path}")
    print(f"{'='*60}")
    
    return {
        'output_path': output_path,
        'stl_path': stl_path,
        'watertight': solid_mesh.is_watertight,
        'dimensions': {
            'width': float(final_dims[0]),
            'height': float(final_dims[1]),
            'depth': float(final_dims[2]),
            'thickness': thickness
        },
        'volume': {
            'mm3': volume_mm3,
            'cm3': volume_cm3
        },
        'material': {
            'name': material,
            'density': density,
            'weight_grams': weight_grams,
            'cost_per_gram': cost_per_gram
        },
        'cost': total_cost,
        'mesh': solid_mesh
    }


# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python mesh_solidifier.py <mesh_path> [thickness] [target_width]")
        print("\nExample:")
        print("  python mesh_solidifier.py ring_matrix_skin.obj 0.8 18.0")
        sys.exit(1)
    
    mesh_path = sys.argv[1]
    thickness = float(sys.argv[2]) if len(sys.argv) > 2 else 0.8
    target_width = float(sys.argv[3]) if len(sys.argv) > 3 else None
    
    result = solidify_and_scale(
        mesh_path,
        thickness=thickness,
        target_width=target_width,
        material="gold",
        cost_per_gram=65.0
    )
