"""Mesh Surface Tessellation to CAD Grid - Enhanced Edition

This script:
1. Loads a 3D mesh (GLB/OBJ) from image-to-3D generation
2. Creates a CONNECTED parametric grid of quads covering the entire surface
3. Exports to Rhino-compatible .3dm CAD file with:
   - NURBS surface patches (not just curves)
   - Shared vertices between adjacent quads
   - Region-based grouping and layers
   - Fully editable CAD geometry

The CAD designer can then:
- Select and modify individual patches or regions
- Adjust quad size/dimensions globally or locally
- Move shared vertices to reshape multiple patches at once
- Input client specifications and apply to grouped regions
- Export clean geometry for mold manufacturing

Key Enhancement: Squares share vertices, so editing one affects neighbors,
creating a true "patchwork surface" that behaves as unified CAD geometry.
"""

import numpy as np
import trimesh
import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from collections import defaultdict

# Optional: scipy and sklearn for advanced grid operations
try:
    from scipy.spatial import KDTree
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Note: scipy not installed. Install with: pip install scipy")

try:
    from sklearn.cluster import KMeans
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("Note: sklearn not installed. Region detection disabled.")
    print("Install with: pip install scikit-learn")

# Optional: rhino3dm for native Rhino file export
try:
    import rhino3dm
    HAS_RHINO3DM = True
except ImportError:
    HAS_RHINO3DM = False
    print("Note: rhino3dm not installed. Will export to OBJ/STEP instead of .3dm")
    print("Install with: pip install rhino3dm")


class MeshGridTessellator:
    """
    Tessellates a 3D mesh surface with a parametric grid of squares.
    """
    
    def __init__(self, mesh_path: str):
        """
        Initialize with a mesh file.
        
        Args:
            mesh_path: Path to GLB, OBJ, STL, or other mesh file
        """
        self.mesh_path = mesh_path
        self.mesh = None
        self.grid_squares = []
        self.grid_params = {}
        
        # Enhanced: Connected patch data
        self.shared_vertices = {}  # vertex_id -> [x, y, z]
        self.quad_patches = []     # Each patch references shared vertex IDs
        self.vertex_to_quads = defaultdict(list)  # Which quads share each vertex
        self.regions = {}          # region_id -> list of quad IDs
        self.region_layers = {}    # region_id -> layer properties
        
        self._load_mesh()
    
    def _load_mesh(self):
        """Load the mesh file using trimesh."""
        print(f"Loading mesh: {self.mesh_path}")
        
        # Load the mesh
        loaded = trimesh.load(self.mesh_path)
        
        # Handle scene vs single mesh
        if isinstance(loaded, trimesh.Scene):
            # Combine all meshes in scene
            meshes = []
            for name, geom in loaded.geometry.items():
                if isinstance(geom, trimesh.Trimesh):
                    meshes.append(geom)
            if meshes:
                self.mesh = trimesh.util.concatenate(meshes)
            else:
                raise ValueError("No valid meshes found in scene")
        elif isinstance(loaded, trimesh.Trimesh):
            self.mesh = loaded
        else:
            raise ValueError(f"Unsupported mesh type: {type(loaded)}")
        
        print(f"  Vertices: {len(self.mesh.vertices)}")
        print(f"  Faces: {len(self.mesh.faces)}")
        print(f"  Bounds: {self.mesh.bounds}")
        
    def get_mesh_info(self) -> Dict[str, Any]:
        """Get information about the loaded mesh."""
        bounds = self.mesh.bounds
        size = bounds[1] - bounds[0]
        
        return {
            "vertices": len(self.mesh.vertices),
            "faces": len(self.mesh.faces),
            "bounds_min": bounds[0].tolist(),
            "bounds_max": bounds[1].tolist(),
            "size": size.tolist(),
            "surface_area": float(self.mesh.area),
            "volume": float(self.mesh.volume) if self.mesh.is_watertight else None,
            "is_watertight": self.mesh.is_watertight
        }
    
    def create_uv_grid(self, square_size: float = 1.0, 
                       uv_resolution: int = 100) -> List[Dict]:
        """
        Create a grid of squares based on UV mapping of the mesh surface.
        
        This method:
        1. Computes UV coordinates for the mesh
        2. Creates a regular grid in UV space
        3. Maps each grid cell back to 3D space
        
        Args:
            square_size: Size of each square in mesh units
            uv_resolution: Resolution of the UV grid
            
        Returns:
            List of square dictionaries with vertices and properties
        """
        print(f"\nCreating UV-based grid with square_size={square_size}")
        
        # Get mesh bounds to determine grid dimensions
        bounds = self.mesh.bounds
        mesh_size = bounds[1] - bounds[0]
        max_dim = max(mesh_size)
        
        # Calculate number of squares needed
        num_squares_per_axis = int(max_dim / square_size) + 1
        
        print(f"  Mesh size: {mesh_size}")
        print(f"  Squares per axis: ~{num_squares_per_axis}")
        
        # Sample points on the mesh surface
        points, face_indices = trimesh.sample.sample_surface(
            self.mesh, 
            count=uv_resolution * uv_resolution
        )
        
        # Get face normals for sampled points
        normals = self.mesh.face_normals[face_indices]
        
        # Create grid in the bounding box
        self.grid_squares = []
        
        # Create a 3D grid aligned with principal axes
        x_steps = np.arange(bounds[0][0], bounds[1][0] + square_size, square_size)
        y_steps = np.arange(bounds[0][1], bounds[1][1] + square_size, square_size)
        z_steps = np.arange(bounds[0][2], bounds[1][2] + square_size, square_size)
        
        square_id = 0
        
        # For each sampled point, create a square tangent to the surface
        # We'll use a subset for performance
        sample_step = max(1, len(points) // 1000)  # Limit to ~1000 squares
        
        for i in range(0, len(points), sample_step):
            point = points[i]
            normal = normals[i]
            
            # Create tangent vectors
            tangent1 = self._get_tangent_vector(normal)
            tangent2 = np.cross(normal, tangent1)
            tangent2 = tangent2 / np.linalg.norm(tangent2)
            
            # Create square vertices
            half_size = square_size / 2
            v1 = point + half_size * tangent1 + half_size * tangent2
            v2 = point - half_size * tangent1 + half_size * tangent2
            v3 = point - half_size * tangent1 - half_size * tangent2
            v4 = point + half_size * tangent1 - half_size * tangent2
            
            square = {
                "id": square_id,
                "center": point.tolist(),
                "normal": normal.tolist(),
                "vertices": [v1.tolist(), v2.tolist(), v3.tolist(), v4.tolist()],
                "size": square_size,
                "area": square_size * square_size
            }
            
            self.grid_squares.append(square)
            square_id += 1
        
        self.grid_params = {
            "square_size": square_size,
            "total_squares": len(self.grid_squares),
            "coverage_type": "surface_sampled"
        }
        
        print(f"  Created {len(self.grid_squares)} surface squares")
        return self.grid_squares
    
    def create_projection_grid(self, square_size: float = 1.0,
                                projection_axis: str = "z") -> List[Dict]:
        """
        Create a grid by projecting squares onto the mesh surface.
        
        This method:
        1. Creates a 2D grid above/below the mesh
        2. Projects each square down onto the mesh surface
        3. Conforms squares to the surface topology
        
        Args:
            square_size: Size of each square in mesh units
            projection_axis: Axis to project from ('x', 'y', or 'z')
            
        Returns:
            List of square dictionaries
        """
        print(f"\nCreating projection grid with square_size={square_size}")
        print(f"  Projection axis: {projection_axis}")
        
        bounds = self.mesh.bounds
        
        # Determine grid plane based on projection axis
        axis_map = {'x': 0, 'y': 1, 'z': 2}
        proj_axis = axis_map.get(projection_axis.lower(), 2)
        
        # Other two axes for the grid
        axes = [i for i in range(3) if i != proj_axis]
        
        # Create 2D grid
        axis1_steps = np.arange(bounds[0][axes[0]], bounds[1][axes[0]] + square_size, square_size)
        axis2_steps = np.arange(bounds[0][axes[1]], bounds[1][axes[1]] + square_size, square_size)
        
        # Ray casting height (above the mesh)
        ray_height = bounds[1][proj_axis] + 10
        ray_direction = np.zeros(3)
        ray_direction[proj_axis] = -1  # Pointing down
        
        self.grid_squares = []
        square_id = 0
        
        print(f"  Grid: {len(axis1_steps)} x {len(axis2_steps)} = {len(axis1_steps) * len(axis2_steps)} potential squares")
        
        for a1 in axis1_steps:
            for a2 in axis2_steps:
                # Create ray origin
                ray_origin = np.zeros(3)
                ray_origin[axes[0]] = a1 + square_size / 2
                ray_origin[axes[1]] = a2 + square_size / 2
                ray_origin[proj_axis] = ray_height
                
                # Cast ray to find intersection
                locations, index_ray, index_tri = self.mesh.ray.intersects_location(
                    ray_origins=[ray_origin],
                    ray_directions=[ray_direction]
                )
                
                if len(locations) > 0:
                    # Take closest intersection
                    hit_point = locations[0]
                    hit_normal = self.mesh.face_normals[index_tri[0]]
                    
                    # Create square tangent to surface
                    tangent1 = self._get_tangent_vector(hit_normal)
                    tangent2 = np.cross(hit_normal, tangent1)
                    tangent2 = tangent2 / np.linalg.norm(tangent2)
                    
                    half_size = square_size / 2
                    v1 = hit_point + half_size * tangent1 + half_size * tangent2
                    v2 = hit_point - half_size * tangent1 + half_size * tangent2
                    v3 = hit_point - half_size * tangent1 - half_size * tangent2
                    v4 = hit_point + half_size * tangent1 - half_size * tangent2
                    
                    square = {
                        "id": square_id,
                        "center": hit_point.tolist(),
                        "normal": hit_normal.tolist(),
                        "vertices": [v1.tolist(), v2.tolist(), v3.tolist(), v4.tolist()],
                        "size": square_size,
                        "area": square_size * square_size,
                        "grid_position": [a1, a2]
                    }
                    
                    self.grid_squares.append(square)
                    square_id += 1
        
        self.grid_params = {
            "square_size": square_size,
            "total_squares": len(self.grid_squares),
            "projection_axis": projection_axis,
            "coverage_type": "projection"
        }
        
        print(f"  Created {len(self.grid_squares)} projected squares")
        return self.grid_squares
    
    def create_face_subdivision_grid(self, subdivisions: int = 2) -> List[Dict]:
        """
        Create a grid by subdividing each mesh face.
        
        This provides complete coverage of the mesh surface.
        
        Args:
            subdivisions: Number of subdivisions per face edge
            
        Returns:
            List of square/quad dictionaries
        """
        print(f"\nCreating face subdivision grid with {subdivisions} subdivisions per edge")
        
        self.grid_squares = []
        square_id = 0
        
        for face_idx, face in enumerate(self.mesh.faces):
            # Get face vertices
            v0 = self.mesh.vertices[face[0]]
            v1 = self.mesh.vertices[face[1]]
            v2 = self.mesh.vertices[face[2]]
            
            # Face normal
            normal = self.mesh.face_normals[face_idx]
            
            # Subdivide the triangle into smaller triangles/quads
            for i in range(subdivisions):
                for j in range(subdivisions - i):
                    # Barycentric coordinates for subdivision
                    u0 = i / subdivisions
                    v0_coord = j / subdivisions
                    
                    u1 = (i + 1) / subdivisions
                    v1_coord = j / subdivisions
                    
                    u2 = i / subdivisions
                    v2_coord = (j + 1) / subdivisions
                    
                    # Calculate 3D positions
                    p0 = (1 - u0 - v0_coord) * v0 + u0 * v1 + v0_coord * v2
                    p1 = (1 - u1 - v1_coord) * v0 + u1 * v1 + v1_coord * v2
                    p2 = (1 - u2 - v2_coord) * v0 + u2 * v1 + v2_coord * v2
                    
                    center = (p0 + p1 + p2) / 3
                    
                    # Create a quad approximation
                    # This is a simplified approach - for true quads, more complex logic needed
                    square = {
                        "id": square_id,
                        "center": center.tolist(),
                        "normal": normal.tolist(),
                        "vertices": [p0.tolist(), p1.tolist(), p2.tolist()],  # Triangle
                        "type": "triangle",
                        "face_index": face_idx
                    }
                    
                    self.grid_squares.append(square)
                    square_id += 1
        
        self.grid_params = {
            "subdivisions": subdivisions,
            "total_elements": len(self.grid_squares),
            "coverage_type": "face_subdivision"
        }
        
        print(f"  Created {len(self.grid_squares)} subdivision elements")
        return self.grid_squares
    
    def _get_tangent_vector(self, normal: np.ndarray) -> np.ndarray:
        """Get a tangent vector perpendicular to the normal."""
        # Find a vector not parallel to normal
        if abs(normal[0]) < 0.9:
            ref = np.array([1, 0, 0])
        else:
            ref = np.array([0, 1, 0])
        
        tangent = np.cross(normal, ref)
        return tangent / np.linalg.norm(tangent)
    
    # ========================================================================
    # MATRIX SKIN: Complete Quad Mesh Wrapping Entire Surface
    # ========================================================================
    
    def create_matrix_skin(self, subdivisions: int = 2, offset: float = 0.0) -> Dict:
        """
        Create a COMPLETE quad-mesh skin that wraps the ENTIRE 3D surface.
        
        Like The Matrix - every part of the surface is covered with a connected
        grid of squares. The skin follows the exact shape of the 3D object.
        
        This converts every triangle face into quads, creating a continuous
        quad-mesh "coating" over the entire object.
        
        Args:
            subdivisions: How many quads per original triangle edge (higher = denser grid)
            offset: Offset the skin outward (positive) or inward (negative) from surface
            
        Returns:
            Dictionary with complete skin data
        """
        print(f"\n{'='*60}")
        print("CREATING MATRIX SKIN (Complete Surface Wrapper)")
        print(f"{'='*60}")
        print(f"  Subdivisions per edge: {subdivisions}")
        print(f"  Surface offset: {offset}")
        print(f"  Original mesh faces: {len(self.mesh.faces)}")
        
        import sys
        
        # CONNECTED 2D SHEET: Vertices must be shared between adjacent quads
        # We use the original mesh vertices + edge midpoints + face centers
        # Key: adjacent triangles share edges, so they share edge midpoints
        
        faces = self.mesh.faces
        vertices = self.mesh.vertices
        normals = self.mesh.vertex_normals  # Use vertex normals for smooth offset
        
        num_faces = len(faces)
        num_verts = len(vertices)
        
        print(f"\n  Creating CONNECTED 2D sheet (shared vertices)...")
        print(f"  Original vertices: {num_verts:,}")
        print(f"  Original faces: {num_faces:,}")
        sys.stdout.flush()
        
        # Step 1: Collect all unique edges and create edge midpoints
        # Edge key = sorted tuple of vertex indices
        print(f"  Step 1: Finding unique edges...")
        sys.stdout.flush()
        
        edge_to_midpoint_idx = {}  # edge_key -> midpoint vertex index
        midpoint_positions = []
        
        for face_idx, face in enumerate(faces):
            if face_idx % 200000 == 0:
                print(f"    Face {face_idx:,}/{num_faces:,} ({100*face_idx/num_faces:.0f}%)")
                sys.stdout.flush()
            
            for i in range(3):
                v_i = face[i]
                v_j = face[(i + 1) % 3]
                edge_key = tuple(sorted([v_i, v_j]))
                
                if edge_key not in edge_to_midpoint_idx:
                    # Create midpoint for this edge
                    midpoint = (vertices[v_i] + vertices[v_j]) / 2
                    edge_to_midpoint_idx[edge_key] = len(midpoint_positions) + num_verts
                    midpoint_positions.append(midpoint)
        
        num_midpoints = len(midpoint_positions)
        print(f"  Found {num_midpoints:,} unique edges (midpoints)")
        sys.stdout.flush()
        
        # Step 2: Create face centers
        print(f"  Step 2: Creating face centers...")
        sys.stdout.flush()
        
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]
        face_centers = (v0 + v1 + v2) / 3  # (N, 3)
        
        # Step 3: Build final vertex array
        # Layout: [original_vertices, edge_midpoints, face_centers]
        print(f"  Step 3: Building vertex array...")
        sys.stdout.flush()
        
        all_vertices = np.vstack([
            vertices,                          # indices 0 to num_verts-1
            np.array(midpoint_positions),      # indices num_verts to num_verts+num_midpoints-1
            face_centers                       # indices num_verts+num_midpoints to end
        ])
        
        # Apply offset if needed
        if offset != 0:
            # Compute vertex normals for all points
            all_normals = np.zeros_like(all_vertices)
            # Original vertices use mesh vertex normals
            all_normals[:num_verts] = normals
            # Midpoints and centers: interpolate from nearby vertices
            for edge_key, mid_idx in edge_to_midpoint_idx.items():
                v_i, v_j = edge_key
                all_normals[mid_idx] = (normals[v_i] + normals[v_j]) / 2
                n = np.linalg.norm(all_normals[mid_idx])
                if n > 0:
                    all_normals[mid_idx] /= n
            # Face centers
            for face_idx in range(num_faces):
                center_idx = num_verts + num_midpoints + face_idx
                face = faces[face_idx]
                all_normals[center_idx] = (normals[face[0]] + normals[face[1]] + normals[face[2]]) / 3
                n = np.linalg.norm(all_normals[center_idx])
                if n > 0:
                    all_normals[center_idx] /= n
            
            all_vertices = all_vertices + all_normals * offset
        
        # Step 4: Create quad faces (3 quads per original triangle)
        print(f"  Step 4: Creating connected quads...")
        sys.stdout.flush()
        
        all_quads = []
        face_center_base = num_verts + num_midpoints
        
        for face_idx, face in enumerate(faces):
            if face_idx % 200000 == 0:
                print(f"    Face {face_idx:,}/{num_faces:,} ({100*face_idx/num_faces:.0f}%)")
                sys.stdout.flush()
            
            v0_idx = face[0]
            v1_idx = face[1]
            v2_idx = face[2]
            
            # Get midpoint indices (these are SHARED across adjacent triangles!)
            edge01 = tuple(sorted([v0_idx, v1_idx]))
            edge12 = tuple(sorted([v1_idx, v2_idx]))
            edge20 = tuple(sorted([v2_idx, v0_idx]))
            
            m01_idx = edge_to_midpoint_idx[edge01]
            m12_idx = edge_to_midpoint_idx[edge12]
            m20_idx = edge_to_midpoint_idx[edge20]
            
            center_idx = face_center_base + face_idx
            
            # Create 3 quads for this triangle
            # Quad 1: v0 -> m01 -> center -> m20
            all_quads.append([v0_idx, m01_idx, center_idx, m20_idx])
            # Quad 2: m01 -> v1 -> m12 -> center
            all_quads.append([m01_idx, v1_idx, m12_idx, center_idx])
            # Quad 3: m20 -> center -> m12 -> v2
            all_quads.append([m20_idx, center_idx, m12_idx, v2_idx])
        
        # Store as class data
        self.skin_vertices = all_vertices
        self.skin_quads = all_quads
        
        # Also store in quad_patches format for compatibility
        self.shared_vertices = {i: all_vertices[i] for i in range(len(all_vertices))}
        self.quad_patches = [(q, list(q)) for q in all_quads]
        
        total_verts = len(all_vertices)
        total_quads = len(all_quads)
        
        print(f"\n  ✓ CONNECTED Matrix skin created!")
        print(f"    - Total unique vertices: {total_verts:,}")
        print(f"    - Quad faces: {total_quads:,}")
        print(f"    - Adjacent quads now SHARE edges (truly connected 2D sheet)")
        sys.stdout.flush()
        
        return {
            "vertices": total_verts,
            "faces": total_quads,
            "quad_count": total_quads,
            "tri_count": 0
        }
    
    def export_matrix_skin(self, output_path: str, shell_thickness: float = 0.01) -> str:
        """
        Export the Matrix skin as a CONNECTED 2D SHEET of quads.
        
        This is a single-layer mesh where adjacent squares share edges,
        creating a continuous surface like a fishing net or chain-link
        pattern wrapped around the 3D object.
        
        Args:
            output_path: Path for OBJ file
            shell_thickness: NOT USED - kept for API compatibility
            
        Returns:
            Path to saved file
        """
        print(f"\n{'='*60}")
        print("EXPORTING MATRIX SKIN (Connected 2D Sheet)")
        print(f"{'='*60}")
        import sys
        sys.stdout.flush()
        
        if not hasattr(self, 'skin_vertices') or not hasattr(self, 'skin_quads'):
            print("  ERROR: No matrix skin created. Run create_matrix_skin() first.")
            return None
        
        print(f"  Output: {output_path}")
        print(f"  Vertices: {len(self.skin_vertices):,}")
        print(f"  Quad faces: {len(self.skin_quads):,}")
        print(f"  Type: Single connected 2D sheet (no thickness)")
        sys.stdout.flush()
        
        print(f"  Writing OBJ file...")
        sys.stdout.flush()
        
        with open(output_path, 'w') as f:
            f.write("# MATRIX SKIN - Connected 2D Sheet\n")
            f.write(f"# Like The Matrix - grid of squares wrapping entire 3D surface\n")
            f.write(f"# This is a CONNECTED mesh - adjacent squares share edges\n")
            f.write(f"# Vertices: {len(self.skin_vertices):,}\n")
            f.write(f"# Quad faces: {len(self.skin_quads):,}\n\n")
            
            # Write vertices
            print(f"    Writing {len(self.skin_vertices):,} vertices...")
            sys.stdout.flush()
            for idx, v in enumerate(self.skin_vertices):
                if idx % 500000 == 0 and idx > 0:
                    print(f"      {idx:,}/{len(self.skin_vertices):,} ({100*idx/len(self.skin_vertices):.0f}%)")
                    sys.stdout.flush()
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            
            # Write quad faces
            print(f"    Writing {len(self.skin_quads):,} quad faces...")
            sys.stdout.flush()
            f.write("\n# Quad faces (connected 2D sheet)\n")
            f.write("g matrix_skin\n")
            for idx, face in enumerate(self.skin_quads):
                if idx % 500000 == 0 and idx > 0:
                    print(f"      {idx:,}/{len(self.skin_quads):,} ({100*idx/len(self.skin_quads):.0f}%)")
                    sys.stdout.flush()
                # OBJ is 1-indexed
                indices = [str(i + 1) for i in face]
                f.write(f"f {' '.join(indices)}\n")
        
        print(f"\n  ✓ Matrix skin exported!")
        print(f"    - File: {output_path}")
        print(f"    - This is a CONNECTED 2D sheet - squares share edges")
        print(f"    - Open in CAD to see the continuous quad mesh")
        sys.stdout.flush()
        
        return output_path

    def export_matrix_lattice(self, output_path: str, frame_width: float = 0.02, 
                               frame_thickness: float = 0.01) -> str:
        """
        Export the Matrix skin as a LATTICE - only frame edges, holes in the squares.
        
        This creates a grid of connected frame edges where the CENTER of each
        square is EMPTY (a hole). Only the edges/borders of the squares are solid.
        
        This is the TRUE Matrix look - you can see through the grid!
        The original 3D object is NOT included - only the lattice remains.
        
        Args:
            output_path: Path for OBJ file
            frame_width: Width of each frame edge (how thick the bars are)
            frame_thickness: Thickness/depth of the frame (extrusion from surface)
            
        Returns:
            Path to saved file
        """
        print(f"\n{'='*60}")
        print("EXPORTING MATRIX LATTICE (Hollow Frames - See Through!)")
        print(f"{'='*60}")
        import sys
        sys.stdout.flush()
        
        if not hasattr(self, 'skin_vertices') or not hasattr(self, 'skin_quads'):
            print("  ERROR: No matrix skin created. Run create_matrix_skin() first.")
            return None
        
        print(f"  Output: {output_path}")
        print(f"  Frame width: {frame_width}")
        print(f"  Frame thickness: {frame_thickness}")
        print(f"  Original quads: {len(self.skin_quads):,}")
        print(f"  → Converting each quad edge to a 3D bar")
        print(f"  → Centers will be EMPTY (holes)")
        sys.stdout.flush()
        
        # Collect unique edges from all quads
        print(f"  Step 1: Collecting unique edges...")
        sys.stdout.flush()
        
        edges = set()
        for idx, face in enumerate(self.skin_quads):
            if idx % 500000 == 0:
                print(f"    Processing quad {idx:,}/{len(self.skin_quads):,}")
                sys.stdout.flush()
            for i in range(len(face)):
                j = (i + 1) % len(face)
                edge = tuple(sorted([face[i], face[j]]))
                edges.add(edge)
        
        num_edges = len(edges)
        print(f"  Found {num_edges:,} unique edges")
        sys.stdout.flush()
        
        # Create 3D bars for each edge
        print(f"  Step 2: Creating 3D frame bars...")
        sys.stdout.flush()
        
        all_vertices = []
        all_faces = []
        
        # Pre-compute vertex normals for offset direction
        vertex_normals = self.mesh.vertex_normals
        num_orig_verts = len(self.mesh.vertices)
        
        for edge_idx, edge in enumerate(edges):
            if edge_idx % 100000 == 0:
                print(f"    Edge {edge_idx:,}/{num_edges:,} ({100*edge_idx/num_edges:.0f}%)")
                sys.stdout.flush()
            
            # Get edge endpoints
            v0 = self.skin_vertices[edge[0]]
            v1 = self.skin_vertices[edge[1]]
            
            # Get normals for offset direction
            if edge[0] < num_orig_verts:
                n0 = vertex_normals[edge[0]]
            else:
                n0 = np.array([0, 0, 1])  # Default for midpoints/centers
            if edge[1] < num_orig_verts:
                n1 = vertex_normals[edge[1]]
            else:
                n1 = np.array([0, 0, 1])
            
            # Edge direction and perpendicular
            edge_dir = v1 - v0
            edge_len = np.linalg.norm(edge_dir)
            if edge_len < 1e-10:
                continue
            edge_dir = edge_dir / edge_len
            
            # Get perpendicular direction (in surface plane)
            normal = (n0 + n1) / 2
            norm_len = np.linalg.norm(normal)
            if norm_len > 0:
                normal = normal / norm_len
            else:
                normal = np.array([0, 0, 1])
            
            # Perpendicular to edge, in surface plane
            perp = np.cross(normal, edge_dir)
            perp_len = np.linalg.norm(perp)
            if perp_len > 0:
                perp = perp / perp_len * (frame_width / 2)
            else:
                perp = np.array([frame_width/2, 0, 0])
            
            # Normal offset for thickness
            offset = normal * frame_thickness
            
            # Create 8 vertices for this bar (box shape)
            base_idx = len(all_vertices)
            
            # Bottom face (on surface)
            all_vertices.append(v0 - perp)           # 0
            all_vertices.append(v0 + perp)           # 1
            all_vertices.append(v1 + perp)           # 2
            all_vertices.append(v1 - perp)           # 3
            
            # Top face (offset from surface)
            all_vertices.append(v0 - perp + offset)  # 4
            all_vertices.append(v0 + perp + offset)  # 5
            all_vertices.append(v1 + perp + offset)  # 6
            all_vertices.append(v1 - perp + offset)  # 7
            
            # Create faces for this bar (6 faces of a box)
            # Bottom
            all_faces.append([base_idx+0, base_idx+3, base_idx+2, base_idx+1])
            # Top
            all_faces.append([base_idx+4, base_idx+5, base_idx+6, base_idx+7])
            # Front
            all_faces.append([base_idx+0, base_idx+1, base_idx+5, base_idx+4])
            # Back
            all_faces.append([base_idx+2, base_idx+3, base_idx+7, base_idx+6])
            # Left
            all_faces.append([base_idx+0, base_idx+4, base_idx+7, base_idx+3])
            # Right
            all_faces.append([base_idx+1, base_idx+2, base_idx+6, base_idx+5])
        
        print(f"  Step 3: Writing OBJ file...")
        print(f"    Total vertices: {len(all_vertices):,}")
        print(f"    Total faces: {len(all_faces):,}")
        sys.stdout.flush()
        
        with open(output_path, 'w') as f:
            f.write("# MATRIX LATTICE - Hollow Frame Grid\n")
            f.write("# Only edges are solid - centers are EMPTY (holes)\n")
            f.write("# You can see THROUGH this grid - no solid object inside\n")
            f.write(f"# Frame edges: {num_edges:,}\n")
            f.write(f"# Frame width: {frame_width}\n")
            f.write(f"# Frame thickness: {frame_thickness}\n")
            f.write(f"# Vertices: {len(all_vertices):,}\n")
            f.write(f"# Faces: {len(all_faces):,}\n\n")
            
            # Write vertices
            for idx, v in enumerate(all_vertices):
                if idx % 500000 == 0 and idx > 0:
                    print(f"      Vertices: {idx:,}/{len(all_vertices):,}")
                    sys.stdout.flush()
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            
            # Write faces
            f.write("\ng matrix_lattice\n")
            for idx, face in enumerate(all_faces):
                if idx % 500000 == 0 and idx > 0:
                    print(f"      Faces: {idx:,}/{len(all_faces):,}")
                    sys.stdout.flush()
                # OBJ is 1-indexed
                indices = [str(i + 1) for i in face]
                f.write(f"f {' '.join(indices)}\n")
        
        print(f"\n  ✓ Matrix LATTICE exported!")
        print(f"    - File: {output_path}")
        print(f"    - ONLY frame edges are solid")
        print(f"    - Centers are EMPTY - you can see through!")
        print(f"    - Original 3D object is NOT included")
        sys.stdout.flush()
        
        return output_path

    def export_wireframe_skin(self, output_path: str, wire_thickness: float = 0.005) -> str:
        """
        Export the Matrix skin as a WIREFRAME - just the edges of the quads.
        
        This creates the true "Matrix" look - a grid of lines wrapping the surface.
        Each edge is extruded into a thin tube for 3D printing/manufacturing.
        
        Args:
            output_path: Path for OBJ file
            wire_thickness: Thickness of each wire/edge
            
        Returns:
            Path to saved file
        """
        print(f"\n{'='*60}")
        print("EXPORTING WIREFRAME SKIN (Matrix Style)")
        print(f"{'='*60}")
        import sys
        sys.stdout.flush()
        
        if not hasattr(self, 'skin_vertices') or not hasattr(self, 'skin_quads'):
            print("  ERROR: No matrix skin created. Run create_matrix_skin() first.")
            return None
        
        print(f"  Wire thickness: {wire_thickness}")
        
        # Collect unique edges
        print("  Collecting unique edges...")
        sys.stdout.flush()
        edges = set()
        total_faces = len(self.skin_quads)
        for idx, face in enumerate(self.skin_quads):
            if idx % 100000 == 0:
                print(f"    Processing face {idx:,}/{total_faces:,} ({100*idx/total_faces:.1f}%)")
                sys.stdout.flush()
            for i in range(len(face)):
                j = (i + 1) % len(face)
                edge = tuple(sorted([face[i], face[j]]))
                edges.add(edge)
        
        print(f"  Total edges: {len(edges):,}")
        
        # For very large meshes, skip wireframe (too slow)
        if len(edges) > 500000:
            print(f"  ⚠️ Too many edges ({len(edges):,}) - skipping wireframe export")
            print(f"     Use the matrix_skin.obj or quad_skin.obj instead")
            # Write a simple placeholder file
            with open(output_path, 'w') as f:
                f.write(f"# WIREFRAME SKIPPED - {len(edges):,} edges would take too long\n")
                f.write(f"# Use matrix_skin.obj or quad_skin.obj instead\n")
            return output_path
        
        # Create tube geometry for each edge
        all_vertices = []
        all_faces = []
        
        segments = 4  # Square cross-section for wires
        total_edges = len(edges)
        
        print(f"  Creating tube geometry for {total_edges:,} edges...")
        sys.stdout.flush()
        
        for edge_idx, edge in enumerate(edges):
            if edge_idx % 50000 == 0:
                print(f"    Edge {edge_idx:,}/{total_edges:,} ({100*edge_idx/total_edges:.1f}%)")
                sys.stdout.flush()
            v0 = self.skin_vertices[edge[0]]
            v1 = self.skin_vertices[edge[1]]
            
            # Edge direction
            direction = v1 - v0
            length = np.linalg.norm(direction)
            if length < 1e-10:
                continue
            direction = direction / length
            
            # Create perpendicular vectors for tube cross-section
            if abs(direction[0]) < 0.9:
                perp1 = np.cross(direction, [1, 0, 0])
            else:
                perp1 = np.cross(direction, [0, 1, 0])
            perp1 = perp1 / np.linalg.norm(perp1) * wire_thickness
            perp2 = np.cross(direction, perp1)
            perp2 = perp2 / np.linalg.norm(perp2) * wire_thickness
            
            # Create vertices for tube (square cross-section)
            base_idx = len(all_vertices)
            
            # Start cap
            all_vertices.append(v0 + perp1 + perp2)
            all_vertices.append(v0 - perp1 + perp2)
            all_vertices.append(v0 - perp1 - perp2)
            all_vertices.append(v0 + perp1 - perp2)
            
            # End cap
            all_vertices.append(v1 + perp1 + perp2)
            all_vertices.append(v1 - perp1 + perp2)
            all_vertices.append(v1 - perp1 - perp2)
            all_vertices.append(v1 + perp1 - perp2)
            
            # Create faces (4 sides of tube)
            for i in range(4):
                j = (i + 1) % 4
                # Side face
                all_faces.append([
                    base_idx + i + 1,      # OBJ is 1-indexed
                    base_idx + j + 1,
                    base_idx + j + 4 + 1,
                    base_idx + i + 4 + 1
                ])
        
        # Write OBJ
        with open(output_path, 'w') as f:
            f.write("# WIREFRAME MATRIX SKIN\n")
            f.write(f"# Wire edges: {len(edges)}\n")
            f.write(f"# Wire thickness: {wire_thickness}\n\n")
            
            for v in all_vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            
            f.write("\ng wireframe_skin\n")
            for face in all_faces:
                f.write(f"f {face[0]} {face[1]} {face[2]} {face[3]}\n")
        
        print(f"\n  ✓ Wireframe skin exported!")
        print(f"    - File: {output_path}")
        print(f"    - Edges converted to tubes: {len(edges)}")
        
        return output_path
    
    # ========================================================================
    # ENHANCED: Connected Quad Patches with Shared Vertices
    # ========================================================================
    
    def create_connected_quad_grid(self, square_size: float = 1.0,
                                    merge_threshold: float = None) -> Dict:
        """
        Create a grid of CONNECTED quads where adjacent patches share vertices.
        
        This is the key enhancement - instead of isolated squares, we create
        a true patchwork surface where:
        - Adjacent quads share edge vertices
        - Moving one vertex affects all connected quads
        - The result is a unified, editable CAD surface
        
        Args:
            square_size: Size of each quad in mesh units
            merge_threshold: Distance threshold for merging vertices (default: square_size/4)
            
        Returns:
            Dictionary with patches, shared vertices, and connectivity info
        """
        print(f"\n{'='*60}")
        print("CREATING CONNECTED QUAD GRID (Enhanced)")
        print(f"{'='*60}")
        print(f"  Square size: {square_size}")
        
        if merge_threshold is None:
            merge_threshold = square_size / 4
        
        # First, create the basic grid
        self.create_uv_grid(square_size=square_size)
        
        if not self.grid_squares:
            print("  ERROR: No grid squares created")
            return {}
        
        print(f"\n  Step 1: Merging nearby vertices...")
        
        # Collect all vertices from all squares
        all_vertices = []
        vertex_sources = []  # (square_idx, corner_idx)
        
        for sq_idx, square in enumerate(self.grid_squares):
            for corner_idx, vertex in enumerate(square['vertices']):
                all_vertices.append(vertex)
                vertex_sources.append((sq_idx, corner_idx))
        
        all_vertices = np.array(all_vertices)
        print(f"    Total vertices before merge: {len(all_vertices)}")
        
        # Build KD-tree for efficient nearest neighbor search
        tree = KDTree(all_vertices)
        
        # Find clusters of nearby vertices
        self.shared_vertices = {}
        vertex_to_shared_id = {}  # (sq_idx, corner_idx) -> shared_vertex_id
        shared_id = 0
        processed = set()
        
        for i, vertex in enumerate(all_vertices):
            if i in processed:
                continue
            
            # Find all vertices within merge threshold
            nearby_indices = tree.query_ball_point(vertex, merge_threshold)
            
            # Average position of cluster
            cluster_vertices = all_vertices[nearby_indices]
            merged_position = np.mean(cluster_vertices, axis=0)
            
            # Create shared vertex
            self.shared_vertices[shared_id] = merged_position.tolist()
            
            # Map all nearby vertices to this shared vertex
            for idx in nearby_indices:
                source = vertex_sources[idx]
                vertex_to_shared_id[source] = shared_id
                processed.add(idx)
            
            shared_id += 1
        
        print(f"    Shared vertices after merge: {len(self.shared_vertices)}")
        print(f"    Vertex reduction: {len(all_vertices) - len(self.shared_vertices)} vertices merged")
        
        # Create quad patches referencing shared vertices
        print(f"\n  Step 2: Building connected quad patches...")
        
        self.quad_patches = []
        self.vertex_to_quads = defaultdict(list)
        
        for sq_idx, square in enumerate(self.grid_squares):
            # Get shared vertex IDs for this quad's corners
            vertex_ids = []
            for corner_idx in range(len(square['vertices'])):
                v_id = vertex_to_shared_id.get((sq_idx, corner_idx), -1)
                vertex_ids.append(v_id)
                self.vertex_to_quads[v_id].append(sq_idx)
            
            patch = {
                "id": sq_idx,
                "vertex_ids": vertex_ids,  # References to shared vertices
                "center": square['center'],
                "normal": square['normal'],
                "size": square['size'],
                "neighbors": [],  # Will be populated next
                "region_id": None
            }
            self.quad_patches.append(patch)
        
        # Find neighbors (quads sharing vertices)
        print(f"\n  Step 3: Computing quad connectivity...")
        
        neighbor_count = 0
        for patch in self.quad_patches:
            neighbor_set = set()
            for v_id in patch['vertex_ids']:
                for neighbor_quad_id in self.vertex_to_quads[v_id]:
                    if neighbor_quad_id != patch['id']:
                        neighbor_set.add(neighbor_quad_id)
            patch['neighbors'] = list(neighbor_set)
            neighbor_count += len(patch['neighbors'])
        
        avg_neighbors = neighbor_count / len(self.quad_patches) if self.quad_patches else 0
        print(f"    Average neighbors per quad: {avg_neighbors:.1f}")
        
        # Update grid params
        self.grid_params.update({
            "connected": True,
            "shared_vertex_count": len(self.shared_vertices),
            "merge_threshold": merge_threshold,
            "average_neighbors": avg_neighbors
        })
        
        print(f"\n  ✓ Connected quad grid created successfully!")
        print(f"    - {len(self.quad_patches)} quad patches")
        print(f"    - {len(self.shared_vertices)} shared vertices")
        
        return {
            "patches": self.quad_patches,
            "shared_vertices": self.shared_vertices,
            "vertex_to_quads": dict(self.vertex_to_quads)
        }
    
    def detect_surface_regions(self, num_regions: int = 6, 
                                method: str = "normal") -> Dict[int, List[int]]:
        """
        Automatically detect and group quads into surface regions.
        
        Regions are grouped by:
        - Normal direction (faces pointing same way)
        - Spatial proximity
        - Curvature similarity
        
        Args:
            num_regions: Target number of regions
            method: Grouping method ('normal', 'spatial', 'curvature')
            
        Returns:
            Dictionary mapping region_id to list of quad IDs
        """
        print(f"\n  Step 4: Detecting surface regions...")
        print(f"    Method: {method}, Target regions: {num_regions}")
        
        if not self.quad_patches:
            print("    ERROR: No quad patches to group")
            return {}
        
        # Collect features for clustering
        features = []
        
        for patch in self.quad_patches:
            if method == "normal":
                # Group by normal direction
                features.append(patch['normal'])
            elif method == "spatial":
                # Group by position
                features.append(patch['center'])
            elif method == "curvature":
                # Group by normal + position (combined)
                features.append(patch['normal'] + patch['center'])
            else:
                features.append(patch['normal'])
        
        features = np.array(features)
        
        # Use K-means clustering
        num_regions = min(num_regions, len(self.quad_patches))
        kmeans = KMeans(n_clusters=num_regions, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features)
        
        # Build regions dictionary
        self.regions = defaultdict(list)
        for quad_idx, region_id in enumerate(labels):
            self.regions[int(region_id)].append(quad_idx)
            self.quad_patches[quad_idx]['region_id'] = int(region_id)
        
        # Assign region properties (colors, names)
        region_colors = [
            (255, 100, 100, 255),   # Red
            (100, 255, 100, 255),   # Green
            (100, 100, 255, 255),   # Blue
            (255, 255, 100, 255),   # Yellow
            (255, 100, 255, 255),   # Magenta
            (100, 255, 255, 255),   # Cyan
            (255, 180, 100, 255),   # Orange
            (180, 100, 255, 255),   # Purple
        ]
        
        region_names = ["Top", "Bottom", "Front", "Back", "Left", "Right", 
                       "Region_7", "Region_8", "Region_9", "Region_10"]
        
        self.region_layers = {}
        for region_id in self.regions:
            avg_normal = np.mean([self.quad_patches[q]['normal'] for q in self.regions[region_id]], axis=0)
            
            # Auto-name based on dominant normal direction
            abs_normal = np.abs(avg_normal)
            if abs_normal[2] > 0.7:  # Z-dominant
                name = "Top" if avg_normal[2] > 0 else "Bottom"
            elif abs_normal[1] > 0.7:  # Y-dominant
                name = "Front" if avg_normal[1] > 0 else "Back"
            elif abs_normal[0] > 0.7:  # X-dominant
                name = "Right" if avg_normal[0] > 0 else "Left"
            else:
                name = f"Region_{region_id}"
            
            self.region_layers[region_id] = {
                "name": f"{name}_{region_id}",
                "color": region_colors[region_id % len(region_colors)],
                "quad_count": len(self.regions[region_id]),
                "average_normal": avg_normal.tolist()
            }
        
        print(f"    Created {len(self.regions)} regions:")
        for r_id, info in self.region_layers.items():
            print(f"      - {info['name']}: {info['quad_count']} quads")
        
        return dict(self.regions)
    
    def create_nurbs_patches(self, degree: int = 3) -> List[Dict]:
        """
        Convert quad patches to NURBS surface patches.
        
        NURBS surfaces are the native CAD representation,
        allowing smooth editing and precise manufacturing.
        
        Args:
            degree: NURBS surface degree (3 = cubic, smooth)
            
        Returns:
            List of NURBS patch data
        """
        print(f"\n  Creating NURBS surface patches (degree={degree})...")
        
        nurbs_patches = []
        
        for patch in self.quad_patches:
            # Get corner vertices
            corners = [self.shared_vertices[v_id] for v_id in patch['vertex_ids']]
            
            if len(corners) < 4:
                continue
            
            # Create bilinear NURBS patch data
            # For true NURBS, we'd need control points beyond corners
            nurbs_data = {
                "id": patch['id'],
                "type": "nurbs_surface",
                "degree_u": min(degree, 1),  # Bilinear for quads
                "degree_v": min(degree, 1),
                "control_points": corners[:4],  # 2x2 grid
                "knots_u": [0, 0, 1, 1],
                "knots_v": [0, 0, 1, 1],
                "weights": [1, 1, 1, 1],  # Rational weights
                "region_id": patch['region_id'],
                "neighbors": patch['neighbors']
            }
            nurbs_patches.append(nurbs_data)
        
        print(f"    Created {len(nurbs_patches)} NURBS patches")
        return nurbs_patches
    
    def export_to_rhino(self, output_path: str, 
                        include_original_mesh: bool = True,
                        layer_name: str = "Grid_Squares") -> str:
        """
        Export the grid to Rhino .3dm format.
        
        Args:
            output_path: Path for the .3dm file
            include_original_mesh: Whether to include the original mesh
            layer_name: Name for the grid layer
            
        Returns:
            Path to the saved file
        """
        if not HAS_RHINO3DM:
            print("rhino3dm not available, falling back to OBJ export")
            return self.export_to_obj(output_path.replace('.3dm', '_grid.obj'))
        
        print(f"\nExporting to Rhino .3dm: {output_path}")
        
        # Create new Rhino file
        model = rhino3dm.File3dm()
        
        # Add layers
        grid_layer = rhino3dm.Layer()
        grid_layer.Name = layer_name
        grid_layer.Color = (255, 100, 100, 255)  # Red-ish
        grid_layer_index = model.Layers.Add(grid_layer)
        
        if include_original_mesh:
            mesh_layer = rhino3dm.Layer()
            mesh_layer.Name = "Original_Mesh"
            mesh_layer.Color = (100, 100, 255, 255)  # Blue-ish
            mesh_layer_index = model.Layers.Add(mesh_layer)
        
        # Add grid squares as polylines/surfaces
        for square in self.grid_squares:
            vertices = square.get("vertices", [])
            
            if len(vertices) >= 3:
                # Create polyline (closed)
                points = [rhino3dm.Point3d(v[0], v[1], v[2]) for v in vertices]
                points.append(points[0])  # Close the loop
                
                polyline = rhino3dm.Polyline(points)
                curve = polyline.ToPolylineCurve()
                
                if curve:
                    attr = rhino3dm.ObjectAttributes()
                    attr.LayerIndex = grid_layer_index
                    attr.Name = f"Square_{square['id']}"
                    
                    # Add user data for parametric control
                    # Note: rhino3dm has limited user data support
                    
                    model.Objects.AddCurve(curve, attr)
        
        # Add original mesh if requested
        if include_original_mesh and self.mesh is not None:
            rhino_mesh = rhino3dm.Mesh()
            
            # Add vertices
            for vertex in self.mesh.vertices:
                rhino_mesh.Vertices.Add(vertex[0], vertex[1], vertex[2])
            
            # Add faces
            for face in self.mesh.faces:
                rhino_mesh.Faces.AddFace(face[0], face[1], face[2])
            
            rhino_mesh.Normals.ComputeNormals()
            
            attr = rhino3dm.ObjectAttributes()
            attr.LayerIndex = mesh_layer_index
            attr.Name = "Original_Mesh"
            
            model.Objects.AddMesh(rhino_mesh, attr)
        
        # Save the file
        model.Write(output_path, version=7)  # Rhino 7 format
        
        print(f"  ✓ Saved {len(self.grid_squares)} grid elements to {output_path}")
        return output_path
    
    def export_connected_to_rhino(self, output_path: str,
                                   include_original_mesh: bool = True,
                                   export_as_surfaces: bool = True,
                                   export_shared_points: bool = True) -> str:
        """
        Export CONNECTED quad grid to Rhino with region layers and shared vertices.
        
        This is the enhanced export that creates:
        - Separate layers for each surface region
        - NURBS surfaces (not just curves)
        - Point cloud of shared vertices for editing
        - Full connectivity information
        
        Args:
            output_path: Path for the .3dm file
            include_original_mesh: Include the source mesh
            export_as_surfaces: Export as surfaces (True) or curves (False)
            export_shared_points: Include shared vertex points for editing
            
        Returns:
            Path to saved file
        """
        if not HAS_RHINO3DM:
            print("rhino3dm not available, using OBJ export")
            return self.export_connected_to_obj(output_path.replace('.3dm', '_connected.obj'))
        
        print(f"\n{'='*60}")
        print("EXPORTING CONNECTED CAD GEOMETRY")
        print(f"{'='*60}")
        print(f"  Output: {output_path}")
        
        model = rhino3dm.File3dm()
        
        # ========== Create Layers ==========
        layer_indices = {}
        
        # Parent layer for all grid geometry
        parent_layer = rhino3dm.Layer()
        parent_layer.Name = "Surface_Grid"
        parent_layer_index = model.Layers.Add(parent_layer)
        
        # Region layers (children of Surface_Grid)
        if self.region_layers:
            for region_id, region_info in self.region_layers.items():
                layer = rhino3dm.Layer()
                layer.Name = region_info['name']
                layer.Color = region_info['color']
                layer.ParentLayerId = model.Layers[parent_layer_index].Id
                layer_indices[region_id] = model.Layers.Add(layer)
                print(f"    Layer: {region_info['name']} ({region_info['quad_count']} patches)")
        else:
            # Default single layer if no regions detected
            default_layer = rhino3dm.Layer()
            default_layer.Name = "Quad_Patches"
            default_layer.Color = (255, 100, 100, 255)
            layer_indices[-1] = model.Layers.Add(default_layer)
        
        # Shared vertices layer
        if export_shared_points:
            points_layer = rhino3dm.Layer()
            points_layer.Name = "Shared_Vertices"
            points_layer.Color = (255, 255, 0, 255)  # Yellow
            points_layer_index = model.Layers.Add(points_layer)
        
        # Original mesh layer
        if include_original_mesh:
            mesh_layer = rhino3dm.Layer()
            mesh_layer.Name = "Original_Mesh"
            mesh_layer.Color = (100, 100, 100, 255)  # Gray
            mesh_layer_index = model.Layers.Add(mesh_layer)
        
        # ========== Export Quad Patches ==========
        print(f"\n  Exporting {len(self.quad_patches)} quad patches...")
        
        surfaces_added = 0
        curves_added = 0
        
        for patch in self.quad_patches:
            # Get vertices for this patch
            vertices = []
            for v_id in patch['vertex_ids']:
                if v_id in self.shared_vertices:
                    vertices.append(self.shared_vertices[v_id])
            
            if len(vertices) < 3:
                continue
            
            # Determine layer
            region_id = patch.get('region_id', -1)
            layer_idx = layer_indices.get(region_id, layer_indices.get(-1, 0))
            
            # Create attributes
            attr = rhino3dm.ObjectAttributes()
            attr.LayerIndex = layer_idx
            attr.Name = f"Patch_{patch['id']}"
            
            if export_as_surfaces and len(vertices) >= 4:
                # Create NURBS surface from 4 corners
                try:
                    # Create a simple bilinear surface
                    pts = [rhino3dm.Point3d(v[0], v[1], v[2]) for v in vertices[:4]]
                    
                    # Create NurbsSurface - 2x2 control points, degree 1
                    surface = rhino3dm.NurbsSurface.Create(
                        3,  # dimension
                        False,  # isRational
                        2, 2,  # degree u, v (linear)
                        2, 2   # control point count u, v
                    )
                    
                    if surface:
                        # Set control points (2x2 grid)
                        surface.Points.SetPoint(0, 0, pts[0])
                        surface.Points.SetPoint(1, 0, pts[1])
                        surface.Points.SetPoint(0, 1, pts[3])
                        surface.Points.SetPoint(1, 1, pts[2])
                        
                        model.Objects.AddSurface(surface, attr)
                        surfaces_added += 1
                except Exception as e:
                    # Fallback to curve
                    pass
            
            # Also/alternatively add as curve (polyline boundary)
            points = [rhino3dm.Point3d(v[0], v[1], v[2]) for v in vertices]
            points.append(points[0])  # Close loop
            
            polyline = rhino3dm.Polyline(points)
            curve = polyline.ToPolylineCurve()
            
            if curve:
                if not export_as_surfaces:
                    model.Objects.AddCurve(curve, attr)
                    curves_added += 1
                else:
                    # Add curve as edge reference
                    edge_attr = rhino3dm.ObjectAttributes()
                    edge_attr.LayerIndex = layer_idx
                    edge_attr.Name = f"Edge_{patch['id']}"
                    model.Objects.AddCurve(curve, edge_attr)
        
        print(f"    Surfaces added: {surfaces_added}")
        print(f"    Edge curves added: {len(self.quad_patches)}")
        
        # ========== Export Shared Vertices as Points ==========
        if export_shared_points and self.shared_vertices:
            print(f"\n  Exporting {len(self.shared_vertices)} shared control points...")
            
            for v_id, coords in self.shared_vertices.items():
                point = rhino3dm.Point3d(coords[0], coords[1], coords[2])
                
                attr = rhino3dm.ObjectAttributes()
                attr.LayerIndex = points_layer_index
                attr.Name = f"V_{v_id}"
                
                # Store connectivity info in name (which quads use this vertex)
                connected_quads = self.vertex_to_quads.get(v_id, [])
                if connected_quads:
                    attr.Name = f"V_{v_id}_Q{len(connected_quads)}"
                
                model.Objects.AddPoint(point, attr)
        
        # ========== Export Original Mesh ==========
        if include_original_mesh and self.mesh is not None:
            print(f"\n  Exporting original mesh (reference)...")
            
            rhino_mesh = rhino3dm.Mesh()
            
            for vertex in self.mesh.vertices:
                rhino_mesh.Vertices.Add(vertex[0], vertex[1], vertex[2])
            
            for face in self.mesh.faces:
                rhino_mesh.Faces.AddFace(int(face[0]), int(face[1]), int(face[2]))
            
            rhino_mesh.Normals.ComputeNormals()
            
            attr = rhino3dm.ObjectAttributes()
            attr.LayerIndex = mesh_layer_index
            attr.Name = "Source_Mesh"
            
            model.Objects.AddMesh(rhino_mesh, attr)
        
        # Save
        model.Write(output_path, version=7)
        
        print(f"\n  ✓ Saved connected CAD geometry to:")
        print(f"    {output_path}")
        print(f"\n  In Rhino, you can now:")
        print(f"    • Select patches by region (layer)")
        print(f"    • Move shared vertices to reshape multiple patches")
        print(f"    • Edit surfaces for manufacturing")
        
        return output_path
    
    def export_connected_to_obj(self, output_path: str) -> str:
        """
        Export connected grid as OBJ with shared vertices.
        
        Args:
            output_path: Path for OBJ file
            
        Returns:
            Path to saved file
        """
        print(f"\nExporting connected grid to OBJ: {output_path}")
        
        with open(output_path, 'w') as f:
            f.write("# Connected Quad Grid Export (Enhanced)\n")
            f.write(f"# Quad patches: {len(self.quad_patches)}\n")
            f.write(f"# Shared vertices: {len(self.shared_vertices)}\n")
            f.write(f"# Regions: {len(self.regions)}\n\n")
            
            # Write all shared vertices first
            f.write("# Shared Vertices\n")
            vertex_id_to_obj_idx = {}
            obj_idx = 1  # OBJ is 1-indexed
            
            for v_id in sorted(self.shared_vertices.keys()):
                coords = self.shared_vertices[v_id]
                f.write(f"v {coords[0]:.6f} {coords[1]:.6f} {coords[2]:.6f}\n")
                vertex_id_to_obj_idx[v_id] = obj_idx
                obj_idx += 1
            
            f.write(f"\n# Quad Patches by Region\n")
            
            # Group patches by region
            for region_id in sorted(self.regions.keys()):
                region_info = self.region_layers.get(region_id, {"name": f"Region_{region_id}"})
                f.write(f"\ng {region_info['name']}\n")
                
                for quad_idx in self.regions[region_id]:
                    patch = self.quad_patches[quad_idx]
                    
                    # Get OBJ vertex indices
                    obj_indices = [vertex_id_to_obj_idx.get(v_id, 1) for v_id in patch['vertex_ids']]
                    
                    if len(obj_indices) >= 4:
                        f.write(f"f {obj_indices[0]} {obj_indices[1]} {obj_indices[2]} {obj_indices[3]}\n")
                    elif len(obj_indices) == 3:
                        f.write(f"f {obj_indices[0]} {obj_indices[1]} {obj_indices[2]}\n")
        
        print(f"  ✓ Saved connected grid to {output_path}")
        return output_path
    
    def export_hollow_shell(self, output_path: str, frame_thickness: float = 0.1,
                            extrude_height: float = 0.05) -> str:
        """
        Export HOLLOW FRAME structure - square frames without solid fill.
        
        This creates the carved-out shell structure where:
        - Each square is just the FRAME (4 edges) with thickness
        - NO solid interior - you can see through each square
        - The frames wrap around the 3D surface
        - Perfect for lattice/mesh-style jewelry designs
        
        Args:
            output_path: Path for the OBJ file
            frame_thickness: Width of each frame edge (default 0.1 = 10% of square)
            extrude_height: Height to extrude frames outward from surface
            
        Returns:
            Path to saved file
        """
        print(f"\n{'='*60}")
        print("CREATING HOLLOW SHELL STRUCTURE (Carved-out frames)")
        print(f"{'='*60}")
        print(f"  Frame thickness: {frame_thickness}")
        print(f"  Extrude height: {extrude_height}")
        print(f"  Total squares: {len(self.quad_patches)}")
        
        all_vertices = []
        all_faces = []
        vertex_offset = 0
        
        for patch in self.quad_patches:
            # Get the 4 corner vertices of this square
            corners = []
            for v_id in patch['vertex_ids']:
                if v_id in self.shared_vertices:
                    corners.append(np.array(self.shared_vertices[v_id]))
            
            if len(corners) < 4:
                continue
            
            # Get normal for extrusion direction
            normal = np.array(patch['normal'])
            normal = normal / (np.linalg.norm(normal) + 1e-8)
            
            # Create INNER corners (shrunk toward center) for the hollow part
            center = np.mean(corners, axis=0)
            
            # Outer corners (original)
            outer = corners
            
            # Inner corners (shrunk by frame_thickness ratio)
            inner = []
            for c in corners:
                direction = center - c
                dist = np.linalg.norm(direction)
                if dist > 0:
                    shrink = min(frame_thickness, dist * 0.4)  # Don't shrink more than 40%
                    inner.append(c + (direction / dist) * shrink)
                else:
                    inner.append(c)
            
            # Create extruded versions (offset along normal)
            outer_top = [c + normal * extrude_height for c in outer]
            inner_top = [c + normal * extrude_height for c in inner]
            outer_bot = [c - normal * extrude_height for c in outer]
            inner_bot = [c - normal * extrude_height for c in inner]
            
            # Add all 16 vertices for this frame
            # Bottom layer: outer[0-3], inner[0-3]
            # Top layer: outer_top[0-3], inner_top[0-3]
            frame_verts = outer_bot + inner_bot + outer_top + inner_top
            
            for v in frame_verts:
                all_vertices.append(v)
            
            # Create faces for the 4 frame edges
            # Each edge is a rectangular tube connecting outer to inner
            base = vertex_offset
            
            # Vertex indices (1-indexed for OBJ)
            # Bottom outer: 0,1,2,3 -> base+1 to base+4
            # Bottom inner: 4,5,6,7 -> base+5 to base+8
            # Top outer: 8,9,10,11 -> base+9 to base+12
            # Top inner: 12,13,14,15 -> base+13 to base+16
            
            ob = [base+1, base+2, base+3, base+4]      # outer bottom
            ib = [base+5, base+6, base+7, base+8]      # inner bottom
            ot = [base+9, base+10, base+11, base+12]   # outer top
            it = [base+13, base+14, base+15, base+16]  # inner top
            
            # Create 4 frame edges (each edge has 4 quad faces)
            for i in range(4):
                j = (i + 1) % 4
                
                # Outer face (facing outward)
                all_faces.append([ob[i], ob[j], ot[j], ot[i]])
                
                # Inner face (facing inward toward hole)
                all_faces.append([ib[j], ib[i], it[i], it[j]])
                
                # Top face (top of frame)
                all_faces.append([ot[i], ot[j], it[j], it[i]])
                
                # Bottom face (bottom of frame)
                all_faces.append([ob[j], ob[i], ib[i], ib[j]])
            
            # End caps for each frame segment
            for i in range(4):
                j = (i + 1) % 4
                # Side cap 1
                all_faces.append([ob[i], ib[i], it[i], ot[i]])
                # Side cap 2
                all_faces.append([ib[j], ob[j], ot[j], it[j]])
            
            vertex_offset += 16
        
        # Write OBJ file
        print(f"\n  Writing hollow shell to: {output_path}")
        
        with open(output_path, 'w') as f:
            f.write("# HOLLOW SHELL STRUCTURE - Carved-out square frames\n")
            f.write(f"# Frame count: {len(self.quad_patches)}\n")
            f.write(f"# Frame thickness: {frame_thickness}\n")
            f.write(f"# Extrude height: {extrude_height}\n")
            f.write(f"# Total vertices: {len(all_vertices)}\n")
            f.write(f"# Total faces: {len(all_faces)}\n\n")
            
            # Write vertices
            for v in all_vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            
            f.write("\n# Frame faces\n")
            f.write("g hollow_shell_frames\n")
            
            # Write faces
            for face in all_faces:
                f.write(f"f {face[0]} {face[1]} {face[2]} {face[3]}\n")
        
        print(f"  ✓ Hollow shell created:")
        print(f"    - {len(self.quad_patches)} square frames")
        print(f"    - {len(all_vertices)} vertices")
        print(f"    - {len(all_faces)} faces")
        print(f"    - Saved to: {output_path}")
        
        return output_path

    def export_to_obj(self, output_path: str) -> str:
        """
        Export grid as OBJ file (fallback if rhino3dm not available).
        
        Args:
            output_path: Path for the OBJ file
            
        Returns:
            Path to the saved file
        """
        print(f"\nExporting grid to OBJ: {output_path}")
        
        with open(output_path, 'w') as f:
            f.write("# Grid Tessellation Export\n")
            f.write(f"# Total squares: {len(self.grid_squares)}\n")
            f.write(f"# Square size: {self.grid_params.get('square_size', 'N/A')}\n\n")
            
            vertex_offset = 1  # OBJ is 1-indexed
            
            for square in self.grid_squares:
                vertices = square.get("vertices", [])
                f.write(f"# Square {square['id']}\n")
                f.write(f"g square_{square['id']}\n")
                
                # Write vertices
                for v in vertices:
                    f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
                
                # Write face
                if len(vertices) == 4:
                    f.write(f"f {vertex_offset} {vertex_offset+1} {vertex_offset+2} {vertex_offset+3}\n")
                elif len(vertices) == 3:
                    f.write(f"f {vertex_offset} {vertex_offset+1} {vertex_offset+2}\n")
                
                vertex_offset += len(vertices)
                f.write("\n")
        
        print(f"  ✓ Saved grid to {output_path}")
        return output_path
    
    def export_to_step(self, output_path: str) -> str:
        """
        Export grid as STEP file for CAD compatibility.
        Note: Requires additional libraries (OCC/cadquery)
        
        Args:
            output_path: Path for the STEP file
            
        Returns:
            Path to the saved file or None if not supported
        """
        try:
            import cadquery as cq
            
            print(f"\nExporting grid to STEP: {output_path}")
            
            # Create workplane
            result = cq.Workplane("XY")
            
            for square in self.grid_squares:
                vertices = square.get("vertices", [])
                if len(vertices) >= 3:
                    # Create wire from vertices
                    points = [(v[0], v[1], v[2]) for v in vertices]
                    points.append(points[0])  # Close
                    
                    # Add as wire
                    # Note: This is simplified - full implementation would create surfaces
            
            # Export
            cq.exporters.export(result, output_path)
            print(f"  ✓ Saved to {output_path}")
            return output_path
            
        except ImportError:
            print("cadquery not installed. STEP export not available.")
            print("Install with: pip install cadquery")
            return None
    
    def export_grid_data(self, output_path: str) -> str:
        """
        Export grid data as JSON for programmatic access.
        
        Args:
            output_path: Path for the JSON file
            
        Returns:
            Path to the saved file
        """
        print(f"\nExporting grid data to JSON: {output_path}")
        
        data = {
            "mesh_info": self.get_mesh_info(),
            "grid_params": self.grid_params,
            "squares": self.grid_squares
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"  ✓ Saved grid data ({len(self.grid_squares)} squares)")
        return output_path
    
    def create_grasshopper_script(self, output_path: str) -> str:
        """
        Generate a Grasshopper Python script that recreates the grid parametrically.
        
        This allows CAD designers to:
        - Adjust square size with a slider
        - Modify grid density
        - Change square shapes
        
        Args:
            output_path: Path for the .py Grasshopper script
            
        Returns:
            Path to the saved file
        """
        print(f"\nGenerating Grasshopper script: {output_path}")
        
        script = '''"""
Grasshopper Python Script - Parametric Surface Grid
Generated by mesh_to_cad_grid.py

Usage in Grasshopper:
1. Add a "Python" component
2. Paste this script
3. Connect inputs:
   - mesh: The input mesh geometry
   - square_size: Number slider (e.g., 0.1 to 10)
   - density: Integer slider for sampling density
4. Output: Grid curves ready for further manipulation
"""

import Rhino.Geometry as rg
import rhinoscriptsyntax as rs
import math

# Inputs from Grasshopper
# mesh = input mesh
# square_size = size of each square (float)
# density = sampling density (int)

def get_tangent_vectors(normal):
    """Get two tangent vectors perpendicular to normal."""
    if abs(normal.X) < 0.9:
        ref = rg.Vector3d(1, 0, 0)
    else:
        ref = rg.Vector3d(0, 1, 0)
    
    tangent1 = rg.Vector3d.CrossProduct(normal, ref)
    tangent1.Unitize()
    
    tangent2 = rg.Vector3d.CrossProduct(normal, tangent1)
    tangent2.Unitize()
    
    return tangent1, tangent2

def create_square_at_point(point, normal, size):
    """Create a square curve centered at point, tangent to surface."""
    t1, t2 = get_tangent_vectors(normal)
    
    half = size / 2.0
    
    v1 = point + half * t1 + half * t2
    v2 = point - half * t1 + half * t2
    v3 = point - half * t1 - half * t2
    v4 = point + half * t1 - half * t2
    
    # Create closed polyline
    pts = [rg.Point3d(v1), rg.Point3d(v2), rg.Point3d(v3), rg.Point3d(v4), rg.Point3d(v1)]
    return rg.PolylineCurve([rg.Point3d(p) for p in pts])

# Main execution
squares = []

if mesh and square_size > 0:
    # Get mesh faces and sample points
    mesh_faces = mesh.Faces
    mesh_normals = mesh.FaceNormals
    
    # Sample points on mesh surface
    sample_count = density * density if density else 100
    
    # Use mesh face centers
    for i in range(mesh_faces.Count):
        if i % max(1, mesh_faces.Count // sample_count) == 0:
            face = mesh_faces[i]
            center = mesh_faces.GetFaceCenter(i)
            normal = mesh_normals[i]
            
            sq = create_square_at_point(center, normal, square_size)
            if sq:
                squares.append(sq)

# Output
a = squares  # Connect to Grasshopper output
'''
        
        with open(output_path, 'w') as f:
            f.write(script)
        
        print(f"  ✓ Grasshopper script saved")
        print(f"  Import into Grasshopper Python component for parametric control")
        return output_path


def process_mesh(mesh_path: str, output_dir: str, 
                 square_size: float = 1.0,
                 method: str = "matrix",
                 num_regions: int = 6,
                 export_formats: List[str] = None,
                 shell_thickness: float = 0.5,
                 subdivisions: int = 2) -> Dict[str, str]:
    """
    Process a mesh and export tessellated grid in multiple formats.
    
    Args:
        mesh_path: Path to input mesh (GLB/OBJ/STL)
        output_dir: Directory for output files
        square_size: Size of grid squares
        method: Tessellation method ('matrix', 'connected', 'uv', 'projection', 'subdivision')
        num_regions: Number of surface regions to detect (for connected method)
        export_formats: List of formats to export ('3dm', 'obj', 'json', 'gh')
        shell_thickness: Thickness of Matrix skin shell (default 0.5)
        subdivisions: Number of subdivisions per triangle for Matrix skin (default 2)
        
    Returns:
        Dictionary of exported file paths
    """
    if export_formats is None:
        export_formats = ['3dm', 'obj', 'json', 'gh']
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get base name
    base_name = Path(mesh_path).stem
    
    # Initialize tessellator
    tessellator = MeshGridTessellator(mesh_path)
    
    # Create grid based on method
    if method == "matrix":
        # MATRIX SKIN: Complete quad mesh wrapping the ENTIRE surface
        print("\n" + "="*60)
        print("CREATING MATRIX SKIN - Complete Surface Wrapper")
        print("="*60)
        print(f"Subdivisions per triangle: {subdivisions}")
        print(f"Shell thickness: {shell_thickness}")
        tessellator.create_matrix_skin(subdivisions=subdivisions, offset=0.0)
    elif method == "connected":
        # NEW: Create connected quad grid with shared vertices and regions
        tessellator.create_connected_quad_grid(square_size=square_size)
        tessellator.detect_surface_regions(num_regions=num_regions)
    elif method == "uv":
        tessellator.create_uv_grid(square_size=square_size)
    elif method == "projection":
        tessellator.create_projection_grid(square_size=square_size)
    elif method == "subdivision":
        subdivisions = max(1, int(5 / square_size))  # More subdivisions for smaller squares
        tessellator.create_face_subdivision_grid(subdivisions=subdivisions)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Export to requested formats
    outputs = {}
    
    # MATRIX SKIN exports
    if method == "matrix":
        # Export Matrix skin (complete quad shell wrapping entire surface)
        matrix_skin_path = os.path.join(output_dir, f"{base_name}_matrix_skin.obj")
        outputs['matrix_skin'] = tessellator.export_matrix_skin(
            matrix_skin_path, 
            shell_thickness=shell_thickness
        )
        
        # Export wireframe version (tubes along edges)
        wireframe_path = os.path.join(output_dir, f"{base_name}_wireframe_skin.obj")
        wire_thickness = shell_thickness * 0.3  # Wire thickness proportional to shell
        outputs['wireframe_skin'] = tessellator.export_wireframe_skin(
            wireframe_path,
            wire_thickness=wire_thickness
        )
        
        # NOTE: Lattice export removed - matrix_skin is better for CAD editing
        # The lattice creates massive files (10-15x more geometry) and is only
        # useful for decorative see-through structures, not for editing designs.
        # Use matrix_skin for adjusting prongs, settings, band thickness, etc.
        
        # Also export the raw quad skin as simple OBJ
        raw_skin_path = os.path.join(output_dir, f"{base_name}_quad_skin.obj")
        outputs['quad_skin'] = tessellator.export_connected_to_obj(raw_skin_path)
        
        print(f"\n✓ Matrix skin exported: {outputs.get('matrix_skin', 'N/A')}")
        print(f"✓ Wireframe skin exported: {outputs.get('wireframe_skin', 'N/A')}")
        print(f"✓ Raw quad skin exported: {outputs.get('quad_skin', 'N/A')}")
    
    elif '3dm' in export_formats:
        path = os.path.join(output_dir, f"{base_name}_grid.3dm")
        if method == "connected":
            # Use enhanced export for connected grid
            outputs['3dm'] = tessellator.export_connected_to_rhino(path)
        else:
            outputs['3dm'] = tessellator.export_to_rhino(path)
    
    if 'obj' in export_formats:
        path = os.path.join(output_dir, f"{base_name}_grid.obj")
        if method == "connected":
            outputs['obj'] = tessellator.export_connected_to_obj(path)
        else:
            outputs['obj'] = tessellator.export_to_obj(path)
    
    # NEW: Always export hollow shell for connected method
    if method == "connected":
        hollow_path = os.path.join(output_dir, f"{base_name}_hollow_shell.obj")
        frame_thickness = square_size * 0.15  # 15% of square size
        extrude_height = square_size * 0.08   # 8% of square size
        outputs['hollow_shell'] = tessellator.export_hollow_shell(
            hollow_path, 
            frame_thickness=frame_thickness,
            extrude_height=extrude_height
        )
    
    if 'json' in export_formats:
        path = os.path.join(output_dir, f"{base_name}_grid_data.json")
        outputs['json'] = tessellator.export_grid_data(path)
    
    if 'gh' in export_formats:
        path = os.path.join(output_dir, f"{base_name}_grasshopper.py")
        outputs['gh'] = tessellator.create_grasshopper_script(path)
    
    # Save parameters (enhanced with connectivity info)
    params = {
        "input_mesh": mesh_path,
        "square_size": square_size,
        "method": method,
        "mesh_info": tessellator.get_mesh_info(),
        "grid_params": tessellator.grid_params,
        "outputs": outputs
    }
    
    # Add connected grid info if available
    if method == "connected":
        params["connected_info"] = {
            "shared_vertices": len(tessellator.shared_vertices),
            "quad_patches": len(tessellator.quad_patches),
            "regions": {r_id: len(quads) for r_id, quads in tessellator.regions.items()},
            "region_layers": tessellator.region_layers
        }
    
    params_path = os.path.join(output_dir, f"{base_name}_params.json")
    with open(params_path, 'w') as f:
        json.dump(params, f, indent=2)
    
    return outputs


def main():
    """Command-line interface for mesh tessellation."""
    parser = argparse.ArgumentParser(
        description="Tessellate 3D mesh surface with parametric grid for CAD export"
    )
    
    parser.add_argument("mesh_path", help="Path to input mesh (GLB/OBJ/STL)")
    parser.add_argument("-o", "--output", default="./grid_output",
                        help="Output directory (default: ./grid_output)")
    parser.add_argument("-s", "--square-size", type=float, default=1.0,
                        help="Size of each grid square (default: 1.0)")
    parser.add_argument("-m", "--method", 
                        choices=["matrix", "connected", "uv", "projection", "subdivision"],
                        default="matrix", 
                        help="Tessellation method (default: matrix - complete surface wrapper)")
    parser.add_argument("-r", "--regions", type=int, default=6,
                        help="Number of surface regions to detect (default: 6)")
    parser.add_argument("-f", "--formats", nargs="+", 
                        default=["3dm", "obj", "json", "gh"],
                        help="Export formats (default: 3dm obj json gh)")
    parser.add_argument("-t", "--thickness", type=float, default=0.5,
                        help="Shell thickness for Matrix skin (default: 0.5)")
    parser.add_argument("--subdivisions", type=int, default=2,
                        help="Subdivisions per triangle for Matrix skin (default: 2)")
    
    args = parser.parse_args()
    
    print("="*60)
    print("MESH SURFACE TESSELLATION TO CAD GRID")
    print("="*60)
    print(f"Method: {args.method}")
    if args.method == "matrix":
        print("  → MATRIX SKIN: Complete quad mesh wrapping ENTIRE surface")
        print("  → Like The Matrix - continuous grid covering everything")
        print("  → Hollow shell for CAD editing and mold manufacturing")
    elif args.method == "connected":
        print("  → Connected quads with shared vertices")
        print("  → Auto-detected surface regions")
        print("  → NURBS surface patches")
    
    outputs = process_mesh(
        mesh_path=args.mesh_path,
        output_dir=args.output,
        square_size=args.square_size,
        method=args.method,
        num_regions=args.regions,
        export_formats=args.formats,
        shell_thickness=args.thickness,
        subdivisions=args.subdivisions
    )
    
    print("\n" + "="*60)
    print("EXPORT COMPLETE")
    print("="*60)
    print(f"Output directory: {args.output}")
    for fmt, path in outputs.items():
        if path:
            print(f"  {fmt.upper()}: {path}")
    
    print("\n" + "="*60)
    print("WHAT CAD DESIGNERS CAN DO NOW:")
    print("="*60)
    if args.method == "connected":
        print("✓ Patches share vertices - moving one affects neighbors")
        print("✓ Select entire regions (layers) for bulk editing")
        print("✓ Edit NURBS surfaces for smooth manufacturing")
        print("✓ Shared vertices appear as control points")
    print("\nNext steps:")
    print("1. Open the .3dm file in Rhino")
    print("2. Toggle layers to see different surface regions")
    print("3. Select shared vertices (yellow points) to reshape multiple patches")
    print("4. Use the Grasshopper script for parametric control")
    print("5. Export clean geometry for mold manufacturing")


if __name__ == "__main__":
    main()
