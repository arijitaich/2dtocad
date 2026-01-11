"""
Mesh Dimension Scaler & Volume Calculator

Scale a connected quad mesh to specific dimensions and calculate material volume.
Useful for manufacturing where you need exact sizes and material cost estimation.
"""

import numpy as np
import trimesh
from pathlib import Path
from typing import Dict, Tuple, Optional


class MeshDimensionScaler:
    """
    Scale mesh to target dimensions and calculate manufacturing volume.
    """
    
    def __init__(self, mesh_path: str):
        """Load mesh from file."""
        self.mesh_path = mesh_path
        self.mesh = trimesh.load(mesh_path, force='mesh')
        
        # Get original dimensions
        self.original_bounds = self.mesh.bounds
        self.original_dimensions = self.mesh.bounds[1] - self.mesh.bounds[0]
        
        print(f"Loaded mesh: {Path(mesh_path).name}")
        print(f"  Vertices: {len(self.mesh.vertices):,}")
        print(f"  Faces: {len(self.mesh.faces):,}")
        print(f"\nOriginal dimensions:")
        print(f"  Width (X):  {self.original_dimensions[0]:.4f}")
        print(f"  Height (Y): {self.original_dimensions[1]:.4f}")
        print(f"  Depth (Z):  {self.original_dimensions[2]:.4f}")
    
    def get_current_dimensions(self) -> Dict[str, float]:
        """Get current mesh dimensions."""
        dims = self.mesh.bounds[1] - self.mesh.bounds[0]
        return {
            'width': float(dims[0]),
            'height': float(dims[1]),
            'depth': float(dims[2]),
            'units': 'model_units'
        }
    
    def scale_to_dimensions(self, 
                           target_width: float = None,
                           target_height: float = None,
                           target_depth: float = None,
                           uniform: bool = True) -> Dict:
        """
        Scale mesh to target dimensions.
        
        Args:
            target_width: Target X dimension (mm, cm, inches - your choice)
            target_height: Target Y dimension
            target_depth: Target Z dimension
            uniform: If True, scale uniformly based on the provided dimension(s)
                    If False, scale each axis independently
        
        Returns:
            Dict with old/new dimensions and scale factors
        """
        current_dims = self.original_dimensions
        
        # Calculate scale factors
        scale_factors = [1.0, 1.0, 1.0]
        
        if target_width is not None:
            scale_factors[0] = target_width / current_dims[0]
        if target_height is not None:
            scale_factors[1] = target_height / current_dims[1]
        if target_depth is not None:
            scale_factors[2] = target_depth / current_dims[2]
        
        if uniform:
            # Use the average of specified scale factors
            specified = []
            if target_width is not None:
                specified.append(scale_factors[0])
            if target_height is not None:
                specified.append(scale_factors[1])
            if target_depth is not None:
                specified.append(scale_factors[2])
            
            if specified:
                uniform_scale = np.mean(specified)
                scale_factors = [uniform_scale, uniform_scale, uniform_scale]
        
        # Apply scaling
        self.mesh.vertices = self.mesh.vertices * scale_factors
        
        # Update bounds
        new_dims = self.mesh.bounds[1] - self.mesh.bounds[0]
        
        result = {
            'original_dimensions': {
                'width': float(current_dims[0]),
                'height': float(current_dims[1]),
                'depth': float(current_dims[2])
            },
            'new_dimensions': {
                'width': float(new_dims[0]),
                'height': float(new_dims[1]),
                'depth': float(new_dims[2])
            },
            'scale_factors': {
                'x': float(scale_factors[0]),
                'y': float(scale_factors[1]),
                'z': float(scale_factors[2])
            },
            'uniform_scaling': uniform
        }
        
        print(f"\n✓ Mesh scaled!")
        print(f"  Scale factors: X={scale_factors[0]:.4f}, Y={scale_factors[1]:.4f}, Z={scale_factors[2]:.4f}")
        print(f"\n  New dimensions:")
        print(f"    Width (X):  {new_dims[0]:.4f}")
        print(f"    Height (Y): {new_dims[1]:.4f}")
        print(f"    Depth (Z):  {new_dims[2]:.4f}")
        
        return result
    
    def calculate_volume(self, shell_thickness: float = 0.0) -> Dict:
        """
        Calculate material volume for manufacturing.
        
        Args:
            shell_thickness: If > 0, calculate hollow shell volume
                            If 0, calculate solid volume
        
        Returns:
            Dict with volume calculations
        """
        print(f"\n{'='*50}")
        print("VOLUME CALCULATION")
        print(f"{'='*50}")
        
        # Get mesh dimensions
        dims = self.mesh.bounds[1] - self.mesh.bounds[0]
        
        # Calculate surface area
        surface_area = self.mesh.area
        
        # Try to calculate volume (works best for watertight meshes)
        try:
            # For a proper solid, use mesh volume
            if self.mesh.is_watertight:
                solid_volume = abs(self.mesh.volume)
                print(f"  Mesh is watertight - using exact volume")
            else:
                # Approximate using convex hull
                solid_volume = self.mesh.convex_hull.volume
                print(f"  Mesh not watertight - using convex hull approximation")
        except:
            # Fallback: bounding box volume (rough estimate)
            solid_volume = dims[0] * dims[1] * dims[2]
            print(f"  Using bounding box approximation")
        
        result = {
            'surface_area': float(surface_area),
            'solid_volume': float(solid_volume),
            'bounding_box_volume': float(dims[0] * dims[1] * dims[2]),
            'dimensions': {
                'width': float(dims[0]),
                'height': float(dims[1]),
                'depth': float(dims[2])
            }
        }
        
        # Shell volume calculation
        if shell_thickness > 0:
            # Shell volume ≈ Surface Area × Thickness
            shell_volume = surface_area * shell_thickness
            result['shell_thickness'] = shell_thickness
            result['shell_volume'] = float(shell_volume)
            
            print(f"\n  Shell thickness: {shell_thickness}")
            print(f"  Shell volume (material needed): {shell_volume:.4f} cubic units")
        
        print(f"\n  Results:")
        print(f"    Surface area: {surface_area:.4f} square units")
        print(f"    Solid volume: {solid_volume:.4f} cubic units")
        print(f"    Bounding box: {result['bounding_box_volume']:.4f} cubic units")
        
        return result
    
    def calculate_material_cost(self, 
                               shell_thickness: float,
                               material_density: float,
                               cost_per_gram: float,
                               unit_scale: float = 1.0) -> Dict:
        """
        Calculate material cost for manufacturing.
        
        Args:
            shell_thickness: Thickness of the shell/skin
            material_density: Density of material (g/cm³)
                             Gold: ~19.3 g/cm³
                             Silver: ~10.5 g/cm³
                             Platinum: ~21.45 g/cm³
                             Brass: ~8.5 g/cm³
                             Resin: ~1.2 g/cm³
            cost_per_gram: Cost per gram of material
            unit_scale: Scale to convert model units to cm (e.g., 0.1 if model is in mm)
        
        Returns:
            Dict with cost calculations
        """
        print(f"\n{'='*50}")
        print("MATERIAL COST CALCULATION")
        print(f"{'='*50}")
        
        # Get volume in model units
        vol_result = self.calculate_volume(shell_thickness)
        shell_volume = vol_result.get('shell_volume', vol_result['solid_volume'])
        
        # Convert to cm³
        volume_cm3 = shell_volume * (unit_scale ** 3)
        
        # Calculate weight
        weight_grams = volume_cm3 * material_density
        
        # Calculate cost
        material_cost = weight_grams * cost_per_gram
        
        result = {
            'shell_volume_model_units': float(shell_volume),
            'shell_volume_cm3': float(volume_cm3),
            'material_density_g_cm3': float(material_density),
            'weight_grams': float(weight_grams),
            'cost_per_gram': float(cost_per_gram),
            'material_cost': float(material_cost)
        }
        
        print(f"\n  Material: density = {material_density} g/cm³")
        print(f"  Shell volume: {volume_cm3:.4f} cm³")
        print(f"  Weight: {weight_grams:.4f} grams")
        print(f"  Cost per gram: ${cost_per_gram:.2f}")
        print(f"\n  💰 ESTIMATED MATERIAL COST: ${material_cost:.2f}")
        
        return result
    
    def export_scaled_mesh(self, output_path: str) -> str:
        """Export the scaled mesh to file."""
        # Determine format from extension
        ext = Path(output_path).suffix.lower()
        
        if ext == '.obj':
            self.mesh.export(output_path, file_type='obj')
        elif ext == '.stl':
            self.mesh.export(output_path, file_type='stl')
        elif ext == '.glb':
            self.mesh.export(output_path, file_type='glb')
        elif ext == '.ply':
            self.mesh.export(output_path, file_type='ply')
        else:
            # Default to OBJ
            output_path = str(Path(output_path).with_suffix('.obj'))
            self.mesh.export(output_path, file_type='obj')
        
        print(f"\n✓ Exported scaled mesh: {output_path}")
        return output_path


def scale_and_calculate(mesh_path: str,
                       target_width: float = None,
                       target_height: float = None,
                       target_depth: float = None,
                       shell_thickness: float = 0.5,
                       material: str = "gold",
                       cost_per_gram: float = None,
                       output_path: str = None,
                       output_dir: str = None) -> Dict:
    """
    Convenience function: Scale mesh and calculate manufacturing cost.
    
    Args:
        mesh_path: Path to input mesh
        target_width: Target X dimension in mm (default: keep original)
        target_height: Target Y dimension in mm (default: scale proportionally)
        target_depth: Target Z dimension in mm (default: scale proportionally)
        shell_thickness: Manufacturing shell thickness in mm (default: 0.5mm)
        material: Material name (gold, silver, platinum, brass, resin) or density value
        cost_per_gram: Material cost per gram (default: based on material type)
        output_path: Path to save scaled mesh (optional)
        output_dir: Directory to save scaled mesh with auto-naming (optional)
    
    Returns:
        Dict with all calculations and file paths
    """
    # Material densities (g/cm³) and default costs ($/gram)
    materials = {
        'gold':     {'density': 19.3,  'cost': 60.0},
        'silver':   {'density': 10.5,  'cost': 0.80},
        'platinum': {'density': 21.45, 'cost': 30.0},
        'brass':    {'density': 8.5,   'cost': 0.02},
        'bronze':   {'density': 8.8,   'cost': 0.03},
        'copper':   {'density': 8.96,  'cost': 0.01},
        'steel':    {'density': 7.85,  'cost': 0.005},
        'titanium': {'density': 4.5,   'cost': 0.10},
        'resin':    {'density': 1.2,   'cost': 0.05},
        'wax':      {'density': 0.9,   'cost': 0.02}
    }
    
    # Get material properties
    if isinstance(material, str) and material.lower() in materials:
        mat_info = materials[material.lower()]
        material_density = mat_info['density']
        default_cost = mat_info['cost']
    else:
        material_density = float(material) if not isinstance(material, str) else 19.3
        default_cost = 60.0
    
    # Use provided cost or default
    if cost_per_gram is None:
        cost_per_gram = default_cost
    
    # Process
    scaler = MeshDimensionScaler(mesh_path)
    
    # Store original dimensions
    original_dims = scaler.get_current_dimensions()
    
    # Scale if dimensions provided
    if any([target_width, target_height, target_depth]):
        scale_result = scaler.scale_to_dimensions(
            target_width=target_width,
            target_height=target_height,
            target_depth=target_depth,
            uniform=True
        )
    else:
        scale_result = {
            'message': 'No scaling applied - using original dimensions',
            'original_dimensions': original_dims,
            'new_dimensions': original_dims
        }
    
    # Get final dimensions
    final_dims = scaler.get_current_dimensions()
    
    # Calculate volume and cost
    volume_result = scaler.calculate_volume(shell_thickness)
    cost_result = scaler.calculate_material_cost(
        shell_thickness=shell_thickness,
        material_density=material_density,
        cost_per_gram=cost_per_gram,
        unit_scale=0.1  # Assuming model units are mm, convert to cm
    )
    
    # Generate output path if output_dir provided
    if output_dir and not output_path:
        base_name = Path(mesh_path).stem
        w = final_dims['width']
        h = final_dims['height']
        d = final_dims['depth']
        output_path = f"{output_dir}/{base_name}_scaled_{w:.1f}x{h:.1f}x{d:.1f}mm.obj"
    
    # Export if requested
    exported_file = None
    if output_path:
        exported_file = scaler.export_scaled_mesh(output_path)
    
    # Print comprehensive summary
    print("\n" + "="*60)
    print("📋 MANUFACTURING SPECIFICATION SUMMARY")
    print("="*60)
    print(f"\n📐 DIMENSIONS:")
    print(f"   Width (X):     {final_dims['width']:.2f} mm")
    print(f"   Height (Y):    {final_dims['height']:.2f} mm")
    print(f"   Depth (Z):     {final_dims['depth']:.2f} mm")
    print(f"   Shell thickness: {shell_thickness:.2f} mm")
    
    print(f"\n📦 VOLUME:")
    print(f"   Surface area:  {volume_result['surface_area']:.2f} mm²")
    print(f"   Shell volume:  {volume_result.get('shell_volume', volume_result['solid_volume']):.4f} mm³")
    print(f"   Shell volume:  {cost_result['shell_volume_cm3']:.4f} cm³")
    
    print(f"\n⚖️  MATERIAL ({material.upper() if isinstance(material, str) else 'Custom'}):")
    print(f"   Density:       {material_density} g/cm³")
    print(f"   Weight:        {cost_result['weight_grams']:.2f} grams")
    print(f"   Cost/gram:     ${cost_per_gram:.2f}")
    
    print(f"\n💰 ESTIMATED COST: ${cost_result['material_cost']:.2f}")
    
    if exported_file:
        print(f"\n📁 SCALED CAD FILE:")
        print(f"   {exported_file}")
    
    print("="*60)
    
    return {
        'dimensions': {
            'width_mm': float(final_dims['width']),
            'height_mm': float(final_dims['height']),
            'depth_mm': float(final_dims['depth']),
            'shell_thickness_mm': float(shell_thickness)
        },
        'volume': {
            'surface_area_mm2': float(volume_result['surface_area']),
            'shell_volume_mm3': float(volume_result.get('shell_volume', volume_result['solid_volume'])),
            'shell_volume_cm3': float(cost_result['shell_volume_cm3'])
        },
        'material': {
            'type': material if isinstance(material, str) else 'custom',
            'density_g_cm3': float(material_density),
            'weight_grams': float(cost_result['weight_grams']),
            'cost_per_gram': float(cost_per_gram)
        },
        'cost': {
            'material_cost_usd': float(cost_result['material_cost'])
        },
        'files': {
            'input_mesh': mesh_path,
            'scaled_cad_file': exported_file
        },
        'scaling': scale_result
    }


if __name__ == "__main__":
    import sys
    
    # Example usage
    if len(sys.argv) > 1:
        mesh_path = sys.argv[1]
    else:
        mesh_path = "manufacturing_output/20260111_071245/grids/Gold_Ring_Mounting__Shank_and_Setting__3d_matrix_skin.obj"
    
    print("="*60)
    print("MESH DIMENSION SCALER & VOLUME CALCULATOR")
    print("="*60)
    
    # Example: Scale ring to 18mm width and calculate gold cost
    result = scale_and_calculate(
        mesh_path=mesh_path,
        target_width=18.0,  # 18mm ring width
        shell_thickness=0.8,  # 0.8mm wall thickness
        material="gold",
        cost_per_gram=60.0,  # $60/gram for gold
        output_path="scaled_ring_18mm.obj"
    )
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Material weight: {result['cost']['weight_grams']:.2f} grams")
    print(f"Material cost: ${result['cost']['material_cost']:.2f}")
