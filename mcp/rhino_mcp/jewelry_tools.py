"""Advanced jewelry design tools for Rhino MCP."""
from mcp.server.fastmcp import FastMCP, Context
import logging
from typing import Optional, List
import json

logger = logging.getLogger("JewelryTools")

class JewelryTools:
    """Tools specifically for jewelry design in Rhino."""
    
    def __init__(self, app: FastMCP):
        self.app = app
        self._register_tools()
    
    def _register_tools(self):
        """Register all jewelry tools with the MCP app."""
        
        @self.app.tool()
        def create_ring_band(
            ctx: Context,
            inner_diameter: float = 17.0,
            width: float = 2.0,
            thickness: float = 1.5,
            profile: str = "half_round",
            name: Optional[str] = None,
            layer: str = "Band"
        ) -> str:
            """Create a ring band with specified dimensions.
            
            Args:
                inner_diameter: Inner diameter of the ring in mm (default 17mm = US size 7)
                width: Width of the band in mm
                thickness: Thickness/height of the band in mm
                profile: Band profile type - "flat", "half_round", "comfort_fit", "d_shape"
                name: Optional name for the band
                layer: Layer to create the band on
            
            Returns:
                JSON string with the created band object info
            """
            return self._create_ring_band(ctx, inner_diameter, width, thickness, profile, name, layer)
        
        @self.app.tool()
        def create_gemstone(
            ctx: Context,
            stone_type: str = "round_brilliant",
            diameter: float = 5.0,
            position: List[float] = [0, 0, 0],
            name: Optional[str] = None,
            layer: str = "Gemstones"
        ) -> str:
            """Create a gemstone with proper faceting.
            
            Args:
                stone_type: Type of stone cut - "round_brilliant", "emerald_cut", "princess", "oval", "cushion", "pear"
                diameter: Diameter of stone in mm (for round) or main dimension for other cuts
                position: [x, y, z] position of the stone
                name: Optional name for the stone
                layer: Layer to create the stone on
            
            Returns:
                JSON string with the created gemstone info
            """
            return self._create_gemstone(ctx, stone_type, diameter, position, name, layer)
        
        @self.app.tool()
        def create_prong_setting(
            ctx: Context,
            stone_diameter: float = 5.0,
            prong_count: int = 4,
            prong_height: float = 3.0,
            prong_diameter: float = 0.5,
            position: List[float] = [0, 0, 0],
            style: str = "tapered",
            name: Optional[str] = None,
            layer: str = "Settings"
        ) -> str:
            """Create a prong setting for a gemstone.
            
            Args:
                stone_diameter: Diameter of the stone to hold
                prong_count: Number of prongs (typically 4, 6, or 8)
                prong_height: Height of the prongs in mm
                prong_diameter: Diameter of each prong at base in mm
                position: [x, y, z] position of the setting
                style: Prong style - "straight", "tapered", "v_tip", "claw"
                name: Optional name for the setting
                layer: Layer to create the setting on
            
            Returns:
                JSON string with the created prong setting info
            """
            return self._create_prong_setting(ctx, stone_diameter, prong_count, prong_height, prong_diameter, position, style, name, layer)
        
        @self.app.tool()
        def create_gallery_rail(
            ctx: Context,
            ring_diameter: float = 17.0,
            stone_count: int = 7,
            stone_diameter: float = 3.0,
            rail_width: float = 1.5,
            rail_height: float = 2.0,
            name: Optional[str] = None,
            layer: str = "Gallery"
        ) -> str:
            """Create a gallery rail for multiple stones on a ring.
            
            This creates the structural framework that holds multiple stones in a row,
            common in eternity bands and multi-stone settings.
            
            Args:
                ring_diameter: Inner diameter of the ring in mm
                stone_count: Number of stones to accommodate
                stone_diameter: Diameter of each stone in mm
                rail_width: Width of the gallery rail
                rail_height: Height of the gallery structure
                name: Optional name
                layer: Layer to create on
            
            Returns:
                JSON string with gallery rail info
            """
            return self._create_gallery_rail(ctx, ring_diameter, stone_count, stone_diameter, rail_width, rail_height, name, layer)
        
        @self.app.tool()
        def array_stones_on_ring(
            ctx: Context,
            stone_diameter: float = 3.0,
            stone_count: int = 7,
            ring_diameter: float = 17.0,
            stone_type: str = "round_brilliant",
            start_angle: float = 0,
            arc_angle: float = 180,
            name_prefix: Optional[str] = "Stone",
            layer: str = "Gemstones"
        ) -> str:
            """Array multiple stones along a ring path.
            
            Args:
                stone_diameter: Diameter of each stone in mm
                stone_count: Number of stones to create
                ring_diameter: Diameter of ring to place stones on
                stone_type: Cut type for all stones
                start_angle: Starting angle in degrees (0 = top center)
                arc_angle: Arc span in degrees (360 = full circle, 180 = half)
                name_prefix: Prefix for stone names
                layer: Layer to create stones on
            
            Returns:
                JSON string with info about all created stones
            """
            return self._array_stones_on_ring(ctx, stone_diameter, stone_count, ring_diameter, stone_type, start_angle, arc_angle, name_prefix, layer)
        
        @self.app.tool()
        def create_complete_ring(
            ctx: Context,
            ring_size: float = 17.0,
            band_width: float = 2.0,
            band_thickness: float = 1.5,
            band_profile: str = "half_round",
            stone_count: int = 7,
            stone_diameter: float = 3.0,
            stone_type: str = "round_brilliant",
            prong_count_per_stone: int = 4,
            include_gallery: bool = True,
            name: Optional[str] = "Ring",
            base_layer: str = "Ring"
        ) -> str:
            """Create a complete ring with band, stones, and settings in one operation.
            
            This is a high-level tool that creates a complete ring design including:
            - Ring band
            - Arrayed gemstones
            - Prong settings for each stone
            - Optional gallery rail
            
            Args:
                ring_size: Inner diameter in mm
                band_width: Band width in mm
                band_thickness: Band thickness in mm
                band_profile: Band profile type
                stone_count: Number of stones
                stone_diameter: Diameter of each stone
                stone_type: Cut type for stones
                prong_count_per_stone: Prongs per stone (4, 6, or 8)
                include_gallery: Whether to add gallery rail
                name: Name for the ring assembly
                base_layer: Base layer name (sub-layers will be created)
            
            Returns:
                JSON string with complete ring assembly info
            """
            return self._create_complete_ring(ctx, ring_size, band_width, band_thickness, band_profile, 
                                             stone_count, stone_diameter, stone_type, prong_count_per_stone,
                                             include_gallery, name, base_layer)
    
    # Implementation methods that call Rhino
    def _create_ring_band(self, ctx, inner_diameter, width, thickness, profile, name, layer):
        from .rhino_tools import get_rhino_connection
        
        code = f"""
import rhinoscriptsyntax as rs
import math

# Create ring band with profile: {profile}
inner_radius = {inner_diameter} / 2.0
width = {width}
thickness = {thickness}

# Create layer
if not rs.IsLayer("{layer}"):
    rs.AddLayer("{layer}")
rs.CurrentLayer("{layer}")

# Create base circle
center = [0, 0, 0]
normal = [0, 0, 1]
circle = rs.AddCircle(center, inner_radius + width/2, normal)

# Create profile curve based on type
if "{profile}" == "flat":
    profile_pts = [
        [-width/2, -thickness/2, 0],
        [width/2, -thickness/2, 0],
        [width/2, thickness/2, 0],
        [-width/2, thickness/2, 0],
        [-width/2, -thickness/2, 0]
    ]
    profile_curve = rs.AddPolyline(profile_pts)
elif "{profile}" == "half_round":
    profile_curve = rs.AddArc([-width/2, 0, 0], [0, thickness/2, 0], [width/2, 0, 0])
    base_line = rs.AddLine([-width/2, 0, 0], [width/2, 0, 0])
    profile_curve = rs.JoinCurves([profile_curve, base_line], True)[0]
elif "{profile}" == "comfort_fit":
    # Flat top, rounded inside
    outer_arc = rs.AddLine([-width/2, thickness/2, 0], [width/2, thickness/2, 0])
    inner_arc = rs.AddArc([width/2, thickness/2, 0], [0, -thickness/4, 0], [-width/2, thickness/2, 0])
    profile_curve = rs.JoinCurves([outer_arc, inner_arc], True)[0]
else:  # d_shape
    flat_top = rs.AddLine([-width/2, thickness/2, 0], [width/2, thickness/2, 0])
    curved_bottom = rs.AddArc([width/2, thickness/2, 0], [0, 0, 0], [-width/2, thickness/2, 0])
    profile_curve = rs.JoinCurves([flat_top, curved_bottom], True)[0]

# Sweep to create band
band = rs.AddSweep1(circle, profile_curve)

# Clean up curves
rs.DeleteObject(circle)
rs.DeleteObject(profile_curve)

# Set attributes
if band and "{name}":
    rs.ObjectName(band[0], "{name}")

result = str(band[0]) if band else "Failed"
print(result)
"""
        
        try:
            connection = get_rhino_connection()
            result = connection.send_command("execute", {"code": code})
            return json.dumps({"success": True, "object_id": result.get("result"), "type": "ring_band"})
        except Exception as e:
            logger.error(f"Error creating ring band: {e}")
            return json.dumps({"success": False, "error": str(e)})
    
    def _create_gemstone(self, ctx, stone_type, diameter, position, name, layer):
        from .rhino_tools import get_rhino_connection
        
        # Stone proportions (diameter : height ratios)
        stone_ratios = {
            "round_brilliant": 0.6,  # 60% height to diameter
            "emerald_cut": 0.65,
            "princess": 0.7,
            "oval": 0.6,
            "cushion": 0.65,
            "pear": 0.6
        }
        
        height = diameter * stone_ratios.get(stone_type, 0.6)
        
        code = f"""
import rhinoscriptsyntax as rs
import math

# Create gemstone: {stone_type}
diameter = {diameter}
height = {height}
position = {position}

# Create layer
if not rs.IsLayer("{layer}"):
    rs.AddLayer("{layer}")
rs.CurrentLayer("{layer}")

# Create basic brilliant cut approximation
radius = diameter / 2

# Crown (top part)
crown_height = height * 0.4
crown_center = [position[0], position[1], position[2] + crown_height]
crown = rs.AddCone(position, crown_center, radius)

# Pavilion (bottom part - inverted cone)
pavilion_height = height * 0.6
pavilion_center = [position[0], position[1], position[2] - pavilion_height]
pavilion = rs.AddCone(position, pavilion_center, radius)

# Join into single object
gemstone = rs.BooleanUnion([crown, pavilion])

if gemstone and "{name}":
    rs.ObjectName(gemstone[0], "{name}")

result = str(gemstone[0]) if gemstone else str(crown)
print(result)
"""
        
        try:
            connection = get_rhino_connection()
            result = connection.send_command("execute", {"code": code})
            return json.dumps({"success": True, "object_id": result.get("result"), "type": "gemstone"})
        except Exception as e:
            logger.error(f"Error creating gemstone: {e}")
            return json.dumps({"success": False, "error": str(e)})
    
    def _create_prong_setting(self, ctx, stone_diameter, prong_count, prong_height, prong_diameter, position, style, name, layer):
        from .rhino_tools import get_rhino_connection
        
        code = f"""
import rhinoscriptsyntax as rs
import math

# Create prong setting
stone_diameter = {stone_diameter}
prong_count = {prong_count}
prong_height = {prong_height}
prong_diameter = {prong_diameter}
position = {position}

# Create layer
if not rs.IsLayer("{layer}"):
    rs.AddLayer("{layer}")
rs.CurrentLayer("{layer}")

prongs = []
stone_radius = stone_diameter / 2
offset_radius = stone_radius + prong_diameter / 2

for i in range(prong_count):
    angle = (360.0 / prong_count) * i
    angle_rad = math.radians(angle)
    
    # Prong base position
    x = position[0] + offset_radius * math.cos(angle_rad)
    y = position[1] + offset_radius * math.sin(angle_rad)
    z = position[2]
    
    base_point = [x, y, z]
    
    if "{style}" == "tapered":
        # Tapered prong
        tip_point = [x, y, z + prong_height]
        prong = rs.AddCone(base_point, tip_point, prong_diameter/2, prong_diameter/4)
    elif "{style}" == "claw":
        # Curved claw prong
        mid_point = [x * 0.95, y * 0.95, z + prong_height * 0.6]
        tip_point = [x * 0.9, y * 0.9, z + prong_height]
        prong = rs.AddCylinder(base_point, prong_height, prong_diameter/2)
    else:  # straight
        # Straight cylindrical prong
        prong = rs.AddCylinder(base_point, prong_height, prong_diameter/2)
    
    if prong:
        prongs.append(prong)

# Group prongs
if prongs:
    group = rs.AddGroup()
    rs.AddObjectsToGroup(prongs, group)
    if "{name}":
        rs.GroupName(group, "{name}")

result = str(prongs[0]) if prongs else "Failed"
print(result)
"""
        
        try:
            connection = get_rhino_connection()
            result = connection.send_command("execute", {"code": code})
            return json.dumps({"success": True, "object_id": result.get("result"), "type": "prong_setting", "prong_count": prong_count})
        except Exception as e:
            logger.error(f"Error creating prong setting: {e}")
            return json.dumps({"success": False, "error": str(e)})
    
    def _create_gallery_rail(self, ctx, ring_diameter, stone_count, stone_diameter, rail_width, rail_height, name, layer):
        from .rhino_tools import get_rhino_connection
        
        code = f"""
import rhinoscriptsyntax as rs
import math

# Create gallery rail
ring_radius = {ring_diameter} / 2
stone_count = {stone_count}
stone_diameter = {stone_diameter}
rail_width = {rail_width}
rail_height = {rail_height}

# Create layer
if not rs.IsLayer("{layer}"):
    rs.AddLayer("{layer}")
rs.CurrentLayer("{layer}")

# Calculate arc span for stones
stone_spacing = stone_diameter * 1.2
total_arc = stone_spacing * stone_count
arc_angle = (total_arc / (2 * math.pi * ring_radius)) * 360

# Create rail path (arc on ring)
center = [0, 0, 0]
start_angle = 90 - arc_angle/2
end_angle = 90 + arc_angle/2
rail_path = rs.AddArc3Pt(
    [ring_radius * math.cos(math.radians(start_angle)), ring_radius * math.sin(math.radians(start_angle)), 0],
    [0, ring_radius, rail_height/2],
    [ring_radius * math.cos(math.radians(end_angle)), ring_radius * math.sin(math.radians(end_angle)), 0]
)

# Create rail profile
profile_pts = [
    [-rail_width/2, 0, 0],
    [rail_width/2, 0, 0],
    [rail_width/2, rail_height, 0],
    [-rail_width/2, rail_height, 0],
    [-rail_width/2, 0, 0]
]
profile = rs.AddPolyline(profile_pts)

# Sweep to create gallery
gallery = rs.AddSweep1(rail_path, profile)

# Clean up
rs.DeleteObject(rail_path)
rs.DeleteObject(profile)

if gallery and "{name}":
    rs.ObjectName(gallery[0], "{name}")

result = str(gallery[0]) if gallery else "Failed"
print(result)
"""
        
        try:
            connection = get_rhino_connection()
            result = connection.send_command("execute", {"code": code})
            return json.dumps({"success": True, "object_id": result.get("result"), "type": "gallery_rail"})
        except Exception as e:
            logger.error(f"Error creating gallery rail: {e}")
            return json.dumps({"success": False, "error": str(e)})
    
    def _array_stones_on_ring(self, ctx, stone_diameter, stone_count, ring_diameter, stone_type, start_angle, arc_angle, name_prefix, layer):
        from .rhino_tools import get_rhino_connection
        
        stone_ratios = {
            "round_brilliant": 0.6,
            "emerald_cut": 0.65,
            "princess": 0.7,
            "oval": 0.6,
            "cushion": 0.65,
            "pear": 0.6
        }
        height = stone_diameter * stone_ratios.get(stone_type, 0.6)
        
        code = f"""
import rhinoscriptsyntax as rs
import math

# Array stones on ring
stone_diameter = {stone_diameter}
stone_count = {stone_count}
ring_radius = {ring_diameter} / 2
height = {height}
start_angle = {start_angle}
arc_angle = {arc_angle}

# Create layer
if not rs.IsLayer("{layer}"):
    rs.AddLayer("{layer}")
rs.CurrentLayer("{layer}")

stones = []
angle_step = arc_angle / (stone_count - 1) if stone_count > 1 else 0

for i in range(stone_count):
    angle = start_angle + (angle_step * i)
    angle_rad = math.radians(angle)
    
    # Position on ring
    x = ring_radius * math.cos(angle_rad)
    y = ring_radius * math.sin(angle_rad)
    z = 2.0  # Raise stones above band
    
    position = [x, y, z]
    radius = stone_diameter / 2
    
    # Create stone (simplified brilliant cut)
    crown_height = height * 0.4
    pavilion_height = height * 0.6
    
    crown_center = [x, y, z + crown_height]
    crown = rs.AddCone(position, crown_center, radius)
    
    pavilion_center = [x, y, z - pavilion_height]
    pavilion = rs.AddCone(position, pavilion_center, radius)
    
    # Join
    stone = rs.BooleanUnion([crown, pavilion])
    if stone:
        rs.ObjectName(stone[0], "{name_prefix}_{{}}".format(i+1))
        stones.append(str(stone[0]))

print(",".join(stones))
"""
        
        try:
            connection = get_rhino_connection()
            result = connection.send_command("execute", {"code": code})
            stone_ids = result.get("result", "").split(",")
            return json.dumps({"success": True, "stone_ids": stone_ids, "count": len(stone_ids)})
        except Exception as e:
            logger.error(f"Error arraying stones: {e}")
            return json.dumps({"success": False, "error": str(e)})
    
    def _create_complete_ring(self, ctx, ring_size, band_width, band_thickness, band_profile, 
                             stone_count, stone_diameter, stone_type, prong_count_per_stone,
                             include_gallery, name, base_layer):
        """Create complete ring by calling other tools."""
        results = {}
        
        # 1. Create band
        band_result = self._create_ring_band(ctx, ring_size, band_width, band_thickness, band_profile, 
                                            f"{name}_Band", f"{base_layer}::Band")
        results["band"] = json.loads(band_result)
        
        # 2. Array stones
        stones_result = self._array_stones_on_ring(ctx, stone_diameter, stone_count, ring_size, 
                                                   stone_type, 0, 180, f"{name}_Stone", 
                                                   f"{base_layer}::Stones")
        results["stones"] = json.loads(stones_result)
        
        # 3. Create prongs for each stone
        # (simplified - would need stone positions)
        
        # 4. Optional gallery
        if include_gallery:
            gallery_result = self._create_gallery_rail(ctx, ring_size, stone_count, stone_diameter, 
                                                       band_width, band_thickness * 1.5, 
                                                       f"{name}_Gallery", f"{base_layer}::Gallery")
            results["gallery"] = json.loads(gallery_result)
        
        return json.dumps({"success": True, "components": results, "name": name})
