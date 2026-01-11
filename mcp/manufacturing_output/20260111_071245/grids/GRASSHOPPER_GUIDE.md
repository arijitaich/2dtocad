# MATRIX SKIN - GRASSHOPPER EDITING GUIDE
==========================================

This guide explains how to edit **matrix_skin.obj** files in Grasshopper for parametric control over ANY 3D model.

## What is Matrix Skin?

The matrix skin is a connected 2D sheet of quad faces that wraps around your 3D model's surface:
- Every square shares edges with its neighbors
- You can select and move groups of connected faces
- Perfect for parametric deformations and CAD editing

---

## QUICK START (5 minutes)

### 1. Import the Matrix Skin into Rhino

```
1. Open Rhino 7 or 8
2. File > Import > Select "matrix_skin.obj"
3. The connected quad mesh appears in your viewport
```

### 2. Simple Manual Editing

**No scripting needed - just use Rhino tools:**

| Task | Method |
|------|--------|
| Select region | Click mesh, use Ctrl+Shift to select faces |
| Move region | Gumball drag or `Move` command |
| Scale region | `Scale` command or Gumball handles |
| Stretch vertically | `Scale1D` in Z direction |
| Thicken section | `Scale2D` in XY plane |

---

## OPTION 1: Simple Grasshopper Setup

**For basic parametric control without Python:**

### A. Import Mesh
1. Open Grasshopper (type `Grasshopper` in Rhino)
2. Add `Mesh` parameter (Params > Geometry > Mesh)
3. Right-click > "Set One Mesh" > Select your imported matrix_skin

### B. Add Deformation Components
Useful components for mesh editing:

| Component | Purpose |
|-----------|---------|
| `Mesh Edit` | Move individual vertices |
| `Mesh Scale` | Scale entire mesh |
| `Box Morph` | Deform within a bounding box |
| `Spatial Deform` | Deform based on control points |
| `Mesh Smooth` | Smooth surface |

### C. Create Number Sliders
1. Add `Number Slider` components for each parameter
2. Connect to deformation inputs
3. Slide to see real-time changes!

---

## OPTION 2: Python Script (Full Control)

**Use the included `parametric_editor.gh.py` for automatic region detection.**

### Setup Steps:

1. **Add Python Component**
   - In Grasshopper: Math > Script > Python 3 Script
   - (Or GhPython if using older Rhino)

2. **Copy the Script**
   - Open `parametric_editor.gh.py` in text editor
   - Copy entire contents
   - Paste into the Python component

3. **Add Inputs** (right-click component > + Input):

   | Input Name | Type | Default | Description |
   |------------|------|---------|-------------|
   | mesh | Mesh | - | The matrix skin mesh |
   | top_height_adjust | float | 0.0 | Move top region up/down |
   | top_scale | float | 1.0 | Scale top region XY |
   | middle_thickness | float | 1.0 | Scale middle section |
   | bottom_height_adjust | float | 0.0 | Move bottom region up/down |
   | top_threshold | float | 0.7 | Define "top" region (0-1) |
   | bottom_threshold | float | 0.3 | Define "bottom" region (0-1) |

4. **Add Outputs**:
   - `edited_mesh` (Mesh) - The modified mesh
   - `top_points` (Point List) - Visualization of top region
   - `region_info` (Text) - Info about detected regions

5. **Connect Number Sliders**:
   - Add sliders for each parameter
   - Suggested ranges:
     - Height adjust: -1.0 to 1.0
     - Scale factors: 0.5 to 2.0
     - Thresholds: 0.0 to 1.0

---

## OPTION 3: Using Plugins

### Weaverbird (Free)
Best for mesh subdivision and smoothing:
```
1. Install: Food4Rhino > Weaverbird
2. Components to use:
   - Wb Mesh Thicken
   - Wb Catmull-Clark Subdivision  
   - Wb Constant Quad Subdivision
```

### Kangaroo (Physics-based)
For organic deformations:
```
1. Built into Rhino 7+
2. Use "Grab" component for interactive editing
3. Add "Anchor" to fix certain vertices
```

---

## Region Detection Explained

The script automatically divides your model into regions by **normalized Z-height**:

```
        ┌─────────┐
        │   TOP   │  ← Z > top_threshold (default 70%)
        │         │
        ├─────────┤
        │         │
        │ MIDDLE  │  ← Between thresholds
        │         │
        ├─────────┤
        │ BOTTOM  │  ← Z < bottom_threshold (default 30%)
        └─────────┘
```

**Adjust thresholds** to target different parts of your model:
- `top_threshold = 0.9` → Only the very top 10%
- `bottom_threshold = 0.1` → Only the very bottom 10%

---

## Common Editing Tasks

### Make Object Taller
```
top_height_adjust = 0.5  (move top up)
bottom_height_adjust = -0.3  (move bottom down)
```

### Make Object Wider
```
middle_thickness = 1.5  (150% width in middle)
```

### Make Top Larger
```
top_scale = 1.3  (130% size at top)
```

### Flatten Object
```
top_height_adjust = -0.2
bottom_height_adjust = 0.2
```

---

## Exporting Your Edits

### From Grasshopper:
1. Add `Bake` component or right-click output > Bake
2. Mesh appears in Rhino

### From Rhino:
| Format | Command | Use For |
|--------|---------|---------|
| OBJ | `Export` | Universal 3D format |
| STL | `Export` | 3D printing |
| 3DM | `Save` | Full Rhino format |
| STEP | `Export` | CAD interchange |

---

## Troubleshooting

**"No mesh connected"**
- Ensure the mesh is imported into Rhino first
- Right-click the Mesh parameter and "Set One Mesh"

**Regions not detected correctly**
- Adjust `top_threshold` and `bottom_threshold`
- Try values like 0.8/0.2 or 0.6/0.4

**Mesh appears distorted**
- Reduce adjustment values (use smaller numbers)
- Check that scale values are between 0.5 and 2.0

**Script errors in Python component**
- Ensure you're using Python 3 component (Rhino 7+)
- For older Rhino, use GhPython component

---

## File Locations

```
grids/
├── parametric_editor.gh.py      ← Generic Python script for Grasshopper
├── GRASSHOPPER_GUIDE.md         ← This guide
└── [model_name]/
    └── matrix_skin.obj          ← The connected quad mesh to edit
```

---

## Tips for Best Results

1. **Start small** - Use small adjustment values first (0.1, 0.2)
2. **Preview regions** - Connect `top_points` to a Point component to see what's selected
3. **Smooth after editing** - Use Weaverbird's smoothing for organic results
4. **Save checkpoints** - Bake intermediate results to preserve your work
5. **Combine transforms** - Adjust multiple parameters for complex edits
