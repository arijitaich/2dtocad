"""Test vertex welding fix for matrix skin grouping"""
from mesh_to_cad_grid import MeshGridTessellator
import os
from collections import defaultdict
import numpy as np

# Use existing test model
test_model = 'manufacturing_output/20260111_071245/3d_models/Gold_Ring_Mounting__Shank_and_Setting__3d.glb'
output_dir = 'manufacturing_output/test_welded'
os.makedirs(output_dir, exist_ok=True)

print('Loading mesh...')
tessellator = MeshGridTessellator(test_model)

print('Creating matrix skin (skipping degenerate triangles)...')
skin_result = tessellator.create_matrix_skin(subdivisions=1, offset=0.0)

# Debug: analyze adjacency
print('\n' + '='*60)
print('CHECKING ADJACENCY AFTER FIX')
print('='*60)

quads = tessellator.skin_quads
print(f'Total quads: {len(quads):,}')

# Build full adjacency graph
edge_to_quads = defaultdict(list)
for quad_idx, quad in enumerate(quads):
    for i in range(len(quad)):
        j = (i + 1) % len(quad)
        edge = tuple(sorted([quad[i], quad[j]]))
        edge_to_quads[edge].append(quad_idx)

# Count neighbors per quad
quad_neighbors = defaultdict(set)
for edge, quad_list in edge_to_quads.items():
    if len(quad_list) == 2:
        q1, q2 = quad_list
        quad_neighbors[q1].add(q2)
        quad_neighbors[q2].add(q1)

isolated = sum(1 for q in range(len(quads)) if len(quad_neighbors[q]) == 0)
has_4plus = sum(1 for q in range(len(quads)) if len(quad_neighbors[q]) >= 4)

print(f'Isolated quads: {isolated:,} (should be ~0)')
print(f'Well-connected quads (4+ neighbors): {has_4plus:,}')

if isolated < 100:  # Good enough to run grouping
    print('\nGrouping quads by angle (180 degree threshold)...')
    result = tessellator.group_quads_by_angle(angle_threshold=180.0)
    groups = result['groups']
    
    print('\n' + '='*60)
    print('FINAL GROUPING RESULTS:')
    print('='*60)
    print(f'Total groups: {len(groups)}')
    
    sorted_groups = sorted(groups.items(), key=lambda x: -len(x[1]))
    for i, (group_id, quad_list) in enumerate(sorted_groups[:10]):
        pct = 100 * len(quad_list) / len(quads)
        print(f"  Group {group_id}: {len(quad_list):,} quads ({pct:.1f}%)")
    
    if len(groups) > 10:
        print(f'  ... and {len(groups) - 10} more groups')

print('\nDone!')
