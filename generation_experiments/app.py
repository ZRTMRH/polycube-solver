"""
Streamlit webapp to explore polycube generation modes.
Run with: streamlit run generation_experiments/app.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import streamlit as st
import plotly.graph_objects as go
import numpy as np
from phase2.data_generator import (
    enumerate_polycubes,
    generate_puzzle_instances,
    generate_constructive_puzzle_instances,
)
from robust_generator import build_robust_constructive_case

PIECE_COLORS = [
    '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
    '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabed4',
    '#469990', '#dcbeff', '#9A6324', '#800000', '#aaffc3',
    '#808000', '#ffd8b1', '#000075', '#a9a9a9', '#ffffff',
]


@st.cache_resource
def get_catalog():
    return enumerate_polycubes(max_size=5)


def _unit_cube_mesh(x, y, z, vert_offset):
    """Return vertices and triangle indices for a unit cube at (x,y,z)."""
    vx = [x, x+1, x,   x+1, x,   x+1, x,   x+1]
    vy = [y, y,   y+1, y+1, y,   y,   y+1, y+1]
    vz = [z, z,   z,   z,   z+1, z+1, z+1, z+1]
    o = vert_offset
    faces = [
        (o+0, o+2, o+1), (o+1, o+2, o+3),  # bottom
        (o+4, o+5, o+6), (o+5, o+7, o+6),  # top
        (o+0, o+1, o+4), (o+1, o+5, o+4),  # front
        (o+2, o+6, o+3), (o+3, o+6, o+7),  # back
        (o+0, o+4, o+2), (o+2, o+4, o+6),  # left
        (o+1, o+3, o+5), (o+3, o+7, o+5),  # right
    ]
    return vx, vy, vz, faces


def make_voxel_figure(solution, grid_size, title="", opacity=1.0):
    fig = go.Figure()
    for piece_idx, cells in solution.items():
        color = PIECE_COLORS[piece_idx % len(PIECE_COLORS)]
        all_x, all_y, all_z = [], [], []
        all_i, all_j, all_k = [], [], []

        for cell in cells:
            offset = len(all_x)
            vx, vy, vz, faces = _unit_cube_mesh(*cell, offset)
            all_x.extend(vx)
            all_y.extend(vy)
            all_z.extend(vz)
            for fi, fj, fk in faces:
                all_i.append(fi)
                all_j.append(fj)
                all_k.append(fk)

        group = f"piece_{piece_idx}"
        piece_label = f"Piece {piece_idx} ({len(cells)} cells)"

        fig.add_trace(go.Mesh3d(
            x=all_x, y=all_y, z=all_z,
            i=all_i, j=all_j, k=all_k,
            color=color,
            opacity=opacity,
            flatshading=True,
            lighting=dict(ambient=0.6, diffuse=0.8, specular=0.2),
            lightposition=dict(x=1, y=2, z=3),
            name=piece_label,
            legendgroup=group,
            showlegend=True,
            hovertemplate=f"Piece {piece_idx}<extra></extra>",
        ))

        # Batch all wireframe edges for this piece into one trace
        edges_x, edges_y, edges_z = [], [], []
        edge_pairs = [
            (0,1),(2,3),(4,5),(6,7),
            (0,2),(1,3),(4,6),(5,7),
            (0,4),(1,5),(2,6),(3,7),
        ]
        for (x, y, z) in cells:
            corners = [(x+dx, y+dy, z+dz) for dx in (0,1) for dy in (0,1) for dz in (0,1)]
            for a, b in edge_pairs:
                edges_x += [corners[a][0], corners[b][0], None]
                edges_y += [corners[a][1], corners[b][1], None]
                edges_z += [corners[a][2], corners[b][2], None]
        fig.add_trace(go.Scatter3d(
            x=edges_x, y=edges_y, z=edges_z,
            mode='lines',
            line=dict(color='rgba(0,0,0,0.3)', width=1),
            legendgroup=group,
            showlegend=False,
            hoverinfo='skip',
        ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(range=[-1, grid_size + 1], title='X'),
            yaxis=dict(range=[-1, grid_size + 1], title='Y'),
            zaxis=dict(range=[-1, grid_size + 1], title='Z'),
            aspectmode='cube',
        ),
        showlegend=True,
        legend=dict(x=1.0, y=0.5),
        margin=dict(l=0, r=140, t=40, b=0),
        height=550,
    )
    return fig


# ── UI ────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Polycube Generator", layout="wide")
st.title("Polycube Generation Explorer")

with st.sidebar:
    st.header("Settings")

    mode = st.selectbox("Generation mode", [
        "DLX (random verified)",
        "Constructive: connected",
        "Constructive: mixed",
        "Constructive: striped",
        "Robust (surface peeling)",
    ])

    grid_size = st.number_input("Grid size", min_value=3, value=4, step=1)
    seed = st.number_input("Seed", value=42, step=1)
    n_instances = st.number_input("Number of instances", min_value=1, value=1, step=1)
    opacity = st.slider("Opacity", min_value=0.1, max_value=1.0, value=1.0, step=0.05)

    generate = st.button("Generate", type="primary")

if generate:
    catalog = get_catalog()
    with st.spinner("Generating..."):
        try:
            if mode == "DLX (random verified)":
                instances = generate_puzzle_instances(
                    num_instances=n_instances,
                    grid_size=grid_size,
                    polycube_catalog=catalog,
                    dlx_timeout=15.0,
                    verbose=False,
                    seed=seed,
                )
            elif mode == "Robust (surface peeling)":
                instances = []
                for i in range(n_instances):
                    abs_pieces = build_robust_constructive_case(grid_size, seed=seed + i)
                    solution = {
                        idx: frozenset(tuple(c) for c in piece)
                        for idx, piece in enumerate(abs_pieces)
                    }
                    instances.append({
                        'pieces': abs_pieces,
                        'solution': solution,
                        'grid_size': grid_size,
                        'instance_source': 'robust',
                    })
            else:
                variant = {
                    "Constructive: connected": "connected",
                    "Constructive: mixed": "mixed",
                    "Constructive: striped": "striped",
                }[mode]
                instances = generate_constructive_puzzle_instances(
                    num_instances=n_instances,
                    grid_size=grid_size,
                    seed=seed,
                    large_suite_type=variant,
                    verbose=False,
                )
        except Exception as e:
            st.error(f"Generation failed: {e}")
            st.stop()
    st.session_state['instances'] = instances
    st.session_state['grid_size'] = grid_size

if 'instances' in st.session_state:
    instances = st.session_state['instances']
    gs = st.session_state['grid_size']
    st.success(f"Generated {len(instances)} instances")
    cols_per_row = 2
    for row_start in range(0, len(instances), cols_per_row):
        cols = st.columns(cols_per_row)
        for col_i, inst_i in enumerate(range(row_start, min(row_start + cols_per_row, len(instances)))):
            inst = instances[inst_i]
            n_pieces = len(inst['pieces'])
            source = inst.get('instance_source', 'dlx')
            title = f"Instance {inst_i + 1} — {n_pieces} pieces ({source})"
            fig = make_voxel_figure(inst['solution'], gs, title=title, opacity=opacity)
            with cols[col_i]:
                st.plotly_chart(fig, use_container_width=True)
                sizes = sorted([len(p) for p in inst['pieces']])
                st.caption(f"Piece sizes: {sizes}")
else:
    st.info("Configure settings in the sidebar and click **Generate**.")
