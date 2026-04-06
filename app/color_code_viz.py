"""Colour code lattice visualizations.

Provides Plotly-based hexagonal lattice visualization for colour codes,
with 3-colourable plaquettes (R/G/B), syndrome highlighting, and animated
glow effects for triggered stabilizers.

Reference: Lee & Brown, Quantum 9, 1609 (2025)
"""
from __future__ import annotations

from typing import Any, Optional, Tuple, List, Dict
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_hexagonal_lattice(
    distance: int = 5,
    highlight_syndrome: Optional[np.ndarray] = None,
    highlight_color: Optional[str] = None,
    title: str = "Colour Code Lattice"
) -> go.Figure:
    """Create a hexagonal lattice visualization for colour codes.
    
    Args:
        distance: Code distance (must be odd for triangular patches)
        highlight_syndrome: Optional array of syndrome triggers to highlight
        highlight_color: Optional color to highlight triggered stabilizers
        title: Plot title
    
    Returns:
        Plotly Figure with hexagonal lattice
    """
    # Create hexagonal lattice coordinates
    # Triangular patch on hexagonal lattice
    sqrt3 = np.sqrt(3)
    
    fig = go.Figure()
    
    # Generate hexagonal grid coordinates
    n_rows = distance + 2
    n_cols = distance + 2
    
    data_qubits: List[Tuple[float, float]] = []
    x_ancillas: List[Tuple[float, float]] = []
    z_ancillas: List[Tuple[float, float]] = []
    
    # Colour code has 3-colourable plaquettes
    plaquettes_r: List[List[Tuple[float, float]]] = []  # Red (X) plaquettes
    plaquettes_g: List[List[Tuple[float, float]]] = []  # Green (X/Z) plaquettes
    plaquettes_b: List[List[Tuple[float, float]]] = []  # Blue (Z) plaquettes
    
    # Generate coordinates for a triangular patch
    for row in range(-distance, distance + 1):
        for col in range(-distance, distance + 1):
            # Hexagonal lattice coordinates
            x = col * 1.5
            y = row * sqrt3 + (col % 2) * sqrt3 / 2
            
            # Only include points inside the triangular boundary
            if abs(row) + abs(col) <= distance:
                if (row + col) % 3 == 0:
                    data_qubits.append((x, y))
                elif (row + col) % 3 == 1:
                    x_ancillas.append((x, y))
                else:
                    z_ancillas.append((x, y))
    
    # Create plaquette polygons around ancillas
    for ax, ay in x_ancillas:
        vertices = _get_hexagon_vertices(ax, ay, radius=1.0)
        plaquettes_r.append(vertices)
    
    for ax, ay in z_ancillas:
        vertices = _get_hexagon_vertices(ax, ay, radius=1.0)
        plaquettes_b.append(vertices)
    
    # Draw plaquettes (fill colours based on type)
    for i, vertices in enumerate(plaquettes_r):
        color = "rgba(255, 100, 100, 0.3)"  # Red tint
        line_color = "rgba(200, 50, 50, 0.8)"
        
        fig.add_trace(go.Scatter(
            x=[v[0] for v in vertices] + [vertices[0][0]],
            y=[v[1] for v in vertices] + [vertices[0][1]],
            fill="toself",
            fillcolor=color,
            line=dict(color=line_color, width=1),
            mode="lines",
            name=f"X-plaquette {i}",
            showlegend=False,
            hoverinfo="skip"
        ))
    
    for i, vertices in enumerate(plaquettes_b):
        color = "rgba(100, 100, 255, 0.3)"  # Blue tint
        line_color = "rgba(50, 50, 200, 0.8)"
        
        fig.add_trace(go.Scatter(
            x=[v[0] for v in vertices] + [vertices[0][0]],
            y=[v[1] for v in vertices] + [vertices[0][1]],
            fill="toself",
            fillcolor=color,
            line=dict(color=line_color, width=1),
            mode="lines",
            name=f"Z-plaquette {i}",
            showlegend=False,
            hoverinfo="skip"
        ))
    
    # Draw data qubits (dark circles)
    if data_qubits:
        fig.add_trace(go.Scatter(
            x=[q[0] for q in data_qubits],
            y=[q[1] for q in data_qubits],
            mode="markers",
            marker=dict(size=12, color="rgba(50, 50, 50, 0.9)", line=dict(width=2, color="white")),
            name="Data Qubits",
            hovertemplate="Data qubit<br>x: %{x:.2f}<br>y: %{y:.2f}"
        ))
    
    # Draw X ancillas (red squares)
    if x_ancillas:
        fig.add_trace(go.Scatter(
            x=[a[0] for a in x_ancillas],
            y=[a[1] for a in x_ancillas],
            mode="markers",
            marker=dict(
                size=10, 
                color="rgba(255, 100, 100, 0.8)", 
                symbol="square",
                line=dict(width=2, color="darkred")
            ),
            name="X Ancillas",
            hovertemplate="X ancilla<br>x: %{x:.2f}<br>y: %{y:.2f}"
        ))
    
    # Draw Z ancillas (blue diamonds)
    if z_ancillas:
        fig.add_trace(go.Scatter(
            x=[a[0] for a in z_ancillas],
            y=[a[1] for a in z_ancillas],
            mode="markers",
            marker=dict(
                size=10, 
                color="rgba(100, 100, 255, 0.8)", 
                symbol="diamond",
                line=dict(width=2, color="darkblue")
            ),
            name="Z Ancillas",
            hovertemplate="Z ancilla<br>x: %{x:.2f}<br>y: %{y:.2f}"
        ))
    
    # Layout settings
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=16)),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[-distance * 2, distance * 2]
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            scaleanchor="x",
            scaleratio=1,
            range=[-distance * 2, distance * 2]
        ),
        plot_bgcolor="rgba(10, 15, 30, 1)",
        paper_bgcolor="rgba(10, 15, 30, 1)",
        font=dict(color="white"),
        legend=dict(
            x=0.02, y=0.98,
            bgcolor="rgba(20, 30, 50, 0.8)",
            bordercolor="rgba(100, 150, 255, 0.3)",
            borderwidth=1
        ),
        margin=dict(l=20, r=20, t=60, b=20),
        hovermode="closest"
    )
    
    return fig


def _get_hexagon_vertices(cx: float, cy: float, radius: float = 1.0) -> List[Tuple[float, float]]:
    """Get vertices of a regular hexagon centered at (cx, cy)."""
    vertices = []
    for i in range(6):
        angle = i * np.pi / 3
        x = cx + radius * np.cos(angle)
        y = cy + radius * np.sin(angle)
        vertices.append((x, y))
    return vertices


def create_colour_code_syndrome_heatmap(
    syndrome: np.ndarray,
    distance: int = 5,
    title: str = "Colour Code Syndrome"
) -> go.Figure:
    """Create a heatmap visualization of colour code syndrome.
    
    Args:
        syndrome: Syndrome measurement array
        distance: Code distance
        title: Plot title
    
    Returns:
        Plotly Figure with syndrome heatmap
    """
    # Create grid for syndrome visualization
    n = len(syndrome)
    grid_size = int(np.ceil(np.sqrt(n)))
    
    # Pad syndrome to fit grid
    padded = np.zeros(grid_size * grid_size)
    padded[:n] = syndrome[:n]
    grid = padded.reshape(grid_size, grid_size)
    
    fig = go.Figure(data=go.Heatmap(
        z=grid,
        colorscale=[[0, "rgba(10, 20, 40, 1)"], [1, "rgba(255, 100, 100, 1)"]],
        showscale=True,
        colorbar=dict(
            title="Defect",
            titleside="right",
            tickvals=[0, 1],
            ticktext=["No", "Yes"]
        ),
        hovertemplate="Row: %{y}<br>Col: %{x}<br>Defect: %{z}<extra></extra>"
    ))
    
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=14)),
        xaxis=dict(showticklabels=False, showgrid=False),
        yaxis=dict(showticklabels=False, showgrid=False, scaleanchor="x"),
        plot_bgcolor="rgba(10, 15, 30, 1)",
        paper_bgcolor="rgba(10, 15, 30, 1)",
        font=dict(color="white"),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    
    return fig


def create_live_colour_code_visualization(
    syndrome: np.ndarray,
    action: Optional[np.ndarray] = None,
    distance: int = 5,
    frame_num: int = 0,
    total_frames: int = 30
) -> Dict[str, Any]:
    """Create animated colour code visualization for live RL training.
    
    Args:
        syndrome: Current syndrome state
        action: Optional action taken by agent
        distance: Code distance
        frame_num: Current animation frame
        total_frames: Total number of frames in animation
    
    Returns:
        Dict with plotly_figure and metadata for Streamlit rendering
    """
    fig = create_hexagonal_lattice(
        distance=distance,
        highlight_syndrome=syndrome,
        title=f"Colour Code d={distance} — Live Training"
    )
    
    # Add glow effect based on frame
    pulse = 1.0 + 0.3 * np.sin(2 * np.pi * frame_num / total_frames)
    
    return {
        "plotly_figure": fig,
        "frame_num": frame_num,
        "total_frames": total_frames,
        "pulse": pulse,
        "syndrome_sum": int(np.sum(syndrome)) if syndrome is not None else 0,
        "action": action.tolist() if action is not None else None
    }


def create_colour_code_threshold_figure(
    data: Dict[int, List[Tuple[float, float]]],
    threshold_p: Optional[float] = None,
    title: str = "Colour Code Threshold"
) -> go.Figure:
    """Create threshold plot for colour code families.
    
    Args:
        data: Dict mapping distance -> [(p, p_L), ...] points
        threshold_p: Optional estimated threshold value
        title: Plot title
    
    Returns:
        Plotly Figure with threshold curves
    """
    fig = go.Figure()
    
    colors = ["#ff6b6b", "#4ecdc4", "#45b7d1", "#96ceb4", "#feca57"]
    
    for i, (d, points) in enumerate(sorted(data.items())):
        if not points:
            continue
        p_vals, pL_vals = zip(*sorted(points))
        
        fig.add_trace(go.Scatter(
            x=p_vals,
            y=pL_vals,
            mode="lines+markers",
            name=f"d={d}",
            line=dict(color=colors[i % len(colors)], width=2.5),
            marker=dict(size=8)
        ))
    
    if threshold_p is not None:
        fig.add_vline(
            x=threshold_p,
            line=dict(color="white", width=2, dash="dash"),
            annotation_text=f"p_th ≈ {threshold_p:.4f}",
            annotation_position="top"
        )
    
    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis=dict(
            title="Physical Error Rate (p)",
            type="log",
            gridcolor="rgba(100, 100, 150, 0.2)"
        ),
        yaxis=dict(
            title="Logical Error Rate (p_L)",
            type="log",
            gridcolor="rgba(100, 100, 150, 0.2)"
        ),
        plot_bgcolor="rgba(10, 15, 30, 1)",
        paper_bgcolor="rgba(10, 15, 30, 1)",
        font=dict(color="white"),
        legend=dict(
            x=0.02, y=0.98,
            bgcolor="rgba(20, 30, 50, 0.8)"
        )
    )
    
    return fig
