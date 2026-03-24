"""Real-time error propagation visualization for RL training.

Provides animated, satisfying visual feedback showing:
- Error injection (X errors in red, Z errors in blue)
- Syndrome measurement (ancilla qubits firing)
- Decoder corrections (green overlay)
- Live lattice state updates
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from typing import Optional, Dict, Any
from functools import lru_cache


@lru_cache(maxsize=32)
def create_surface_code_lattice(distance: int) -> dict:
    """Create the surface code lattice structure (cached for performance).
    
    Returns coordinates for data qubits (at vertices) and 
    ancilla qubits (at plaquette centers).
    """
    # Data qubits are on a (2d-1) x (2d-1) grid at odd coordinates
    # Ancilla qubits are at even coordinates (for X-type) and centers (for Z-type)
    d = distance
    
    # Data qubits positions
    data_qubits = []
    for row in range(2*d - 1):
        for col in range(2*d - 1):
            if (row + col) % 2 == 1:  # Data qubits on checkerboard
                data_qubits.append({
                    'id': len(data_qubits),
                    'x': col,
                    'y': row,
                    'type': 'data'
                })
    
    # X-ancilla qubits (measure X stabilizers)
    x_ancilla = []
    for row in range(0, 2*d, 2):
        for col in range(0, 2*d, 2):
            if 0 < row < 2*d-1 or 0 < col < 2*d-1:  # Exclude corners for d>3
                x_ancilla.append({
                    'id': len(x_ancilla),
                    'x': col,
                    'y': row,
                    'type': 'x_ancilla'
                })
    
    # Z-ancilla qubits (measure Z stabilizers)  
    z_ancilla = []
    for row in range(2, 2*d-1, 2):
        for col in range(2, 2*d-1, 2):
            z_ancilla.append({
                'id': len(z_ancilla),
                'x': col,
                'y': row,
                'type': 'z_ancilla'
            })
    
    return {
        'data': data_qubits,
        'x_ancilla': x_ancilla,
        'z_ancilla': z_ancilla,
        'distance': d
    }


def animated_error_propagation_frame(
    lattice: dict,
    x_errors: Optional[np.ndarray] = None,
    z_errors: Optional[np.ndarray] = None,
    x_syndrome: Optional[np.ndarray] = None,
    z_syndrome: Optional[np.ndarray] = None,
    corrections: Optional[np.ndarray] = None,
    frame_num: int = 0,
    total_frames: int = 10,
    title: str = "Live Error Propagation"
) -> go.Figure:
    """Create a single animated frame showing error propagation state.
    
    Args:
        lattice: Lattice structure from create_surface_code_lattice()
        x_errors: Boolean array indicating X errors on data qubits
        z_errors: Boolean array indicating Z errors on data qubits
        x_syndrome: Boolean array for X-type ancilla measurements
        z_syndrome: Boolean array for Z-type ancilla measurements
        corrections: Boolean array indicating decoder corrections applied
        frame_num: Current animation frame number
        total_frames: Total frames in animation
        title: Plot title
    """
    d = lattice['distance']
    
    fig = go.Figure()
    
    # Background gradient for depth effect
    fig.add_vrect(x0=-1, x1=2*d, fillcolor="#0a0c18", opacity=0.8, line_width=0)
    fig.add_vrect(x0=-0.5, x1=2*d-0.5, fillcolor="#111424", opacity=0.5, line_width=0)
    
    # Background grid with glow effect
    for i in range(2*d):
        alpha = 0.3 + 0.2 * np.sin(2 * np.pi * i / (2*d))  # Varying opacity for depth
        fig.add_trace(go.Scatter(
            x=[-0.5, 2*d-0.5],
            y=[i-0.5, i-0.5],
            mode='lines',
            line=dict(color=f'rgba(30, 42, 80, {alpha})', width=1),
            hoverinfo='skip',
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=[i-0.5, i-0.5],
            y=[-0.5, 2*d-0.5],
            mode='lines',
            line=dict(color=f'rgba(30, 42, 80, {alpha})', width=1),
            hoverinfo='skip',
            showlegend=False
        ))
    
    # Data qubits
    data_x = [q['x'] for q in lattice['data']]
    data_y = [q['y'] for q in lattice['data']]
    
    # Determine colors based on errors
    data_colors = []
    data_sizes = []
    data_symbols = []
    
    for i, q in enumerate(lattice['data']):
        has_x = x_errors is not None and i < len(x_errors) and x_errors[i]
        has_z = z_errors is not None and i < len(z_errors) and z_errors[i]
        has_corr = corrections is not None and i < len(corrections) and corrections[i]
        
        if has_x and has_z:
            # Y error (both X and Z)
            color = '#ff00ff'  # Magenta
            size = 25
            symbol = 'diamond'
        elif has_x:
            color = '#ff4444'  # Red for X error
            size = 20
            symbol = 'x'
        elif has_z:
            color = '#4444ff'  # Blue for Z error
            size = 20
            symbol = 'circle'
        elif has_corr:
            color = '#44ff44'  # Green for correction
            size = 18
            symbol = 'circle'
        else:
            color = '#666666'  # Gray for clean
            size = 12
            symbol = 'circle'
        
        # Add glow effect for errors with dynamic pulsing
        if has_x or has_z:
            # Animated pulse based on frame - more dramatic effect
            pulse = 1 + 0.4 * np.sin(2 * np.pi * frame_num / total_frames + i * 0.5)
            size = int(size * pulse)
            # Add extra glow for satisfying effect
            if pulse > 1.3:
                size = int(size * 1.2)
        
        data_colors.append(color)
        data_sizes.append(size)
        data_symbols.append(symbol)
    
    # Add data qubits trace
    fig.add_trace(go.Scatter(
        x=data_x,
        y=data_y,
        mode='markers',
        marker=dict(
            color=data_colors,
            size=data_sizes,
            symbol=data_symbols,
            line=dict(color='#ffffff', width=1),
            opacity=0.9
        ),
        name='Data Qubits',
        hovertemplate='Qubit %{text}<br>X: %{customdata[0]}<br>Z: %{customdata[1]}<extra></extra>',
        text=[f'D{i}' for i in range(len(data_x))],
        customdata=[[
            'X' if (x_errors is not None and i < len(x_errors) and x_errors[i]) else '·',
            'Z' if (z_errors is not None and i < len(z_errors) and z_errors[i]) else '·'
        ] for i in range(len(data_x))]
    ))
    
    # X-ancilla qubits (syndrome measurement)
    x_anc_x = [q['x'] for q in lattice['x_ancilla']]
    x_anc_y = [q['y'] for q in lattice['x_ancilla']]
    
    x_anc_colors = []
    x_anc_sizes = []
    for i, q in enumerate(lattice['x_ancilla']):
        fired = x_syndrome is not None and i < len(x_syndrome) and x_syndrome[i]
        if fired:
            # Glowing red for fired syndrome
            pulse = 1 + 0.5 * np.sin(2 * np.pi * frame_num / total_frames)
            x_anc_colors.append('#ff6600')
            x_anc_sizes.append(30 * pulse)
        else:
            x_anc_colors.append('#333366')
            x_anc_sizes.append(15)
    
    fig.add_trace(go.Scatter(
        x=x_anc_x,
        y=x_anc_y,
        mode='markers',
        marker=dict(
            color=x_anc_colors,
            size=x_anc_sizes,
            symbol='square',
            line=dict(color='#8888ff', width=2),
            opacity=0.8
        ),
        name='X-Ancilla',
        hovertemplate='X-Ancilla %{text}<br>Status: %{customdata}<extra></extra>',
        text=[f'X{i}' for i in range(len(x_anc_x))],
        customdata=['FIRED!' if (x_syndrome is not None and i < len(x_syndrome) and x_syndrome[i]) else 'idle' 
                    for i in range(len(x_anc_x))]
    ))
    
    # Z-ancilla qubits
    z_anc_x = [q['x'] for q in lattice['z_ancilla']]
    z_anc_y = [q['y'] for q in lattice['z_ancilla']]
    
    z_anc_colors = []
    z_anc_sizes = []
    for i, q in enumerate(lattice['z_ancilla']):
        fired = z_syndrome is not None and i < len(z_syndrome) and z_syndrome[i]
        if fired:
            pulse = 1 + 0.5 * np.sin(2 * np.pi * frame_num / total_frames)
            z_anc_colors.append('#00aaff')
            z_anc_sizes.append(30 * pulse)
        else:
            z_anc_colors.append('#333366')
            z_anc_sizes.append(15)
    
    fig.add_trace(go.Scatter(
        x=z_anc_x,
        y=z_anc_y,
        mode='markers',
        marker=dict(
            color=z_anc_colors,
            size=z_anc_sizes,
            symbol='diamond',
            line=dict(color='#00ffff', width=2),
            opacity=0.8
        ),
        name='Z-Ancilla',
        hovertemplate='Z-Ancilla %{text}<br>Status: %{customdata}<extra></extra>',
        text=[f'Z{i}' for i in range(len(z_anc_x))],
        customdata=['FIRED!' if (z_syndrome is not None and i < len(z_syndrome) and z_syndrome[i]) else 'idle'
                    for i in range(len(z_anc_x))]
    ))
    
    # Stabilizer edges (showing which data qubits each ancilla measures)
    # X-stabilizers (measure X of surrounding data qubits)
    for anc in lattice['x_ancilla']:
        # Find neighboring data qubits
        neighbors = []
        for dq in lattice['data']:
            dx = abs(dq['x'] - anc['x'])
            dy = abs(dq['y'] - anc['y'])
            if (dx == 1 and dy == 1) or (dx == 0.5 and dy == 1.5):  # Rough approximation
                neighbors.append(dq)
        
        # Draw edges
        for n in neighbors[:4]:  # Limit to 4 neighbors
            fig.add_trace(go.Scatter(
                x=[anc['x'], n['x']],
                y=[anc['y'], n['y']],
                mode='lines',
                line=dict(color='#4444aa', width=1, dash='dot'),
                hoverinfo='skip',
                showlegend=False,
                opacity=0.3
            ))
    
    # Layout
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(color='#ffffff', size=16),
            x=0.5
        ),
        paper_bgcolor='#0d0f1a',
        plot_bgcolor='#0d0f1a',
        xaxis=dict(
            range=[-1, 2*d],
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            visible=False
        ),
        yaxis=dict(
            range=[-1, 2*d],
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            visible=False,
            scaleanchor='x',
            scaleratio=1
        ),
        showlegend=True,
        legend=dict(
            font=dict(color='#ffffff'),
            bgcolor='#1a1a3a',
            bordercolor='#4444aa',
            borderwidth=1
        ),
        margin=dict(l=20, r=20, t=50, b=20),
        height=500,
        width=500
    )
    
    return fig


def create_error_timeline_visualization(
    error_history: list[dict],
    distance: int = 3,
    current_step: int = 0
) -> go.Figure:
    """Create a timeline visualization showing error propagation over time.
    
    Args:
        error_history: List of error states over time steps
        distance: Code distance
        current_step: Current time step to highlight
    """
    fig = go.Figure()
    
    d = distance
    n_steps = len(error_history)
    
    if n_steps == 0:
        return fig
    
    # Create mini visualizations for each time step
    step_width = 0.8
    step_height = 0.8
    spacing = 1.0
    
    max_display = min(n_steps, 10)  # Show last 10 steps
    start_idx = max(0, n_steps - max_display)
    
    for step_idx in range(start_idx, n_steps):
        step_data = error_history[step_idx]
        x_offset = (step_idx - start_idx) * spacing
        
        is_current = (step_idx == current_step)
        alpha = 1.0 if is_current else 0.4
        border_color = '#ffaa00' if is_current else '#444488'
        border_width = 3 if is_current else 1
        
        # Draw step border
        fig.add_trace(go.Scatter(
            x=[x_offset, x_offset + step_width, x_offset + step_width, x_offset, x_offset],
            y=[0, 0, step_height, step_height, 0],
            mode='lines',
            line=dict(color=border_color, width=border_width),
            fill='toself',
            fillcolor=f'rgba(20, 20, 50, {alpha * 0.5})',
            hoverinfo='skip',
            showlegend=False
        ))
        
        # Add step label
        fig.add_annotation(
            x=x_offset + step_width/2,
            y=-0.1,
            text=f"t={step_idx}",
            showarrow=False,
            font=dict(color='#ffffff' if is_current else '#888888', size=10)
        )
        
        # Draw simplified error state
        x_err = step_data.get('x_errors', [])
        z_err = step_data.get('z_errors', [])
        
        # Show error count as colored dots
        n_x = sum(x_err) if x_err else 0
        n_z = sum(z_err) if z_err else 0
        
        # X errors (red dots)
        for i in range(min(n_x, 5)):
            fig.add_trace(go.Scatter(
                x=[x_offset + 0.1 + i * 0.12],
                y=[step_height - 0.15],
                mode='markers',
                marker=dict(color='#ff4444', size=8, opacity=alpha),
                hoverinfo='skip',
                showlegend=False
            ))
        
        # Z errors (blue dots)  
        for i in range(min(n_z, 5)):
            fig.add_trace(go.Scatter(
                x=[x_offset + 0.1 + i * 0.12],
                y=[step_height - 0.35],
                mode='markers',
                marker=dict(color='#4444ff', size=8, opacity=alpha),
                hoverinfo='skip',
                showlegend=False
            ))
        
        # Syndrome fired indicator
        syndrome = step_data.get('syndrome', [])
        n_syn = sum(syndrome) if syndrome else 0
        if n_syn > 0:
            fig.add_trace(go.Scatter(
                x=[x_offset + step_width - 0.15],
                y=[step_height - 0.25],
                mode='markers',
                marker=dict(
                    color='#ff6600',
                    size=12 + (n_syn * 2),
                    opacity=alpha,
                    symbol='star'
                ),
                hoverinfo='skip',
                showlegend=False
            ))
    
    # Layout
    fig.update_layout(
        title=dict(
            text="Error Timeline",
            font=dict(color='#ffffff', size=14),
            x=0.5
        ),
        paper_bgcolor='#0d0f1a',
        plot_bgcolor='#0d0f1a',
        xaxis=dict(
            range=[-0.5, max_display * spacing + 0.5],
            showgrid=False,
            zeroline=False,
            showticklabels=False
        ),
        yaxis=dict(
            range=[-0.5, 1.2],
            showgrid=False,
            zeroline=False,
            showticklabels=False
        ),
        showlegend=False,
        margin=dict(l=20, r=20, t=40, b=20),
        height=200,
        width=800
    )
    
    return fig


def generate_live_rl_visualization(
    distance: int = 3,
    syndrome: Optional[np.ndarray] = None,
    action: Optional[np.ndarray] = None,
    correct: bool = False,
    episode: int = 0,
    reward: float = 0.0,
    error_locations: Optional[dict] = None
) -> dict:
    """Generate a complete live visualization for the RL training tab.
    
    Returns a dict with:
    - 'lattice': Main lattice figure
    - 'timeline': Error timeline figure
    - 'stats': Text statistics
    """
    lattice = create_surface_code_lattice(distance)
    
    # Parse syndrome into X and Z components
    n_x_anc = len(lattice['x_ancilla'])
    n_z_anc = len(lattice['z_ancilla'])
    
    x_syndrome = None
    z_syndrome = None
    if syndrome is not None:
        if len(syndrome) >= n_x_anc:
            x_syndrome = syndrome[:n_x_anc]
        if len(syndrome) >= n_x_anc + n_z_anc:
            z_syndrome = syndrome[n_x_anc:n_x_anc + n_z_anc]
    
    # Parse errors from error_locations
    x_errors = None
    z_errors = None
    if error_locations:
        x_errors = error_locations.get('x', [])
        z_errors = error_locations.get('z', [])
    
    # Parse corrections from action
    corrections = None
    if action is not None:
        corrections = action.astype(bool)
    
    # Create main lattice visualization
    main_fig = animated_error_propagation_frame(
        lattice=lattice,
        x_errors=x_errors,
        z_errors=z_errors,
        x_syndrome=x_syndrome,
        z_syndrome=z_syndrome,
        corrections=corrections,
        frame_num=episode % 10,
        total_frames=10,
        title=f"Live Environment — Episode {episode} | Reward: {reward:.3f} | {'✅ Correct' if correct else '❌ Error'}"
    )
    
    return {
        'lattice': main_fig,
        'stats': {
            'episode': episode,
            'reward': reward,
            'correct': correct,
            'n_data': len(lattice['data']),
            'n_x_ancilla': n_x_anc,
            'n_z_ancilla': n_z_anc,
            'syndrome_weight': np.sum(syndrome) if syndrome is not None else 0
        }
    }
