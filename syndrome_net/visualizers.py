"""Visualizer implementations for syndrome-net.

Provides concrete implementations of the Visualizer protocol
for rendering QEC circuits and syndromes.
"""
from __future__ import annotations

from typing import Any
import base64

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from numpy.typing import NDArray
import stim

from syndrome_net import Visualizer, Syndrome


class PlotlyVisualizer(Visualizer):
    """Plotly-based interactive visualizer.
    
    Generates interactive HTML visualizations using Plotly,
    suitable for embedding in Streamlit and web applications.
    """
    
    @property
    def name(self) -> str:
        return "plotly"
    
    def render_circuit(self, circuit: stim.Circuit) -> go.Figure:
        """Render a circuit as an interactive diagram.
        
        Args:
            circuit: Stim circuit to visualize
            
        Returns:
            Plotly figure with circuit layout
        """
        # Get coordinates from circuit
        coords = self._extract_coordinates(circuit)
        
        fig = go.Figure()
        
        # Add qubit positions
        if coords:
            x_vals = [c[0] for c in coords.values()]
            y_vals = [c[1] for c in coords.values()]
            labels = list(coords.keys())
            
            fig.add_trace(go.Scatter(
                x=x_vals,
                y=y_vals,
                mode='markers+text',
                text=labels,
                textposition="top center",
                marker=dict(size=15, color='lightblue'),
                name='Qubits'
            ))
        
        fig.update_layout(
            title='QEC Circuit Layout',
            xaxis_title='X Coordinate',
            yaxis_title='Y Coordinate',
            showlegend=True
        )
        
        return fig
    
    def render_syndrome(
        self,
        syndrome: Syndrome,
        layout: dict[int, tuple[float, float]]
    ) -> go.Figure:
        """Render syndrome measurements on the code layout.
        
        Args:
            syndrome: Syndrome to visualize
            layout: Mapping from qubit index to (x, y) coordinates
            
        Returns:
            Plotly figure with syndrome highlighted
        """
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('X Syndrome', 'Z Syndrome'),
            specs=[[{'type': 'scatter'}, {'type': 'scatter'}]]
        )
        
        # X syndrome visualization
        x_coords = []
        y_coords = []
        colors = []
        for i, triggered in enumerate(syndrome.x_syndrome):
            if i in layout:
                x_coords.append(layout[i][0])
                y_coords.append(layout[i][1])
                colors.append('red' if triggered else 'lightgray')
        
        fig.add_trace(
            go.Scatter(
                x=x_coords,
                y=y_coords,
                mode='markers',
                marker=dict(size=15, color=colors),
                name='X Syndrome'
            ),
            row=1, col=1
        )
        
        # Z syndrome visualization
        x_coords = []
        y_coords = []
        colors = []
        for i, triggered in enumerate(syndrome.z_syndrome):
            if i in layout:
                x_coords.append(layout[i][0])
                y_coords.append(layout[i][1])
                colors.append('blue' if triggered else 'lightgray')
        
        fig.add_trace(
            go.Scatter(
                x=x_coords,
                y=y_coords,
                mode='markers',
                marker=dict(size=15, color=colors),
                name='Z Syndrome'
            ),
            row=1, col=2
        )
        
        fig.update_layout(height=400, showlegend=False)
        return fig
    
    def _extract_coordinates(
        self,
        circuit: stim.Circuit
    ) -> dict[int, tuple[float, float]]:
        """Extract qubit coordinates from circuit."""
        coords = {}
        for line in str(circuit).split('\n'):
            if line.startswith('QUBIT_COORDS'):
                # Parse QUBIT_COORDS(x, y) index
                parts = line.split('(')[1].split(')')[0].split(',')
                x, y = float(parts[0]), float(parts[1])
                idx = int(line.split()[-1])
                coords[idx] = (x, y)
        return coords


class SVGVisualizer(Visualizer):
    """SVG-based static visualizer.
    
    Generates SVG diagrams for embedding in documents and
    for situations where interactive plots are not suitable.
    """
    
    @property
    def name(self) -> str:
        return "svg"
    
    def render_circuit(self, circuit: stim.Circuit) -> str:
        """Render a circuit as an SVG string.
        
        Args:
            circuit: Stim circuit to visualize
            
        Returns:
            SVG string representation
        """
        # Generate SVG using Stim's diagram capabilities
        try:
            diagram = circuit.diagram('detslice-with-ops-svg')
            if isinstance(diagram, bytes):
                return diagram.decode('utf-8')
            return str(diagram)
        except Exception:
            # Fallback to simple SVG
            return self._generate_fallback_svg(circuit)
    
    def render_syndrome(
        self,
        syndrome: Syndrome,
        layout: dict[int, tuple[float, float]]
    ) -> str:
        """Render syndrome as SVG.
        
        Args:
            syndrome: Syndrome to visualize
            layout: Qubit coordinate mapping
            
        Returns:
            SVG string
        """
        svg_parts = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<svg xmlns="http://www.w3.org/2000/svg" width="400" height="400">',
        ]
        
        # Draw syndrome triggers
        for i, triggered in enumerate(syndrome.x_syndrome):
            if triggered and i in layout:
                x, y = layout[i]
                color = "red"
                svg_parts.append(
                    f'<circle cx="{x*50+50}" cy="{y*50+50}" r="10" fill="{color}"/>'
                )
        
        for i, triggered in enumerate(syndrome.z_syndrome):
            if triggered and i in layout:
                x, y = layout[i]
                color = "blue"
                svg_parts.append(
                    f'<circle cx="{x*50+50}" cy="{y*50+50}" r="10" fill="{color}"/>'
                )
        
        svg_parts.append('</svg>')
        return '\n'.join(svg_parts)
    
    def _generate_fallback_svg(self, circuit: stim.Circuit) -> str:
        """Generate simple SVG when Stim diagram fails."""
        return '''<?xml version="1.0"?>
        <svg xmlns="http://www.w3.org/2000/svg" width="200" height="200">
            <text x="10" y="20">Circuit visualization</text>
        </svg>'''


class CrumbleVisualizer(Visualizer):
    """Crumble (Stim interactive) visualizer.
    
    Generates URLs for the Crumble interactive Stim circuit viewer.
    """
    
    @property
    def name(self) -> str:
        return "crumble"
    
    def render_circuit(self, circuit: stim.Circuit) -> str:
        """Generate a Crumble URL for interactive circuit viewing.
        
        Args:
            circuit: Stim circuit to visualize
            
        Returns:
            URL to Crumble viewer
        """
        try:
            return circuit.to_crumble_url()
        except Exception as e:
            return f"Error generating Crumble URL: {e}"
    
    def render_syndrome(
        self,
        syndrome: Syndrome,
        layout: dict[int, tuple[float, float]]
    ) -> str:
        """Crumble doesn't support syndrome-only visualization."""
        return "Crumble requires full circuit for visualization"


class LiveVisualizer(Visualizer):
    """Real-time visualization for RL training.
    
    Optimized for updating visualizations during live training,
    showing error propagation, syndrome measurements, and
    decoder corrections in real-time.
    """
    
    def __init__(self) -> None:
        self._history: list[dict] = []
        self._max_history: int = 100
    
    @property
    def name(self) -> str:
        return "live"
    
    def render_circuit(self, circuit: stim.Circuit) -> go.Figure:
        """Render circuit layout optimized for live updates."""
        return PlotlyVisualizer().render_circuit(circuit)
    
    def render_syndrome(
        self,
        syndrome: Syndrome,
        layout: dict[int, tuple[float, float]]
    ) -> go.Figure:
        """Render syndrome with animation support."""
        # Add to history
        self._history.append({
            'x_syndrome': syndrome.x_syndrome.copy(),
            'z_syndrome': syndrome.z_syndrome.copy()
        })
        if len(self._history) > self._max_history:
            self._history.pop(0)
        
        return PlotlyVisualizer().render_syndrome(syndrome, layout)
    
    def render_error_propagation(
        self,
        errors: dict[int, str],
        layout: dict[int, tuple[float, float]]
    ) -> go.Figure:
        """Render error locations on the code layout.
        
        Args:
            errors: Mapping from qubit index to error type ('X', 'Y', 'Z')
            layout: Qubit coordinate mapping
            
        Returns:
            Plotly figure with errors highlighted
        """
        fig = go.Figure()
        
        error_colors = {
            'X': 'red',
            'Y': 'purple',
            'Z': 'blue',
            'I': 'lightgray'
        }
        
        for qubit, error_type in errors.items():
            if qubit in layout:
                x, y = layout[qubit]
                color = error_colors.get(error_type, 'gray')
                fig.add_trace(go.Scatter(
                    x=[x],
                    y=[y],
                    mode='markers',
                    marker=dict(size=20, color=color, symbol='x'),
                    name=f'Qubit {qubit}: {error_type}'
                ))
        
        fig.update_layout(
            title='Error Propagation',
            xaxis_title='X Coordinate',
            yaxis_title='Y Coordinate'
        )
        
        return fig
