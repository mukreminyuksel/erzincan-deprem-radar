import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import plotly.graph_objects as go
import numpy as np

x = np.linspace(0, 100, 20)
z = np.linspace(-50, 0, 10)
xx, zz = np.meshgrid(x, z)
xx = xx.flatten()
zz = zz.flatten()

frames = []
for t in range(0, 50, 2):
    # P wave from (0, -25)
    dist = np.sqrt((xx - 0)**2 + (zz + 25)**2)
    p_radius = t * 3
    # S wave
    s_radius = t * 1.5
    
    # Displacement
    dx = np.zeros_like(xx)
    dz = np.zeros_like(zz)
    
    # P wave displacement (longitudinal)
    p_active = np.abs(dist - p_radius) < 5
    dx[p_active] += 2 * (xx[p_active] - 0) / dist[p_active] * np.sin(dist[p_active] - p_radius)
    dz[p_active] += 2 * (zz[p_active] + 25) / dist[p_active] * np.sin(dist[p_active] - p_radius)
    
    # S wave displacement (transverse)
    s_active = np.abs(dist - s_radius) < 5
    # transverse vector is (-dz, dx) relative to radius vector
    nx = -(zz[s_active] + 25) / dist[s_active]
    nz = (xx[s_active] - 0) / dist[s_active]
    dx[s_active] += 2 * nx * np.sin(dist[s_active] - s_radius)
    dz[s_active] += 2 * nz * np.sin(dist[s_active] - s_radius)
    
    frames.append(go.Frame(data=[go.Scatter(x=xx+dx, y=zz+dz, mode='markers')]))

fig = go.Figure(
    data=[go.Scatter(x=xx, y=zz, mode='markers')],
    frames=frames
)
print("Frames generated:", len(frames))
