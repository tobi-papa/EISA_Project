# To run the app: python3 -m streamlit run streamlit_editor.py

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- 1. GLOBAL CONFIGURATION ---
L_width_m = 17.0
L_height_m = 8.0
dx = 0.5
nx = int(L_width_m / dx)
ny = int(L_height_m / dx)

# Simulation Parameters
steps = 960  # Approximately 200 minutes simulation time
alpha = 0.25
alpha_real = 0.005  # Turbulent diffusivity for real time calculation

# Calculation of real time per step
dt_seconds = (alpha * dx**2) / alpha_real
total_minutes = (steps * dt_seconds) / 60

# --- 2. STATE AND DATA MANAGEMENT ---
if 'mask' not in st.session_state:
    if os.path.exists('room_data.npz'):
        data = np.load('room_data.npz')
        st.session_state.mask = data['mask']
        # Retrieve heaters if they exist
        saved_heaters = data['heaters'].tolist() if 'heaters' in data else []
        st.session_state.heaters = saved_heaters
        # Retrieve windows
        saved_windows = data['windows'].tolist() if 'windows' in data else []
        st.session_state.windows = saved_windows
        # Retrieve parameters
        st.session_state.loss_factor_wall = data['loss_factor_wall'].item() if 'loss_factor_wall' in data else 0.05
        st.session_state.loss_factor_window = data['loss_factor_window'].item() if 'loss_factor_window' in data else 0.2
        st.session_state.heater_temp = data['heater_temp'].item() if 'heater_temp' in data else 30.0
        st.session_state.initial_temp = data['initial_temp'].item() if 'initial_temp' in data else 14.0
        st.session_state.external_temp = data['external_temp'].item() if 'external_temp' in data else 10.0
    else:
        st.session_state.mask = np.ones((ny, nx), dtype=bool)
        st.session_state.heaters = []
        st.session_state.windows = []
        st.session_state.loss_factor_wall = 0.05
        st.session_state.loss_factor_window = 0.2
        st.session_state.heater_temp = 30.0
        st.session_state.initial_temp = 14.0
        st.session_state.external_temp = 10.0

# --- 3. USER INTERFACE (SIDEBAR & EDITOR) ---
st.set_page_config(layout="wide") # Wide layout to see the graphs better
st.title("🌡️ Interactive Room Thermal Simulator")

col_main, col_controls = st.columns([2, 1])

with col_main:
    st.subheader("1. Draw the Room")
    st.info("Black (True) = Room/Air. White (False) = Solid Wall/Empty. Each cell = 0.5m x 0.5m.")
    
    # DataFrame Editor
    df = pd.DataFrame(st.session_state.mask)
    edited_df = st.data_editor(df, height=400, width='stretch', key="editor")
    
    # Update the mask in real time
    st.session_state.mask = edited_df.values.astype(bool)

with col_controls:
    st.subheader("Quick Tools")
    
    # Input for quick rectangular modification
    c1, c2 = st.columns(2)
    rect_x1 = c1.number_input("X1 (Grid Units)", 0, nx, 0)
    rect_y1 = c1.number_input("Y1 (Grid Units)", 0, ny, 0)
    rect_x2 = c2.number_input("X2 (Grid Units)", 0, nx, 10)
    rect_y2 = c2.number_input("Y2 (Grid Units)", 0, ny, 10)
    
    col_btn1, col_btn2 = st.columns(2)
    if col_btn1.button("Set 'Room'"):
        st.session_state.mask[rect_y1:rect_y2, rect_x1:rect_x2] = True
        st.rerun()
    if col_btn2.button("Set 'Wall'"):
        st.session_state.mask[rect_y1:rect_y2, rect_x1:rect_x2] = False
        st.rerun()
        
    st.divider()
    if st.button("Reset: All Full"):
        st.session_state.mask[:, :] = True
        st.rerun()
    if st.button("Reset: All Empty"):
        st.session_state.mask[:, :] = False
        st.rerun()
    if st.button("Create Test Rectangular Room"):
        # Central room: 20x12 grid units (10m x 6m), with 2 grid unit borders (1m)
        # x: from 7 to 27, y: from 2 to 14
        st.session_state.mask[:, :] = False  # All wall
        st.session_state.mask[2:14, 7:27] = True  # Room
        st.rerun()

st.divider()

# --- 4. HEATER CONFIGURATION ---
st.subheader("2. Position the Air Conditioners (Grid Units)")
h_col1, h_col2 = st.columns(2)

with h_col1:
    st.write("Heater 1 (Main)")
    h1_x1 = st.number_input("H1 X1", 0, nx, 2)
    h1_x2 = st.number_input("H1 X2", 0, nx, 10)
    h1_y1 = st.number_input("H1 Y1", 0, ny, 2)
    h1_y2 = st.number_input("H1 Y2", 0, ny, 6)

with h_col2:
    st.write("Heater 2 (Optional - Leave 0 to disable)")
    h2_x1 = st.number_input("H2 X1", 0, nx, 0)
    h2_x2 = st.number_input("H2 X2", 0, nx, 0)
    h2_y1 = st.number_input("H2 Y1", 0, ny, 0)
    h2_y2 = st.number_input("H2 Y2", 0, ny, 0)

# Update the heater list
current_heaters = [(h1_x1, h1_x2, h1_y1, h1_y2)]
if h2_x2 > h2_x1 and h2_y2 > h2_y1:
    current_heaters.append((h2_x1, h2_x2, h2_y1, h2_y2))

# --- 4.5 WINDOW CONFIGURATION ---
st.subheader("2. Position the Windows (Grid Units) - For greater dispersion")

# First row: Window 1 and 2
f_row1_col1, f_row1_col2 = st.columns(2)

with f_row1_col1:
    st.write("Window 1")
    f1_x1 = st.number_input("F1 X1", 0, nx, 0)
    f1_x2 = st.number_input("F1 X2", 0, nx, 0)
    f1_y1 = st.number_input("F1 Y1", 0, ny, 0)
    f1_y2 = st.number_input("F1 Y2", 0, ny, 0)

with f_row1_col2:
    st.write("Window 2 (Optional)")
    f2_x1 = st.number_input("F2 X1", 0, nx, 0)
    f2_x2 = st.number_input("F2 X2", 0, nx, 0)
    f2_y1 = st.number_input("F2 Y1", 0, ny, 0)
    f2_y2 = st.number_input("F2 Y2", 0, ny, 0)

# Second row: Window 3 and 4
f_row2_col1, f_row2_col2 = st.columns(2)

with f_row2_col1:
    st.write("Window 3 (Optional)")
    f3_x1 = st.number_input("F3 X1", 0, nx, 0)
    f3_x2 = st.number_input("F3 X2", 0, nx, 0)
    f3_y1 = st.number_input("F3 Y1", 0, ny, 0)
    f3_y2 = st.number_input("F3 Y2", 0, ny, 0)

with f_row2_col2:
    st.write("Window 4 (Optional)")
    f4_x1 = st.number_input("F4 X1", 0, nx, 0)
    f4_x2 = st.number_input("F4 X2", 0, nx, 0)
    f4_y1 = st.number_input("F4 Y1", 0, ny, 0)
    f4_y2 = st.number_input("F4 Y2", 0, ny, 0)

# Update the window list
current_windows = []
if f1_x2 > f1_x1 and f1_y2 > f1_y1:
    current_windows.append((f1_x1, f1_x2, f1_y1, f1_y2))
if f2_x2 > f2_x1 and f2_y2 > f2_y1:
    current_windows.append((f2_x1, f2_x2, f2_y1, f2_y2))
if f3_x2 > f3_x1 and f3_y2 > f3_y1:
    current_windows.append((f3_x1, f3_x2, f3_y1, f3_y2))
if f4_x2 > f4_x1 and f4_y2 > f4_y1:
    current_windows.append((f4_x1, f4_x2, f4_y1, f4_y2))

# --- 4. PHYSICAL PARAMETERS ---
st.subheader("2.5 Physical Parameters")
loss_factor_wall = st.number_input("Wall Loss Factor", 0.0, 1.0, st.session_state.get('loss_factor_wall', 0.05), 0.01, key='loss_factor_wall')
loss_factor_window = st.number_input("Window Loss Factor", 0.0, 1.0, st.session_state.get('loss_factor_window', 0.2), 0.01, key='loss_factor_window')
heater_temp = st.number_input("Heater Temperature (°C)", 0.0, 100.0, st.session_state.get('heater_temp', 30.0), 1.0, key='heater_temp')
initial_temp = st.number_input("Initial Room Temperature (°C)", -50.0, 50.0, st.session_state.get('initial_temp', 14.0), 1.0, key='initial_temp')
external_temp = st.number_input("External Temperature (°C)", -50.0, 50.0, st.session_state.get('external_temp', 10.0), 1.0, key='external_temp')

# --- 5. SIMULATION LOGIC ---
st.subheader("3. Results")

# Function to apply boundary conditions
def apply_conditions_live(grid, mask, heaters_list_m, windows_list_m, loss_factor_wall, loss_factor_window, heater_temp, external_temp):
    # Create mask for windows
    window_mask = np.zeros_like(mask, dtype=bool)
    for w in windows_list_m:
        wx1, wx2, wy1, wy2 = w
        ix1, ix2 = int(wx1), int(wx2)
        iy1, iy2 = int(wy1), int(wy2)
        ix1, ix2 = max(0, ix1), min(nx, ix2)
        iy1, iy2 = max(0, iy1), min(ny, iy2)
        if ix2 > ix1 and iy2 > iy1:
            window_mask[iy1:iy2, ix1:ix2] = True
    
    # Apply loss at the room edges (True cells adjacent to False)
    for i in range(ny):
        for j in range(nx):
            if mask[i, j]:  # If it's room
                # Check if it's boundary (has at least one False neighbor)
                is_boundary = False
                neighbors = []
                if i > 0 and mask[i-1, j]: neighbors.append(grid[i-1, j])
                if i < ny-1 and mask[i+1, j]: neighbors.append(grid[i+1, j])
                if j > 0 and mask[i, j-1]: neighbors.append(grid[i, j-1])
                if j < nx-1 and mask[i, j+1]: neighbors.append(grid[i, j+1])
                
                if len(neighbors) < 4:  # It's boundary if it doesn't have 4 internal neighbors
                    is_boundary = True
                
                if is_boundary:
                    # Loss based on whether it's window or wall
                    loss = loss_factor_window if window_mask[i, j] else loss_factor_wall
                    if neighbors:
                        avg_neighbor = np.mean(neighbors)
                        grid[i, j] = (1 - loss) * avg_neighbor + loss * external_temp
                    else:
                        grid[i, j] = external_temp  # If isolated, goes to external
    
    # Apply heaters
    for h in heaters_list_m:
        hx1, hx2, hy1, hy2 = h
        ix1, ix2 = int(hx1), int(hx2)
        iy1, iy2 = int(hy1), int(hy2)
        
        ix1, ix2 = max(0, ix1), min(nx, ix2)
        iy1, iy2 = max(0, iy1), min(ny, iy2)
        
        if ix2 > ix1 and iy2 > iy1:
            grid[iy1:iy2, ix1:ix2] = heater_temp
            
    return grid

if st.button('🚀 Start Simulation', type="primary"):
    
    # Save current state
    np.savez('room_data.npz', mask=st.session_state.mask, heaters=np.array(current_heaters), 
             windows=np.array(current_windows), loss_factor_wall=loss_factor_wall, 
             loss_factor_window=loss_factor_window, heater_temp=heater_temp, initial_temp=initial_temp, external_temp=external_temp)
    st.success("Configuration saved!")
    
    # Initial Setup
    room_mask = st.session_state.mask
    u = np.ones((ny, nx)) * initial_temp
    
    avg_temp_history = []
    time_axis = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Simulation Loop
    for k in range(steps):
        # Diffusion Calculation
        u[1:-1, 1:-1] += alpha * (
            u[2:, 1:-1] + u[:-2, 1:-1] + u[1:-1, 2:] + u[1:-1, :-2] - 4*u[1:-1, 1:-1]
        )
        
        # Reapply conditions
        u = apply_conditions_live(u, room_mask, current_heaters, current_windows, loss_factor_wall, loss_factor_window, heater_temp, external_temp)
        
        # Save data occasionally (to speed up)
        avg_temp_history.append(np.mean(u[room_mask]))
        time_axis.append(k * dt_seconds / 60) # Minutes
        
        # Update progress bar every 5%
        if k % (steps // 20) == 0:
            progress_bar.progress(k / steps)
            status_text.text(f"Simulation: {k}/{steps} steps ({int(k*dt_seconds/60)} min simulated)")

    progress_bar.progress(100)
    status_text.text("Simulation Completed!")
    
    # --- FINAL VISUALIZATION ---
    u_visual = u.copy()
    u_visual[~room_mask] = np.nan 

    fig = plt.figure(figsize=(16, 7))
    gs = fig.add_gridspec(1, 2)
    
    # PLOT 1: Heatmap
    ax1 = fig.add_subplot(gs[0, 0])
    sns.heatmap(u_visual, ax=ax1, cmap="coolwarm", vmin=10, vmax=25, 
                cbar_kws={'label': 'Temp (°C)'})
    ax1.set_facecolor('lightgray')
    ax1.set_title(f"Heat Distribution ({total_minutes:.0f} min elapsed)")
    
    # Axes in Meters
    ticks_step = 2 # Every 2 meters
    xticks = np.arange(0, L_width_m+1, ticks_step)
    yticks = np.arange(0, L_height_m+1, ticks_step)
    ax1.set_xticks([int(x / dx) for x in xticks])
    ax1.set_xticklabels(xticks)
    ax1.set_yticks([int(y / dx) for y in yticks])
    ax1.set_yticklabels(yticks)
    ax1.set_xlabel("Meters")
    ax1.set_ylabel("Meters")

    # PLOT 2: Efficiency
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(time_axis, avg_temp_history, color='#d62728', linewidth=2.5)
    ax2.axhline(y=20, color='green', linestyle='--', label='Target Comfort (20°C)')
    
    # Find when we exceeded 20 degrees
    cross_target = next((t for t, temp in zip(time_axis, avg_temp_history) if temp >= 20), None)
    if cross_target:
        ax2.scatter([cross_target], [20], color='green', zorder=5)
        ax2.annotate(f"{cross_target:.1f} min", (cross_target, 20), 
                     xytext=(cross_target+5, 18), arrowprops=dict(facecolor='black', shrink=0.05))

    ax2.set_title("Heating Speed Analysis")
    ax2.set_xlabel("Real Time (Minutes)")
    ax2.set_ylabel("Avg Room Temp (°C)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    st.pyplot(fig)
    
    # Final statistics for the report
    st.success(f"🌡️ Average final temperature: {avg_temp_history[-1]:.2f}°C")
    if cross_target:
        st.info(f"✅ Target (20°C) reached in: {cross_target:.1f} minutes")
    else:
        st.warning(f"⚠️ Target (20°C) NOT reached. Increase heaters or time.")