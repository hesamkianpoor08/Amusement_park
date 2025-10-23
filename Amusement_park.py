import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Page Configuration ---
st.set_page_config(
    page_title="Amusement Park Ride Designer",
    page_icon="🎡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Enhanced Light Mode CSS ---
st.markdown("""
<style>
    :root { color-scheme: light only !important; }
    html, body, [data-testid="stAppViewContainer"], .main, .stApp {
        color-scheme: light !important;
        background-color: #FFFFFF !important;
        color: #000000 !important;
    }
    * {
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
    }
    input, textarea, select, button, label, p, span, div {
        color: #000000 !important;
        background-color: #FFFFFF !important;
    }
    input[type="number"], input[type="text"] {
        background-color: #F5F5F5 !important;
        color: #000000 !important;
        border: 1px solid #CCCCCC !important;
        border-radius: 4px !important;
        padding: 8px !important;
    }
    input[type="number"]:focus, input[type="text"]:focus {
        background-color: #FFFFFF !important;
        border: 2px solid #2196F3 !important;
    }
    h1 {
        color: #1976D2 !important;
        font-size: 40px !important;
        text-align: center;
        font-weight: bold;
    }
    h2, h3 { color: #000000 !important; }
    
    /* Navigation Buttons */
    div.stButton > button {
        background-color: #2196F3 !important;
        color: white !important;
        -webkit-text-fill-color: white !important;
        border-radius: 8px;
        height: 3em;
        font-size: 16px;
        border: none;
        padding: 0 30px;
    }
    div.stButton > button:hover {
        background-color: #1976D2 !important;
    }
    
    /* Error box */
    .error-box {
        background-color: #FFCDD2 !important;
        border-left: 4px solid #F44336 !important;
        padding: 10px;
        margin: 10px 0;
        border-radius: 4px;
    }
    .error-box p {
        color: #C62828 !important;
        -webkit-text-fill-color: #C62828 !important;
        font-weight: bold;
    }
    
    /* Success box */
    .stSuccess {
        background-color: #C8E6C9 !important;
        color: #1B5E20 !important;
    }
    
    /* Metrics */
    .stMetric {
        background-color: #F5F5F5 !important;
        padding: 10px;
        border-radius: 8px;
    }
    
    /* Info boxes */
    .info-box {
        background-color: #E3F2FD !important;
        border-left: 4px solid #2196F3 !important;
        padding: 10px;
        margin: 10px 0;
        border-radius: 4px;
    }
    
    /* Checkbox styling */
    .stCheckbox label {
        color: #000000 !important;
        font-weight: bold;
    }

    /* Force Plotly SVG text to be black and fully opaque (helps against global CSS overrides) */
    .stPlotlyChart svg, .stPlotlyChart svg text, .js-plotly-plot svg text {
        fill: #000000 !important;
        color: #000000 !important;
        opacity: 1 !important;
        -webkit-text-fill-color: #000000 !important;
        stroke: none !important;
    }
</style>
""", unsafe_allow_html=True)

# --- Initialize Session State ---
if 'step' not in st.session_state:
    st.session_state.step = 0
if 'ride_type' not in st.session_state:
    st.session_state.ride_type = None
if 'basic_params' not in st.session_state:
    st.session_state.basic_params = {}
# Set default wind_force True so checkbox is checked by default
if 'advanced_params' not in st.session_state:
    st.session_state.advanced_params = {
        'wind_force': True,              # default checked
        'earthquake_force': False,
        'snow_force': False,
        # optional defaults for numeric fields
        'height': 66.7,
        'gravity': 9.81,
        'air_density': 1.225,
        'safety_factor': 1.5
    }
if 'validation_errors' not in st.session_state:
    st.session_state.validation_errors = []

# --- Wind Load Calculation Function (cached for performance) ---
@st.cache_data(show_spinner=False)
def calculate_wind_load(H, omega, g, rho_air, Ax=303.3, Ay=592.5, z0=0.01, c_dir=1, c_season=1, c0=1, cp=1.2):
    z = np.arange(1, H + 1)
    v_b0 = 100 / 3.6
    vb = c_dir * c_season * v_b0
    kr = 0.19 * (z0 / 0.05) ** 0.07
    cr = kr * np.log(z / z0)
    vm = cr * c0 * vb
    vm_max = vm[-1]
    kl = 1
    Iv = kl / (c0 * np.log(z / z0))
    q_p = 0.5 * rho_air * (vm ** 2) * (1 + 7 * Iv)
    Fwy = q_p * cp * Ay / 1e3
    Fwx = q_p * cp * Ax / 1e3
    return {'z': z, 'vm': vm, 'vm_max': vm_max, 'Fwy': Fwy, 'Fwx': Fwx, 'q_p': q_p, 'Iv': Iv}

# --- Plotly Plots (updated to force white background and black text) ---
def create_wind_plots(results):
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Wind Load Distribution', 'Wind Velocity Profile'),
        horizontal_spacing=0.12
    )
    
    fig.add_trace(
        go.Scatter(x=results['Fwy'], y=results['z'], mode='lines', name='Y-direction',
                   line=dict(color='#00BCD4', width=3)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=results['Fwx'], y=results['z'], mode='lines', name='X-direction',
                   line=dict(color='#FF9800', width=3)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=results['vm'], y=results['z'], mode='lines', name='Mean Wind Velocity',
                   line=dict(color='#4CAF50', width=3)),
        row=1, col=2
    )

    axis_common = dict(
        gridcolor='#E0E0E0',
        linecolor='#000000',
        zerolinecolor='#BDBDBD',
        tickfont=dict(color='#000000', size=11),
        title_font=dict(color='#000000', size=13),
        showgrid=True
    )

    fig.update_xaxes(title_text="Wind Load [kN]", row=1, col=1, **axis_common)
    fig.update_xaxes(title_text="Wind Velocity [m/s]", row=1, col=2, **axis_common)
    fig.update_yaxes(title_text="Height [m]", row=1, col=1, **axis_common)
    fig.update_yaxes(title_text="Height [m]", row=1, col=2, **axis_common)

    fig.update_layout(
        template="plotly_white",
        font=dict(color='black', size=12),
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=520,
        margin=dict(l=90, r=60, t=90, b=70),
        showlegend=True,
        legend=dict(bgcolor='rgba(255,255,255,0.95)', bordercolor='#BDBDBD', borderwidth=1, font=dict(color='#000000')),
        hovermode='closest',
        hoverlabel=dict(bgcolor="white", font_size=13, font_family="sans-serif", font_color="black", bordercolor="#2196F3")
    )

    for ann in fig.layout.annotations:
        ann.font = dict(color='#000000', size=13)

    return fig

def create_placeholder_plot(title):
    fig = go.Figure()
    fig.add_annotation(
        text=f"{title}<br>(Analysis Not Implemented Yet)",
        xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=18, color="black")
    )
    fig.update_layout(
        height=420,
        template="plotly_white",
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=60, r=60, t=60, b=60)
    )
    return fig

def create_component_diagram(diameter, height, capacity, motor_power):
    fig = go.Figure()
    
    theta = np.linspace(0, 2*np.pi, 200)
    x_wheel = diameter/2 * np.cos(theta)
    y_wheel = diameter/2 * np.sin(theta) + height/2
    
    fig.add_trace(go.Scatter(x=x_wheel, y=y_wheel, mode='lines', 
                             name='Wheel Structure', line=dict(color='#2196F3', width=3)))
    
    fig.add_trace(go.Scatter(x=[0, 0], y=[0, height/2], mode='lines',
                             name='Support Tower', line=dict(color='#FF5722', width=6)))
    
    annotations = [
        dict(x=0, y=height + diameter*0.05 + 2, text=f"Height: {height} m", showarrow=False, font=dict(color='black')),
        dict(x=diameter/2 + 2, y=height/2, text=f"Diameter: {diameter} m", showarrow=False, font=dict(color='black')),
        dict(x=0, y=-5, text=f"Motor Power: {motor_power:.1f} kW", showarrow=False, font=dict(color='black')),
        dict(x=0, y=-8, text=f"Capacity: {capacity} passengers", showarrow=False, font=dict(color='black'))
    ]
    
    fig.update_layout(
        title=dict(text="Ferris Wheel Components & Specifications", font=dict(color='black')),
        height=620,
        template="plotly_white",
        plot_bgcolor='white',
        paper_bgcolor='white',
        showlegend=True,
        xaxis=dict(title="Width [m]", gridcolor='#E0E0E0', zeroline=True, linecolor='#000000', tickfont=dict(color='#000000')),
        yaxis=dict(title="Height [m]", gridcolor='#E0E0E0', zeroline=True, linecolor='#000000', tickfont=dict(color='#000000')),
        annotations=annotations,
        margin=dict(l=80, r=80, t=80, b=80)
    )

    for a in fig.layout.annotations:
        a.font = dict(color='#000000')
    return fig

# --- Validation Function ---
def validate_basic_params(params, ride_type):
    errors = []
    if ride_type == "Ferris Wheel":
        if 'num_cabins' not in params or params['num_cabins'] <= 0:
            errors.append("Number of cabins must be greater than 0")
        if 'diameter' not in params or params['diameter'] <= 0:
            errors.append("Diameter must be greater than 0")
        if 'capacity' not in params or params['capacity'] <= 0:
            errors.append("Cabin capacity must be greater than 0")
    return errors

# --- Navigation Functions ---
def next_step():
    st.session_state.step += 1

def prev_step():
    if st.session_state.step > 0:
        st.session_state.step -= 1

# --- Main App ---
st.title("🎡 Amusement Park Ride Designer")

# Progress indicator
progress = st.session_state.step / 4
st.progress(progress)
st.markdown(f"**Step {st.session_state.step + 1} of 5**")
st.markdown("---")

# === STEP 0: Ride Type Selection ===
if st.session_state.step == 0:
    st.header("Step 1: Select Ride Type")
    st.write("What type of amusement park ride would you like to design?")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🎡 Ferris Wheel", key="ferris", use_container_width=True):
            st.session_state.ride_type = "Ferris Wheel"
            next_step()
    
    with col2:
        if st.button("🎢 Roller Coaster", key="coaster", use_container_width=True):
            st.session_state.ride_type = "Roller Coaster"
            st.info("Roller Coaster design module coming soon!")
    
    with col3:
        if st.button("🎠 Other Rides", key="other", use_container_width=True):
            st.info("Additional ride types coming soon!")
    
    if st.session_state.ride_type:
        st.success(f"✅ Selected: {st.session_state.ride_type}")

# === STEP 1: Basic Parameters ===
elif st.session_state.step == 1:
    st.header(f"Step 2: Basic Parameters - {st.session_state.ride_type}")
    
    if st.session_state.ride_type == "Ferris Wheel":
        col1, col2 = st.columns(2)
        
        with col1:
            num_cabins = st.number_input(
                "Number of Cabins", 
                min_value=1, 
                value=st.session_state.basic_params.get('num_cabins', 12),
                help="Total number of passenger cabins"
            )
            st.caption("Unit: count | Standard: ASTM F2291")
            
            diameter = st.number_input(
                "Wheel Diameter", 
                min_value=1.0, 
                value=st.session_state.basic_params.get('diameter', 60.0),
                help="Diameter of the ferris wheel"
            )
            st.caption("Unit: meters [m] | Standard: EN 13814")
        
        with col2:
            capacity = st.number_input(
                "Cabin Capacity", 
                min_value=1, 
                value=st.session_state.basic_params.get('capacity', 6),
                help="Number of passengers per cabin"
            )
            st.caption("Unit: passengers | Standard: ASTM F2291")
            
            rotation_speed = st.number_input(
                "Rotation Speed", 
                min_value=0.1, 
                value=st.session_state.basic_params.get('rotation_speed', 2.0),
                format="%.2f",
                help="Rotational speed in RPM"
            )
            st.caption("Unit: RPM | Standard: EN 13814")
        
        st.session_state.basic_params.update({
            'num_cabins': int(num_cabins),
            'diameter': float(diameter),
            'capacity': int(capacity),
            'rotation_speed': float(rotation_speed)
        })
    
    # Display validation errors
    if st.session_state.validation_errors:
        st.markdown('<div class="error-box">', unsafe_allow_html=True)
        for error in st.session_state.validation_errors:
            st.markdown(f'<p>❌ {error}</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Navigation
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("⬅️ Back", use_container_width=True):
            prev_step()
    with col3:
        if st.button("Next ➡️", use_container_width=True):
            errors = validate_basic_params(st.session_state.basic_params, st.session_state.ride_type)
            if errors:
                st.session_state.validation_errors = errors
                st.rerun()
            else:
                st.session_state.validation_errors = []
                next_step()

# === STEP 2: Advanced Parameters ===
elif st.session_state.step == 2:
    st.header("Step 3: Advanced Load Analysis Parameters")
    
    st.subheader("📋 Environmental Load Analysis (Optional)")
    col1, col2, col3 = st.columns(3)
    
    # Wind checkbox: default checked (True) and user CAN change it
    with col1:
        wind = st.checkbox("🌬️ Wind Force Analysis", value=st.session_state.advanced_params.get('wind_force', True))
        st.caption("Standard: BS EN 1991-1-4")
    with col2:
        earthquake = st.checkbox("🏚️ Earthquake Analysis", 
                                value=st.session_state.advanced_params.get('earthquake_force', False))
        st.caption("Standard: EN 1998-1")
    with col3:
        snow = st.checkbox("❄️ Snow Load Analysis", 
                          value=st.session_state.advanced_params.get('snow_force', False))
        st.caption("Standard: EN 1991-1-3")
    
    # Update session state with the user's choices
    st.session_state.advanced_params.update({
        'wind_force': bool(wind),
        'earthquake_force': bool(earthquake),
        'snow_force': bool(snow)
    })
    
    st.markdown("---")
    st.subheader("⚙️ Mandatory Structural Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        height = st.number_input(
            "Total Structure Height *", 
            min_value=1.0, 
            value=st.session_state.advanced_params.get('height', 66.7),
            help="Total height from ground to top"
        )
        st.caption("Unit: meters [m] | Standard: EN 13814")
        
        gravity = st.number_input(
            "Gravitational Acceleration *", 
            value=st.session_state.advanced_params.get('gravity', 9.81),
            format="%.3f"
        )
        st.caption("Unit: m/s² | Standard: SI Units")
    
    with col2:
        air_density = st.number_input(
            "Air Density *", 
            value=st.session_state.advanced_params.get('air_density', 1.225),
            format="%.3f",
            help="Standard air density at sea level"
        )
        st.caption("Unit: kg/m³ | Standard: ISO 2533")
        
        safety_factor = st.number_input(
            "Safety Factor *", 
            min_value=1.0,
            value=st.session_state.advanced_params.get('safety_factor', 1.5),
            format="%.2f"
        )
        st.caption("Unit: dimensionless | Standard: EN 13814")
    
    st.session_state.advanced_params.update({
        'height': float(height),
        'gravity': float(gravity),
        'air_density': float(air_density),
        'safety_factor': float(safety_factor)
    })
    
    # Validation
    mandatory_fields = ['height', 'gravity', 'air_density', 'safety_factor']
    missing_fields = [f for f in mandatory_fields if f not in st.session_state.advanced_params]
    
    if missing_fields:
        st.markdown('<div class="error-box">', unsafe_allow_html=True)
        st.markdown(f'<p>❌ Please fill in all mandatory fields marked with *</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Navigation
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("⬅️ Back", use_container_width=True):
            prev_step()
    with col3:
        if st.button("Next ➡️", use_container_width=True):
            if not missing_fields:
                next_step()

# === STEP 3: Results - Force Analysis ===
elif st.session_state.step == 3:
    st.header("Step 4: Force Analysis Results")
    
    # Wind Force Analysis (runs if the checkbox is checked)
    if st.session_state.advanced_params.get('wind_force', True):
        st.subheader("🌬️ Wind Force Analysis (BS EN 1991-1-4)")
        
        # Calculate wind loads
        omega = st.session_state.basic_params['rotation_speed'] * 2 * np.pi / 60
        results = calculate_wind_load(
            int(st.session_state.advanced_params['height']),
            omega,
            st.session_state.advanced_params['gravity'],
            st.session_state.advanced_params['air_density']
        )
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Max Wind Velocity", f"{results['vm_max']:.2f} m/s")
        col2.metric("Max Load (X-dir)", f"{results['Fwx'][-1]:.2f} kN")
        col3.metric("Max Load (Y-dir)", f"{results['Fwy'][-1]:.2f} kN")
        
        # Display plots
        fig = create_wind_plots(results)
        st.plotly_chart(fig, use_container_width=True, config={'responsive': True})
    else:
        st.info("🌬️ Wind Force Analysis: Not selected")
    
    st.markdown("---")
    
    # Earthquake Analysis
    if st.session_state.advanced_params.get('earthquake_force'):
        st.subheader("🏚️ Earthquake Force Analysis (EN 1998-1)")
        fig = create_placeholder_plot("Earthquake Force Analysis")
        st.plotly_chart(fig, use_container_width=True, config={'responsive': True})
    else:
        st.info("🏚️ Earthquake Analysis: Not selected")
    
    st.markdown("---")
    
    # Snow Load Analysis
    if st.session_state.advanced_params.get('snow_force'):
        st.subheader("❄️ Snow Load Analysis (EN 1991-1-3)")
        fig = create_placeholder_plot("Snow Load Analysis")
        st.plotly_chart(fig, use_container_width=True, config={'responsive': True})
    else:
        st.info("❄️ Snow Load Analysis: Not selected")
    
    # Navigation
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("⬅️ Back", use_container_width=True):
            prev_step()
    with col3:
        if st.button("Next ➡️", use_container_width=True):
            next_step()

# === STEP 4: Final Results - Component Diagram ===
elif st.session_state.step == 4:
    st.header("Step 5: Component Specifications & System Overview")
    
    # Calculate derived parameters
    diameter = st.session_state.basic_params['diameter']
    height = st.session_state.advanced_params['height']
    capacity = st.session_state.basic_params['capacity']
    num_cabins = st.session_state.basic_params['num_cabins']
    rotation_speed = st.session_state.basic_params['rotation_speed']
    
    # Estimate motor power (simplified calculation)
    total_mass = num_cabins * capacity * 80  # Assume 80kg per person
    angular_velocity = rotation_speed * 2 * np.pi / 60
    moment_of_inertia = total_mass * (diameter/2)**2
    motor_power = moment_of_inertia * angular_velocity**2 / 1000  # kW
    
    # System specifications
    st.subheader("📊 System Specifications")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Wheel Diameter", f"{diameter} m")
    col2.metric("Total Height", f"{height} m")
    col3.metric("Total Capacity", f"{num_cabins * capacity} passengers")
    col4.metric("Est. Motor Power", f"{motor_power:.1f} kW")
    
    st.markdown("---")
    
    # Component diagram
    st.subheader("🔧 Component Layout & Structure")
    fig = create_component_diagram(diameter, height, num_cabins * capacity, motor_power)
    st.plotly_chart(fig, use_container_width=True, config={'responsive': True})
    
    st.markdown("---")
    
    # Key components list
    st.subheader("📦 Required Components")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Structural Components:**")
        st.markdown(f"- Main wheel structure (Ø{diameter}m)")
        st.markdown(f"- Support tower ({height}m)")
        st.markdown(f"- {num_cabins} passenger cabins")
        st.markdown("- Rotation axis and bearings")
        st.markdown("- Foundation and anchor bolts")
    
    with col2:
        st.markdown("**Mechanical & Electrical:**")
        st.markdown(f"- Electric motor ({motor_power:.1f} kW minimum)")
        st.markdown("- Gear reduction system")
        st.markdown("- Emergency braking system")
        st.markdown("- Safety restraints and sensors")
        st.markdown("- Control system and PLC")
    
    # Navigation
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("⬅️ Back", use_container_width=True):
            prev_step()
    with col2:
        if st.button("🔄 Start New Design", use_container_width=True):
            st.session_state.step = 0
            st.session_state.ride_type = None
            st.session_state.basic_params = {}
            st.session_state.advanced_params = {'wind_force': True, 'earthquake_force': False, 'snow_force': False, 'height':66.7, 'gravity':9.81, 'air_density':1.225, 'safety_factor':1.5}
            st.rerun()
    with col3:
        st.success("✅ Design Complete!")
