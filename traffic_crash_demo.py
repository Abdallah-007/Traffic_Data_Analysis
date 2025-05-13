import streamlit as st
import pandas as pd
import numpy as np
import os
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Chicago Traffic Crashes Analysis",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for dark mode theme
st.markdown("""
<style>
    /* Dark background with white text */
    body {
        color: #FFFFFF;
        background-color: #121212;
    }
    
    /* Main header */
    .main-header {
        font-size: 2.5rem;
        color: #FFFFFF;
        text-align: center;
        margin-bottom: 1rem;
        padding: 1rem;
        background-color: #1E1E1E;
        border-radius: 0.5rem;
        box-shadow: 0 2px 5px rgba(0,0,0,0.3);
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.8rem;
        color: #00B4D8;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
        border-bottom: 2px solid #00B4D8;
        padding-bottom: 0.3rem;
    }
    
    /* Subsection headers */
    .subsection-header {
        font-size: 1.4rem;
        color: #90E0EF;
        margin-top: 0.8rem;
        margin-bottom: 0.3rem;
    }
    
    /* Insight text blocks */
    .insight-text {
        background-color: #1E1E1E;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #00B4D8;
        margin-bottom: 1rem;
        color: #FFFFFF;
    }
    
    /* Metric cards */
    .metric-card {
        background-color: #00B4D8;
        padding: 1.2rem 1rem;
        border-radius: 0.5rem;
        text-align: center;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.3);
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: #FFFFFF;
    }
    .metric-label {
        font-size: 1rem;
        color: #FFFFFF;
        margin-top: 0.3rem;
    }
    
    /* Highlighted text */
    .highlight {
        color: #FF9F1C;
        font-weight: bold;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #1E1E1E;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-left: 16px;
        padding-right: 16px;
        color: #FFFFFF;
    }
    .stTabs [aria-selected="true"] {
        background-color: #00B4D8;
        color: white;
    }
    
    /* Dark mode for backgrounds */
    .reportview-container, .main .block-container, .stApp {
        background-color: #121212;
    }
    
    /* Sidebar */
    .sidebar .sidebar-content {
        background-color: #1E1E1E;
        color: white !important;
    }
    
    /* Fix sidebar headers */
    .sidebar h1, .sidebar h2, .sidebar h3, .sidebar h4, .sidebar h5, .sidebar h6 {
        color: #00B4D8 !important;
    }
    
    /* Fix sidebar text */
    .sidebar .sidebar-content p, 
    .sidebar .sidebar-content div,
    .sidebar .sidebar-content span,
    .sidebar .sidebar-content label {
        color: white !important;
    }
    
    /* Image styling */
    .element-container img {
        border: 1px solid #333;
        border-radius: 5px;
        max-width: 100%;
    }
    [data-testid="stImage"] {
        background-color: #1E1E1E;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #333333;
        margin-bottom: 1rem;
    }
    
    /* Ensure plotly charts have visible text */
    .js-plotly-plot .plotly .main-svg {
        background-color: #242424 !important;
    }
    .js-plotly-plot .plotly .main-svg text {
        fill: #FFFFFF !important;
    }
    
    /* Warning and table styles */
    .stAlert {
        background-color: #332700;
        color: #FFD166;
        border-color: #665200;
    }
    .dataframe {
        background-color: #1E1E1E;
        color: #FFFFFF;
        border: 1px solid #333333;
    }
    .dataframe th {
        background-color: #00B4D8;
        color: #FFFFFF;
    }
    
    /* Fix for CSS issues */
    [data-testid="stMarkdownContainer"] { color: white !important; }
    .css-10trblm { color: #00B4D8 !important; }
    .css-1oe6o3n, .css-1d391kg { color: white !important; }
</style>
""", unsafe_allow_html=True)

# Helper function to load data
@st.cache_data
def load_data():
    try:
        # Try to load ML-ready data
        df = pd.read_csv('traffic_data.crashes_ml_ready.csv')
        
        # Drop MongoDB _id column if it exists
        if '_id' in df.columns:
            df = df.drop('_id', axis=1)
            
        # Convert crash_date to datetime if it exists
        if 'crash_date' in df.columns:
            df['crash_date'] = pd.to_datetime(df['crash_date'])
            
        # Create derived columns if needed
        if 'day_of_week' in df.columns:
            day_map = {1: 'Sunday', 2: 'Monday', 3: 'Tuesday', 4: 'Wednesday', 5: 'Thursday', 6: 'Friday', 7: 'Saturday'}
            df['day_name'] = df['day_of_week'].map(day_map)
            df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x in [1, 7] else 0)
            
        if 'hour' in df.columns:
            df['time_category'] = pd.cut(
                df['hour'], 
                bins=[0, 6, 12, 18, 24], 
                labels=['Night (0-6)', 'Morning (6-12)', 'Afternoon (12-18)', 'Evening (18-24)']
            )
            
        return df
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        # Create sample data for demonstration
        return create_sample_data()

def create_sample_data():
    """Create sample data if the real dataset is not available"""
    st.warning("Using sample data for demonstration purposes. Real dataset not found.")
    
    np.random.seed(42)
    n_samples = 10000
    
    # Generate days of week and hours
    days = np.random.randint(1, 8, n_samples)
    hours = np.random.randint(0, 24, n_samples)
    
    # Generate categorical features with realistic distributions
    weather_conditions = np.random.choice(
        ['CLEAR', 'RAIN', 'SNOW', 'CLOUDY/OVERCAST', 'FOG/SMOKE/HAZE', 'SLEET/HAIL', 'UNKNOWN'], 
        n_samples, 
        p=[0.7, 0.1, 0.05, 0.08, 0.03, 0.02, 0.02]
    )
    
    lighting_conditions = np.random.choice(
        ['DAYLIGHT', 'DARKNESS, LIGHTED ROAD', 'DARKNESS', 'DAWN', 'DUSK', 'UNKNOWN'], 
        n_samples, 
        p=[0.6, 0.2, 0.1, 0.04, 0.05, 0.01]
    )
    
    road_conditions = np.random.choice(
        ['DRY', 'WET', 'SNOW OR SLUSH', 'ICE', 'UNKNOWN'], 
        n_samples,
        p=[0.7, 0.15, 0.05, 0.05, 0.05]
    )
    
    # Generate injuries and fatalities
    injuries = np.random.poisson(0.5, n_samples)
    fatal = np.zeros(n_samples)
    fatal_rate = 0.001  # 0.1% fatal rate
    fatal_idx = np.random.choice(n_samples, int(n_samples * fatal_rate), replace=False)
    fatal[fatal_idx] = 1
    
    # Create DataFrame
    df = pd.DataFrame({
        'day_of_week': days,
        'hour': hours,
        'weather': weather_conditions,
        'lighting': lighting_conditions,
        'road_condition': road_conditions,
        'injuries': injuries,
        'fatal': fatal.astype(int)
    })
    
    # Create derived columns
    day_map = {1: 'Sunday', 2: 'Monday', 3: 'Tuesday', 4: 'Wednesday', 5: 'Thursday', 6: 'Friday', 7: 'Saturday'}
    df['day_name'] = df['day_of_week'].map(day_map)
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x in [1, 7] else 0)
    
    df['time_category'] = pd.cut(
        df['hour'], 
        bins=[0, 6, 12, 18, 24], 
        labels=['Night (0-6)', 'Morning (6-12)', 'Afternoon (12-18)', 'Evening (18-24)']
    )
    
    return df

# Function to display a visualization
def display_visualization(image_path, alt_text="Visualization not available"):
    try:
        # Ensure correct path handling
        if not os.path.isabs(image_path):
            # Check various possible paths
            if os.path.exists(image_path):
                full_path = image_path
            elif os.path.exists(os.path.join(os.getcwd(), image_path)):
                full_path = os.path.join(os.getcwd(), image_path)
            else:
                full_path = os.path.join('/home/thor/Projects/Big_Data', image_path)
        else:
            full_path = image_path
            
        # Load and display the image
        image = Image.open(full_path)
        
        # Add a white background for better visibility in dark mode
        bg = Image.new("RGB", image.size, (255, 255, 255))
        if image.mode == 'RGBA':
            bg.paste(image, mask=image.split()[3])  # Use alpha channel as mask
        else:
            bg.paste(image, (0, 0))
            
        # Display with caption
        st.image(
            bg, 
            use_container_width=True, 
            caption=os.path.basename(image_path).replace('.png', '').replace('_', ' ').title()
        )
    except Exception as e:
        st.warning(f"{alt_text} (Error: {str(e)})")

# Load the data
df = load_data()
df_filtered = df

# Sidebar
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/9/9b/Flag_of_Chicago%2C_Illinois.svg/1200px-Flag_of_Chicago%2C_Illinois.svg.png", width=200)
st.sidebar.title("Chicago Traffic Crashes")

# Add project information
st.sidebar.markdown("---")
st.sidebar.markdown("<h3 style='color:#00B4D8;'>Project Information</h3>", unsafe_allow_html=True)
st.sidebar.markdown("""
This dashboard visualizes Chicago traffic crash data from 2015-2023.
""")

# Data source info
st.sidebar.markdown("---")
st.sidebar.markdown("<h3 style='color:#00B4D8;'>Data Source</h3>", unsafe_allow_html=True)
st.sidebar.markdown("City of Chicago Data Portal")
st.sidebar.markdown("Traffic Crashes dataset: 2015-2023")

# Add dataset statistics to sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("<h3 style='color:#00B4D8;'>Dataset Statistics</h3>", unsafe_allow_html=True)
st.sidebar.markdown(f"**Total Records:** {len(df):,}")
if 'fatal' in df.columns:
    fatal_count = int(df['fatal'].sum())
    st.sidebar.markdown(f"**Fatal Crashes:** {fatal_count:,}")
    st.sidebar.markdown(f"**Fatality Rate:** {fatal_count/len(df)*100:.3f}%")
if 'injuries' in df.columns:
    injury_count = int(df['injuries'].sum())
    st.sidebar.markdown(f"**Total Injuries:** {injury_count:,}")

# Main content
st.markdown("<h1 class='main-header'>Chicago Traffic Crashes Analysis</h1>", unsafe_allow_html=True)

# Dataset summary
st.markdown("<h2 class='section-header'>Dataset Overview</h2>", unsafe_allow_html=True)
total_crashes = len(df)
total_fatal = int(df['fatal'].sum()) if 'fatal' in df.columns else 0
total_injuries = int(df['injuries'].sum()) if 'injuries' in df.columns else 0

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-value'>{total_crashes:,}</div>
        <div class='metric-label'>Total Crashes</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-value'>{total_injuries:,}</div>
        <div class='metric-label'>Total Injuries</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-value'>{total_fatal:,}</div>
        <div class='metric-label'>Fatal Crashes</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    fatal_rate = round((total_fatal / total_crashes) * 100, 3) if total_crashes > 0 else 0
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-value'>{fatal_rate}%</div>
        <div class='metric-label'>Fatality Rate</div>
    </div>
    """, unsafe_allow_html=True)

# Class imbalance note
st.markdown("""
<div class='insight-text'>
💡 <span class='highlight'>Class Imbalance Challenge:</span> This dataset exhibits a severe class imbalance with only about 0.1% of crashes being fatal.
This presents challenges for predictive modeling and requires specialized techniques like class weighting and threshold optimization.
</div>
""", unsafe_allow_html=True)

# Main visualizations in tabs
st.markdown("<h2 class='section-header'>Crash Analysis</h2>", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["Temporal Patterns", "Environmental Factors", "Severity Analysis", "Predictive Insights"])

with tab1:
    st.markdown("<h3 class='subsection-header'>Crashes by Day of Week</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        display_visualization('improved_visuals/crashes_by_day.png', "Day of week visualization not available")
    
    with col2:
        st.markdown("""
        <div class='insight-text'>
        🔍 <span class='highlight'>Key Insights:</span>
        <ul>
            <li>Friday has the highest number of crashes</li>
            <li>Weekends show lower crash volumes</li>
            <li>Workday commutes contribute significantly to accident rates</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<h3 class='subsection-header'>Crashes by Hour of Day</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        display_visualization('improved_visuals/crashes_by_hour.png', "Hour of day visualization not available")
    
    with col2:
        st.markdown("""
        <div class='insight-text'>
        🔍 <span class='highlight'>Key Insights:</span>
        <ul>
            <li>Rush hour peaks at 7-9 AM and 4-6 PM</li>
            <li>Lowest crash rates occur between midnight and 5 AM</li>
            <li>Afternoon peak is higher than morning peak</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

with tab2:
    st.markdown("<h3 class='subsection-header'>Crashes by Weather Condition</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        display_visualization('improved_visuals/crashes_by_weather.png', "Weather visualization not available")
    
    with col2:
        st.markdown("""
        <div class='insight-text'>
        🔍 <span class='highlight'>Key Insights:</span>
        <ul>
            <li>Most crashes occur during clear weather (70%+)</li>
            <li>Rain is associated with the second highest number of crashes</li>
            <li>The "per-day" crash rate during adverse weather is higher</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<h3 class='subsection-header'>Crashes by Road Condition</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        display_visualization('improved_visuals/crashes_by_road_condition.png', "Road condition visualization not available")
    
    with col2:
        st.markdown("""
        <div class='insight-text'>
        🔍 <span class='highlight'>Key Insights:</span>
        <ul>
            <li>Dry road conditions account for most crashes</li>
            <li>Wet and icy conditions have higher risk per exposure time</li>
            <li>Snow/slush conditions reduce volume but increase risk</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<h3 class='subsection-header'>Crashes by Lighting Condition</h3>", unsafe_allow_html=True)
    display_visualization('improved_visuals/crashes_by_lighting.png', "Lighting condition visualization not available")

with tab3:
    st.markdown("<h3 class='subsection-header'>Fatal Crashes by Time of Day</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        display_visualization('improved_visuals/fatal_by_time.png', "Fatal by time visualization not available")
    
    with col2:
        st.markdown("""
        <div class='insight-text'>
        🔍 <span class='highlight'>Key Insights:</span>
        <ul>
            <li>Night hours (0-6) have the highest fatality rate</li>
            <li>Poor visibility and potentially impaired driving contribute to night fatalities</li>
            <li>Afternoon hours have high crash volume but lower fatality rate</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<h3 class='subsection-header'>Fatal Crashes by Weather Condition</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        display_visualization('improved_visuals/fatal_by_weather.png', "Fatal by weather visualization not available")
    
    with col2:
        st.markdown("""
        <div class='insight-text'>
        🔍 <span class='highlight'>Key Insights:</span>
        <ul>
            <li>Clear weather has most fatal crashes by volume</li>
            <li>Severe cross wind has highest fatality rate</li>
            <li>Rain shows increase in both crash volume and fatality rate</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<h3 class='subsection-header'>Injuries by Hour of Day</h3>", unsafe_allow_html=True)
    display_visualization('improved_visuals/injuries_by_hour.png', "Injuries by hour visualization not available")

with tab4:
    st.markdown("<h3 class='subsection-header'>Model Feature Importance</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        display_visualization('improved_visuals/feature_importance.png', "Feature importance visualization not available")
    
    with col2:
        st.markdown("""
        <div class='insight-text'>
        🔍 <span class='highlight'>Predictive Insights:</span>
        <ul>
            <li>Hour of day is the most important predictor</li>
            <li>Lighting conditions are strongly correlated with fatal crashes</li>
            <li>Weather plays a significant but lesser role</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<h3 class='subsection-header'>Model Performance on Imbalanced Data</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        display_visualization('improved_visuals/confusion_matrix.png', "Confusion matrix visualization not available")
    
    with col2:
        st.markdown("""
        <div class='insight-text'>
        🔍 <span class='highlight'>Class Imbalance Challenge:</span>
        <ul>
            <li>Baseline model: 99.89% accuracy but 0% recall for fatal crashes</li>
            <li>With class weighting: Improved fatal crash recall to 9%</li>
            <li>Demonstrates the challenge of predicting rare events</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

# Key findings section
st.markdown("<h2 class='section-header'>Key Findings</h2>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class='insight-text'>
    📊 <span class='highlight'>Temporal Patterns:</span>
    <ul>
        <li>Rush hours (7-9 AM, 4-6 PM) show highest crash volumes</li>
        <li>Friday has highest crash frequency among days</li>
        <li>Late night hours (midnight-5 AM) have fewer crashes but higher fatality rates</li>
    </ul>
    </div>
    
    <div class='insight-text'>
    🌦️ <span class='highlight'>Environmental Factors:</span>
    <ul>
        <li>Most crashes occur in clear weather and dry road conditions</li>
        <li>Adverse conditions (rain, snow, ice) have higher risk per exposure hour</li>
        <li>Lighting conditions are critical for crash severity</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class='insight-text'>
    ⚠️ <span class='highlight'>Severity Factors:</span>
    <ul>
        <li>Night driving has highest fatality rate</li>
        <li>Weekend crashes more likely to be severe</li>
        <li>Poor lighting and adverse weather combine to increase severity</li>
    </ul>
    </div>
    
    <div class='insight-text'>
    🧠 <span class='highlight'>Predictive Modeling:</span>
    <ul>
        <li>Class imbalance makes fatal crash prediction challenging</li>
        <li>Time and lighting features are most predictive</li>
        <li>Specialized techniques needed for rare event prediction</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# Recommendations section
st.markdown("<h2 class='section-header'>Recommendations</h2>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class='insight-text'>
    🚨 <span class='highlight'>For Traffic Management:</span>
    <ul>
        <li>Increase patrol presence during peak crash hours</li>
        <li>Target Friday afternoon rush hour for special attention</li>
        <li>Focus on night-time driving safety campaigns</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class='insight-text'>
    🛣️ <span class='highlight'>For Infrastructure:</span>
    <ul>
        <li>Improve street lighting in high-fatality areas</li>
        <li>Enhance road drainage in areas prone to weather-related crashes</li>
        <li>Add traffic calming measures in high-risk corridors</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class='insight-text'>
    📱 <span class='highlight'>For Public Awareness:</span>
    <ul>
        <li>Develop targeted safety campaigns for night drivers</li>
        <li>Create weather alerts that include driving safety tips</li>
        <li>Educate on the dangers of rush hour distracted driving</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("© 2023 Chicago Traffic Safety Analysis Project | Data source: City of Chicago Data Portal") 