import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
import folium
from folium.plugins import HeatMap
from streamlit_folium import folium_static
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Chicago Traffic Crashes Dashboard",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3D59;
        text-align: center;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.8rem;
        color: #1E3D59;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .insight-text {
        background-color: #F5F5F5;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1E3D59;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #F0F2F6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E3D59;
    }
    .metric-label {
        font-size: 1rem;
        color: #666;
    }
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
            
        # Create day_name and time_category if needed
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
        st.stop()

# Load the data
df = load_data()

# Define the sidebar
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/9/9b/Flag_of_Chicago%2C_Illinois.svg/1200px-Flag_of_Chicago%2C_Illinois.svg.png", width=200)
st.sidebar.title("Chicago Traffic Crashes")

# Date range filter if crash_date exists
if 'crash_date' in df.columns:
    min_date = df['crash_date'].min()
    max_date = df['crash_date'].max()
    
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(min_date.date(), max_date.date()),
        min_value=min_date.date(),
        max_value=max_date.date(),
    )
    
    filtered_df = df[
        (df['crash_date'].dt.date >= date_range[0]) & 
        (df['crash_date'].dt.date <= date_range[1])
    ]
else:
    filtered_df = df

# Additional filters
if 'weather' in filtered_df.columns:
    weather_options = ['All'] + list(filtered_df['weather'].unique())
    selected_weather = st.sidebar.selectbox("Weather Condition", weather_options)
    
    if selected_weather != 'All':
        filtered_df = filtered_df[filtered_df['weather'] == selected_weather]

if 'time_category' in filtered_df.columns:
    time_options = ['All'] + list(filtered_df['time_category'].unique())
    selected_time = st.sidebar.selectbox("Time of Day", time_options)
    
    if selected_time != 'All':
        filtered_df = filtered_df[filtered_df['time_category'] == selected_time]

# Additional sidebar info
st.sidebar.markdown("---")
st.sidebar.info(
    "This dashboard visualizes Chicago traffic crash data to identify patterns and insights."
)
st.sidebar.markdown("---")
st.sidebar.markdown("Data source: City of Chicago")

# Header section
st.markdown("<h1 class='main-header'>Chicago Traffic Crashes Dashboard</h1>", unsafe_allow_html=True)

# Key metrics
st.markdown("<h2 class='section-header'>Key Metrics</h2>", unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)

total_crashes = len(filtered_df)
with col1:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-value'>{total_crashes:,}</div>
        <div class='metric-label'>Total Crashes</div>
    </div>
    """, unsafe_allow_html=True)

if 'injuries' in filtered_df.columns:
    total_injuries = int(filtered_df['injuries'].sum())
    with col2:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>{total_injuries:,}</div>
            <div class='metric-label'>Total Injuries</div>
        </div>
        """, unsafe_allow_html=True)

if 'fatal' in filtered_df.columns:
    fatal_crashes = int(filtered_df['fatal'].sum())
    with col3:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>{fatal_crashes:,}</div>
            <div class='metric-label'>Fatal Crashes</div>
        </div>
        """, unsafe_allow_html=True)

if 'is_weekend' in filtered_df.columns:
    weekend_pct = filtered_df['is_weekend'].mean() * 100
    with col4:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>{weekend_pct:.1f}%</div>
            <div class='metric-label'>Weekend Crashes</div>
        </div>
        """, unsafe_allow_html=True)

# Main visualizations
st.markdown("<h2 class='section-header'>Crash Patterns</h2>", unsafe_allow_html=True)

# Crashes by time tab panel
tab1, tab2, tab3, tab4 = st.tabs(["Day & Time", "Weather & Road", "Severity Analysis", "Map View"])

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        # Crashes by day of week
        if 'day_name' in filtered_df.columns:
            day_counts = filtered_df['day_name'].value_counts().reindex(['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'])
            
            # Calculate percentages
            day_pct = (day_counts / day_counts.sum() * 100).round(1)
            
            # Create figure with primary y-axis
            fig = go.Figure()
            
            # Add bar chart for count
            fig.add_trace(go.Bar(
                x=day_counts.index,
                y=day_counts.values,
                name='Crash Count',
                marker_color=px.colors.sequential.Viridis,
                text=day_counts.values,
                textposition='outside'
            ))
            
            # Update layout
            fig.update_layout(
                title='Crashes by Day of Week',
                xaxis_title='Day of Week',
                yaxis_title='Number of Crashes',
                template='plotly_white',
                height=450
            )
            
            # Add percentage labels inside bars
            for i, (day, count, pct) in enumerate(zip(day_counts.index, day_counts.values, day_pct)):
                fig.add_annotation(
                    x=day,
                    y=count/2,
                    text=f"{pct}%",
                    font=dict(color='white', size=12),
                    showarrow=False
                )
                
            st.plotly_chart(fig, use_container_width=True)
            
            weekday_crashes = int(filtered_df[filtered_df['is_weekend'] == 0]['injuries'].sum())
            weekend_crashes = int(filtered_df[filtered_df['is_weekend'] == 1]['injuries'].sum())
            
            st.markdown(
                "<div class='insight-text'>💡 " + 
                f"Accidents on weekdays account for {(1 - weekend_pct/100):.1%} of all crashes. " +
                f"Peak crash day is typically {day_counts.idxmax()}." +
                "</div>", 
                unsafe_allow_html=True
            )
    
    with col2:
        # Crashes by hour of day
        if 'hour' in filtered_df.columns:
            hour_counts = filtered_df['hour'].value_counts().sort_index()
            
            # Create figure
            fig = go.Figure()
            
            # Add line chart
            fig.add_trace(go.Scatter(
                x=hour_counts.index,
                y=hour_counts.values,
                mode='lines+markers',
                name='Hourly Crashes',
                line=dict(color='royalblue', width=3),
                marker=dict(size=8)
            ))
            
            # Add rush hour highlights
            rush_hours_morning = list(range(7, 10))
            rush_hours_evening = list(range(16, 19))
            
            # Add rectangles for rush hours
            for hour in rush_hours_morning + rush_hours_evening:
                if hour in hour_counts.index:
                    fig.add_vrect(
                        x0=hour-0.5, 
                        x1=hour+0.5, 
                        fillcolor="orange", 
                        opacity=0.15, 
                        layer="below", 
                        line_width=0
                    )
            
            # Update layout
            fig.update_layout(
                title='Crashes by Hour of Day',
                xaxis_title='Hour (24-hour format)',
                yaxis_title='Number of Crashes',
                xaxis=dict(
                    tickmode='array',
                    tickvals=list(range(0, 24)),
                    ticktext=[f'{h:02d}:00' for h in range(0, 24)]
                ),
                template='plotly_white',
                height=450
            )
            
            # Find peak hour
            peak_hour = hour_counts.idxmax()
            peak_value = hour_counts.max()
            
            # Add annotation for peak
            fig.add_annotation(
                x=peak_hour,
                y=peak_value,
                text=f"Peak: {peak_value:,}",
                showarrow=True,
                arrowhead=1,
                ax=40,
                ay=-40
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown(
                "<div class='insight-text'>💡 " + 
                f"Peak crash hour is {peak_hour:02d}:00 with {peak_value:,} crashes. " +
                "Morning (7-9 AM) and evening (4-6 PM) rush hours show significantly higher crash rates." +
                "</div>", 
                unsafe_allow_html=True
            )

with tab2:
    col1, col2 = st.columns(2)
    
    with col1:
        # Weather conditions
        if 'weather' in filtered_df.columns:
            weather_counts = filtered_df['weather'].value_counts().head(10)
            weather_pct = (weather_counts / weather_counts.sum() * 100).round(1)
            
            # Create horizontal bar chart
            fig = px.bar(
                x=weather_counts.values,
                y=weather_counts.index,
                orientation='h',
                color=weather_counts.values,
                color_continuous_scale=px.colors.sequential.Viridis,
                labels={'x': 'Number of Crashes', 'y': 'Weather Condition'}
            )
            
            # Update layout
            fig.update_layout(
                title='Top 10 Weather Conditions During Crashes',
                xaxis_title='Number of Crashes',
                yaxis_title='',
                coloraxis_showscale=False,
                height=450
            )
            
            # Add percentage annotations
            for i, (val, pct) in enumerate(zip(weather_counts.values, weather_pct)):
                fig.add_annotation(
                    x=val + max(weather_counts.values) * 0.02,
                    y=i,
                    text=f"{val:,} ({pct}%)",
                    showarrow=False,
                    font=dict(size=10)
                )
                
            st.plotly_chart(fig, use_container_width=True)
            
            clear_pct = weather_pct.loc["CLEAR"] if "CLEAR" in weather_pct.index else 0
            
            st.markdown(
                "<div class='insight-text'>💡 " + 
                f"Most crashes ({clear_pct}%) occur during clear weather conditions, " +
                "highlighting that driver behavior rather than weather is often the primary factor." +
                "</div>", 
                unsafe_allow_html=True
            )
    
    with col2:
        # Road conditions
        if 'road_condition' in filtered_df.columns:
            road_counts = filtered_df['road_condition'].value_counts().head(10)
            road_pct = (road_counts / road_counts.sum() * 100).round(1)
            
            # Create horizontal bar chart with blues color scale
            fig = px.bar(
                x=road_counts.values,
                y=road_counts.index,
                orientation='h',
                color=road_counts.values,
                color_continuous_scale=px.colors.sequential.Blues,
                labels={'x': 'Number of Crashes', 'y': 'Road Condition'}
            )
            
            # Update layout
            fig.update_layout(
                title='Top 10 Road Conditions During Crashes',
                xaxis_title='Number of Crashes',
                yaxis_title='',
                coloraxis_showscale=False,
                height=450
            )
            
            # Add percentage annotations
            for i, (val, pct) in enumerate(zip(road_counts.values, road_pct)):
                fig.add_annotation(
                    x=val + max(road_counts.values) * 0.02,
                    y=i,
                    text=f"{val:,} ({pct}%)",
                    showarrow=False,
                    font=dict(size=10)
                )
                
            st.plotly_chart(fig, use_container_width=True)
            
            dry_pct = road_pct.loc["DRY"] if "DRY" in road_pct.index else 0
            
            st.markdown(
                "<div class='insight-text'>💡 " + 
                f"Dry road conditions account for {dry_pct}% of crashes. " +
                "Adverse road conditions like wet, snow, or ice significantly increase crash risk per mile driven." +
                "</div>", 
                unsafe_allow_html=True
            )

with tab3:
    col1, col2 = st.columns(2)
    
    with col1:
        # Fatal crashes by time category
        if 'time_category' in filtered_df.columns and 'fatal' in filtered_df.columns:
            fatal_by_time = filtered_df.groupby('time_category', observed=True)['fatal'].sum()
            total_by_time = filtered_df.groupby('time_category', observed=True).size()
            percentage_by_time = (fatal_by_time / total_by_time * 100).round(2)
            
            # Create figure with dual y-axes
            fig = go.Figure()
            
            # Add bar chart for count
            fig.add_trace(go.Bar(
                x=fatal_by_time.index,
                y=fatal_by_time.values,
                name='Fatal Crashes',
                marker_color=px.colors.sequential.Reds
            ))
            
            # Add line chart for percentage
            fig.add_trace(go.Scatter(
                x=percentage_by_time.index,
                y=percentage_by_time.values,
                name='Fatality Rate (%)',
                mode='lines+markers',
                line=dict(color='darkred', width=3),
                marker=dict(size=10),
                yaxis='y2'
            ))
            
            # Update layout with secondary y-axis
            fig.update_layout(
                title='Fatal Crashes by Time of Day',
                xaxis_title='Time of Day',
                yaxis_title='Number of Fatal Crashes',
                yaxis2=dict(
                    title='Fatality Rate (%)',
                    titlefont=dict(color='darkred'),
                    tickfont=dict(color='darkred'),
                    overlaying='y',
                    side='right'
                ),
                legend=dict(
                    orientation='h',
                    yanchor='bottom',
                    y=1.02,
                    xanchor='right',
                    x=1
                ),
                template='plotly_white',
                height=450
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            most_fatal_time = fatal_by_time.idxmax()
            highest_rate_time = percentage_by_time.idxmax()
            
            st.markdown(
                "<div class='insight-text'>💡 " + 
                f"The {most_fatal_time} period has the highest number of fatal crashes, " +
                f"while the {highest_rate_time} period has the highest fatality rate at {percentage_by_time.max():.2f}%." +
                "</div>", 
                unsafe_allow_html=True
            )
    
    with col2:
        # Injuries by hour
        if 'hour' in filtered_df.columns and 'injuries' in filtered_df.columns:
            hourly_injuries = filtered_df.groupby('hour')['injuries'].mean().reset_index()
            
            # Create figure
            fig = px.line(
                hourly_injuries, 
                x='hour', 
                y='injuries',
                labels={'hour': 'Hour of Day', 'injuries': 'Average Injuries per Crash'},
                markers=True,
                color_discrete_sequence=['purple']
            )
            
            # Add night hour shading
            fig.add_vrect(
                x0=0, x1=6, 
                fillcolor="gray", opacity=0.2, 
                layer="below", line_width=0,
                annotation_text="Night Hours",
                annotation_position="top left"
            )
            fig.add_vrect(
                x0=22, x1=24, 
                fillcolor="gray", opacity=0.2, 
                layer="below", line_width=0
            )
            
            # Update layout
            fig.update_layout(
                title='Average Injuries per Crash by Hour of Day',
                xaxis=dict(
                    tickmode='array',
                    tickvals=list(range(0, 24)),
                    ticktext=[f'{h:02d}:00' for h in range(0, 24)]
                ),
                template='plotly_white',
                height=450
            )
            
            # Highlight peak injury hour
            peak_hour = hourly_injuries.loc[hourly_injuries['injuries'].idxmax()]
            
            fig.add_annotation(
                x=peak_hour['hour'],
                y=peak_hour['injuries'],
                text=f"Peak: {peak_hour['injuries']:.2f}",
                showarrow=True,
                arrowhead=1,
                ax=40,
                ay=-40
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown(
                "<div class='insight-text'>💡 " + 
                f"Crashes at {int(peak_hour['hour']):02d}:00 result in the highest average injuries ({peak_hour['injuries']:.2f} per crash). " +
                "Late night/early morning crashes tend to be more severe." +
                "</div>", 
                unsafe_allow_html=True
            )

with tab4:
    # Map visualization if coordinates are available
    if 'latitude' in filtered_df.columns and 'longitude' in filtered_df.columns:
        st.subheader("Crash Locations in Chicago")
        
        # Filter out invalid coordinates
        map_data = filtered_df.dropna(subset=['latitude', 'longitude'])
        
        # Sample data if it's too large
        if len(map_data) > 5000:
            st.info(f"Showing a sample of 5,000 crash locations out of {len(map_data):,} total.")
            map_data = map_data.sample(5000, random_state=42)
        
        # Create a base map centered on Chicago
        m = folium.Map(location=[41.8781, -87.6298], zoom_start=11)
        
        # Add a heatmap layer
        heat_data = [[row['latitude'], row['longitude']] for _, row in map_data.iterrows()]
        HeatMap(heat_data, radius=10).add_to(m)
        
        # Add cluster markers for fatal crashes if available
        if 'fatal' in map_data.columns:
            fatal_crashes = map_data[map_data['fatal'] == 1]
            if len(fatal_crashes) > 0:
                fatal_group = folium.FeatureGroup(name='Fatal Crashes')
                
                for _, crash in fatal_crashes.iterrows():
                    folium.CircleMarker(
                        location=[crash['latitude'], crash['longitude']],
                        radius=4,
                        color='red',
                        fill=True,
                        fill_color='red',
                        fill_opacity=0.7,
                        tooltip='Fatal Crash'
                    ).add_to(fatal_group)
                
                fatal_group.add_to(m)
        
        # Display the map
        folium_static(m, width=1200, height=600)
        
        st.markdown(
            "<div class='insight-text'>💡 " + 
            "Crash hotspots are concentrated along major highways and in downtown areas. " +
            "Red markers indicate fatal crashes, which are distributed throughout the city." +
            "</div>", 
            unsafe_allow_html=True
        )
    else:
        st.info("Map view requires latitude and longitude data, which is not available in the current dataset.")

# ML Insights section
if 'fatal' in df.columns:
    st.markdown("<h2 class='section-header'>Predictive Insights</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Fatal Crashes by Weather")
        # Fatal crashes by weather
        fatal_by_weather = filtered_df.groupby('weather')['fatal'].sum().sort_values(ascending=False).head(10)
        total_by_weather = filtered_df.groupby('weather').size().loc[fatal_by_weather.index]
        fatality_rate = (fatal_by_weather / total_by_weather * 1000).round(2)  # per 1000 crashes
        
        # Create a DataFrame for the table
        weather_data = pd.DataFrame({
            'Weather Condition': fatal_by_weather.index,
            'Fatal Crashes': fatal_by_weather.values,
            'Total Crashes': total_by_weather.values,
            'Fatality Rate (per 1,000)': fatality_rate.values
        })
        
        # Display the table
        st.dataframe(
            weather_data,
            column_config={
                "Weather Condition": st.column_config.TextColumn("Weather Condition"),
                "Fatal Crashes": st.column_config.NumberColumn("Fatal Crashes", format="%d"),
                "Total Crashes": st.column_config.NumberColumn("Total Crashes", format="%d"),
                "Fatality Rate (per 1,000)": st.column_config.NumberColumn("Fatality Rate (per 1,000)", format="%.2f")
            },
            use_container_width=True,
            hide_index=True
        )
        
        # Find highest fatality rate
        highest_rate_condition = fatality_rate.idxmax()
        highest_rate = fatality_rate.max()
        
        st.markdown(
            "<div class='insight-text'>💡 " + 
            f"The highest fatality rate is in {highest_rate_condition} conditions " +
            f"with {highest_rate:.2f} fatal crashes per 1,000 accidents." +
            "</div>", 
            unsafe_allow_html=True
        )
        
    with col2:
        st.subheader("Crash Severity Predictors")
        
        # Display a sample ML feature importance visualization
        st.image('improved_visuals/feature_importance.png', use_column_width=True)
        
        st.markdown(
            "<div class='insight-text'>💡 " + 
            "The machine learning model identifies time of day and environmental conditions " +
            "as key predictors for crash severity. These insights can help target safety initiatives." +
            "</div>", 
            unsafe_allow_html=True
        )

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #666;">
        Chicago Traffic Crashes Analysis Dashboard | Big Data Project | Created with Streamlit
    </div>
    """,
    unsafe_allow_html=True
) 