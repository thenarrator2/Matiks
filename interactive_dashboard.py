import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import warnings
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from operator import attrgetter
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Gaming Analytics Dashboard",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize theme in session state
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = True

# Custom CSS for dark/light theme
def load_css():
    # Add dynamic theme switching with JavaScript
    st.markdown(f"""
    <style>
    /* Dynamic theme variables */
    :root {{
        --theme-bg: {'linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%)' if st.session_state.dark_mode else 'linear-gradient(135deg, #f8f9fa 0%, #e9ecef 50%, #dee2e6 100%)'};
        --theme-text: {'#ffffff' if st.session_state.dark_mode else '#212529'};
        --theme-text-secondary: {'#e2e8f0' if st.session_state.dark_mode else '#495057'};
        --theme-sidebar: {'linear-gradient(180deg, #2d3436 0%, #636e72 100%)' if st.session_state.dark_mode else 'linear-gradient(180deg, #ffffff 0%, #f8f9fa 100%)'};
    }}
    </style>
    """, unsafe_allow_html=True)
    
    if st.session_state.dark_mode:
        # Dark mode styles with better contrast
        st.markdown("""
        <style>
        .main > div {
            padding-top: 2rem;
        }
        
        .stApp {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%) !important;
            color: #ffffff !important;
        }
        
        .stSidebar {
            background: linear-gradient(180deg, #2d3436 0%, #636e72 100%) !important;
        }
        
        .stSidebar .stMarkdown {
            color: #ffffff !important;
        }
        
        .metric-card {
            background: linear-gradient(145deg, rgba(255,255,255,0.15) 0%, rgba(255,255,255,0.08) 100%);
            padding: 1.5rem;
            border-radius: 15px;
            border: 1px solid rgba(255,255,255,0.2);
            backdrop-filter: blur(10px);
            box-shadow: 0 8px 32px rgba(0,0,0,0.4);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        
        .metric-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 12px 40px rgba(0,0,0,0.6);
        }
        
        .metric-title {
            font-size: 0.9rem;
            color: #e2e8f0;
            margin-bottom: 0.5rem;
            font-weight: 600;
        }
        
        .metric-value {
            font-size: 2rem;
            font-weight: 700;
            color: #ffffff;
            margin-bottom: 0.3rem;
        }
        
        .metric-delta {
            font-size: 0.8rem;
            color: #74c0fc;
            font-weight: 600;
        }
        
        .section-header {
            color: #ffffff !important;
            font-size: 1.8rem;
            font-weight: 700;
            margin: 2rem 0 1rem 0;
        }
        
        .stSelectbox > div > div {
            background: rgba(255,255,255,0.15) !important;
            border: 1px solid rgba(255,255,255,0.3) !important;
            border-radius: 10px !important;
            color: #ffffff !important;
        }
        
        .stMultiSelect > div > div {
            background: rgba(255,255,255,0.15) !important;
            border: 1px solid rgba(255,255,255,0.3) !important;
            border-radius: 10px !important;
            color: #ffffff !important;
        }
        
        .stSelectbox label, .stMultiSelect label, .stDateInput label {
            color: #ffffff !important;
            font-weight: 600 !important;
        }
        
        .stTabs [data-baseweb="tab-list"] {
            background: rgba(255,255,255,0.1);
            border-radius: 10px;
        }
        
        .stTabs [data-baseweb="tab"] {
            color: #e2e8f0 !important;
            font-weight: 600;
        }
        
        .stTabs [aria-selected="true"] {
            background: rgba(255,255,255,0.2) !important;
            color: #ffffff !important;
        }
        
        .footer {
            background: linear-gradient(135deg, rgba(255,255,255,0.15) 0%, rgba(255,255,255,0.08) 100%);
            border-radius: 15px;
            padding: 2rem;
            text-align: center;
            margin-top: 3rem;
            border: 1px solid rgba(255,255,255,0.2);
        }
        
        div[data-testid="stDataFrame"] {
            background: rgba(255,255,255,0.08) !important;
            border-radius: 10px;
            border: 1px solid rgba(255,255,255,0.2);
        }
        
        div[data-testid="stDataFrame"] table {
            color: #ffffff !important;
        }
        
        div[data-testid="stDataFrame"] th {
            background: rgba(255,255,255,0.1) !important;
            color: #ffffff !important;
            font-weight: 600 !important;
        }
        
        div[data-testid="stDataFrame"] td {
            color: #e2e8f0 !important;
        }
        
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
            color: #ffffff !important;
        }
        
        .stButton > button {
            background: linear-gradient(45deg, #4facfe, #00f2fe) !important;
            color: #ffffff !important;
            border: none !important;
            border-radius: 25px !important;
            font-weight: 600 !important;
            transition: all 0.3s ease !important;
        }
        
        .stButton > button:hover {
            transform: scale(1.05) !important;
            box-shadow: 0 8px 25px rgba(79, 172, 254, 0.4) !important;
        }
        
        /* Fix header gradient */
        .main-header {
            background: linear-gradient(45deg, #4facfe, #00f2fe) !important;
            -webkit-background-clip: text !important;
            -webkit-text-fill-color: transparent !important;
            background-clip: text !important;
        }
        </style>
        """, unsafe_allow_html=True)
    else:
        # Light mode styles
        st.markdown("""
        <style>
        .main > div {
            padding-top: 2rem;
        }
        
        .stApp {
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 50%, #dee2e6 100%) !important;
            color: #212529 !important;
        }
        
        .stSidebar {
            background: linear-gradient(180deg, #ffffff 0%, #f8f9fa 100%) !important;
        }
        
        .stSidebar .stMarkdown {
            color: #212529 !important;
        }
        
        .metric-card {
            background: linear-gradient(145deg, rgba(255,255,255,0.95) 0%, rgba(255,255,255,0.85) 100%);
            padding: 1.5rem;
            border-radius: 15px;
            border: 1px solid rgba(0,0,0,0.1);
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        
        .metric-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 12px 40px rgba(0,0,0,0.15);
        }
        
        .metric-title {
            font-size: 0.9rem;
            color: #495057;
            margin-bottom: 0.5rem;
            font-weight: 600;
        }
        
        .metric-value {
            font-size: 2rem;
            font-weight: 700;
            color: #212529;
            margin-bottom: 0.3rem;
        }
        
        .metric-delta {
            font-size: 0.8rem;
            color: #0066cc;
            font-weight: 600;
        }
        
        .section-header {
            color: #212529 !important;
            font-size: 1.8rem;
            font-weight: 700;
            margin: 2rem 0 1rem 0;
        }
        
        .footer {
            background: linear-gradient(135deg, rgba(255,255,255,0.95) 0%, rgba(255,255,255,0.85) 100%);
            border-radius: 15px;
            padding: 2rem;
            text-align: center;
            margin-top: 3rem;
            border: 1px solid rgba(0,0,0,0.1);
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        }
        
        .stButton > button {
            background: linear-gradient(45deg, #0066cc, #004499) !important;
            color: #ffffff !important;
            border: none !important;
            border-radius: 25px !important;
            font-weight: 600 !important;
            transition: all 0.3s ease !important;
        }
        
        .stButton > button:hover {
            transform: scale(1.05) !important;
            box-shadow: 0 8px 25px rgba(0, 102, 204, 0.4) !important;
        }
        
        /* Fix header gradient */
        .main-header {
            background: linear-gradient(45deg, #0066cc, #004499) !important;
            -webkit-background-clip: text !important;
            -webkit-text-fill-color: transparent !important;
            background-clip: text !important;
        }
        </style>
        """, unsafe_allow_html=True)

# Load CSS
load_css()

# Theme toggle button with better handling
col1, col2, col3 = st.columns([6, 1, 1])
with col2:
    theme_icon = "🌙" if st.session_state.dark_mode else "☀️"
    theme_text = "Light Mode" if st.session_state.dark_mode else "Dark Mode"
    
    if st.button(f"{theme_icon}", help=f"Switch to {theme_text}"):
        st.session_state.dark_mode = not st.session_state.dark_mode
        # No rerun needed - let Streamlit handle the natural refresh

# Load and cache data
@st.cache_data
def load_data():
    """Load and preprocess the gaming data"""
    try:
        df = pd.read_excel('Matiks - Data Analyst Data_cleaned.xlsx')
    except:
        df = pd.read_excel('Matiks - Data Analyst Data.xlsx')
    
    # Convert date columns
    df['Signup_Date'] = pd.to_datetime(df['Signup_Date'])
    df['Last_Login'] = pd.to_datetime(df['Last_Login'])
    
    # Fix negative values
    df['Total_Hours_Played'] = df['Total_Hours_Played'].abs()
    df['Avg_Session_Duration_Min'] = df['Avg_Session_Duration_Min'].abs()
    
    # Create additional metrics
    df['Days_Since_Signup'] = (df['Last_Login'] - df['Signup_Date']).dt.days
    df['Revenue_Per_Session'] = df['Total_Revenue_USD'] / df['Total_Play_Sessions'].replace(0, 1)
    df['Revenue_Per_Hour'] = df['Total_Revenue_USD'] / df['Total_Hours_Played'].replace(0, 1)
    
    # Create user segments
    df['Spending_Segment'] = pd.cut(df['Total_Revenue_USD'], 
                                   bins=[0, 1, 25, 75, float('inf')], 
                                   labels=['Non-Spender', 'Low Spender', 'Medium Spender', 'High Spender'])
    
    df['Engagement_Segment'] = pd.cut(df['Total_Hours_Played'], 
                                     bins=[0, 10, 50, 150, float('inf')], 
                                     labels=['Casual', 'Regular', 'Engaged', 'Super User'])
    
    return df

# Calculate DAU/WAU/MAU
@st.cache_data
def calculate_active_users(df):
    """Calculate Daily, Weekly, Monthly Active Users"""
    today = df['Last_Login'].max()
    
    dau = df[df['Last_Login'] >= today - timedelta(days=1)].shape[0]
    wau = df[df['Last_Login'] >= today - timedelta(days=7)].shape[0]
    mau = df[df['Last_Login'] >= today - timedelta(days=30)].shape[0]
    
    return dau, wau, mau

# Cohort Analysis Functions
@st.cache_data
def perform_cohort_analysis(df):
    """Perform cohort analysis based on signup date"""
    df_cohort = df.copy()
    df_cohort['Signup_Period'] = df_cohort['Signup_Date'].dt.to_period('M')
    df_cohort['Last_Login_Period'] = df_cohort['Last_Login'].dt.to_period('M')
    
    df_cohort['Period_Number'] = (
        df_cohort['Last_Login_Period'] - df_cohort['Signup_Period']
    ).apply(attrgetter('n'))
    
    # Ensure Period_Number is non-negative
    df_cohort = df_cohort[df_cohort['Period_Number'] >= 0]
    
    cohort_data = df_cohort.groupby('Signup_Period')['User_ID'].nunique().reset_index()
    cohort_data['Signup_Period'] = cohort_data['Signup_Period'].astype(str)
    cohort_data.rename(columns={'User_ID': 'Total_Users'}, inplace=True)
    
    cohort_sizes = df_cohort.groupby(['Signup_Period', 'Period_Number'])['User_ID'].nunique().reset_index()
    cohort_sizes['Signup_Period'] = cohort_sizes['Signup_Period'].astype(str)
    
    if cohort_sizes.empty:
        return pd.DataFrame(), pd.DataFrame(), cohort_data
        
    cohort_table = cohort_sizes.pivot(index='Signup_Period', columns='Period_Number', values='User_ID')
    
    # Get initial cohort sizes (users in period 0)
    initial_cohort_sizes = cohort_table.iloc[:, 0]
    
    # Prevent division by zero or NaN issues, then clip to ensure 0-1 range
    cohort_table_pct = cohort_table.divide(initial_cohort_sizes + 1e-9, axis=0)
    cohort_table_pct = cohort_table_pct.fillna(0).clip(lower=0, upper=1)
    
    return cohort_table, cohort_table_pct, cohort_data

# Funnel Analysis Functions
@st.cache_data
def calculate_funnel_metrics(df):
    """Calculate funnel metrics for user journey"""
    funnel_data = {}
    
    # Stage 1: Total Signups
    total_signups = len(df)
    funnel_data['Signups'] = total_signups
    
    # Stage 2: Users who played at least one game (have play sessions > 0)
    first_game_users = len(df[df['Total_Play_Sessions'] > 0])
    funnel_data['First Game'] = first_game_users
    
    # Stage 3: Users who had repeat sessions (more than 3 sessions to be more realistic)
    repeat_users = len(df[df['Total_Play_Sessions'] > 3])
    funnel_data['Repeat Sessions'] = repeat_users
    
    # Stage 4: Users who made a purchase (revenue > 0)
    paying_users = len(df[df['Total_Revenue_USD'] > 0])
    funnel_data['Made Purchase'] = paying_users
    
    # Stage 5: High value users (revenue > $50)
    high_value_users = len(df[df['Total_Revenue_USD'] > 50])
    funnel_data['High Value Users'] = high_value_users
    
    return funnel_data

# User Clustering Functions
@st.cache_data
def perform_user_clustering(df, n_clusters=4):
    """Perform K-means clustering on user data"""
    # Prepare features for clustering
    available_features = ['Total_Play_Sessions', 'Total_Revenue_USD', 'Total_Hours_Played', 'Achievement_Score']
    features = [f for f in available_features if f in df.columns]
    
    if len(features) < 2:
        raise ValueError("Need at least 2 numeric features for clustering")
    
    # Clean data for clustering
    cluster_df = df[features].fillna(0)
    
    # Remove any infinite values
    cluster_df = cluster_df.replace([np.inf, -np.inf], 0)
    
    # Ensure we have enough data points
    if len(cluster_df) < n_clusters:
        raise ValueError(f"Need at least {n_clusters} data points for {n_clusters} clusters")
    
    # Standardize features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(cluster_df)
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(scaled_features)
    
    # Add cluster labels to dataframe
    df_clustered = df.copy()
    df_clustered['Cluster'] = clusters
    
    # Calculate cluster statistics
    cluster_stats = df_clustered.groupby('Cluster').agg({
        'Total_Play_Sessions': 'mean',
        'Total_Revenue_USD': 'mean',
        'Total_Hours_Played': 'mean',
        'Achievement_Score': 'mean',
        'User_ID': 'count'
    }).round(2)
    
    cluster_stats.rename(columns={'User_ID': 'User_Count'}, inplace=True)
    
    return df_clustered, cluster_stats

# Get theme colors
def get_theme_colors():
    if st.session_state.dark_mode:
        return {
            'primary': '#4facfe',
            'secondary': '#00f2fe',
            'accent': '#74c0fc',
            'background': '#1a1a2e',
            'surface': 'rgba(255,255,255,0.15)',
            'text': '#ffffff',
            'text_secondary': '#e2e8f0'
        }
    else:
        return {
            'primary': '#0066cc',
            'secondary': '#004499',
            'accent': '#28a745',
            'background': '#f8f9fa',
            'surface': 'rgba(255,255,255,0.95)',
            'text': '#212529',
            'text_secondary': '#495057'
        }

# Create themed plotly template
def get_plotly_theme():
    colors = get_theme_colors()
    template = 'plotly_dark' if st.session_state.dark_mode else 'plotly_white'
    
    return {
        'template': template,
        'color_discrete_sequence': [colors['primary'], colors['secondary'], colors['accent'], '#ff6b6b', '#4ecdc4', '#45b7d1', '#f39c12', '#e74c3c'],
        'paper_bgcolor': 'rgba(0,0,0,0)',
        'plot_bgcolor': 'rgba(0,0,0,0)',
        'font_color': colors['text']
    }

# Load data
df = load_data()
dau, wau, mau = calculate_active_users(df)
colors = get_theme_colors()
theme = get_plotly_theme()

# Sidebar with modern styling
st.sidebar.markdown(f"""
<div style="background: linear-gradient(135deg, {colors['primary']}, {colors['secondary']}); 
            padding: 1.5rem; border-radius: 15px; margin-bottom: 2rem; text-align: center;">
    <h1 style="color: white; margin: 0; font-size: 1.5rem;">🎮 Gaming Analytics</h1>
    <p style="color: rgba(255,255,255,0.8); margin: 0.5rem 0 0 0;">Real-time insights dashboard</p>
</div>
""", unsafe_allow_html=True)

# Filters
st.sidebar.markdown("### 📊 Filters")

selected_games = st.sidebar.multiselect(
    "Select Games", 
    options=df['Game_Title'].unique(), 
    default=df['Game_Title'].unique()
)

selected_devices = st.sidebar.multiselect(
    "Select Device Types", 
    options=df['Device_Type'].unique(), 
    default=df['Device_Type'].unique()
)

selected_tiers = st.sidebar.multiselect(
    "Select Subscription Tiers", 
    options=df['Subscription_Tier'].unique(), 
    default=df['Subscription_Tier'].unique()
)

date_range = st.sidebar.date_input(
    "Select Date Range",
    value=[df['Signup_Date'].min(), df['Signup_Date'].max()],
    min_value=df['Signup_Date'].min(),
    max_value=df['Signup_Date'].max()
)

# Apply filters
filtered_df = df[
    (df['Game_Title'].isin(selected_games)) &
    (df['Device_Type'].isin(selected_devices)) &
    (df['Subscription_Tier'].isin(selected_tiers)) &
    (df['Signup_Date'] >= pd.to_datetime(date_range[0])) &
    (df['Signup_Date'] <= pd.to_datetime(date_range[1]))
]

# Main Dashboard Header
st.markdown(f"""
<div style="text-align: center; margin-bottom: 3rem;">
    <h1 class="main-header" style="font-size: 3rem; font-weight: 800; margin: 0;">
        🎮 Gaming Analytics Dashboard
    </h1>
    <p style="color: {colors['text_secondary']}; font-size: 1.2rem; margin: 0.5rem 0 0 0;">
        Comprehensive insights for data-driven gaming decisions
    </p>
</div>
""", unsafe_allow_html=True)

# Key Metrics Row with custom cards
col1, col2, col3, col4 = st.columns(4)

metrics = [
    ("📱", "Daily Active Users", f"{dau:,}", f"{(dau/mau*100):.1f}% of MAU"),
    ("📅", "Weekly Active Users", f"{wau:,}", f"{(wau/mau*100):.1f}% of MAU"),
    ("📊", "Monthly Active Users", f"{mau:,}", "Total Active Users"),
    ("💰", "Total Revenue", f"${filtered_df['Total_Revenue_USD'].sum():,.2f}", f"${filtered_df['Total_Revenue_USD'].sum()/len(filtered_df):.2f} ARPU")
]

for i, (icon, title, value, delta) in enumerate(metrics):
    with [col1, col2, col3, col4][i]:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">{icon}</div>
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-delta">{delta}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Revenue Trends Section
st.markdown('<h2 class="section-header">📈 Revenue Trends Over Time</h2>', unsafe_allow_html=True)

col1, col2 = st.columns([4, 1])

with col2:
    revenue_metric = st.selectbox(
        "Revenue Metric",
        ["Total Revenue", "Average Revenue per User", "Revenue per Session"]
    )

# Calculate revenue trends
revenue_trends = filtered_df.groupby(filtered_df['Signup_Date'].dt.to_period('M')).agg({
    'Total_Revenue_USD': ['sum', 'mean'],
    'Total_Play_Sessions': 'sum',
    'User_ID': 'count'
}).round(2)

revenue_trends.columns = ['Total_Revenue', 'Avg_Revenue_Per_User', 'Total_Sessions', 'User_Count']
revenue_trends['Revenue_Per_Session'] = revenue_trends['Total_Revenue'] / revenue_trends['Total_Sessions']
revenue_trends.index = revenue_trends.index.to_timestamp()

with col1:
    fig_revenue = go.Figure()
    
    if revenue_metric == "Total Revenue":
        y_data = revenue_trends['Total_Revenue']
        y_title = "Total Revenue ($)"
    elif revenue_metric == "Average Revenue per User":
        y_data = revenue_trends['Avg_Revenue_Per_User']
        y_title = "ARPU ($)"
    else:
        y_data = revenue_trends['Revenue_Per_Session']
        y_title = "Revenue per Session ($)"
    
    fig_revenue.add_trace(go.Scatter(
        x=revenue_trends.index,
        y=y_data,
        mode='lines+markers',
        line=dict(color=colors['primary'], width=4),
        marker=dict(size=8, color=colors['secondary']),
        fill='tonexty',
        fillcolor=f"rgba{(*[int(colors['primary'][i:i+2], 16) for i in (1, 3, 5)], 0.1)}"
    ))
    
    fig_revenue.update_layout(
        title=f"{revenue_metric} Trends by Signup Cohort",
        xaxis_title="Signup Month",
        yaxis_title=y_title,
        height=400,
        template=theme['template'],
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color=colors['text']
    )
    
    st.plotly_chart(fig_revenue, use_container_width=True)

# Segmentation Analysis
st.markdown('<h2 class="section-header">🎯 User Segmentation Analysis</h2>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["📱 Device Type", "👥 User Segments", "🎮 Game Mode", "🏆 Subscription Tiers"])

with tab1:
    col1, col2 = st.columns(2)
    
    device_stats = filtered_df.groupby('Device_Type').agg({
        'User_ID': 'count',
        'Total_Revenue_USD': 'sum',
        'Total_Hours_Played': 'sum'
    }).reset_index()
    
    with col1:
        fig_device_pie = px.pie(
            device_stats, 
            values='User_ID', 
            names='Device_Type',
            title="User Distribution by Device Type",
            color_discrete_sequence=theme['color_discrete_sequence']
        )
        fig_device_pie.update_layout(
            template=theme['template'],
            paper_bgcolor='rgba(0,0,0,0)',
            font_color=colors['text']
        )
        st.plotly_chart(fig_device_pie, use_container_width=True)
    
    with col2:
        fig_device_revenue = px.bar(
            device_stats,
            x='Device_Type',
            y='Total_Revenue_USD',
            title="Revenue by Device Type",
            color='Device_Type',
            color_discrete_sequence=theme['color_discrete_sequence']
        )
        fig_device_revenue.update_layout(
            template=theme['template'],
            paper_bgcolor='rgba(0,0,0,0)',
            font_color=colors['text'],
            showlegend=False
        )
        st.plotly_chart(fig_device_revenue, use_container_width=True)
    
    # Device metrics table
    device_metrics = filtered_df.groupby('Device_Type').agg({
        'User_ID': 'count',
        'Total_Revenue_USD': ['sum', 'mean'],
        'Total_Hours_Played': 'mean',
        'Total_Play_Sessions': 'mean',
        'Avg_Session_Duration_Min': 'mean'
    }).round(2)
    
    device_metrics.columns = ['Users', 'Total Revenue', 'ARPU', 'Avg Hours', 'Avg Sessions', 'Avg Session Duration']
    st.dataframe(device_metrics, use_container_width=True)

with tab2:
    col1, col2 = st.columns(2)
    
    with col1:
        spending_dist = filtered_df['Spending_Segment'].value_counts()
        fig_spending = px.pie(
            values=spending_dist.values,
            names=spending_dist.index,
            title="Distribution by Spending Segment",
            color_discrete_sequence=theme['color_discrete_sequence']
        )
        fig_spending.update_layout(
            template=theme['template'],
            paper_bgcolor='rgba(0,0,0,0)',
            font_color=colors['text']
        )
        st.plotly_chart(fig_spending, use_container_width=True)
    
    with col2:
        engagement_dist = filtered_df['Engagement_Segment'].value_counts()
        fig_engagement = px.pie(
            values=engagement_dist.values,
            names=engagement_dist.index,
            title="Distribution by Engagement Level",
            color_discrete_sequence=theme['color_discrete_sequence']
        )
        fig_engagement.update_layout(
            template=theme['template'],
            paper_bgcolor='rgba(0,0,0,0)',
            font_color=colors['text']
        )
        st.plotly_chart(fig_engagement, use_container_width=True)
    
    # Segment correlation heatmap
    segment_matrix = pd.crosstab(filtered_df['Spending_Segment'], filtered_df['Engagement_Segment'], normalize='index') * 100
    fig_heatmap = px.imshow(
        segment_matrix.values,
        x=segment_matrix.columns,
        y=segment_matrix.index,
        color_continuous_scale='Viridis' if st.session_state.dark_mode else 'Blues',
        title="Spending vs Engagement Segment Correlation (%)",
        text_auto=".1f"
    )
    fig_heatmap.update_layout(
        template=theme['template'],
        paper_bgcolor='rgba(0,0,0,0)',
        font_color=colors['text']
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)

with tab3:
    col1, col2 = st.columns(2)
    
    mode_stats = filtered_df.groupby('Preferred_Game_Mode').agg({
        'User_ID': 'count',
        'Total_Revenue_USD': 'mean',
        'Total_Hours_Played': 'mean'
    }).reset_index()
    
    with col1:
        fig_mode_users = px.bar(
            mode_stats,
            x='Preferred_Game_Mode',
            y='User_ID',
            title="Users by Game Mode",
            color='Preferred_Game_Mode',
            color_discrete_sequence=theme['color_discrete_sequence']
        )
        fig_mode_users.update_layout(
            template=theme['template'],
            paper_bgcolor='rgba(0,0,0,0)',
            font_color=colors['text'],
            showlegend=False
        )
        st.plotly_chart(fig_mode_users, use_container_width=True)
    
    with col2:
        fig_mode_hours = px.bar(
            mode_stats,
            x='Preferred_Game_Mode',
            y='Total_Hours_Played',
            title="Average Hours Played by Game Mode",
            color='Preferred_Game_Mode',
            color_discrete_sequence=theme['color_discrete_sequence']
        )
        fig_mode_hours.update_layout(
            template=theme['template'],
            paper_bgcolor='rgba(0,0,0,0)',
            font_color=colors['text'],
            showlegend=False
        )
        st.plotly_chart(fig_mode_hours, use_container_width=True)

with tab4:
    tier_stats = filtered_df.groupby('Subscription_Tier').agg({
        'User_ID': 'count',
        'Total_Revenue_USD': ['sum', 'mean'],
        'Total_Hours_Played': 'mean',
        'Achievement_Score': 'mean'
    }).round(2)
    
    tier_stats.columns = ['Users', 'Total Revenue', 'ARPU', 'Avg Hours', 'Avg Achievement Score']
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_tier_revenue = px.bar(
            x=tier_stats.index,
            y=tier_stats['Total Revenue'],
            title="Total Revenue by Subscription Tier",
            color=tier_stats.index,
            color_discrete_sequence=theme['color_discrete_sequence']
        )
        fig_tier_revenue.update_layout(
            template=theme['template'],
            paper_bgcolor='rgba(0,0,0,0)',
            font_color=colors['text']
        )
        st.plotly_chart(fig_tier_revenue, use_container_width=True)
    
    with col2:
        fig_tier_arpu = px.bar(
            x=tier_stats.index,
            y=tier_stats['ARPU'],
            title="ARPU by Subscription Tier",
            color=tier_stats.index,
            color_discrete_sequence=theme['color_discrete_sequence']
        )
        fig_tier_arpu.update_layout(
            template=theme['template'],
            paper_bgcolor='rgba(0,0,0,0)',
            font_color=colors['text']
        )
        st.plotly_chart(fig_tier_arpu, use_container_width=True)
    
    st.dataframe(tier_stats, use_container_width=True)

# Advanced Analytics Section
st.markdown('<h2 class="section-header">🔬 Advanced Analytics</h2>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📅 Monthly Signup Cohorts")
    
    cohort_data = filtered_df.groupby(filtered_df['Signup_Date'].dt.to_period('M')).agg({
        'User_ID': 'count',
        'Total_Revenue_USD': 'sum',
        'Total_Hours_Played': 'sum'
    })
    cohort_data.index = cohort_data.index.to_timestamp()
    
    fig_cohort = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig_cohort.add_trace(
        go.Bar(x=cohort_data.index, y=cohort_data['User_ID'], name="New Users", 
               marker_color=colors['primary']),
        secondary_y=False,
    )
    
    fig_cohort.add_trace(
        go.Scatter(x=cohort_data.index, y=cohort_data['Total_Revenue_USD'], 
                  mode='lines+markers', name="Revenue", 
                  line=dict(color=colors['secondary'], width=3),
                  marker=dict(size=8)),
        secondary_y=True,
    )
    
    fig_cohort.update_xaxes(title_text="Signup Month")
    fig_cohort.update_yaxes(title_text="New Users", secondary_y=False)
    fig_cohort.update_yaxes(title_text="Revenue ($)", secondary_y=True)
    fig_cohort.update_layout(
        title_text="User Acquisition vs Revenue by Cohort",
        template=theme['template'],
        paper_bgcolor='rgba(0,0,0,0)',
        font_color=colors['text']
    )
    
    st.plotly_chart(fig_cohort, use_container_width=True)

with col2:
    st.markdown("### 🏆 Top Performers")
    
    metric_choice = st.selectbox("Select Metric", ["Revenue", "Hours Played", "Sessions"])
    
    if metric_choice == "Revenue":
        top_users = filtered_df.nlargest(10, 'Total_Revenue_USD')[['Username', 'Total_Revenue_USD', 'Game_Title', 'Subscription_Tier']]
        top_users.columns = ['Username', 'Revenue ($)', 'Game', 'Tier']
    elif metric_choice == "Hours Played":
        top_users = filtered_df.nlargest(10, 'Total_Hours_Played')[['Username', 'Total_Hours_Played', 'Game_Title', 'Subscription_Tier']]
        top_users.columns = ['Username', 'Hours', 'Game', 'Tier']
    else:
        top_users = filtered_df.nlargest(10, 'Total_Play_Sessions')[['Username', 'Total_Play_Sessions', 'Game_Title', 'Subscription_Tier']]
        top_users.columns = ['Username', 'Sessions', 'Game', 'Tier']
    
    st.dataframe(top_users, use_container_width=True, hide_index=True)

# Enhanced Advanced Analytics Section
st.markdown('<h2 class="section-header">🧬 Enhanced Analytics & Intelligence</h2>', unsafe_allow_html=True)

# Create tabs for the new advanced features
advanced_tab1, advanced_tab2, advanced_tab3 = st.tabs(["📊 Cohort Analysis", "🔄 Funnel Tracking", "🎯 User Clustering"])

with advanced_tab1:
    st.markdown("### 📈 User Retention Cohort Analysis")
    
    try:
        cohort_table, cohort_table_pct, cohort_sizes = perform_cohort_analysis(filtered_df)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Cohort retention heatmap
            if not cohort_table_pct.empty and cohort_table_pct.shape[1] > 1:
                # Limit to first 12 months for better visualization
                cohort_display = cohort_table_pct.iloc[:, :min(12, cohort_table_pct.shape[1])]
                
                fig_cohort_heatmap = px.imshow(
                    cohort_display.values,
                    x=[f"Month {i}" for i in range(cohort_display.shape[1])],
                    y=[str(idx) for idx in cohort_display.index],
                    color_continuous_scale='Viridis' if st.session_state.dark_mode else 'Blues',
                    title="Cohort Retention Rates (%)",
                    text_auto=".1%",
                    aspect="auto",
                    zmin=0,  # Ensure color scale starts at 0
                    zmax=1   # Ensure color scale ends at 1 (100%)
                )
                fig_cohort_heatmap.update_layout(
                    template=theme['template'],
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color=colors['text'],
                    xaxis_title="Period (Months after signup)",
                    yaxis_title="Signup Cohort"
                )
                st.plotly_chart(fig_cohort_heatmap, use_container_width=True)
            else:
                st.info("Not enough data for cohort analysis. Need users with different signup periods and activity.")
        
        with col2:
            # Cohort size chart
            if not cohort_sizes.empty:
                fig_cohort_sizes = px.bar(
                    cohort_sizes,
                    x='Signup_Period',
                    y='Total_Users',
                    title="Cohort Sizes by Signup Period",
                    color='Total_Users',
                    color_continuous_scale='Viridis' if st.session_state.dark_mode else 'Blues'
                )
                fig_cohort_sizes.update_layout(
                    template=theme['template'],
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color=colors['text'],
                    xaxis_title="Signup Period",
                    yaxis_title="Number of Users"
                )
                st.plotly_chart(fig_cohort_sizes, use_container_width=True)
            
            # Display cohort summary stats
            if not cohort_table_pct.empty and cohort_table_pct.shape[1] > 0:
                st.markdown("#### Retention Summary")
                avg_retention_values = []
                for i in range(min(6, cohort_table_pct.shape[1])):
                    # Calculate mean for valid, non-NaN retention rates
                    valid_rates = cohort_table_pct.iloc[:, i].dropna()
                    avg_rate = valid_rates.mean() if not valid_rates.empty else 0
                    avg_retention_values.append(f"{avg_rate:.1%}")
                
                retention_summary = pd.DataFrame({
                    'Period': [f"Month {i}" for i in range(min(6, cohort_table_pct.shape[1]))],
                    'Avg Retention': avg_retention_values
                })
                st.dataframe(retention_summary, use_container_width=True, hide_index=True)
    
    except Exception as e:
        st.error(f"Error in cohort analysis: {str(e)}")
        st.info("Cohort analysis requires users with varied signup dates and activity periods. Try adjusting filters.")

with advanced_tab2:
    st.markdown("### 🚀 User Journey Funnel Analysis")
    
    funnel_data = calculate_funnel_metrics(filtered_df)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Create funnel chart
        stages = list(funnel_data.keys())
        values = list(funnel_data.values())
        
        # Calculate conversion rates
        conversion_rates = []
        for i in range(1, len(values)):
            rate = (values[i] / values[i-1]) * 100 if values[i-1] > 0 else 0
            conversion_rates.append(f"{rate:.1f}%")
        
        # Create funnel visualization
        fig_funnel = go.Figure()
        
        for i, (stage, value) in enumerate(zip(stages, values)):
            fig_funnel.add_trace(go.Funnel(
                name=stage,  # Use stage name for legend
                y=[stage],
                x=[value],
                textinfo="value+percent initial",
                marker_color=theme['color_discrete_sequence'][i % len(theme['color_discrete_sequence'])],
                connector={"line": {"color": colors['primary'], "dash": "dot", "width": 3}},
            ))
        
        fig_funnel.update_layout(
            title="User Journey Funnel",
            template=theme['template'],
            paper_bgcolor='rgba(0,0,0,0)',
            font_color=colors['text'],
            height=500
        )
        st.plotly_chart(fig_funnel, use_container_width=True)
    
    with col2:
        st.markdown("#### Funnel Metrics")
        
        # Create funnel metrics table
        funnel_df = pd.DataFrame({
            'Stage': stages,
            'Users': values,
            'Conversion': ['100%'] + conversion_rates
        })
        
        st.dataframe(funnel_df, use_container_width=True, hide_index=True)
        
        st.markdown("#### Key Insights")
        
        # Calculate key insights
        signup_to_play = (funnel_data['First Game'] / funnel_data['Signups']) * 100
        play_to_repeat = (funnel_data['Repeat Sessions'] / funnel_data['First Game']) * 100 if funnel_data['First Game'] > 0 else 0
        repeat_to_pay = (funnel_data['Made Purchase'] / funnel_data['Repeat Sessions']) * 100 if funnel_data['Repeat Sessions'] > 0 else 0
        pay_to_high_value = (funnel_data['High Value Users'] / funnel_data['Made Purchase']) * 100 if funnel_data['Made Purchase'] > 0 else 0
        
        insights = [
            f"🎮 **{signup_to_play:.1f}%** of signups play first game",
            f"🔄 **{play_to_repeat:.1f}%** return for repeat sessions (3+ sessions)",
            f"💰 **{repeat_to_pay:.1f}%** of repeat players make purchases",
            f"💎 **{pay_to_high_value:.1f}%** of paying users become high value ($50+)"
        ]
        
        for insight in insights:
            st.markdown(insight)

with advanced_tab3:
    st.markdown("### 🎯 Machine Learning User Segmentation")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        n_clusters = st.slider("Number of Clusters", min_value=2, max_value=8, value=4)
        cluster_features = st.multiselect(
            "Select Features for Clustering",
            ["Total_Play_Sessions", "Total_Revenue_USD", "Total_Hours_Played", "Achievement_Score"],
            default=["Total_Play_Sessions", "Total_Revenue_USD"]
        )
    
    if len(cluster_features) >= 2:
        try:
            # Create a custom clustering function for selected features
            def perform_custom_clustering(df, features, n_clusters):
                # Clean data for clustering
                cluster_df = df[features].fillna(0)
                
                # Remove any infinite values
                cluster_df = cluster_df.replace([np.inf, -np.inf], 0)
                
                # Ensure we have enough data points
                if len(cluster_df) < n_clusters:
                    raise ValueError(f"Need at least {n_clusters} data points for {n_clusters} clusters")
                
                # Standardize features
                scaler = StandardScaler()
                scaled_features = scaler.fit_transform(cluster_df)
                
                # Perform K-means clustering
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                clusters = kmeans.fit_predict(scaled_features)
                
                # Add cluster labels to dataframe
                df_clustered = df.copy()
                df_clustered['Cluster'] = clusters
                
                # Calculate cluster statistics for selected features
                agg_dict = {feature: 'mean' for feature in features}
                agg_dict['User_ID'] = 'count'
                
                cluster_stats = df_clustered.groupby('Cluster').agg(agg_dict).round(2)
                cluster_stats.rename(columns={'User_ID': 'User_Count'}, inplace=True)
                
                return df_clustered, cluster_stats
            
            # Perform clustering with selected features
            df_clustered, cluster_stats = perform_custom_clustering(filtered_df, cluster_features, n_clusters)
            
            with col2:
                # Create cluster visualization
                if len(cluster_features) >= 2:
                    fig_clusters = px.scatter(
                        df_clustered,
                        x=cluster_features[0],
                        y=cluster_features[1],
                        color='Cluster',
                        size='Total_Hours_Played' if 'Total_Hours_Played' in df_clustered.columns else None,
                        hover_data=['Username', 'Game_Title', 'Subscription_Tier'],
                        title=f"User Clusters: {cluster_features[0]} vs {cluster_features[1]}",
                        color_discrete_sequence=theme['color_discrete_sequence']
                    )
                    fig_clusters.update_layout(
                        template=theme['template'],
                        paper_bgcolor='rgba(0,0,0,0)',
                        font_color=colors['text']
                    )
                    st.plotly_chart(fig_clusters, use_container_width=True)
            
            # Cluster characteristics
            st.markdown("#### 📊 Cluster Characteristics")
            
            # Add cluster labels for better interpretation
            cluster_labels = {
                0: "🎯 High Value Players",
                1: "👥 Casual Players", 
                2: "🏆 Engaged Players",
                3: "💰 Spenders"
            }
            
            cluster_stats_display = cluster_stats.copy()
            cluster_stats_display.index = [cluster_labels.get(i, f"Cluster {i}") for i in cluster_stats_display.index]
            
            st.dataframe(cluster_stats_display, use_container_width=True)
            
            # Cluster distribution
            col1, col2 = st.columns(2)
            
            with col1:
                cluster_dist = df_clustered['Cluster'].value_counts().sort_index()
                fig_cluster_dist = px.pie(
                    values=cluster_dist.values,
                    names=[cluster_labels.get(i, f"Cluster {i}") for i in cluster_dist.index],
                    title="User Distribution by Cluster",
                    color_discrete_sequence=theme['color_discrete_sequence']
                )
                fig_cluster_dist.update_layout(
                    template=theme['template'],
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color=colors['text']
                )
                st.plotly_chart(fig_cluster_dist, use_container_width=True)
            
            with col2:
                # Cluster revenue comparison
                cluster_revenue = df_clustered.groupby('Cluster')['Total_Revenue_USD'].sum().sort_index()
                fig_cluster_revenue = px.bar(
                    x=[cluster_labels.get(i, f"Cluster {i}") for i in cluster_revenue.index],
                    y=cluster_revenue.values,
                    title="Total Revenue by Cluster",
                    color=cluster_revenue.values,
                    color_continuous_scale='Viridis' if st.session_state.dark_mode else 'Blues'
                )
                fig_cluster_revenue.update_layout(
                    template=theme['template'],
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color=colors['text'],
                    showlegend=False,
                    xaxis_title="Cluster",
                    yaxis_title="Total Revenue ($)"
                )
                st.plotly_chart(fig_cluster_revenue, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error in clustering analysis: {str(e)}")
            st.info("Please ensure you have sufficient data for clustering analysis.")
    else:
        st.info("Please select at least 2 features for clustering analysis.")

# Game-specific analysis
st.markdown('<h2 class="section-header">🎮 Game-Specific Performance</h2>', unsafe_allow_html=True)

game_analysis = filtered_df.groupby('Game_Title').agg({
    'User_ID': 'count',
    'Total_Revenue_USD': ['sum', 'mean'],
    'Total_Hours_Played': 'mean',
    'Total_Play_Sessions': 'mean',
    'Achievement_Score': 'mean'
}).round(2)

game_analysis.columns = ['Players', 'Total Revenue', 'ARPU', 'Avg Hours', 'Avg Sessions', 'Avg Achievement']

col1, col2 = st.columns([3, 1])

with col1:
    fig_games = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Players by Game', 'Total Revenue by Game', 
                       'ARPU by Game', 'Average Hours by Game'),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "bar"}]]
    )
    
    fig_games.add_trace(go.Bar(x=game_analysis.index, y=game_analysis['Players'], 
                              marker_color=colors['primary'], name="Players"), row=1, col=1)
    fig_games.add_trace(go.Bar(x=game_analysis.index, y=game_analysis['Total Revenue'], 
                              marker_color=colors['secondary'], name="Revenue"), row=1, col=2)
    fig_games.add_trace(go.Bar(x=game_analysis.index, y=game_analysis['ARPU'], 
                              marker_color=colors['accent'], name="ARPU"), row=2, col=1)
    fig_games.add_trace(go.Bar(x=game_analysis.index, y=game_analysis['Avg Hours'], 
                              marker_color='#ff6b6b', name="Hours"), row=2, col=2)
    
    fig_games.update_layout(
        height=600, 
        showlegend=False, 
        title_text="Game Performance Dashboard",
        template=theme['template'],
        paper_bgcolor='rgba(0,0,0,0)',
        font_color=colors['text']
    )
    st.plotly_chart(fig_games, use_container_width=True)

with col2:
    st.markdown("### Game Stats")
    st.dataframe(game_analysis, use_container_width=True)

# Footer
st.markdown(f"""
<div class="footer">
    <h3 style="color: {colors['text']}; margin-bottom: 1rem;">🎮 Gaming Analytics Dashboard</h3>
    <p style="color: {colors['text_secondary']}; margin: 0;">
        Built with ❤️ using Streamlit & Plotly • Real-time insights for data-driven decisions
    </p>
    <div style="margin-top: 1rem;">
        <span style="background: linear-gradient(45deg, {colors['primary']}, {colors['secondary']}); 
                     padding: 0.5rem 1rem; border-radius: 25px; color: white; font-weight: 600;">
            Theme: {'🌙 Dark Mode' if st.session_state.dark_mode else '☀️ Light Mode'}
        </span>
    </div>
</div>
""", unsafe_allow_html=True) 