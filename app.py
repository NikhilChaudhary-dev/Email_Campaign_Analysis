import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from prophet import Prophet
import warnings
import os
warnings.filterwarnings('ignore')

# Page config for wide layout and custom title
st.set_page_config(
    page_title="📧 Email Campaign Analytics Pro",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Backup solution: Add sidebar toggle in main content if needed
col_toggle, col_spacer = st.columns([1, 10])
with col_toggle:
    if st.button("☰ Menu", help="Toggle sidebar if not visible"):
        st.rerun()

# Define dark and light CSS separately
dark_css = """
/* Dark Theme Styling */
.stApp {
    background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
    color: #ffffff;
    font-family: 'Inter', sans-serif;
}
.section-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
}
.metric-container {
    background: linear-gradient(135deg, #2a2a5a 0%, #1e1e3f 100%);
    border: 1px solid rgba(102, 126, 234, 0.3);
}
.plotly-graph-div {
    background: linear-gradient(135deg, #2a2a5a 0%, #1e1e3f 100%);
    border: 1px solid rgba(102, 126, 234, 0.3);
}
.custom-footer {
    background: linear-gradient(135deg, #1e1e3f 0%, #2a2a5a 100%);
    border: 1px solid rgba(102, 126, 234, 0.3);
}
"""

light_css = """
/* Light Theme Styling */
.stApp {
    background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
    color: #2d3748;
    font-family: 'Inter', sans-serif;
}
.section-header {
    background: linear-gradient(135deg, #4299e1 0%, #3182ce 100%);
    color: white;
}
.metric-container {
    background: linear-gradient(135deg, #ffffff 0%, #f7fafc 100%);
    border: 1px solid rgba(66, 153, 225, 0.3);
    box-shadow: 0 15px 35px rgba(0,0,0,0.1);
}
.plotly-graph-div {
    background: linear-gradient(135deg, #ffffff 0%, #f7fafc 100%);
    border: 1px solid rgba(66, 153, 225, 0.3);
}
.custom-footer {
    background: linear-gradient(135deg, #f7fafc 0%, #ffffff 100%);
    border: 1px solid rgba(66, 153, 225, 0.3);
}
.metric-value {
    color: #3182ce !important;
}
"""

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

body.dark {{
    {dark_css}
}}

body.light {{
    {light_css}
}}

/* Remove Streamlit branding but keep essential controls */
#MainMenu {{visibility: hidden;}}
footer {{visibility: hidden;}}

/* CRITICAL: Keep sidebar toggle button visible with multiple selectors */
button[kind="headerNoPadding"],
button[data-testid="collapsedControl"], 
.css-1rs6os,
.css-14xtw13.e8zbici0,
[data-testid="stSidebarNav"] button,
[data-baseweb="button"][kind="headerNoPadding"] {{
    visibility: visible !important;
    display: flex !important;
    opacity: 1 !important;
    position: relative !important;
    z-index: 999999 !important;
}}

/* Force show any hidden sidebar controls */
div[data-testid="stSidebar"] button {{
    visibility: visible !important;
    display: block !important;
}}

/* Make sure sidebar nav is always visible */
.css-1d391kg .css-1rs6os {{
    visibility: visible !important;
    display: block !important;
}}

/* Override any hiding of sidebar toggle */
.stApp > header button {{
    visibility: visible !important;
    display: flex !important;
}}

/* Custom Header */
.main-header {{
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2rem 2rem;
    border-radius: 20px;
    margin: 1rem 0 2rem 0;
    text-align: center;
    box-shadow: 0 20px 60px rgba(102, 126, 234, 0.3);
    border: 1px solid rgba(255,255,255,0.1);
    position: relative;
    overflow: hidden;
}}

.main-header::before {{
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
    animation: shimmer 3s infinite;
}}

@keyframes shimmer {{
    0% {{ left: -100%; }}
    100% {{ left: 100%; }}
}}

.main-header h1 {{
    font-size: 3.5rem;
    font-weight: 800;
    margin: 0;
    background: linear-gradient(45deg, #ffffff, #e3f2fd);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-shadow: 0 0 30px rgba(255,255,255,0.3);
}}

.main-header p {{
    font-size: 1.3rem;
    margin: 0.5rem 0 0 0;
    color: rgba(255,255,255,0.9);
    font-weight: 400;
}}

/* Sidebar Styling */
.css-1d391kg, .css-1cypcdb {{
    background: linear-gradient(180deg, #1e1e3f 0%, #2a2a5a 100%);
    border-right: 2px solid #667eea;
}}

.css-1d391kg .stSelectbox label,
.css-1d391kg .stMultiSelect label,
.css-1d391kg .stFileUploader label {{
    color: #ffffff;
    font-weight: 600;
    font-size: 1.1rem;
}}

.stSidebar .stFileUploader {{
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1.5rem;
    border-radius: 15px;
    border: 2px solid rgba(255,255,255,0.1);
    box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    margin-bottom: 1rem;
}}

.stSidebar .stFileUploader div[role="button"] {{
    background: linear-gradient(45deg, #4facfe 0%, #00f2fe 100%);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.8rem 1.5rem;
    font-weight: 600;
    font-size: 1rem;
    transition: all 0.3s ease;
    box-shadow: 0 5px 15px rgba(79, 172, 254, 0.4);
}}

.stSidebar .stFileUploader div[role="button"]:hover {{
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(79, 172, 254, 0.6);
}}

/* Metric Cards */
.metric-container {{
    background: linear-gradient(135deg, #2a2a5a 0%, #1e1e3f 100%);
    border-radius: 20px;
    padding: 1.5rem;
    border: 1px solid rgba(102, 126, 234, 0.3);
    box-shadow: 0 15px 35px rgba(0,0,0,0.2);
    text-align: center;
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}}

.metric-container::before {{
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 2px;
    background: linear-gradient(90deg, #667eea, #764ba2);
}}

.metric-container:hover {{
    transform: translateY(-10px);
    box-shadow: 0 25px 50px rgba(102, 126, 234, 0.3);
    border-color: #667eea;
}}

.metric-value {{
    font-size: 2rem;
    font-weight: 800;
    color: #4facfe;
    margin: 0;
    text-shadow: 0 0 20px rgba(79, 172, 254, 0.5);
    word-break: break-word;
    overflow: hidden;
    text-overflow: ellipsis;
}}

.metric-label {{
    font-size: 1rem;
    color: rgba(255,255,255,0.8);
    margin-top: 0.5rem;
    font-weight: 500;
}}

/* Buttons */
.stButton > button {{
    background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.8rem 2rem;
    font-weight: 600;
    font-size: 1rem;
    transition: all 0.3s ease;
    box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3);
    min-height: 3rem;
}}

.stButton > button:hover {{
    transform: translateY(-3px);
    box-shadow: 0 15px 35px rgba(102, 126, 234, 0.4);
    background: linear-gradient(45deg, #764ba2 0%, #667eea 100%);
}}

/* Charts */
.plotly-graph-div {{
    background: linear-gradient(135deg, #2a2a5a 0%, #1e1e3f 100%);
    border-radius: 20px;
    padding: 1rem;
    border: 1px solid rgba(102, 126, 234, 0.3);
    box-shadow: 0 15px 35px rgba(0,0,0,0.2);
    margin-bottom: 1rem;
}}

/* Section Headers */
.section-header {{
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 1.5rem 2rem;
    border-radius: 15px;
    margin: 2rem 0 1rem 0;
    font-size: 1.8rem;
    font-weight: 700;
    text-align: center;
    box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    border: 1px solid rgba(255,255,255,0.1);
}}

/* Insight Buttons */
.insight-btn {{
    background: linear-gradient(45deg, #ff6b6b 0%, #4ecdc4 100%);
    color: white;
    border: none;
    border-radius: 50%;
    width: 60px;
    height: 60px;
    font-size: 1.5rem;
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 10px 25px rgba(255, 107, 107, 0.3);
    transition: all 0.3s ease;
    margin: auto;
}}

.insight-btn:hover {{
    transform: scale(1.1) rotate(15deg);
    box-shadow: 0 15px 35px rgba(255, 107, 107, 0.5);
}}

/* Data Tables */
.dataframe {{
    background: #2a2a5a;
    border-radius: 15px;
    border: 1px solid rgba(102, 126, 234, 0.3);
    overflow: hidden;
}}

/* Footer */
.custom-footer {{
    background: linear-gradient(135deg, #1e1e3f 0%, #2a2a5a 100%);
    padding: 2rem;
    border-radius: 20px;
    text-align: center;
    margin-top: 3rem;
    border: 1px solid rgba(102, 126, 234, 0.3);
    box-shadow: 0 15px 35px rgba(0,0,0,0.2);
}}

.custom-footer p {{
    margin: 0.5rem 0;
    color: rgba(255,255,255,0.8);
}}

/* Theme Toggle */
.theme-toggle {{
    position: fixed;
    top: 20px;
    right: 20px;
    z-index: 1000;
    background: linear-gradient(45deg, #667eea, #764ba2);
    border: 2px solid rgba(255,255,255,0.2);
    border-radius: 50px;
    padding: 12px 24px;
    color: white;
    font-weight: 600;
    box-shadow: 0 10px 25px rgba(0,0,0,0.3);
    transition: all 0.3s ease;
}}

.theme-toggle:hover {{
    transform: translateY(-3px);
    box-shadow: 0 15px 35px rgba(102, 126, 234, 0.4);
}}

/* Progress Bars */
.stProgress > div > div > div > div {{
    background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
    border-radius: 10px;
}}

/* Selectbox and Multiselect */
.stSelectbox > div > div {{
    background: #2a2a5a;
    border: 1px solid #667eea;
    border-radius: 10px;
}}

.stMultiSelect > div > div {{
    background: #2a2a5a;
    border: 1px solid #667eea;
    border-radius: 10px;
}}

/* Animation Classes */
.fade-in {{
    animation: fadeIn 1s ease-in;
}}

@keyframes fadeIn {{
    0% {{ opacity: 0; transform: translateY(20px); }}
    100% {{ opacity: 1; transform: translateY(0); }}
}}

.slide-up {{
    animation: slideUp 0.8s ease-out;
}}

@keyframes slideUp {{
    0% {{ transform: translateY(50px); opacity: 0; }}
    100% {{ transform: translateY(0); opacity: 1; }}
}}

/* Warning and Info boxes */
.stAlert > div {{
    background: linear-gradient(135deg, #2a2a5a 0%, #1e1e3f 100%);
    border: 1px solid #667eea;
    border-radius: 15px;
    color: white;
}}
</style>
""", unsafe_allow_html=True)

# Client-side theme toggle button
st.sidebar.markdown("""
<button onclick="toggleTheme()">🌓 Toggle Theme</button>
<script>
function toggleTheme() {{
    const body = document.body;
    if (body.classList.contains('dark')) {{
        body.classList.remove('dark');
        body.classList.add('light');
        localStorage.setItem('theme', 'light');
    }} else {{
        body.classList.remove('light');
        body.classList.add('dark');
        localStorage.setItem('theme', 'dark');
    }}
}}
const savedTheme = localStorage.getItem('theme') || 'dark';
document.body.classList.add(savedTheme);
</script>
""", unsafe_allow_html=True)

# Enhanced Header
st.markdown("""
<div class="main-header fade-in">
    <h1>🚀 Welcome to CamML Analytics</h1>
    <p>✨ Unleash the Power of Your Campaigns with AI-Driven Insights ✨</p>
</div>
""", unsafe_allow_html=True)

# Function to load data (CSV or Excel)
@st.cache_data(max_entries=1)
def load_data(file, file_type):
    try:
        if file_type == "csv":
            chunk_size = 100000
            chunks = pd.read_csv(file, chunksize=chunk_size, low_memory=False)
            df = pd.concat(chunks, ignore_index=True)
        else:
            df = pd.read_excel(file, engine='openpyxl')
        
        df = df.dropna(how='all')  # Remove completely empty rows
        
        # Check if 'Status' column exists; if not, skip bounce/failure filtering
        if 'Status' in df.columns:
            df = df[~df['Status'].str.contains('bounced|failed', case=False, na=False)]  # Remove bounced/failed emails
        
        # Data cleaning
        df['Sent_Date'] = pd.to_datetime(df['Sent_Date'], errors='coerce')
        df['Opened Time'] = pd.to_datetime(df['Opened Time'], errors='coerce')
        df['Sent_Year'] = df['Sent_Date'].dt.year
        df['Sent_Month'] = df['Sent_Date'].dt.month
        df['Quarter'] = df['Sent_Date'].dt.quarter
        df['Sent_DayOfWeek'] = df['Sent_Date'].dt.dayofweek
        df['Response_Time'] = (df['Opened Time'] - df['Sent_Date']).dt.total_seconds().fillna(0)
        df['Is_Unsubscribed'] = df['Is Unsubscribed'].astype(bool) if 'Is Unsubscribed' in df.columns else False
        
        # Clean Reply Message and Positive Reply columns
        if 'Reply Message' in df.columns:
            df['Reply Message'] = df['Reply Message'].fillna('').astype(str)
            df['Has_Reply'] = df['Reply Message'].str.strip().ne('')  # Boolean for non-empty replies
        else:
            df['Reply Message'] = ''
            df['Has_Reply'] = False
        
        if 'Positive Reply(Yes/No)' in df.columns:
            df['Positive_Reply'] = df['Positive Reply(Yes/No)'].str.lower().eq('yes')
        else:
            df['Positive_Reply'] = False
        
        # Debug Engagement column
        st.write("Engagement unique values in raw data:", df['Engagement'].unique())
        st.write("Engagement value counts in raw data:", df['Engagement'].value_counts(dropna=False))
        
        return df
    except Exception as e:
        st.error(f"❌ Error loading file: {e}")
        return None

# Enhanced insights function with more professional language
def generate_insights(df, section_name):
    try:
        insights = {
            "Top Opens by Time Range": f"⏰ **Timing Intelligence**: Peak engagement occurs during optimal send windows. Schedule campaigns during high-performing time slots to increase open rates by up to 40%.",
            "Top Opens by City": f"📍 **Geographic Strategy**: Location-based targeting shows clear performance patterns. Top-performing cities should receive increased campaign frequency and localized content.",
            "Top Opens by Campaign": f"🎪 **Campaign Optimization**: Top-performing campaigns demonstrate winning formulas. Analyze subject lines, content structure, and timing of high-performers for replication.",
            "Top Opens by ESP": f"📧 **Platform Intelligence**: ESP performance varies significantly. Focus resources on platforms with highest engagement rates and consider migration strategies.",
            "Top Opens by State": f"🗺️ **Regional Insights**: State-level performance indicates regional preferences. Tailor messaging and offers based on geographic performance patterns.",
            "Top Clicks by Campaign": f"🖱️ **Conversion Intelligence**: Click-through patterns reveal content effectiveness. High-clicking campaigns should be analyzed for CTA placement, design, and copy.",
            "Unsubscribes by Campaign": f"🚪 **Retention Analysis**: Unsubscribe patterns indicate content-audience mismatch. Review high-unsubscribe campaigns for frequency, relevance, and targeting issues.",
            "Reply Rate": f"💬 **Engagement Quality**: Direct replies indicate genuine interest and engagement. High reply rates correlate with better conversion and customer lifetime value.",
            "Reply vs Positive Reply": f"✅ **Sentiment Analysis**: Positive replies are leading indicators of conversion. Focus on campaigns generating positive sentiment for scaling opportunities.",
            "Traffic Sources": f"🌐 **Channel Performance**: Multi-channel attribution reveals strongest traffic sources. Optimize budget allocation based on channel performance data.",
            "Geographic Clustering (KMeans)": "🗺️ **AI Geographic Intelligence**: Machine learning identifies natural geographic clusters of high-performing regions. Target similar demographics in adjacent areas for expansion.",
            "Lead Behavior Segmentation (KMeans)": "🧠 **AI Behavioral Intelligence**: ML segmentation reveals 4 distinct user personas. Tailor campaigns for each segment to increase relevance and performance.",
            "Open Probability (Random Forest)": "🎯 **Predictive Intelligence**: AI predicts email open likelihood with high accuracy. Focus on high-probability leads for better resource allocation and ROI.",
            "Predicted Opens (Prophet)": "📈 **Forecasting Intelligence**: Time series forecasting predicts future performance trends. Plan capacity and content strategy based on predicted demand.",
            "Enhanced Bot Probability (Random Forest)": f"🛡️ **Quality Assurance**: Advanced ML bot detection improves data quality. Clean datasets lead to better strategic decisions and accurate performance metrics.",
            "Top Companies by HE": "🏢 **High Engagement Focus**: Top companies with high engagement are prime targets for follow-ups. Analyze their interaction patterns to replicate success.",
            "Top Cities by Opens and Clicks": "📍 **City Engagement Insights**: Top cities by opens and clicks indicate high-potential markets. Prioritize these locations for targeted campaigns."
        }
        return insights.get(section_name, "📊 **Analytics Insight**: This visualization reveals important patterns in your email performance. Use these insights to optimize your campaign strategy.")
    except Exception as e:
        return f"⚠️ **Analysis Error**: {str(e)}. Please verify your data structure and try again."

# Enhanced file uploader for main data
st.sidebar.markdown("""
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1rem; border-radius: 15px; margin-bottom: 1rem; text-align: center;">
    <h3 style="color: white; margin: 0;">📁 Main Data Upload</h3>
    <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0; font-size: 0.9rem;">Upload your campaign data to begin analysis</p>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.sidebar.file_uploader("", type=["csv", "xlsx", "xls"], accept_multiple_files=False, key="main_file")
if uploaded_file is not None:
    file_size = uploaded_file.size
    max_size = 500 * 1024 * 1024
    if file_size > max_size:
        st.error(f"❌ File size ({file_size / (1024 * 1024):.2f} MB) exceeds 500MB limit.")
        st.stop()
    
    with st.spinner("🔄 Processing your main data..."):
        file_type = uploaded_file.name.split('.')[-1].lower()
        df = load_data(uploaded_file, file_type)
        if df is None:
            st.stop()
        st.session_state.df = df  # Persist df in session state
    
    st.sidebar.success(f"✅ Loaded {len(st.session_state.df):,} records successfully!")
    
    # Debug: Show column names in an expandable section
    with st.sidebar.expander("📋 Available Columns (Main)"):
        st.write(st.session_state.df.columns.tolist())
        st.info("If 'Status' is missing, the script skips bounce/failure filtering. Rename your status column to 'Status' if needed.")
elif 'df' in st.session_state:
    df = st.session_state.df
else:
    st.markdown("""
    <div style="text-align: center; padding: 3rem; background: linear-gradient(135deg, #2a2a5a 0%, #1e1e3f 100%); border-radius: 20px; border: 2px dashed #667eea; margin: 2rem 0;">
        <h2 style="color: #4facfe; margin-bottom: 1rem;">📊 Ready to Analyze Your Campaigns?</h2>
        <p style="color: rgba(255,255,255,0.8); font-size: 1.2rem;">Upload your email campaign data to unlock powerful insights and AI-driven recommendations.</p>
        <div style="margin-top: 2rem;">
            <span style="color: #667eea; font-size: 1.1rem;">📁 Supported formats: CSV, XLSX, XLS • Max size: 500MB</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

df = st.session_state.df  # Use persisted df

# Note indicating reply data is now in main data-set
st.sidebar.markdown("""
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1rem; border-radius: 15px; margin-bottom: 1rem; text-align: center;">
    <h3 style="color: white; margin: 0;">📋 Reply Data</h3>
    <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0; font-size: 0.9rem;">Reply data is now included in the main dataset under 'Reply Message' and 'Positive Reply(Yes/No)' columns.</p>
</div>
""", unsafe_allow_html=True)

# Enhanced sidebar navigation
st.sidebar.markdown("""
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1rem; border-radius: 15px; margin: 1rem 0; text-align: center;">
    <h3 style="color: white; margin: 0;">🧭 Navigation</h3>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.selectbox("", ["🏠 Dashboard Home", "📊 Compare Quarters", "🤖 AI Predictions", "👑 Boss Dashboard"], label_visibility="collapsed")

# Enhanced filters section
st.sidebar.markdown("""
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1rem; border-radius: 15px; margin: 1rem 0; text-align: center;">
    <h3 style="color: white; margin: 0;">⚙️ Smart Filters</h3>
</div>
""", unsafe_allow_html=True)

# Checkbox for excluding invalid entries
exclude_invalid = st.sidebar.checkbox("Exclude Invalid Entries (blanks, --, 0, etc.)", value=True)

# Initialize session state for show_full_numbers
if 'show_full_numbers' not in st.session_state:
    st.session_state.show_full_numbers = False

# Sidebar for full numbers toggle with dynamic label
button_label = "🔢 Short Form" if st.session_state.show_full_numbers else "🔢 In Details"
if st.sidebar.button(button_label):
    st.session_state.show_full_numbers = not st.session_state.show_full_numbers
    st.rerun()

show_full_numbers = st.session_state.show_full_numbers

# Function to format number with enhanced styling
def format_number(num, full=False):
    if full:
        return f"{num:,}"
    if num >= 1_000_000:
        return f"{num / 1_000_000:.1f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.1f}K"
    else:
        return str(num)

# Global filters (slicers) with enhanced styling
valid_years = sorted([year for year in df['Sent_Year'].unique() if not pd.isna(year)])
selected_year = st.sidebar.multiselect("📅 Select Year", options=valid_years, default=valid_years)
selected_quarter = st.sidebar.multiselect("🗓️ Select Quarter", options=[f"Q{q}" for q in range(1,5)], default=[f"Q{q}" for q in range(1,5)])
selected_campaign = st.sidebar.multiselect("🎯 Select Campaign", options=sorted(df['Campaign Name'].unique()), default=sorted(df['Campaign Name'].unique()))
df['Quarter_Label'] = 'Q' + df['Quarter'].astype(str)
selected_quarter_num = [int(q[1:]) for q in selected_quarter]
bot_filter = st.sidebar.multiselect("🤖 Bot/Human Filter", options=df['Bot Check'].unique(), default=df['Bot Check'].unique())
time_range_filter = st.sidebar.multiselect("⏰ Time Range Filter", options=df['Opend Time Range'].unique(), default=df['Opend Time Range'].unique())

# Apply filters to main df
filtered_df = df[
    (df['Sent_Year'].isin(selected_year)) &
    (df['Quarter'].isin(selected_quarter_num)) &
    (df['Campaign Name'].isin(selected_campaign)) &
    (df['Bot Check'].isin(bot_filter)) &
    (df['Opend Time Range'].isin(time_range_filter))
].copy()

# Debug Engagement in filtered_df
st.write("Engagement value counts in filtered_df:", filtered_df['Engagement'].value_counts(dropna=False))

# Warning if no HE in filtered data
if 'HE' not in filtered_df['Engagement'].values:
    st.warning("⚠️ No 'HE' engagements found in filtered data. Try adjusting filters.")

# Ensure 'Website' column is handled correctly for unique brands
if 'Website' in filtered_df.columns:
    filtered_df['Website'] = filtered_df['Website'].fillna('Unknown')
    if exclude_invalid:
        valid_brands = filtered_df[filtered_df['Website'].notna() & (filtered_df['Website'] != 'Unknown') & (filtered_df['Website'] != '--') & (filtered_df['Website'] != '')]
        total_brands = len(valid_brands['Website'].unique())
    else:
        total_brands = len(filtered_df['Website'].dropna().unique())
else:
    total_brands = 0

# Enhanced color schemes for charts
st.sidebar.markdown("""
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1rem; border-radius: 15px; margin: 1rem 0; text-align: center;">
    <h3 style="color: white; margin: 0;">📈 Chart Settings</h3>
</div>
""", unsafe_allow_html=True)

top_n = st.sidebar.selectbox("📊 Top N for Charts", options=[5, 10, 20, 'All'], index=0)
if top_n == 'All':
    top_n_val = len(filtered_df)
else:
    top_n_val = top_n

color_schemes = {
    'primary': ['#667eea', '#764ba2', '#4facfe', '#00f2fe'],
    'gradient': ['#ff6b6b', '#4ecdc4', '#45b7d1', '#f9ca24'],
    'professional': ['#2c3e50', '#3498db', '#e74c3c', '#f39c12'],
    'neon': ['#00f2fe', '#4facfe', '#667eea', '#764ba2']
}

if page == "🏠 Dashboard Home":
    # Enhanced Key Metrics with modern cards
    st.markdown('<div class="section-header slide-up">📈 Performance Dashboard</div>', unsafe_allow_html=True)

    # Calculate metrics
    filtered_df['Engagement'] = filtered_df['Engagement'].astype(str).str.strip().str.upper()
    total_campaigns = filtered_df['Campaign Name'].nunique()
    total_sent = len(filtered_df)
    unique_prospects = filtered_df['Lead Email'].nunique()
    total_opens = len(filtered_df[filtered_df['Open Count'] > 0])
    total_clicks = len(filtered_df[filtered_df['Click Count'] > 0])
    he_count = len(filtered_df[filtered_df['Engagement'] == 'HE'])
    le_count = len(filtered_df[filtered_df['Engagement'] == 'LE'])
    no_count = len(filtered_df[filtered_df['Engagement'] == 'NO'])
    total_replies = filtered_df['Has_Reply'].sum()
    total_positive_replies = len(filtered_df[filtered_df['Positive_Reply'] == True])
    bot_count = len(filtered_df[filtered_df['Bot Check'] == 'Bot'])
    human_count = len(filtered_df[filtered_df['Bot Check'] == 'Human'])
    reply_percentage = (total_replies / total_brands * 100) if total_brands > 0 else 0

    # Debug raw counts
    st.write("Raw counts - HE:", he_count, "LE:", le_count, "NO:", no_count)

    # Key metrics layout (adjusted to accommodate Reply Percentage)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f"""
        <div class="metric-container fade-in">
            <div class="metric-value">{format_number(total_campaigns, show_full_numbers)}</div>
            <div class="metric-label">🎯 Total Campaigns</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-container fade-in">
            <div class="metric-value">{format_number(total_sent, show_full_numbers)}</div>
            <div class="metric-label">📧 Total Emails Sent</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-container fade-in">
            <div class="metric-value">{format_number(total_brands, show_full_numbers)}</div>
            <div class="metric-label">🏢 Total Brands</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-container fade-in">
            <div class="metric-value">{format_number(unique_prospects, show_full_numbers)}</div>
            <div class="metric-label">👥 Unique Prospects</div>
        </div>
        """, unsafe_allow_html=True)
    with col5:
        st.markdown(f"""
        <div class="metric-container fade-in">
            <div class="metric-value">{format_number(total_opens, show_full_numbers)}</div>
            <div class="metric-label">👀 Total Opens</div>
        </div>
        """, unsafe_allow_html=True)

    col6, col7, col8, col9, col10 = st.columns(5)
    with col6:
        st.markdown(f"""
        <div class="metric-container fade-in">
            <div class="metric-value">{format_number(total_clicks, show_full_numbers)}</div>
            <div class="metric-label">🖱️ Total Clicks</div>
        </div>
        """, unsafe_allow_html=True)
    with col7:
        open_rate = (total_opens / total_sent * 100) if total_sent > 0 else 0
        st.markdown(f"""
        <div class="metric-container fade-in">
            <div class="metric-value">{open_rate:.1f}%</div>
            <div class="metric-label">📈 Open Rate</div>
        </div>
        """, unsafe_allow_html=True)
    with col8:
        click_rate = (total_clicks / total_opens * 100) if total_opens > 0 else 0
        st.markdown(f"""
        <div class="metric-container fade-in">
            <div class="metric-value">{click_rate:.1f}%</div>
            <div class="metric-label">🖱️ Click Rate</div>
        </div>
        """, unsafe_allow_html=True)
    with col9:
        st.markdown(f"""
        <div class="metric-container fade-in">
            <div class="metric-value">{format_number(total_replies, show_full_numbers)}</div>
            <div class="metric-label">💬 Total Replies</div>
        </div>
        """, unsafe_allow_html=True)
    with col10:
        st.markdown(f"""
        <div class="metric-container fade-in">
            <div class="metric-value">{format_number(total_positive_replies, show_full_numbers)}</div>
            <div class="metric-label">✅ Total Positive Replies</div>
        </div>
        """, unsafe_allow_html=True)

    col11, col12, col13, col14, col15 = st.columns(5)
    with col11:
        st.markdown(f"""
        <div class="metric-container fade-in">
            <div class="metric-value">{format_number(he_count, show_full_numbers)}</div>
            <div class="metric-label">HE Count</div>
        </div>
        """, unsafe_allow_html=True)
    with col12:
        st.markdown(f"""
        <div class="metric-container fade-in">
            <div class="metric-value">{format_number(le_count, show_full_numbers)}</div>
            <div class="metric-label">LE Count</div>
        </div>
        """, unsafe_allow_html=True)
    with col13:
        st.markdown(f"""
        <div class="metric-container fade-in">
            <div class="metric-value">{format_number(no_count, show_full_numbers)}</div>
            <div class="metric-label">NO Count</div>
        </div>
        """, unsafe_allow_html=True)
    with col14:
        st.markdown(f"""
        <div class="metric-container fade-in">
            <div class="metric-value">{format_number(bot_count, show_full_numbers)}</div>
            <div class="metric-label">🤖 Bot Count</div>
        </div>
        """, unsafe_allow_html=True)
    with col15:
        st.markdown(f"""
        <div class="metric-container fade-in">
            <div class="metric-value">{reply_percentage:.1f}%</div>
            <div class="metric-label">💬 Reply Rate</div>
        </div>
        """, unsafe_allow_html=True)

    col16, col17, col18 = st.columns([1, 1, 3])
    with col16:
        st.markdown(f"""
        """, unsafe_allow_html=True)
    with col17:
        st.markdown("")  # Empty column for spacing
    with col18:
        st.markdown("")  # Empty column for spacing

    # Enhanced Charts Section
    st.markdown('<div class="section-header slide-up">📊 Performance Visualizations</div>', unsafe_allow_html=True)

    # Opens by Sent Time Range with modern styling
    col1, col2 = st.columns([4, 1])
    with col1:
        time_range_data = filtered_df[filtered_df['Open Count'] > 0]['Opend Time Range'].value_counts().head(top_n_val)
        fig_time = px.bar(
            x=time_range_data.index, 
            y=time_range_data.values, 
            title="<b>Peak Performance by Time Range</b>",
            color_discrete_sequence=color_schemes['primary']
        )
        fig_time.update_layout(
            font=dict(color='white'),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            title_font_size=18,
            title_x=0.5,
            xaxis_title="Time Range",
            yaxis_title="Opens"
        )
        fig_time.update_traces(marker_line_width=0, marker_cornerradius="15%")
        st.plotly_chart(fig_time, use_container_width=True)
    with col2:
        st.markdown('<div style="padding: 1rem;">', unsafe_allow_html=True)
        if st.button("💡 AI Insights", key="time_insight", help="Get timing insights"):
            st.info(generate_insights(filtered_df, "Top Opens by Time Range"))
        st.markdown('</div>', unsafe_allow_html=True)

    # Top Cities by Opens and Clicks (Map-Based)
    if 'latitude' in filtered_df.columns and 'longitude' in filtered_df.columns:
        col3, col4 = st.columns([4, 1])
        with col3:
            city_data = filtered_df[filtered_df['Open Count'] > 0].groupby(['City', 'latitude', 'longitude']).agg({
                'Open Count': 'sum',
                'Click Count': 'sum'
            }).reset_index()
            if exclude_invalid:
                city_data = city_data[
                    (city_data['City'].notna()) &
                    (city_data['City'] != '') &
                    (city_data['City'] != '0') &
                    (city_data['City'] != '--') &
                    (city_data['City'] != 'Unknown')
                ]
            city_data = city_data.nlargest(top_n_val, 'Open Count')
            city_data['Size'] = city_data['Open Count'] + city_data['Click Count'] * 2  # Weighted size for visualization
            fig_city_map = px.scatter_mapbox(
                city_data,
                lat='latitude',
                lon='longitude',
                size='Size',
                color='Open Count',
                hover_name='City',
                hover_data={'Open Count': True, 'Click Count': True, 'latitude': False, 'longitude': False},
                title="<b>Top Cities by Opens and Clicks</b>",
                color_continuous_scale='Viridis',
                size_max=20,
                zoom=3
            )
            fig_city_map.update_layout(
                mapbox_style="carto-darkmatter",
                font=dict(color='white'),
                paper_bgcolor='rgba(0,0,0,0)',
                title_font_size=18,
                title_x=0.5
            )
            st.plotly_chart(fig_city_map, use_container_width=True)
        with col4:
            st.markdown('<div style="padding: 1rem;">', unsafe_allow_html=True)
            if st.button("💡 AI Insights", key="city_map_insight", help="Get city insights"):
                st.info(generate_insights(filtered_df, "Top Cities by Opens and Clicks"))
            st.markdown('</div>', unsafe_allow_html=True)

    # Opens by City (Bar Chart)
    col3, col4 = st.columns([4, 1])
    with col3:
        city_data = filtered_df[filtered_df['Open Count'] > 0].groupby('City').size().reset_index(name='Opens')
        if exclude_invalid:
            city_data = city_data[
                (city_data['City'].notna()) &
                (city_data['City'] != '') &
                (city_data['City'] != '0') &
                (city_data['City'] != '--') &
                (city_data['City'] != 'Unknown')
            ]
        city_data = city_data.nlargest(top_n_val, 'Opens')
        fig_city = px.bar(
            city_data, 
            x='City', 
            y='Opens', 
            title="<b>Top Opens by Cities</b>", 
            color='Opens',
            color_continuous_scale='Viridis'
        )
        fig_city.update_layout(
            font=dict(color='white'),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            title_font_size=18,
            title_x=0.5,
            xaxis_title="City",
            yaxis_title="Opens"
        )
        fig_city.update_traces(marker_line_width=0, marker_cornerradius="15%")
        st.plotly_chart(fig_city, use_container_width=True)
    with col4:
        st.markdown('<div style="padding: 1rem;">', unsafe_allow_html=True)
        if st.button("💡 AI Insights", key="city_insight", help="Get geographic insights"):
            st.info(generate_insights(filtered_df, "Top Opens by City"))
        st.markdown('</div>', unsafe_allow_html=True)

    # Opens by Campaign with gradient colors
    row1_col1, row1_col2 = st.columns([4, 1])
    with row1_col1:
        campaign_data = filtered_df[filtered_df['Open Count'] > 0].groupby('Campaign Name').size().nlargest(top_n_val).reset_index(name='Opens')
        fig_campaign = px.bar(
            campaign_data, 
            x='Campaign Name', 
            y='Opens', 
            title="<b>Top Performing Campaigns</b>",
            color='Opens',
            color_continuous_scale='Plasma'
        )
        fig_campaign.update_layout(
            font=dict(color='white'),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            title_font_size=18,
            title_x=0.5,
            xaxis_title="Campaign Name",
            yaxis_title="Opens"
        )
        fig_campaign.update_traces(marker_line_width=0, marker_cornerradius="15%")
        fig_campaign.update_xaxes(tickangle=45)
        st.plotly_chart(fig_campaign, use_container_width=True)
    with row1_col2:
        st.markdown('<div style="padding: 1rem;">', unsafe_allow_html=True)
        if st.button("💡 AI Insights", key="campaign_insight", help="Get campaign insights"):
            st.info(generate_insights(filtered_df, "Top Opens by Campaign"))
        st.markdown('</div>', unsafe_allow_html=True)

    # Opens by ESP with modern styling
    row2_col1, row2_col2 = st.columns([4, 1])
    with row2_col1:
        esp_data = filtered_df[filtered_df['Open Count'] > 0].groupby('ESP Type').size().nlargest(top_n_val).reset_index(name='Opens')
        fig_esp = px.bar(
            esp_data, 
            x='ESP Type', 
            y='Opens', 
            title="<b>Email Service Provider Performance</b>", 
            color='Opens',
            color_continuous_scale='Turbo'
        )
        fig_esp.update_layout(
            font=dict(color='white'),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            title_font_size=18,
            title_x=0.5,
            xaxis_title="ESP Type",
            yaxis_title="Opens"
        )
        fig_esp.update_traces(marker_line_width=0, marker_cornerradius="15%")
        st.plotly_chart(fig_esp, use_container_width=True)
    with row2_col2:
        st.markdown('<div style="padding: 1rem;">', unsafe_allow_html=True)
        if st.button("💡 AI Insights", key="esp_insight", help="Get ESP insights"):
            st.info(generate_insights(filtered_df, "Top Opens by ESP"))
        st.markdown('</div>', unsafe_allow_html=True)

    # Additional Enhanced Insights
    st.markdown('<div class="section-header slide-up">🔍 Advanced Analytics</div>', unsafe_allow_html=True)

    # Top Opens by State if no lat/long
    if 'latitude' not in filtered_df.columns or 'longitude' not in filtered_df.columns:
        st.markdown("### 🗺️ Top Opens by State")
        col1, col2 = st.columns([4, 1])
        with col1:
            state_data = filtered_df[filtered_df['Open Count'] > 0].groupby('State').size().nlargest(top_n_val).reset_index(name='Opens')
            if exclude_invalid:
                state_data = state_data[
                    (state_data['State'].notna()) &
                    (state_data['State'] != '') &
                    (state_data['State'] != '--') &
                    (state_data['State'] != 'Unknown')
                ]
            fig_state = px.bar(
                state_data, 
                x='State', 
                y='Opens', 
                title="<b>Top Performing States</b>", 
                color='Opens',
                color_continuous_scale='Viridis'
            )
            fig_state.update_layout(
                font=dict(color='white'),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                title_font_size=18,
                title_x=0.5
            )
            fig_state.update_traces(marker_cornerradius="15%")
            st.plotly_chart(fig_state, use_container_width=True)
        with col2:
            st.markdown('<div style="padding: 1rem;">', unsafe_allow_html=True)
            if st.button("💡 AI Insights", key="state_insight", help="Get state insights"):
                st.info(generate_insights(filtered_df, "Top Opens by State"))
            st.markdown('</div>', unsafe_allow_html=True)

    # Clicks by Campaign with enhanced styling
    col3, col4 = st.columns([4, 1])
    with col3:
        clicks_campaign = filtered_df[filtered_df['Click Count'] > 0].groupby('Campaign Name').size().nlargest(top_n_val).reset_index(name='Clicks')
        fig_clicks = px.bar(
            clicks_campaign, 
            x='Campaign Name', 
            y='Clicks', 
            title="<b>Click Performance by Campaign</b>",
            color='Clicks',
            color_continuous_scale='Cividis'
        )
        fig_clicks.update_layout(
            font=dict(color='white'),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            title_font_size=18,
            title_x=0.5
        )
        fig_clicks.update_traces(marker_cornerradius="15%")
        fig_clicks.update_xaxes(tickangle=45)
        st.plotly_chart(fig_clicks, use_container_width=True)
    with col4:
        st.markdown('<div style="padding: 1rem;">', unsafe_allow_html=True)
        if st.button("💡 AI Insights", key="clicks_insight", help="Get click insights"):
            st.info(generate_insights(filtered_df, "Top Clicks by Campaign"))
        st.markdown('</div>', unsafe_allow_html=True)

    # Enhanced Unsubscribes Table
    st.markdown("### 🚪 Unsubscribe Analysis")
    col1, col2 = st.columns([4, 1])
    with col1:
        unsub_data = filtered_df[filtered_df['Is_Unsubscribed'] == True].groupby('Campaign Name').size().nlargest(top_n_val).reset_index(name='Unsubscribes')
        if not unsub_data.empty:
            st.dataframe(
                unsub_data.style.background_gradient(subset=['Unsubscribes'], cmap='Reds'),
                use_container_width=True
            )
        else:
            st.success("🎉 Great news! No unsubscribes found in the selected data.")
    with col2:
        st.markdown('<div style="padding: 1rem;">', unsafe_allow_html=True)
        if st.button("💡 AI Insights", key="unsub_insight", help="Get unsubscribe insights"):
            st.info(generate_insights(filtered_df, "Unsubscribes by Campaign"))
        st.markdown('</div>', unsafe_allow_html=True)

    # Reply Analysis with modern charts using main df
    st.markdown("### 💬 Reply Intelligence")
    col1, col2 = st.columns([4, 1])
    with col1:
        reply_data = filtered_df.groupby('Campaign Name').agg({
            'Has_Reply': 'sum',
            'Positive_Reply': 'sum',
            'Sent_Date': 'min'
        }).reset_index()
        reply_data = reply_data.rename(columns={'Has_Reply': 'Replied Count', 'Positive_Reply': 'Positive Reply Count'})
        fig_reply = px.bar(
            reply_data, 
            x='Campaign Name', 
            y=['Replied Count', 'Positive Reply Count'], 
            title="<b>Reply Performance Analysis</b>",
            barmode='group', 
            color_discrete_sequence=color_schemes['gradient']
        )
        fig_reply.update_layout(
            font=dict(color='white'),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            title_font_size=18,
            title_x=0.5
        )
        fig_reply.update_traces(marker_cornerradius="15%")
        fig_reply.update_xaxes(tickangle=45)
        st.plotly_chart(fig_reply, use_container_width=True)
        
        st.dataframe(
            reply_data[['Campaign Name', 'Sent_Date', 'Replied Count', 'Positive Reply Count']].style.background_gradient(
                subset=['Replied Count', 'Positive Reply Count'], cmap='Blues'
            ),
            use_container_width=True
        )
    with col2:
        st.markdown('<div style="padding: 1rem;">', unsafe_allow_html=True)
        if st.button("💡 AI Insights", key="reply_vs_positive_insight", help="Get reply insights"):
            st.info(generate_insights(filtered_df, "Reply vs Positive Reply"))
        st.markdown('</div>', unsafe_allow_html=True)

    # Reply Rate Table
    col3, col4 = st.columns([4, 1])
    with col3:
        reply_data = filtered_df.groupby('Campaign Name').agg({
            'Has_Reply': 'sum',
            'Sent_Date': 'min',
            'Lead Email': 'count'  # Count of emails sent per campaign
        }).reset_index()
        reply_data = reply_data.rename(columns={'Has_Reply': 'Replied Count', 'Lead Email': 'Sent'})
        reply_data['Reply Rate (%)'] = (reply_data['Replied Count'] / reply_data['Sent'] * 100).round(2)
        st.dataframe(
            reply_data[['Campaign Name', 'Sent_Date', 'Replied Count', 'Reply Rate (%)']].style.background_gradient(
                subset=['Reply Rate (%)'], cmap='Greens'
            ),
            use_container_width=True
        )
    with col4:
        st.markdown('<div style="padding: 1rem;">', unsafe_allow_html=True)
        if st.button("💡 AI Insights", key="reply_insight", help="Get reply rate insights"):
            st.info(generate_insights(filtered_df, "Reply Rate"))
        st.markdown('</div>', unsafe_allow_html=True)

    # Traffic Sources with enhanced pie chart
    if 'Traffic' in filtered_df.columns and filtered_df['Traffic'].dtype == 'object':
        col1, col2 = st.columns([4, 1])
        with col1:
            traffic_data = filtered_df.groupby('Traffic').size().nlargest(top_n_val).reset_index(name='Count')
            if exclude_invalid:
                traffic_data = traffic_data[
                    (traffic_data['Traffic'].notna()) &
                    (traffic_data['Traffic'] != '') &
                    (traffic_data['Traffic'] != '--') &
                    (traffic_data['Traffic'] != 'Unknown')
                ]
            fig_traffic = px.pie(
                traffic_data, 
                values='Count', 
                names='Traffic', 
                title="<b>Traffic Source Distribution</b>",
                color_discrete_sequence=color_schemes['neon'],
                hole=0.4
            )
            fig_traffic.update_layout(
                font=dict(color='white'),
                paper_bgcolor='rgba(0,0,0,0)',
                title_font_size=18,
                title_x=0.5
            )
            fig_traffic.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_traffic, use_container_width=True)
        with col2:
            st.markdown('<div style="padding: 1rem;">', unsafe_allow_html=True)
            if st.button("💡 AI Insights", key="traffic_insight", help="Get traffic insights"):
                st.info(generate_insights(filtered_df, "Traffic Sources"))
            st.markdown('</div>', unsafe_allow_html=True)

    # New chart: Unique companies with most HE
    if 'Website' in filtered_df.columns and 'Engagement' in filtered_df.columns:
        st.markdown("### 🏢 Top Companies by High Engagement (HE)")
        col1, col2 = st.columns([4, 1])
        with col1:
            he_company_data = filtered_df[filtered_df['Engagement'] == 'HE'].groupby('Website').size().reset_index(name='HE Count')
            if exclude_invalid:
                he_company_data = he_company_data[
                    (he_company_data['Website'].notna()) &
                    (he_company_data['Website'] != '--') &
                    (he_company_data['Website'] != '') &
                    (he_company_data['Website'] != 'Unknown')
                ]
            he_company_data = he_company_data.nlargest(top_n_val, 'HE Count')
            fig_he_company = px.bar(
                he_company_data, 
                x='Website', 
                y='HE Count', 
                title="<b>Top Unique Companies with High Engagement</b>",
                color='HE Count',
                color_continuous_scale='Viridis'
            )
            fig_he_company.update_layout(
                font=dict(color='white'),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                title_font_size=18,
                title_x=0.5
            )
            fig_he_company.update_traces(marker_cornerradius="15%")
            fig_he_company.update_xaxes(tickangle=45)
            st.plotly_chart(fig_he_company, use_container_width=True)
        with col2:
            st.markdown('<div style="padding: 1rem;">', unsafe_allow_html=True)
            if st.button("💡 AI Insights", key="he_company_insight", help="Get insights"):
                st.info(generate_insights(filtered_df, "Top Companies by HE"))
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("No high engagement data available.")

elif page == "📊 Compare Quarters":
    st.markdown('<div class="section-header slide-up">📊 Quarterly Performance Comparison</div>', unsafe_allow_html=True)
    
    selected_quarters = st.multiselect(
        "🗓️ Select Quarters to Compare", 
        options=[f"Q{q}" for q in range(1,5)], 
        default=[f"Q1", f"Q2", f"Q3"]
    )
    
    if len(selected_quarters) < 2:
        st.warning("⚠️ Please select at least two quarters for meaningful comparison.")
    else:
        selected_quarter_nums = [int(q[1:]) for q in selected_quarters]
        df_list = [df[df['Quarter'] == q_num].copy() for q_num in selected_quarter_nums]
        
        # Enhanced comparison metrics
        st.markdown("### 📈 Key Performance Metrics")
        metrics = [
            "Total Campaigns", "Total Emails Sent", "Total Brands", "Unique Prospects", 
            "Total Opens", "Total Clicks", "Open Rate (%)", "Click Rate (%)", 
            "HE Count", "LE Count", "NO Count", "Total Replies", "Total Positive Replies", "Reply Rate"
        ]
        
        data = []
        for metric in metrics:
            row = [f"📊 {metric}"]
            for i, df_q in enumerate(df_list):
                q_num = selected_quarter_nums[i]
                if metric == "Total Campaigns":
                    val = df_q['Campaign Name'].nunique()
                elif metric == "Total Emails Sent":
                    val = len(df_q)
                elif metric == "Total Brands":
                    if 'Website' in df_q.columns:
                        df_q['Website'] = df_q['Website'].fillna('Unknown')
                        if exclude_invalid:
                            valid_brands = df_q[df_q['Website'].notna() & (df_q['Website'] != 'Unknown') & (df_q['Website'] != '--') & (df_q['Website'] != '')]
                            val = len(valid_brands['Website'].unique())
                        else:
                            val = len(df_q['Website'].dropna().unique())
                    else:
                        val = 0
                elif metric == "Unique Prospects":
                    val = df_q['Lead Email'].nunique()
                elif metric == "Total Opens":
                    val = len(df_q[df_q['Open Count'] > 0])
                elif metric == "Total Clicks":
                    val = len(df_q[df_q['Click Count'] > 0])
                elif metric == "Open Rate (%)":
                    val = f"{(len(df_q[df_q['Open Count'] > 0]) / len(df_q) * 100):.1f}%" if len(df_q) > 0 else "0.0%"
                elif metric == "Click Rate (%)":
                    val = f"{(len(df_q[df_q['Click Count'] > 0]) / len(df_q[df_q['Open Count'] > 0]) * 100):.1f}%" if len(df_q[df_q['Open Count'] > 0]) > 0 else "0.0%"
                elif metric == "HE Count":
                    df_q['Engagement'] = df_q['Engagement'].astype(str).str.strip().str.upper()
                    val = len(df_q[df_q['Engagement'] == 'HE'])
                elif metric == "LE Count":
                    df_q['Engagement'] = df_q['Engagement'].astype(str).str.strip().str.upper()
                    val = len(df_q[df_q['Engagement'] == 'LE'])
                elif metric == "NO Count":
                    df_q['Engagement'] = df_q['Engagement'].astype(str).str.strip().str.upper()
                    val = len(df_q[df_q['Engagement'] == 'NO'])
                elif metric == "Total Replies":
                    val = df_q['Has_Reply'].sum()
                elif metric == "Total Positive Replies":
                    val = df_q['Positive_Reply'].sum()
                elif metric == "Reply Rate":
                    if 'Website' in df_q.columns:
                        df_q['Website'] = df_q['Website'].fillna('Unknown')
                        if exclude_invalid:
                            valid_brands = df_q[df_q['Website'].notna() & (df_q['Website'] != 'Unknown') & (df_q['Website'] != '--') & (df_q['Website'] != '')]
                            total_brands_q = len(valid_brands['Website'].unique())
                        else:
                            total_brands_q = len(df_q['Website'].dropna().unique())
                    else:
                        total_brands_q = 0
                    val = f"{(df_q['Has_Reply'].sum() / total_brands_q * 100):.1f}%" if total_brands_q > 0 else "0.0%"
                
                if "Rate" not in metric and "Percentage" not in metric:
                    val = format_number(val, show_full_numbers)
                row.append(val)
            data.append(row)
        
        compare_df = pd.DataFrame(data, columns=["📊 Metric"] + [f"📅 {q}" for q in selected_quarters])
        st.dataframe(
            compare_df.style.set_properties(**{
                'background-color': '#2a2a5a',
                'color': 'white',
                'border-color': '#667eea'
            }),
            use_container_width=True
        )
        
        # Enhanced comparison visualization
        st.markdown("### 🎯 Campaign Performance Comparison")
        combined_campaign = pd.DataFrame()
        for i, df_q in enumerate(df_list):
            campaign_data = df_q[df_q['Open Count'] > 0].groupby('Campaign Name').size().nlargest(top_n_val).reset_index(name=f'Opens {selected_quarters[i]}')
            if combined_campaign.empty:
                combined_campaign = campaign_data
            else:
                combined_campaign = combined_campaign.merge(campaign_data, on='Campaign Name', how='outer')
        
        combined_campaign = combined_campaign.fillna(0)
        fig_compare = px.bar(
            combined_campaign, 
            x='Campaign Name', 
            y=[f'Opens {q}' for q in selected_quarters], 
            title="<b>Opens by Campaign - Quarterly Comparison</b>",
            barmode='group',
            color_discrete_sequence=color_schemes['gradient']
        )
        fig_compare.update_layout(
            font=dict(color='white'),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            title_font_size=18,
            title_x=0.5
        )
        fig_compare.update_traces(marker_cornerradius="15%")
        fig_compare.update_xaxes(tickangle=45)
        st.plotly_chart(fig_compare, use_container_width=True)

elif page == "🤖 AI Predictions":
    st.markdown('<div class="section-header slide-up">🧠 AI-Powered Predictive Analytics</div>', unsafe_allow_html=True)
    
    # Geographic Clustering with enhanced visualization
    if 'latitude' in filtered_df.columns and 'longitude' in filtered_df.columns:
        st.markdown("### 🗺️ Geographic Clustering Intelligence")
        col1, col2 = st.columns([4, 1])
        with col1:
            geo_df = filtered_df[['latitude', 'longitude', 'Open Count']].dropna()
            if len(geo_df) > 0:
                n_clusters_geo = min(3, len(geo_df))
                kmeans = KMeans(n_clusters=n_clusters_geo, random_state=42, n_init=10)
                geo_df['Cluster'] = kmeans.fit_predict(geo_df[['latitude', 'longitude']])
                fig_cluster = px.scatter_mapbox(
                    geo_df,
                    lat='latitude',
                    lon='longitude',
                    size='Open Count',
                    color='Cluster',
                    title="<b>AI Geographic Clusters (K-Means)</b>",
                    color_discrete_sequence=color_schemes['neon'],
                    size_max=20,
                    zoom=3,
                    mapbox_style="carto-darkmatter"
                )
                fig_cluster.update_layout(
                    font=dict(color='white'),
                    paper_bgcolor='rgba(0,0,0,0)',
                    title_font_size=18,
                    title_x=0.5
                )
                st.plotly_chart(fig_cluster, use_container_width=True)
            else:
                st.warning("⚠️ No valid geographic data available for clustering analysis.")
        with col2:
            st.markdown('<div style="padding: 1rem;">', unsafe_allow_html=True)
            if st.button("💡 AI Insights", key="geo_cluster_insight", help="Get clustering insights"):
                st.info(generate_insights(filtered_df, "Geographic Clustering (KMeans)"))
            st.markdown('</div>', unsafe_allow_html=True)

    # Lead Behavior Segmentation with enhanced styling
    st.markdown("### 🧠 Behavioral Segmentation Intelligence")
    col1, col2 = st.columns([4, 1])
    with col1:
        behavior_features = filtered_df[['Open Count', 'Click Count', 'Response_Time']].fillna(0)
        if len(behavior_features) > 0:
            n_clusters_behavior = min(4, len(behavior_features))
            kmeans_behavior = KMeans(n_clusters=n_clusters_behavior, random_state=42, n_init=10)
            filtered_df['Behavior_Cluster'] = kmeans_behavior.fit_predict(behavior_features)
            fig_segment = px.scatter(
                filtered_df, 
                x='Open Count', 
                y='Click Count', 
                color='Behavior_Cluster', 
                size='Response_Time',
                title="<b>AI Lead Behavior Segments</b>", 
                color_discrete_sequence=color_schemes['gradient'],
                size_max=20
            )
            fig_segment.update_layout(
                font=dict(color='white'),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                title_font_size=18,
                title_x=0.5
            )
            st.plotly_chart(fig_segment, use_container_width=True)
        else:
            st.warning("⚠️ No behavior data available for segmentation analysis.")
    with col2:
        st.markdown('<div style="padding: 1rem;">', unsafe_allow_html=True)
        if st.button("💡 AI Insights", key="behavior_insight", help="Get behavioral insights"):
            st.info(generate_insights(filtered_df, "Lead Behavior Segmentation (KMeans)"))
        st.markdown('</div>', unsafe_allow_html=True)

    # Enhanced Open Prediction Model
    st.markdown("### 🎯 Email Open Probability Prediction")
    col1, col2 = st.columns([4, 1])
    with col1:
        @st.cache_resource
        def train_open_model(_df):
            features = ['Sent_Year', 'Sent_Month', 'Sent_DayOfWeek', 'Quarter']
            categorical_cols = []
            if 'ESP Type' in _df.columns:
                features.append('ESP Type')
                categorical_cols.append('ESP Type')
            if 'Traffic' in _df.columns:
                features.append('Traffic')
                categorical_cols.append('Traffic')
            
            X = _df[features].copy()
            if 'City' in _df.columns and len(_df['City'].unique()) <= 50:
                features.append('City')
                categorical_cols.append('City')
            
            le_dict = {}
            for col in categorical_cols:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                le_dict[col] = le
            y = (_df['Open Count'] > 0).astype(int)
            
            if len(X) > 100:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
                model.fit(X_train, y_train)
                accuracy = model.score(X_test, y_test)
                
                # Display enhanced accuracy metric
                st.markdown(f"""
                <div class="metric-container fade-in">
                    <div class="metric-value">{accuracy:.2f}</div>
                    <div class="metric-label">🎯 Model Accuracy</div>
                </div>
                """, unsafe_allow_html=True)
                
                _df['Open_Probability'] = model.predict_proba(X)[:, 1]
                return model, _df
            return None, _df

        model, filtered_df_with_pred = train_open_model(filtered_df)
        if model is not None:
            pred_data = filtered_df_with_pred[['Lead Email', 'Open_Probability', 'Campaign Name']].head(top_n_val).sort_values('Open_Probability', ascending=False)
            st.dataframe(
                pred_data.style.background_gradient(subset=['Open_Probability'], cmap='Greens'),
                use_container_width=True
            )
        else:
            st.warning("⚠️ Insufficient data for training prediction model (minimum 100 rows required).")
    with col2:
        st.markdown('<div style="padding: 1rem;">', unsafe_allow_html=True)
        if st.button("💡 AI Insights", key="open_pred_insight", help="Get prediction insights"):
            st.info(generate_insights(filtered_df, "Open Probability (Random Forest)"))
        st.markdown('</div>', unsafe_allow_html=True)

    # Enhanced Forecasting with Prophet
    # Enhanced Forecasting with Prophet
    st.markdown("### 📈 Future Opens Forecasting")
    col1, col2 = st.columns([4, 1])
    with col1:
        @st.cache_resource
        def forecast_opens(_df):
            time_data = _df[_df['Open Count'] > 0].groupby('Sent_Date').size().reset_index(name='Opens')
            if len(time_data) < 10:
                st.warning("⚠️ Insufficient time-series data for forecasting (minimum 10 data points required).")
                return None
            time_data.columns = ['ds', 'y']
            model_prophet = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
            model_prophet.fit(time_data)
            future = model_prophet.make_future_dataframe(periods=30)
            forecast = model_prophet.predict(future)
            return forecast

        forecast = forecast_opens(filtered_df)
        if forecast is not None:
            fig_forecast = px.line(
                forecast, 
                x='ds', 
                y='yhat', 
                title="<b>AI-Powered Opens Forecast (Next 30 Days)</b>", 
                color_discrete_sequence=['#4facfe']
            )
            fig_forecast.add_scatter(
                x=forecast['ds'], 
                y='yhat_lower', 
                mode='lines', 
                line=dict(color='rgba(255,255,255,0)'), 
                showlegend=False, 
                fill='tonexty', 
                fillcolor='rgba(79, 172, 254, 0.2)'
            )
            fig_forecast.add_scatter(
                x=forecast['ds'], 
                y='yhat_upper', 
                mode='lines', 
                line=dict(color='rgba(255,255,255,0)'), 
                showlegend=False, 
                fill='tonexty', 
                fillcolor='rgba(79, 172, 254, 0.2)'
            )
            fig_forecast.update_layout(
                font=dict(color='white'),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                title_font_size=18,
                title_x=0.5
            )
            st.plotly_chart(fig_forecast, use_container_width=True)
        else:
            st.warning("⚠️ No forecast generated due to insufficient data.")
    with col2:
        st.markdown('<div style="padding: 1rem;">', unsafe_allow_html=True)
        if st.button("💡 AI Insights", key="forecast_insight", help="Get forecasting insights"):
            st.info(generate_insights(filtered_df, "Predicted Opens (Prophet)"))
        st.markdown('</div>', unsafe_allow_html=True)

    # Enhanced Bot Detection
    st.markdown("### 🛡️ Enhanced Bot Detection")
    col1, col2 = st.columns([4, 1])
    with col1:
        @st.cache_resource
        def bot_detection_model(_df):
            features = ['Open Count', 'Click Count', 'Response_Time']
            X = _df[features].fillna(0)
            y = _df['Bot Check'].map({'Bot': 1, 'Human': 0}).fillna(0).astype(int)
            
            if len(X) > 100 and len(y.unique()) > 1:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
                model.fit(X_train, y_train)
                accuracy = model.score(X_test, y_test)
                
                st.markdown(f"""
                <div class="metric-container fade-in">
                    <div class="metric-value">{accuracy:.2f}</div>
                    <div class="metric-label">🛡️ Bot Detection Accuracy</div>
                </div>
                """, unsafe_allow_html=True)
                
                _df['Bot_Probability'] = model.predict_proba(X)[:, 1]
                return model, _df
            return None, _df

        bot_model, filtered_df_with_bot = bot_detection_model(filtered_df)
        if bot_model is not None:
            bot_data = filtered_df_with_bot[['Lead Email', 'Bot_Probability', 'Campaign Name']].head(top_n_val).sort_values('Bot_Probability', ascending=False)
            st.dataframe(
                bot_data.style.background_gradient(subset=['Bot_Probability'], cmap='Oranges'),
                use_container_width=True
            )
        else:
            st.warning("⚠️ Insufficient data for bot detection model (minimum 100 rows with both Bot and Human labels required).")
    with col2:
        st.markdown('<div style="padding: 1rem;">', unsafe_allow_html=True)
        if st.button("💡 AI Insights", key="bot_insight", help="Get bot detection insights"):
            st.info(generate_insights(filtered_df, "Enhanced Bot Probability (Random Forest)"))
        st.markdown('</div>', unsafe_allow_html=True)

elif page == "👑 Boss Dashboard":
    st.markdown('<div class="section-header slide-up">👑 Executive Insights Dashboard</div>', unsafe_allow_html=True)
    
    # Enhanced executive summary
    st.markdown("### 📊 Executive Summary")
    filtered_df['Engagement'] = filtered_df['Engagement'].astype(str).str.strip().str.upper()
    total_campaigns = filtered_df['Campaign Name'].nunique()
    total_sent = len(filtered_df)
    unique_prospects = filtered_df['Lead Email'].nunique()
    total_opens = len(filtered_df[filtered_df['Open Count'] > 0])
    total_clicks = len(filtered_df[filtered_df['Click Count'] > 0])
    he_count = len(filtered_df[filtered_df['Engagement'] == 'HE'])
    le_count = len(filtered_df[filtered_df['Engagement'] == 'LE'])
    no_count = len(filtered_df[filtered_df['Engagement'] == 'NO'])
    total_replies = filtered_df['Has_Reply'].sum()
    total_positive_replies = filtered_df['Positive_Reply'].sum()
    # Calculate total_brands for Reply Percentage
    if 'Website' in filtered_df.columns:
        filtered_df['Website'] = filtered_df['Website'].fillna('Unknown')
        if exclude_invalid:
            valid_brands = filtered_df[filtered_df['Website'].notna() & (filtered_df['Website'] != 'Unknown') & (filtered_df['Website'] != '--') & (filtered_df['Website'] != '')]
            total_brands = len(valid_brands['Website'].unique())
        else:
            total_brands = len(filtered_df['Website'].dropna().unique())
    else:
        total_brands = 0
    reply_percentage = (total_replies / total_brands * 100) if total_brands > 0 else 0
    
    # Debug raw counts for Boss Dashboard
    st.write("Boss Dashboard - Raw counts - HE:", he_count, "LE:", le_count, "NO:", no_count)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-container fade-in">
            <div class="metric-value">{format_number(total_campaigns, show_full_numbers)}</div>
            <div class="metric-label">🎯 Total Campaigns</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        open_rate = (total_opens / total_sent * 100) if total_sent > 0 else 0
        st.markdown(f"""
        <div class="metric-container fade-in">
            <div class="metric-value">{open_rate:.1f}%</div>
            <div class="metric-label">📈 Open Rate</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        click_rate = (total_clicks / total_opens * 100) if total_opens > 0 else 0
        st.markdown(f"""
        <div class="metric-container fade-in">
            <div class="metric-value">{click_rate:.1f}%</div>
            <div class="metric-label">🖱️ Click Rate</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-container fade-in">
            <div class="metric-value">{reply_percentage:.1f}%</div>
            <div class="metric-label">💬 Reply Rate</div>
        </div>
        """, unsafe_allow_html=True)

    # Top performing campaigns
    st.markdown("### 🎯 Top Performing Campaigns")
    campaign_summary = filtered_df.groupby('Campaign Name').agg({
        'Open Count': 'sum',
        'Click Count': 'sum',
        'Has_Reply': 'sum',
        'Positive_Reply': 'sum'
    }).reset_index()
    campaign_summary = campaign_summary.nlargest(5, 'Open Count')
    st.dataframe(
        campaign_summary.style.background_gradient(cmap='Blues'),
        use_container_width=True
    )
    if st.button("💡 AI Insights", key="boss_campaign_insight", help="Get campaign insights"):
        st.info(generate_insights(filtered_df, "Top Opens by Campaign"))

    # Engagement Breakdown
    st.markdown("### 📊 Engagement Breakdown")
    col1, col2 = st.columns([3, 2])
    with col1:
        engagement_data = filtered_df['Engagement'].value_counts().reset_index(name='Count')
        engagement_data.columns = ['Engagement', 'Count']
        fig_engagement = px.pie(
            engagement_data, 
            values='Count', 
            names='Engagement', 
            title="<b>Engagement Distribution</b>", 
            color_discrete_sequence=color_schemes['primary'],
            hole=0.4
        )
        fig_engagement.update_layout(
            font=dict(color='white'),
            paper_bgcolor='rgba(0,0,0,0)',
            title_font_size=18,
            title_x=0.5
        )
        fig_engagement.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_engagement, use_container_width=True)
    with col2:
        st.markdown('<div style="padding: 1rem;">', unsafe_allow_html=True)
        if st.button("💡 AI Insights", key="engagement_insight", help="Get engagement insights"):
            st.info("📊 **Engagement Insight**: The distribution of High (HE), Low (LE), and No (NO) engagement highlights campaign effectiveness. Focus on strategies that boost HE to improve overall ROI.")
        st.markdown('</div>', unsafe_allow_html=True)

    # Geographic Performance (if available)
    if 'latitude' in filtered_df.columns and 'longitude' in filtered_df.columns:
        st.markdown("### 🗺️ Geographic Performance")
        geo_data = filtered_df[filtered_df['Open Count'] > 0].groupby(['City', 'latitude', 'longitude']).agg({
            'Open Count': 'sum',
            'Click Count': 'sum'
        }).reset_index()
        if exclude_invalid:
            geo_data = geo_data[
                (geo_data['City'].notna()) &
                (geo_data['City'] != '') &
                (geo_data['City'] != '--') &
                (geo_data['City'] != 'Unknown')
            ]
        geo_data = geo_data.nlargest(5, 'Open Count')
        fig_geo = px.scatter_mapbox(
            geo_data,
            lat='latitude',
            lon='longitude',
            size='Open Count',
            color='Click Count',
            hover_name='City',
            hover_data={'Open Count': True, 'Click Count': True},
            title="<b>Top Cities by Engagement</b>",
            color_continuous_scale='Viridis',
            size_max=20,
            zoom=3
        )
        fig_geo.update_layout(
            mapbox_style="carto-darkmatter",
            font=dict(color='white'),
            paper_bgcolor='rgba(0,0,0,0)',
            title_font_size=18,
            title_x=0.5
        )
        st.plotly_chart(fig_geo, use_container_width=True)
        if st.button("💡 AI Insights", key="boss_geo_insight", help="Get geographic insights"):
            st.info(generate_insights(filtered_df, "Top Cities by Opens and Clicks"))

    # Key Takeaways
    st.markdown("### 🔑 Key Takeaways")
    takeaways = [
        f"🎯 **Campaign Reach**: {format_number(total_campaigns, show_full_numbers)} campaigns reached {format_number(unique_prospects, show_full_numbers)} unique prospects.",
        f"📈 **Engagement Metrics**: Achieved an open rate of {open_rate:.1f}% and click rate of {click_rate:.1f}%.",
        f"💬 **Reply Performance**: {format_number(total_replies, show_full_numbers)} total replies with {format_number(total_positive_replies, show_full_numbers)} positive replies and a reply rate of {reply_percentage:.1f}% (based on {format_number(total_brands, show_full_numbers)} unique brands).",
        f"🤖 **Bot Detection**: {format_number(len(filtered_df[filtered_df['Bot Check'] == 'Bot']), show_full_numbers)} bot interactions detected."
    ]
    for takeaway in takeaways:
        st.markdown(f"<div style='padding: 0.5rem;'>{takeaway}</div>", unsafe_allow_html=True)

# Enhanced Footer
st.markdown("""
<div class="custom-footer fade-in">
    <p>🚀 Powered by CamML Analytics | Designed for Data-Driven Success</p>
    <p>📧 Contact us at support@camml.com | © 2025</p>
</div>
""", unsafe_allow_html=True)