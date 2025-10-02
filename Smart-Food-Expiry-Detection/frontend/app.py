"""
Streamlit Frontend for Smart Food Expiry Detection System

This is the main dashboard interface that allows users to:
- Upload fridge images for food detection
- View expiry predictions and timelines
- Manage their food inventory
- Receive notifications about expiring items
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json
from datetime import datetime, timedelta
from PIL import Image
import io
import base64
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Configure Streamlit page
st.set_page_config(
    page_title="Smart Food Expiry Detection",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    
    .status-fresh {
        color: #28a745;
        font-weight: bold;
    }
    
    .status-warning {
        color: #ffc107;
        font-weight: bold;
    }
    
    .status-critical {
        color: #fd7e14;
        font-weight: bold;
    }
    
    .status-expired {
        color: #dc3545;
        font-weight: bold;
    }
    
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
        margin: 0.5rem 0;
    }
    
    .stAlert {
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'detected_items' not in st.session_state:
    st.session_state.detected_items = []

if 'user_inventory' not in st.session_state:
    st.session_state.user_inventory = []

if 'api_base_url' not in st.session_state:
    st.session_state.api_base_url = "http://localhost:8000"


def call_api(endpoint: str, method: str = "GET", data: dict = None, files: dict = None):
    """Helper function to call the FastAPI backend."""
    url = f"{st.session_state.api_base_url}{endpoint}"
    
    try:
        if method == "GET":
            response = requests.get(url, params=data)
        elif method == "POST":
            if files:
                response = requests.post(url, files=files, data=data)
            else:
                response = requests.post(url, json=data)
        elif method == "DELETE":
            response = requests.delete(url)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        response.raise_for_status()
        return response.json()
    
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to API server. Please ensure the backend is running on localhost:8000")
        return {"error": "Connection failed"}
    except requests.exceptions.HTTPError as e:
        st.error(f"❌ API Error: {e}")
        return {"error": str(e)}
    except Exception as e:
        st.error(f"❌ Unexpected error: {e}")
        return {"error": str(e)}


def display_status_badge(status: str) -> str:
    """Return HTML for status badge."""
    status_colors = {
        "fresh": ("🟢", "status-fresh"),
        "warning": ("🟡", "status-warning"), 
        "critical": ("🔴", "status-critical"),
        "expired": ("⛔", "status-expired")
    }
    
    emoji, css_class = status_colors.get(status.lower(), ("❓", ""))
    return f'<span class="{css_class}">{emoji} {status.title()}</span>'


def create_freshness_timeline(prediction_data: dict):
    """Create timeline visualization for freshness prediction."""
    if not prediction_data or 'best_prediction' not in prediction_data:
        return None
    
    best_pred = prediction_data['best_prediction']
    
    # Create timeline data
    dates = []
    freshness_values = []
    
    # Current freshness
    current_date = datetime.now()
    current_freshness = best_pred.get('current_freshness', 100)
    
    dates.append(current_date.strftime('%Y-%m-%d'))
    freshness_values.append(current_freshness)
    
    # Predicted values
    predictions = [
        (1, best_pred.get('predicted_freshness_1d')),
        (3, best_pred.get('predicted_freshness_3d')),
        (7, best_pred.get('predicted_freshness_7d'))
    ]
    
    for days, freshness in predictions:
        if freshness is not None:
            future_date = current_date + timedelta(days=days)
            dates.append(future_date.strftime('%Y-%m-%d'))
            freshness_values.append(freshness)
    
    # Create the plot
    fig = go.Figure()
    
    # Add freshness line
    fig.add_trace(go.Scatter(
        x=dates,
        y=freshness_values,
        mode='lines+markers',
        name='Predicted Freshness',
        line=dict(color='#2E8B57', width=3),
        marker=dict(size=8)
    ))
    
    # Add threshold lines
    fig.add_hline(y=50, line_dash="dash", line_color="orange", 
                  annotation_text="Warning Threshold (50%)")
    fig.add_hline(y=20, line_dash="dash", line_color="red", 
                  annotation_text="Spoilage Threshold (20%)")
    
    # Update layout
    fig.update_layout(
        title="Food Freshness Timeline",
        xaxis_title="Date",
        yaxis_title="Freshness (%)",
        yaxis=dict(range=[0, 100]),
        height=400,
        showlegend=True
    )
    
    return fig


def main_dashboard():
    """Main dashboard interface."""
    # Header
    st.markdown('<h1 class="main-header">🍎 Smart Food Expiry Detection</h1>', unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Choose a function:",
        ["🏠 Dashboard", "📸 Food Detection", "📊 Expiry Analysis", "🔔 Notifications", "⚙️ Settings"]
    )
    
    if page == "🏠 Dashboard":
        dashboard_home()
    elif page == "📸 Food Detection":
        food_detection_page()
    elif page == "📊 Expiry Analysis":
        expiry_analysis_page()
    elif page == "🔔 Notifications":
        notifications_page()
    elif page == "⚙️ Settings":
        settings_page()


def dashboard_home():
    """Home dashboard with overview metrics."""
    st.header("📊 Dashboard Overview")
    
    # Fetch statistics from API
    stats_data = call_api("/statistics")
    
    if "error" not in stats_data:
        # Display key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        stats = stats_data.get("statistics", {})
        
        with col1:
            st.metric(
                label="Total Foods in DB",
                value=stats.get("total_foods_in_db", 0)
            )
        
        with col2:
            st.metric(
                label="Tracked Items",
                value=stats.get("total_user_items", 0)
            )
        
        with col3:
            items_by_status = stats.get("items_by_status", {})
            expiring_count = items_by_status.get("warning", 0) + items_by_status.get("critical", 0)
            st.metric(
                label="Items Expiring Soon",
                value=expiring_count,
                delta=-expiring_count if expiring_count > 0 else 0
            )
        
        with col4:
            expired_count = items_by_status.get("expired", 0)
            st.metric(
                label="Expired Items",
                value=expired_count,
                delta=-expired_count if expired_count > 0 else 0
            )
    
    # Quick expiring items check
    st.subheader("⚠️ Items Expiring Soon")
    
    expiring_data = call_api("/expiring", data={"days_ahead": 5})
    
    if "error" not in expiring_data and expiring_data.get("items"):
        items_df = pd.DataFrame(expiring_data["items"])
        
        # Display as table with color coding
        for _, item in items_df.iterrows():
            days_remaining = item.get('days_remaining', 0)
            
            if days_remaining <= 0:
                status = "expired"
            elif days_remaining <= 1:
                status = "critical"
            elif days_remaining <= 3:
                status = "warning"
            else:
                status = "fresh"
            
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.write(f"**{item['food_name'].title()}**")
            
            with col2:
                st.markdown(display_status_badge(status), unsafe_allow_html=True)
            
            with col3:
                days_text = f"{int(days_remaining)} days" if days_remaining > 0 else "Expired"
                st.write(days_text)
    
    else:
        st.success("🎉 No items expiring soon!")
    
    # Recent activity (placeholder)
    st.subheader("📈 Recent Activity")
    
    # Create sample activity chart
    activity_dates = pd.date_range(
        start=datetime.now() - timedelta(days=7),
        end=datetime.now(),
        freq='D'
    )
    
    activity_data = {
        'Date': activity_dates,
        'Items Added': np.random.randint(0, 5, len(activity_dates)),
        'Items Consumed': np.random.randint(0, 3, len(activity_dates)),
        'Items Expired': np.random.randint(0, 2, len(activity_dates))
    }
    
    activity_df = pd.DataFrame(activity_data)
    
    fig = px.bar(
        activity_df.melt(id_vars=['Date'], var_name='Activity', value_name='Count'),
        x='Date',
        y='Count',
        color='Activity',
        title="Weekly Activity Summary",
        color_discrete_map={
            'Items Added': '#28a745',
            'Items Consumed': '#007bff',
            'Items Expired': '#dc3545'
        }
    )
    
    st.plotly_chart(fig, use_container_width=True)


def food_detection_page():
    """Food detection and image upload page."""
    st.header("📸 Food Detection")
    st.write("Upload an image of your fridge or food items to automatically detect what's inside.")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        help="Upload a clear image of your fridge contents or individual food items"
    )
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Detection options
        col1, col2 = st.columns(2)
        
        with col1:
            save_results = st.checkbox("Save annotated results", value=True)
        
        with col2:
            confidence_threshold = st.slider("Confidence threshold", 0.1, 1.0, 0.5, 0.1)
        
        # Detect button
        if st.button("🔍 Detect Food Items", type="primary"):
            with st.spinner("Analyzing image..."):
                # Prepare file for API call
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                data = {"save_results": save_results}
                
                # Call detection API
                result = call_api("/detect", method="POST", files=files, data=data)
                
                if "error" not in result and result.get("success"):
                    # Display results
                    st.success(f"✅ Detection completed! Found {result['total_items']} items.")
                    
                    # Show summary
                    st.subheader("📋 Detection Summary")
                    st.write(result.get("summary", "No summary available"))
                    
                    # Display detections in a table
                    if result.get("detections"):
                        detections_df = pd.DataFrame(result["detections"])
                        
                        # Format for display
                        display_df = detections_df[[
                            'food_name', 'confidence', 'class_name'
                        ]].copy()
                        
                        display_df['confidence'] = display_df['confidence'].round(3)
                        display_df.columns = ['Food Item', 'Confidence', 'Class']
                        
                        st.subheader("🍽️ Detected Items")
                        st.dataframe(display_df, use_container_width=True)
                        
                        # Store in session state
                        st.session_state.detected_items = result["detections"]
                        
                        # Add to inventory button
                        if st.button("➕ Add All Items to Inventory"):
                            add_detected_items_to_inventory()
                
                else:
                    st.error("❌ Detection failed. Please try again.")


def add_detected_items_to_inventory():
    """Add detected items to user inventory."""
    st.subheader("➕ Add Items to Inventory")
    
    if not st.session_state.detected_items:
        st.warning("No detected items to add.")
        return
    
    with st.form("add_items_form"):
        st.write("Configure details for detected items:")
        
        items_to_add = []
        
        for i, item in enumerate(st.session_state.detected_items):
            st.write(f"**{item['food_name'].title()}**")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                purchase_date = st.date_input(
                    f"Purchase date",
                    value=datetime.now().date(),
                    key=f"date_{i}"
                )
            
            with col2:
                storage = st.selectbox(
                    f"Storage location",
                    ["fridge", "room", "freezer"],
                    key=f"storage_{i}"
                )
            
            with col3:
                quantity = st.number_input(
                    f"Quantity",
                    min_value=1,
                    value=1,
                    key=f"quantity_{i}"
                )
            
            items_to_add.append({
                "food_name": item['food_name'],
                "purchase_date": purchase_date.strftime('%Y-%m-%d'),
                "storage_location": storage,
                "quantity": quantity
            })
        
        submit = st.form_submit_button("Add All Items", type="primary")
        
        if submit:
            success_count = 0
            
            for item_data in items_to_add:
                result = call_api("/expiry", method="POST", data=item_data)
                
                if "error" not in result:
                    success_count += 1
            
            if success_count == len(items_to_add):
                st.success(f"✅ Successfully added {success_count} items to inventory!")
            else:
                st.warning(f"⚠️ Added {success_count} out of {len(items_to_add)} items.")
            
            # Clear detected items
            st.session_state.detected_items = []
            
            # Rerun to refresh the page
            st.rerun()


def expiry_analysis_page():
    """Expiry analysis and prediction page."""
    st.header("📊 Expiry Analysis")
    
    # Manual item addition
    with st.expander("➕ Add New Item Manually", expanded=False):
        add_manual_item_form()
    
    # Freshness prediction
    st.subheader("🔮 Freshness Prediction")
    
    with st.form("freshness_prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            food_item = st.text_input("Food item name", placeholder="e.g., apple, milk, bread")
            purchase_date = st.date_input("Purchase date", value=datetime.now().date())
        
        with col2:
            temperature = st.number_input("Storage temperature (°C)", value=4.0)
            humidity = st.number_input("Storage humidity (%)", value=60.0, min_value=0.0, max_value=100.0)
        
        prediction_method = st.selectbox(
            "Prediction method",
            ["auto", "prophet", "arima", "linear"],
            help="Auto will choose the best available method"
        )
        
        predict_button = st.form_submit_button("🔮 Predict Freshness", type="primary")
        
        if predict_button and food_item:
            with st.spinner("Predicting freshness..."):
                prediction_data = {
                    "food_item": food_item,
                    "purchase_date": purchase_date.strftime('%Y-%m-%d'),
                    "storage_conditions": {
                        "temperature": temperature,
                        "humidity": humidity
                    },
                    "prediction_method": prediction_method
                }
                
                result = call_api("/freshness", method="POST", data=prediction_data)
                
                if "error" not in result and result.get("success"):
                    prediction = result["prediction"]
                    best_pred = prediction["best_prediction"]
                    
                    # Display results
                    st.subheader("📈 Prediction Results")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        current_freshness = best_pred.get('current_freshness', 0)
                        st.metric(
                            "Current Freshness",
                            f"{current_freshness:.1f}%",
                            delta=None
                        )
                    
                    with col2:
                        pred_3d = best_pred.get('predicted_freshness_3d', 0)
                        st.metric(
                            "Freshness in 3 days",
                            f"{pred_3d:.1f}%" if pred_3d else "N/A",
                            delta=f"{pred_3d - current_freshness:.1f}%" if pred_3d else None
                        )
                    
                    with col3:
                        pred_7d = best_pred.get('predicted_freshness_7d', 0)
                        st.metric(
                            "Freshness in 7 days",
                            f"{pred_7d:.1f}%" if pred_7d else "N/A",
                            delta=f"{pred_7d - current_freshness:.1f}%" if pred_7d else None
                        )
                    
                    # Display recommendation
                    st.info(f"📝 **Recommendation:** {prediction['recommendation']}")
                    
                    # Show timeline chart
                    timeline_fig = create_freshness_timeline(prediction)
                    if timeline_fig:
                        st.plotly_chart(timeline_fig, use_container_width=True)
                    
                    # Show model details
                    with st.expander("🔧 Model Details"):
                        st.json(best_pred)
                
                else:
                    st.error("❌ Freshness prediction failed. Please try again.")


def add_manual_item_form():
    """Form to manually add items to inventory."""
    with st.form("manual_item_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            food_name = st.text_input("Food item name", placeholder="e.g., apple, milk, bread")
            purchase_date = st.date_input("Purchase date", value=datetime.now().date())
        
        with col2:
            storage_location = st.selectbox("Storage location", ["fridge", "room", "freezer"])
            quantity = st.number_input("Quantity", min_value=1, value=1)
        
        submit = st.form_submit_button("Add Item", type="primary")
        
        if submit and food_name:
            item_data = {
                "food_name": food_name,
                "purchase_date": purchase_date.strftime('%Y-%m-%d'),
                "storage_location": storage_location,
                "quantity": quantity
            }
            
            result = call_api("/expiry", method="POST", data=item_data)
            
            if "error" not in result:
                st.success(f"✅ Added {food_name} to inventory!")
            else:
                st.error("❌ Failed to add item. Please try again.")


def notifications_page():
    """Notifications and alerts page."""
    st.header("🔔 Notifications")
    
    # Notification settings
    st.subheader("⚙️ Notification Settings")
    
    with st.form("notification_settings"):
        col1, col2 = st.columns(2)
        
        with col1:
            email = st.text_input("Email address", placeholder="your.email@example.com")
            notification_type = st.selectbox(
                "Notification type",
                ["email", "sms", "whatsapp", "telegram"]
            )
        
        with col2:
            phone = st.text_input("Phone number", placeholder="+1234567890")
            days_ahead = st.number_input("Days ahead to check", min_value=1, max_value=30, value=3)
        
        send_test = st.form_submit_button("📧 Send Test Notification", type="primary")
        
        if send_test:
            notification_data = {
                "user_email": email if email else None,
                "phone_number": phone if phone else None,
                "notification_type": notification_type,
                "days_ahead": days_ahead
            }
            
            result = call_api("/notify", method="POST", data=notification_data)
            
            if "error" not in result and result.get("success"):
                st.success("✅ Test notification sent successfully!")
                
                # Display notification content
                st.subheader("📧 Notification Content")
                st.text_area(
                    "Message content:",
                    value=result.get("notification_content", ""),
                    height=200,
                    disabled=True
                )
            
            else:
                st.error("❌ Failed to send notification.")
    
    # Current expiring items
    st.subheader("⚠️ Current Expiring Items")
    
    expiring_data = call_api("/expiring", data={"days_ahead": 7})
    
    if "error" not in expiring_data and expiring_data.get("items"):
        items_df = pd.DataFrame(expiring_data["items"])
        
        # Create chart of expiring items
        fig = px.bar(
            items_df,
            x='food_name',
            y='days_remaining',
            color='status',
            title="Items by Days Remaining",
            color_discrete_map={
                'fresh': '#28a745',
                'warning': '#ffc107',
                'critical': '#fd7e14',
                'expired': '#dc3545'
            }
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display detailed table
        display_df = items_df[[
            'food_name', 'purchase_date', 'expiry_date', 'days_remaining', 'status'
        ]].copy()
        
        display_df.columns = ['Food Item', 'Purchase Date', 'Expiry Date', 'Days Remaining', 'Status']
        st.dataframe(display_df, use_container_width=True)
    
    else:
        st.success("🎉 No items expiring soon!")


def settings_page():
    """Settings and configuration page."""
    st.header("⚙️ Settings")
    
    # API Configuration
    st.subheader("🔧 API Configuration")
    
    current_api_url = st.session_state.get('api_base_url', 'http://localhost:8000')
    
    new_api_url = st.text_input(
        "API Base URL",
        value=current_api_url,
        help="URL of the FastAPI backend server"
    )
    
    if st.button("Update API URL"):
        st.session_state.api_base_url = new_api_url
        st.success(f"✅ API URL updated to: {new_api_url}")
    
    # Test API connection
    if st.button("🔗 Test API Connection"):
        with st.spinner("Testing connection..."):
            health_check = call_api("/health")
            
            if "error" not in health_check:
                st.success("✅ API connection successful!")
                st.json(health_check)
            else:
                st.error("❌ API connection failed!")
    
    # Database management
    st.subheader("🗄️ Database Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 View Statistics"):
            stats = call_api("/statistics")
            if "error" not in stats:
                st.json(stats)
    
    with col2:
        if st.button("🔍 Search Foods"):
            search_query = st.text_input("Search for food items:")
            if search_query:
                search_results = call_api("/foods/search", data={"query": search_query})
                if "error" not in search_results:
                    st.dataframe(pd.DataFrame(search_results.get("results", [])))
    
    # System information
    st.subheader("ℹ️ System Information")
    
    system_info = {
        "Frontend": "Streamlit",
        "Backend": "FastAPI",
        "Computer Vision": "YOLOv8 (Ultralytics)",
        "Time Series": "Prophet / ARIMA",
        "Database": "SQLite",
        "Version": "1.0.0"
    }
    
    for key, value in system_info.items():
        st.write(f"**{key}:** {value}")


if __name__ == "__main__":
    main_dashboard()