"""
Streamlit Dashboard for Train Crowd Density Detection and Prediction.
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.detector import CrowdDetector
from src.predictor import OccupancyPredictor
from src.config import MAX_CAPACITY, HISTORY_FILE


# Page configuration
st.set_page_config(
    page_title="Train Crowd Density Monitor",
    page_icon="🚆",
    layout="wide"
)

st.title("🚆 Train Crowd Density Detection & Prediction")
st.markdown("---")


@st.cache_resource
def load_detector():
    """Load YOLO detector (cached)."""
    return CrowdDetector()


@st.cache_resource
def load_predictor():
    """Load and train predictor (cached)."""
    predictor = OccupancyPredictor()
    try:
        predictor.load_history()
        predictor.train()
    except FileNotFoundError:
        st.warning("No history file found. Run generate_mock_history.py first.")
    return predictor


# Sidebar
st.sidebar.header("⚙️ Settings")
max_cap = st.sidebar.slider("Max Capacity", 50, 500, MAX_CAPACITY, step=10)

tab1, tab2, tab3 = st.tabs([
    "📸 Live Detection",
    "📊 Occupancy Prediction",
    "📈 Historical Data"
])


# ── Tab 1: Live Detection ──
with tab1:
    st.header("Upload an Image for Detection")

    uploaded_file = st.file_uploader(
        "Choose a train/platform image...",
        type=["jpg", "jpeg", "png", "bmp"]
    )

    if uploaded_file is not None:
        # Read image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original Image")
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)

        # Run detection
        with st.spinner("Detecting people..."):
            detector = load_detector()
            detector.max_capacity = max_cap
            result = detector.process_image(image)

        detection = result['detection']
        occupancy = result['occupancy']
        annotated = result['annotated_image']

        with col2:
            st.subheader("Detection Result")
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

        # Metrics row
        st.markdown("---")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("👥 People", f"{occupancy['count']}detected / {occupancy['count']}estimated")
        m2.metric("📊 Occupancy", f"{occupancy['occupancy_percent']}%")
        m3.metric("🏷️ Density Level", occupancy['density_level'])
        m4.metric("🚃 Max Capacity", occupancy['max_capacity'])

        # Density color indicator
        level = occupancy['density_level']
        color_map = {
            "Low": "🟢", "Medium": "🟡",
            "High": "🟠", "Overcrowded": "🔴"
        }
        emoji = color_map.get(level, "⚪")
        st.markdown(f"### Status: {emoji} **{level}**")
    else:
        st.info("👆 Upload an image of a train carriage or platform to begin detection.")


# ── Tab 2: Occupancy Prediction ──
with tab2:
    st.header("Predict Crowd Density")

    predictor = load_predictor()

    if predictor.is_trained:
        pcol1, pcol2 = st.columns(2)

        with pcol1:
            st.subheader("Single Prediction")
            pred_hour = st.slider("Hour of Day", 5, 23, datetime.now().hour)
            pred_dow = st.selectbox(
                "Day of Week",
                options=list(range(7)),
                format_func=lambda x: [
                    "Monday", "Tuesday", "Wednesday", "Thursday",
                    "Friday", "Saturday", "Sunday"
                ][x],
                index=datetime.now().weekday()
            )
            pred_month = st.slider("Month", 1, 12, datetime.now().month)

            if st.button("🔮 Predict"):
                pred = predictor.predict(pred_hour, pred_dow, pred_month)
                st.success(f"Predicted Passengers: **{pred['predicted_count']}**")
                st.info(f"Density Level: **{pred['density_level']}** "
                        f"({pred['occupancy_percent']}%)")

        with pcol2:
            st.subheader("Full Day Forecast")
            fc_dow = st.selectbox(
                "Forecast Day",
                options=list(range(7)),
                format_func=lambda x: [
                    "Monday", "Tuesday", "Wednesday", "Thursday",
                    "Friday", "Saturday", "Sunday"
                ][x],
                index=datetime.now().weekday(),
                key="forecast_dow"
            )

            if st.button("📈 Generate Forecast"):
                day_preds = predictor.predict_day(fc_dow)
                df_pred = pd.DataFrame(day_preds)
                st.line_chart(
                    df_pred.set_index('hour')['predicted_count'],
                    use_container_width=True
                )
                st.dataframe(
                    df_pred[['hour', 'predicted_count', 'occupancy_percent', 'density_level']],
                    use_container_width=True
                )
    else:
        st.warning("Predictor could not be trained. Make sure history.csv exists.")


# ── Tab 3: Historical Data ──
with tab3:
    st.header("Historical Passenger Data")

    if os.path.exists(HISTORY_FILE):
        df_hist = pd.read_csv(HISTORY_FILE)
        df_hist['date'] = pd.to_datetime(df_hist['date'])

        st.subheader("Data Overview")
        st.write(f"Total records: **{len(df_hist):,}**")
        st.write(f"Date range: **{df_hist['date'].min().date()}** to "
                 f"**{df_hist['date'].max().date()}**")

        # Average by hour
        st.subheader("Average Passengers by Hour")
        hourly_avg = df_hist.groupby('hour')['passenger_count'].mean()
        st.bar_chart(hourly_avg, use_container_width=True)

        # Weekday vs Weekend
        st.subheader("Weekday vs Weekend")
        day_type = df_hist.copy()
        day_type['type'] = day_type['is_weekend'].map({0: 'Weekday', 1: 'Weekend'})
        comparison = day_type.groupby(['hour', 'type'])['passenger_count'].mean().unstack()
        st.line_chart(comparison, use_container_width=True)

        # Raw data viewer
        with st.expander("📋 View Raw Data"):
            st.dataframe(df_hist.head(100), use_container_width=True)
    else:
        st.warning("No historical data found. Run generate_mock_history.py first.")


# Footer
st.markdown("---")
st.markdown(
    "*Built with YOLOv8, Streamlit, and scikit-learn* | "
    "Train Crowd Density Detection System"
)