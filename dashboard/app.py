"""
HEAL Security Dashboard — Streamlit entry point.
Run with: streamlit run dashboard/app.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

st.set_page_config(
    page_title="HEAL Security Dashboard",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🏢 HEAL Security")
    st.caption("Face Detection & Tracking System")
    st.divider()

    # API connection
    if "api_url" not in st.session_state:
        st.session_state.api_url = "http://localhost:8000"

    api_url = st.text_input("API Server", value=st.session_state.api_url)
    st.session_state.api_url = api_url

    # Health check
    from dashboard.components.api_client import APIClient
    client = APIClient(api_url)
    health = client.health()

    if health.get("status") == "ok":
        st.success("API Connected")
        st.metric("Enrolled Faces", health.get("enrolled_faces", 0))
        stream_active = health.get("stream_active", False)
        if stream_active:
            st.success("Stream Active")
        else:
            st.warning("No Active Stream")
    else:
        st.error("API Unreachable")
        st.caption(f"Make sure API is running at {api_url}")

    st.divider()
    st.caption("Navigate using the pages in the sidebar above.")

# ─── Home Page ───────────────────────────────────────────────────────────────
st.title("HEAL Face Recognition System")

col1, col2, col3, col4 = st.columns(4)

health = client.health()
with col1:
    st.metric("Enrolled Employees", health.get("enrolled_faces", 0))
with col2:
    status = client.get_stream_status()
    st.metric("Active Tracks", status.get("active_tracks", 0))
with col3:
    alert_count = client.get_alert_count(resolved=False)
    st.metric("Open Alerts", alert_count, delta=None)
with col4:
    st.metric("Stream Status", "Active" if status.get("running") else "Inactive")

st.divider()

st.markdown("""
### Quick Navigation

| Page | Description |
|------|-------------|
| **Live Monitor** | Real-time CCTV feed with face detection overlays |
| **Enrollment** | Register new employees and manage existing ones |
| **Attendance Log** | View daily attendance and event history |
| **Alerts** | Review and resolve unknown person detections |

Use the **sidebar** to navigate between pages.
""")
