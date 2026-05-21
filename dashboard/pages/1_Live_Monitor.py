"""
Live Monitor page — real-time CCTV feed with face detection overlays.
Polls /api/streams/frame at configurable interval.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import time
import streamlit as st

st.set_page_config(page_title="Live Monitor", layout="wide")
st.title("Live Monitor")

from dashboard.components.api_client import APIClient

api_url = st.session_state.get("api_url", "http://localhost:8000")
client = APIClient(api_url)

# ─── Stream Controls ─────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Stream Controls")
    stream_url = st.text_input("Camera URL", value="0", help="RTSP URL or webcam index (0, 1, 2)")
    camera_id = st.text_input("Camera ID", value="main")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start", use_container_width=True, type="primary"):
            result = client.start_stream(stream_url, camera_id)
            if "error" in result:
                st.error(result["error"])
            else:
                st.success("Stream started!")
                st.rerun()
    with col2:
        if st.button("Stop", use_container_width=True):
            client.stop_stream()
            st.rerun()

    st.divider()
    refresh_ms = st.slider("Refresh interval (ms)", 200, 2000, 500, step=100)
    st.caption(f"~{1000 // refresh_ms} fps display")

# ─── Status Bar ──────────────────────────────────────────────────────────────
status = client.get_stream_status()

stat_cols = st.columns(4)
stat_cols[0].metric("Status", "Running" if status["running"] else "Stopped")
stat_cols[1].metric("Camera", status.get("camera_id") or "—")
stat_cols[2].metric("Frames", status.get("frame_count", 0))
stat_cols[3].metric("Active Faces", status.get("active_tracks", 0))

# ─── Live Feed ───────────────────────────────────────────────────────────────
st.divider()
main_col, info_col = st.columns([3, 1])

with main_col:
    frame_placeholder = st.empty()

with info_col:
    st.subheader("Detected Faces")
    faces_placeholder = st.empty()
    st.divider()
    alert_count = client.get_alert_count(resolved=False)
    st.metric("Open Alerts", alert_count)
    if alert_count > 0:
        st.warning(f"{alert_count} unresolved alerts")

# ─── Auto-refresh Loop ───────────────────────────────────────────────────────
if status["running"]:
    frame_bytes = client.get_latest_frame()
    if frame_bytes:
        frame_placeholder.image(frame_bytes, channels="BGR", use_container_width=True)
    else:
        frame_placeholder.info("Waiting for first frame...")

    # Update face list
    data = client.get_frame_b64()
    if data and data.get("results"):
        face_info = []
        for r in data["results"]:
            icon = "🔴" if r["is_unknown"] else "🟢"
            face_info.append(
                f"{icon} **{r['name']}**  \nTrack #{r['track_id']} | {r['confidence']:.2f}"
            )
        faces_placeholder.markdown("\n\n".join(face_info))
    else:
        faces_placeholder.caption("No faces detected")

    time.sleep(refresh_ms / 1000)
    st.rerun()
else:
    frame_placeholder.info(
        "Stream is not running. Enter a camera URL and click **Start**."
    )
