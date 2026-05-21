"""
Attendance Log page — view daily attendance and event history.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from datetime import date, timedelta

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Attendance Log", layout="wide")
st.title("Attendance Log")

from dashboard.components.api_client import APIClient

api_url = st.session_state.get("api_url", "http://localhost:8000")
client = APIClient(api_url)

tab_summary, tab_events = st.tabs(["Daily Summary", "Event History"])

# ─── Daily Summary ────────────────────────────────────────────────────────────
with tab_summary:
    col1, col2 = st.columns([2, 1])
    with col1:
        selected_date = st.date_input("Date", value=date.today())
    with col2:
        if st.button("Refresh", use_container_width=True):
            st.rerun()

    with st.spinner("Loading attendance..."):
        summary = client.get_attendance(target_date=str(selected_date))

    if not summary:
        st.info(f"No attendance data for {selected_date}.")
    else:
        st.metric("Employees Present", len(summary))
        st.divider()

        df = pd.DataFrame(summary)
        # Format datetime columns
        for col in ["first_seen", "last_seen"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col]).dt.strftime("%H:%M:%S")

        st.dataframe(
            df[["name", "department", "first_seen", "last_seen", "appearances"]].rename(
                columns={
                    "name": "Name",
                    "department": "Department",
                    "first_seen": "First Seen",
                    "last_seen": "Last Seen",
                    "appearances": "Detections",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        # Bar chart
        if len(df) > 0:
            st.subheader("Detections per Employee")
            chart_df = df.set_index("name")["appearances"]
            st.bar_chart(chart_df)

# ─── Event History ────────────────────────────────────────────────────────────
with tab_events:
    st.subheader("Detailed Event Log")

    filter_col1, filter_col2, filter_col3 = st.columns(3)
    with filter_col1:
        from_date = st.date_input("From", value=date.today() - timedelta(days=1), key="ev_from")
    with filter_col2:
        to_date = st.date_input("To", value=date.today(), key="ev_to")
    with filter_col3:
        event_type_filter = st.selectbox(
            "Event Type", ["All", "recognized", "unknown"], key="ev_type"
        )

    employees = client.get_employees()
    emp_options = {"All": None}
    emp_options.update({e["name"]: e["id"] for e in employees})
    selected_emp_name = st.selectbox("Employee", list(emp_options.keys()))
    selected_emp_id = emp_options[selected_emp_name]

    with st.spinner("Loading events..."):
        events = client.get_events(
            employee_id=selected_emp_id,
            event_type=event_type_filter if event_type_filter != "All" else None,
            from_dt=str(from_date) + "T00:00:00",
            to_dt=str(to_date) + "T23:59:59",
            limit=500,
        )

    if not events:
        st.info("No events found for the selected filters.")
    else:
        st.caption(f"Showing {len(events)} events")
        df = pd.DataFrame(events)
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")

        # Color unknown events
        display_cols = ["timestamp", "employee_name", "event_type", "camera_id", "confidence", "track_id"]
        available_cols = [c for c in display_cols if c in df.columns]

        st.dataframe(
            df[available_cols].rename(columns={
                "timestamp": "Time",
                "employee_name": "Name",
                "event_type": "Type",
                "camera_id": "Camera",
                "confidence": "Confidence",
                "track_id": "Track ID",
            }),
            use_container_width=True,
            hide_index=True,
        )
