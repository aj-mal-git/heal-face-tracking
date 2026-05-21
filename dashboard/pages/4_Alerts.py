"""
Alerts page — view unknown person detections with snapshots.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from datetime import datetime

import streamlit as st

st.set_page_config(page_title="Alerts", layout="wide")
st.title("Security Alerts")

from dashboard.components.api_client import APIClient

api_url = st.session_state.get("api_url", "http://localhost:8000")
client = APIClient(api_url)

# ─── Filter Bar ──────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    show_resolved = st.checkbox("Show Resolved Alerts", value=False)
with col2:
    if st.button("Refresh", use_container_width=True):
        st.rerun()
with col3:
    open_count = client.get_alert_count(resolved=False)
    resolved_count = client.get_alert_count(resolved=True)
    st.metric("Open Alerts", open_count)

# ─── Alert List ──────────────────────────────────────────────────────────────
with st.spinner("Loading alerts..."):
    alerts = client.get_alerts(resolved=show_resolved if not show_resolved else None)

if not alerts:
    if show_resolved:
        st.info("No alerts found.")
    else:
        st.success("No open alerts. All clear!")
else:
    st.caption(f"Showing {len(alerts)} alerts")
    st.divider()

    for alert in alerts:
        is_resolved = alert.get("resolved", False)
        status_icon = "✅" if is_resolved else "🚨"
        created_at = alert.get("created_at", "")[:19].replace("T", " ")

        with st.expander(
            f"{status_icon} Alert #{alert['id']} — {alert['alert_type']} — {created_at}",
            expanded=not is_resolved,
        ):
            img_col, info_col = st.columns([1, 1])

            with img_col:
                snapshot_url = client.get_snapshot_url(alert)
                try:
                    import httpx
                    resp = httpx.get(snapshot_url, timeout=5.0)
                    if resp.status_code == 200:
                        st.image(resp.content, caption="Snapshot", use_container_width=True)
                    else:
                        st.warning("Snapshot not available")
                except Exception:
                    st.warning("Could not load snapshot")

            with info_col:
                st.write(f"**Alert ID:** #{alert['id']}")
                st.write(f"**Type:** {alert['alert_type']}")
                st.write(f"**Detected:** {created_at}")
                st.write(f"**Status:** {'Resolved' if is_resolved else 'Open'}")

                if alert.get("notes"):
                    st.write(f"**Notes:** {alert['notes']}")

                if not is_resolved:
                    st.divider()
                    notes_input = st.text_area(
                        "Resolution Notes (optional)",
                        key=f"notes_{alert['id']}",
                        height=80,
                    )
                    if st.button("Mark as Resolved", key=f"resolve_{alert['id']}", type="primary"):
                        result = client.resolve_alert(alert["id"], notes=notes_input)
                        if "error" in result:
                            st.error(result["error"])
                        else:
                            st.success("Alert resolved!")
                            st.rerun()

# ─── Summary Stats ───────────────────────────────────────────────────────────
st.divider()
st.subheader("Alert Statistics")
stat_cols = st.columns(3)
stat_cols[0].metric("Total Open", open_count)
stat_cols[1].metric("Total Resolved", resolved_count)
total = open_count + resolved_count
resolution_rate = f"{resolved_count / total * 100:.0f}%" if total > 0 else "N/A"
stat_cols[2].metric("Resolution Rate", resolution_rate)
