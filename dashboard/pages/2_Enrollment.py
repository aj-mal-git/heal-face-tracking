"""
Enrollment page — register employees and manage existing ones.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st

st.set_page_config(page_title="Enrollment", layout="wide")
st.title("Employee Enrollment")

from dashboard.components.api_client import APIClient

api_url = st.session_state.get("api_url", "http://localhost:8000")
client = APIClient(api_url)

tab_new, tab_manage = st.tabs(["Enroll New Employee", "Manage Employees"])

# ─── Enroll New Employee ─────────────────────────────────────────────────────
with tab_new:
    st.subheader("Register New Employee")
    st.info(
        "Take a clear, front-facing photo in good lighting. "
        "The system needs to see the full face."
    )

    col_form, col_preview = st.columns([1, 1])

    with col_form:
        with st.form("enroll_form", clear_on_submit=True):
            name = st.text_input("Full Name *", placeholder="John Doe")
            department = st.selectbox(
                "Department",
                ["", "Engineering", "HR", "Finance", "Operations",
                 "Security", "Management", "Sales", "IT", "Other"],
            )
            email = st.text_input("Email", placeholder="john.doe@heal.com")

            photo_source = st.radio("Photo Source", ["Upload File", "Use Camera"], horizontal=True)

            photo_file = None
            if photo_source == "Upload File":
                photo_file = st.file_uploader(
                    "Upload Photo", type=["jpg", "jpeg", "png"],
                    help="Front-facing photo, at least 200x200 pixels"
                )
            else:
                photo_file = st.camera_input("Take Photo")

            submitted = st.form_submit_button("Enroll Employee", type="primary", use_container_width=True)

    with col_preview:
        if photo_file:
            st.image(photo_file, caption="Preview", use_container_width=True)

    if submitted:
        if not name:
            st.error("Name is required.")
        elif not photo_file:
            st.error("Please provide a photo.")
        else:
            with st.spinner(f"Enrolling {name}..."):
                result = client.enroll_employee(
                    name=name,
                    department=department or "",
                    email=email or "",
                    photo_bytes=photo_file.getvalue(),
                    filename=getattr(photo_file, "name", "photo.jpg"),
                )

            if "error" in result:
                st.error(f"Enrollment failed: {result['error']}")
            else:
                st.success(f"Successfully enrolled **{result['name']}** (ID: {result['id']})")
                st.balloons()

# ─── Manage Employees ────────────────────────────────────────────────────────
with tab_manage:
    st.subheader("Registered Employees")

    if st.button("Refresh List"):
        st.rerun()

    employees = client.get_employees(active_only=False)

    if not employees:
        st.info("No employees enrolled yet.")
    else:
        # Stats
        active = sum(1 for e in employees if e["is_active"])
        col1, col2 = st.columns(2)
        col1.metric("Total Enrolled", len(employees))
        col2.metric("Active", active)

        st.divider()

        # Employee cards
        for emp in employees:
            status_icon = "🟢" if emp["is_active"] else "🔴"
            with st.expander(f"{status_icon} {emp['name']} | {emp.get('department', 'N/A')} | ID: {emp['id']}"):
                detail_col, action_col = st.columns([2, 1])

                with detail_col:
                    st.write(f"**Email:** {emp.get('email') or 'Not set'}")
                    st.write(f"**Department:** {emp.get('department') or 'Not set'}")
                    st.write(f"**Enrolled:** {emp.get('enrolled_at', '')[:10]}")
                    st.write(f"**Status:** {'Active' if emp['is_active'] else 'Inactive'}")

                with action_col:
                    # Re-enroll with new photo
                    new_photo = st.file_uploader(
                        "Update Photo",
                        type=["jpg", "jpeg", "png"],
                        key=f"re_enroll_{emp['id']}",
                    )
                    if new_photo and st.button("Update Face", key=f"btn_re_{emp['id']}"):
                        with st.spinner("Updating..."):
                            r = client.re_enroll(emp["id"], new_photo.getvalue(), new_photo.name)
                        if "error" in r:
                            st.error(r["error"])
                        else:
                            st.success("Face updated!")
                            st.rerun()

                    if emp["is_active"]:
                        if st.button("Deactivate", key=f"del_{emp['id']}", type="secondary"):
                            r = client.deactivate_employee(emp["id"])
                            st.warning(r.get("message", "Deactivated"))
                            st.rerun()
