#type: ignore
"""
Report History page — accessible to all authenticated, approved users.

Shows previously generated reports for the current user with:
  - Date range filter
  - Checkbox selection
  - Download buttons (PDF / CSV) via signed URLs
"""

import html
from io import BytesIO
import zipfile

import streamlit as st
from datetime import datetime, timedelta, timezone


def _fmt_ist(dt_str: str) -> str:
    """Convert a UTC ISO timestamp to a human-readable IST string."""
    if not dt_str:
        return "—"
    try:
        from datetime import datetime, timezone, timedelta
        dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        ist = dt.astimezone(timezone(timedelta(hours=5, minutes=30)))
        return ist.strftime("%b %d, %Y %I:%M %p IST")
    except Exception:
        return dt_str[:16].replace("T", " ")


def render():
    """Render the Report History page for the current user."""
    from storage import list_reports_for_user, list_reports_for_org, download_file

    user_id = st.session_state.get("user_id")
    org_id = st.session_state.get("organization_id")
    role = st.session_state.get("user_role", "viewer")

    from auth import is_authenticated, is_approved, render_pending_approval_screen  # local import to avoid cycles

    if not is_authenticated():
        st.error("Not authenticated.")
        return
    if not is_approved():
        render_pending_approval_screen()
        return

    # Header with close button
    _title_col, _close_col = st.columns([0.88, 0.12])
    with _title_col:
        st.markdown("## 📄 Report History")
    with _close_col:
        if role != "viewer":
            if st.button("✕ Close", key="report_history_close_btn", use_container_width=True):
                st.session_state.show_report_history = False
                # Restore previous page — viewers go to front page, others to upload area
                from auth import check_role
                if check_role(["viewer"]):
                    st.session_state.show_front_page = True
                else:
                    prev_upload = st.session_state.get("prev_page_show_upload_area", False)
                    prev_submitted = st.session_state.get("prev_page_files_submitted", False)
                    if prev_upload or prev_submitted:
                        st.session_state.show_upload_area = prev_upload
                        st.session_state.files_submitted = prev_submitted
                    else:
                        st.session_state.show_front_page = True
                st.rerun()

    st.markdown("---")

    # Date range filter
    fc1, fc2, fc3 = st.columns([1, 1, 2])
    with fc1:
        start_date = st.date_input(
            "From",
            value=datetime.now() - timedelta(days=90),
            key="rh_from",
        )
    with fc2:
        end_date = st.date_input("To", value=datetime.now(), key="rh_to")

    selected_editor_id = "All"
    if role == "viewer" and org_id:
        with fc3:
            from db_queries import fetch_all_profiles
            profiles = fetch_all_profiles(org_id=org_id)
            # Find all potential report generators (editors/admins)
            generators = [{"id": p["id"], "name": p["full_name"] or p["id"][:8]} for p in profiles if p["role"] in ("editor", "org_admin", "super_admin")]
            options = ["All"] + [g["name"] for g in generators]
            selected_name = st.selectbox("Editor", options=options, key="rh_creator_filter", label_visibility="collapsed")
            if selected_name != "All":
                selected_editor_id = next((g["id"] for g in generators if g["name"] == selected_name), "All")

    # Fetch reports based on role
    reports = []
    if org_id:
        try:
            if role in ("org_admin", "super_admin", "viewer"):
                # Org admin + viewer → all org reports
                all_reports = list_reports_for_org(org_id, limit=200)
            else:
                # Editor → own reports only
                all_reports = list_reports_for_user(org_id, user_id, limit=200)
            # Client-side date filter
            for r in all_reports:
                if selected_editor_id != "All" and r.get("user_id") != selected_editor_id:
                    continue
                gen_at = r.get("generated_at", "")
                if isinstance(gen_at, str) and len(gen_at) >= 10:
                    try:
                        r_date = datetime.fromisoformat(gen_at[:10]).date()
                        if start_date <= r_date <= end_date:
                            reports.append(r)
                    except (ValueError, TypeError):
                        reports.append(r)  # Include if date parsing fails
                else:
                    reports.append(r)
        except Exception as e:
            st.error(f"Failed to load reports: {e}")
    else:
        st.info("No organization assigned. Reports will appear once you're assigned to an organization.")
        return

    if not reports:
        st.info("No reports found for the selected date range.")
        st.caption(f"Showing reports from {start_date} to {end_date}.")
        return

    st.caption(f"**{len(reports)}** report(s) found")

    col_sel, col_mode, col_dl = st.columns([1, 3, 2])

    prev_select_all = st.session_state.get("rh_select_all_prev", False)

    with col_sel:
        select_all = st.checkbox("Select all", key="rh_select_all")
        if select_all:
            for i in range(len(reports)):
                st.session_state[f"rh_cb_{i}"] = True
        elif prev_select_all and not select_all:
            for i in range(len(reports)):
                st.session_state[f"rh_cb_{i}"] = False

    st.session_state["rh_select_all_prev"] = select_all

    with col_mode:
        download_mode = st.radio(
            "Download as",
            ["PDF", "CSV", "Both"],
            index=0,
            horizontal=True,
            key="rh_download_mode",
        )

    selected_indices = [
        i for i in range(len(reports))
        if st.session_state.get(f"rh_cb_{i}", False)
    ]

    include_pdf = download_mode in ("PDF", "Both")
    include_csv = download_mode in ("CSV", "Both")

    with col_dl:
        if not selected_indices or not (include_pdf or include_csv):
            st.caption("Select reports and download type")
        else:
            selected_reports = [reports[i] for i in selected_indices]

            # Case 1: single report, single format -> direct styled download button
            if len(selected_reports) == 1 and (include_pdf ^ include_csv):
                r = selected_reports[0]
                rname = r.get("report_name", "report")
                rname_base = os.path.splitext(str(rname))[0]
                safe_name = "".join(
                    c if c.isalnum() or c in "-_ " else "_"
                    for c in str(rname_base)
                ) or "report"

                if include_pdf and r.get("pdf_storage_path"):
                    pdf_bytes = download_file(r["pdf_storage_path"], silent=True)
                    if pdf_bytes:
                        st.download_button(
                            "Download PDF",
                            data=pdf_bytes,
                            file_name=f"{safe_name}.pdf",
                            mime="application/pdf",
                            use_container_width=True,
                            key="rh_download_pdf_btn",
                        )
                    else:
                        # Single concise warning only when the user explicitly requests a missing file
                        st.warning("⚠️ File not found in storage. It may have been deleted or moved.")
                elif include_csv and r.get("csv_storage_path"):
                    csv_bytes = download_file(r["csv_storage_path"], silent=True)
                    if csv_bytes:
                        st.download_button(
                            "Download CSV",
                            data=csv_bytes,
                            file_name=f"{safe_name}.csv",
                            mime="text/csv",
                            use_container_width=True,
                            key="rh_download_csv_btn",
                        )
                    else:
                        st.warning("⚠️ File not found in storage. It may have been deleted or moved.")
                else:
                    st.caption("No files available for the chosen type.")
            else:
                # Case 2: multiple reports OR both formats -> ZIP
                zip_buf = BytesIO()
                with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                    used_names: set[str] = set()
                    for r in selected_reports:
                        rname = r.get("report_name", "report")
                        rname_base = os.path.splitext(str(rname))[0]
                        safe_name = "".join(
                            c if c.isalnum() or c in "-_ " else "_"
                            for c in str(rname_base)
                        ) or "report"
                        if include_pdf and r.get("pdf_storage_path"):
                            pdf_bytes = download_file(r["pdf_storage_path"], silent=True) or None
                            if pdf_bytes:
                                pdf_name = f"{safe_name}.pdf"
                                # Avoid duplicate names inside the ZIP
                                while pdf_name in used_names:
                                    pdf_name = pdf_name.replace(".pdf", "_1.pdf")
                                used_names.add(pdf_name)
                                zf.writestr(pdf_name, pdf_bytes)
                        if include_csv and r.get("csv_storage_path"):
                            csv_bytes = download_file(r["csv_storage_path"], silent=True) or None
                            if csv_bytes:
                                csv_name = f"{safe_name}.csv"
                                while csv_name in used_names:
                                    csv_name = csv_name.replace(".csv", "_1.csv")
                                used_names.add(csv_name)
                                zf.writestr(csv_name, csv_bytes)
                zip_buf.seek(0)
                zip_bytes = zip_buf.getvalue()

                if zip_bytes:
                    company = (
                        (st.session_state.get("author_company") or "")
                        or (st.session_state.get("organization_name") or "")
                        or "Reports"
                    ).strip() or "Reports"
                    company_safe = company.replace(" ", "_")
                    date_str = datetime.now().strftime("%Y%m%d")
                    zip_name = f"{company_safe}_report_{date_str}.zip"
                    st.download_button(
                        "Download selected reports",
                        data=zip_bytes,
                        file_name=zip_name,
                        mime="application/zip",
                        use_container_width=True,
                        key="rh_zip_dl",
                    )
                else:
                    st.caption("No files available for the chosen type.")

    # Report list
    with st.container(height=500):
        for i, report in enumerate(reports):
            rname = report.get("report_name") or "Report"
            gen_at = report.get("generated_at") or ""
            gen_at_display = _fmt_ist(gen_at)

            pdf_path = report.get("pdf_storage_path")
            csv_path = report.get("csv_storage_path")
            indicators = []
            if pdf_path:
                indicators.append("PDF")
            if csv_path:
                indicators.append("CSV")
            type_str = " | ".join(indicators) if indicators else ""

            label = f"{html.escape(rname)} · {gen_at_display}"
            if type_str:
                label += f" [{type_str}]"

            st.checkbox(label, key=f"rh_cb_{i}")
