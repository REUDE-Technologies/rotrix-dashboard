#type: ignore
"""
Organization Admin Dashboard for Rotrix Dashboard.

Accessible to users with role = 'admin'. Provides org-scoped management:
  - Tab 0: Pending Approvals (own org only, quota-enforced)
  - Tab 1: Overview (org KPIs + charts)
  - Tab 2: Members (manage org users)
  - Tab 3: Activity (org-scoped usage events)
"""

import html
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta, timezone


def _format_ist(dt_str):
    if not dt_str or dt_str == "—":
        return "—"
    try:
        dt = pd.to_datetime(dt_str)
        if dt.tzinfo is None:
            dt = dt.tz_localize("UTC")
        return dt.tz_convert("Asia/Kolkata").strftime("%b %d, %Y %H:%M (IST)")
    except Exception:
        return str(dt_str)[:16].replace("T", " ")


def _use_local_auth() -> bool:
    from auth import USE_LOCAL_AUTH
    return USE_LOCAL_AUTH


def _get_supabase():
    """Get the Supabase client (anon, current user session)."""
    if _use_local_auth():
        return None
    from auth import get_supabase
    return get_supabase()


def _get_supabase_service():
    """Get the Supabase service-role client for admin mutations (bypasses RLS)."""
    if _use_local_auth():
        return None
    from auth import get_supabase_service
    return get_supabase_service()


def _require_admin():
    """Check that the current user is an org_admin."""
    role = st.session_state.get("user_role", "")
    if role != "org_admin":
        st.error("🚫 Access denied. Only organization administrators can access this page.")
        st.stop()


def _get_org_context():
    """Return (org_id, org_name) for the current admin."""
    org_id = st.session_state.get("organization_id")
    org_name = st.session_state.get("organization_name", "My Organization")
    return org_id, org_name


# ======================================================================
# MAIN RENDER FUNCTION
# ======================================================================
def render():
    """Render the org admin dashboard."""
    _require_admin()
    org_id, org_name = _get_org_context()

    if not org_id:
        st.error("⚠️ No organization assigned. Please contact your Rotrix administrator.")
        st.stop()

    safe_org_name = html.escape(org_name)
    # Close button to exit org admin dashboard and return to previous page (analysis/upload)
    _title_col, _close_col = st.columns([0.88, 0.12])
    with _title_col:
        st.markdown(f"## 🛡️ {safe_org_name} — Admin Dashboard")
    with _close_col:
        if st.button("✕ Close", key="org_admin_close_btn", use_container_width=True):
            st.session_state.show_author_form = False
            st.session_state.show_upload_area = st.session_state.get("prev_page_show_upload_area", True)
            st.session_state.files_submitted = st.session_state.get("prev_page_files_submitted", False)
            st.rerun()
    # st.markdown("---")

    tab0, tab1, tab2, tab3 = st.tabs([
        "📈 Activity", "📊 Overview", "👥 Members", "📄 Reports"
    ])

    with tab0:
        _render_activity(org_id)
    with tab1:
        _render_overview(org_id, org_name)
    with tab2:
        _render_members(org_id)
    with tab3:
        _render_reports(org_id)


# ======================================================================
# TAB 0: PENDING APPROVALS (ORG-SCOPED)
# ======================================================================
def _render_pending_approvals(org_id: str):
    """Show profiles awaiting approval within this org with approve/reject actions."""
    from auth import get_pending_profiles, approve_profile, reject_profile, get_org_user_count, get_org_max_users

    st.markdown("### 🔔 Pending Profile Approvals")

    # Show quota usage
    current_count = get_org_user_count(org_id)
    max_users = get_org_max_users(org_id)
    remaining = max(0, max_users - current_count)

    qc1, qc2, qc3 = st.columns(3)
    qc1.metric("✅ Approved Users", current_count)
    qc2.metric("📋 User Quota", max_users)
    qc3.metric("🆓 Slots Remaining", remaining)

    if remaining == 0:
        st.warning(f"⚠️ Your organization has reached the maximum user limit ({max_users}). Contact your Rotrix administrator to increase the quota.")

    st.markdown("---")

    pending = get_pending_profiles(org_id=org_id)

    if not pending:
        st.success("✅ No pending approvals — all clear!")
        return

    st.info(f"**{len(pending)}** profile(s) awaiting your review.")

    for i, profile in enumerate(pending):
        org_data = profile.get("organizations")
        org_name_display = org_data.get("name", "—") if isinstance(org_data, dict) else "Not specified"
        name = profile.get("full_name", "Unknown")
        email = profile.get("email", "")
        role = profile.get("role", "viewer")
        role_display = role.replace("_", " ").title()
        safe_name = html.escape(str(name))
        safe_email = html.escape(str(email))
        safe_role_display = html.escape(role_display)

        with st.container():
            st.markdown(f"""
            <div style="
                padding: 1rem 1.25rem;
                background: #fffbeb;
                border-radius: 12px;
                border: 1px solid #fbbf24;
                margin-bottom: 0.75rem;
            ">
                <div style="font-weight: 600; font-size: 1rem; color: #0f172a;">👤 {safe_name}</div>
                <div style="font-size: 0.85rem; color: #64748b; margin-top: 2px;">✉️ {safe_email}</div>
                <div style="font-size: 0.85rem; color: #64748b; margin-top: 2px;">🔖 Requested role: {safe_role_display}</div>
            </div>
            """, unsafe_allow_html=True)

            ac1, ac2, _ = st.columns([1, 1, 3])
            with ac1:
                if st.button("✅ Approve", key=f"org_approve_{profile['id']}_{i}", type="primary", use_container_width=True):
                    success, msg = approve_profile(profile["id"])
                    if success:
                        st.toast(f"Approved {name}!", icon="✅")
                        st.rerun()
                    else:
                        st.error(msg)
            with ac2:
                if st.button("❌ Reject", key=f"org_reject_{profile['id']}_{i}", use_container_width=True):
                    success, msg = reject_profile(profile["id"])
                    if success:
                        st.toast(f"Rejected {name}.", icon="❌")
                        st.rerun()
                    else:
                        st.error(msg)

            st.markdown("---")


# ======================================================================
# TAB 1: OVERVIEW (ORG-SCOPED)
# ======================================================================
def _render_overview(org_id: str, org_name: str):
    """Org KPIs, status indicators, member breakdown, recent activity, and usage charts."""
    from auth import get_org_user_count, get_org_max_users

    service = _get_supabase_service()
    if not _use_local_auth() and not service:
        st.error("Server configuration error.")
        return

    safe_org = html.escape(org_name)

    # ── Shared plotly styling ──
    _plotly_layout = dict(
        font=dict(family="Inter, -apple-system, sans-serif", size=15, color="#000000"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FAFCFE",
        margin=dict(t=48, b=24, l=48, r=24),
        title_font=dict(size=18, color="#000000", family="Inter, sans-serif"),
        legend=dict(font=dict(size=14, color="#000000"), orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
        xaxis=dict(gridcolor="#E2E8F0", linecolor="#000000", linewidth=1.5, showline=True, title_font=dict(size=14, color="#000000"), tickfont=dict(size=13, color="#000000")),
        yaxis=dict(gridcolor="#E2E8F0", linecolor="#000000", linewidth=1.5, showline=True, title_font=dict(size=14, color="#000000"), tickfont=dict(size=13, color="#000000")),
    )
    _color_seq = ["#1B6CA8", "#0A2E42", "#B8941F", "#10b981", "#8B5CF6", "#EC4899"]

    # ── Fetch KPIs ──
    current_users = get_org_user_count(org_id)
    max_users = get_org_max_users(org_id)

    _local = _use_local_auth()
    if _local:
        import db_queries as dbq
        total_files = dbq.count_events_where(event_type="file_uploaded", org_id=org_id)
        total_reports = dbq.count_reports_where(org_id=org_id)
        pending_count = dbq.count_profiles_where(org_id=org_id, profile_status="pending_approval")
        thirty_days_ago = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        active_users_30d = len(dbq.get_active_user_ids(thirty_days_ago, org_id=org_id))
        week_ago = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
        reports_this_week = dbq.count_reports_where(org_id=org_id, generated_after=week_ago)
    else:
        try:
            files_result = service.table("usage_events").select("id", count="exact").eq("organization_id", org_id).eq("event_type", "file_uploaded").execute()
            total_files = files_result.count or 0
        except Exception:
            total_files = 0

        try:
            reports_result = service.table("report_metadata").select("id", count="exact").eq("organization_id", org_id).execute()
            total_reports = reports_result.count or 0
        except Exception:
            total_reports = 0

        try:
            pending_result = service.table("profiles").select("id", count="exact") \
                .eq("organization_id", org_id).eq("profile_status", "pending_approval").execute()
            pending_count = pending_result.count or 0
        except Exception:
            pending_count = 0

        # Active users (last 30 days)
        try:
            thirty_days_ago = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
            active = service.table("usage_events") \
                .select("user_id") \
                .eq("organization_id", org_id) \
                .eq("event_type", "login") \
                .gte("created_at", thirty_days_ago) \
                .execute()
            active_users_30d = len(set(e["user_id"] for e in (active.data or [])))
        except Exception:
            active_users_30d = 0

        # Reports this week
        try:
            week_ago = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
            rw = service.table("report_metadata").select("id", count="exact") \
                .eq("organization_id", org_id).gte("generated_at", week_ago).execute()
            reports_this_week = rw.count or 0
        except Exception:
            reports_this_week = 0

    # ── KPI Row ──
    st.markdown(f"### 📊 {safe_org} — Overview")
    st.markdown("""
        <style>
        div[data-testid="stMetric"] {
            height: 95px !important;
            display: block !important;
        }
        div[data-testid="stMetricValue"] {
            display: inline-block !important;
            margin-right: 12px !important;
            vertical-align: baseline !important;
        }
        div[data-testid="stMetricDelta"] {
            display: inline-flex !important;
            vertical-align: baseline !important;
        }
        </style>
    """, unsafe_allow_html=True)
    # KPI metrics (members/quota removed)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📁 Files", total_files)
    col2.metric("📄 Reports", total_reports, delta=f"+{reports_this_week} this week" if reports_this_week else None)
    col3.metric("🔔 Pending", pending_count)
    col4.metric("🟢 Active (30d)", active_users_30d)

    # ── Status Indicator Cards ──
    s1, s2, s3 = st.columns(3)
    with s1:
        _badge_color = "#ef4444" if pending_count > 0 else "#22c55e"
        _badge_label = f"{pending_count} pending" if pending_count > 0 else "All clear"
        st.markdown(f"""
        <div style="padding: 0.75rem 1rem; border-radius: 10px; border: 1px solid #e2e8f0;
                    background: linear-gradient(135deg, #fefce8, #fef9c3);
                    height: 95px; box-sizing: border-box; display: flex; flex-direction: column; justify-content: space-between;">
            <div style="font-size: 0.8rem; color: #92400e; font-weight: 600;">🔔 Approval Queue</div>
            <div style="font-size: 1.4rem; font-weight: 700; color: #78350f;">{pending_count}
                <span style="font-size: 0.75rem; color: {_badge_color}; font-weight: 500;">{_badge_label}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with s2:
        _adoption = round(active_users_30d / max(current_users, 1) * 100)
        _adopt_color = "#22c55e" if _adoption >= 50 else ("#f59e0b" if _adoption >= 25 else "#ef4444")
        st.markdown(f"""
        <div style="padding: 0.75rem 1rem; border-radius: 10px; border: 1px solid #e2e8f0;
                    background: linear-gradient(135deg, #ecfdf5, #d1fae5);
                    height: 95px; box-sizing: border-box; display: flex; flex-direction: column; justify-content: space-between;">
            <div style="font-size: 0.8rem; color: #065f46; font-weight: 600;">📈 Adoption Rate</div>
            <div style="font-size: 1.4rem; font-weight: 700; color: {_adopt_color};">{_adoption}%
                <span style="font-size: 0.75rem; color: #6b7280; font-weight: 400;">of members active</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with s3:
        _reports_per_user = round(total_reports / max(current_users, 1), 1)
        st.markdown(f"""
        <div style="padding: 0.75rem 1rem; border-radius: 10px; border: 1px solid #e2e8f0;
                    background: linear-gradient(135deg, #eff6ff, #dbeafe);
                    height: 95px; box-sizing: border-box; display: flex; flex-direction: column; justify-content: space-between;">
            <div style="font-size: 0.8rem; color: #1e40af; font-weight: 600;">📄 Reports Per Member</div>
            <div style="font-size: 1.4rem; font-weight: 700; color: #1e3a5f;">{_reports_per_user}
                <span style="font-size: 0.75rem; color: #6b7280; font-weight: 400;">avg reports/member</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Charts: Usage + Role Distribution ──
    ch1, ch2 = st.columns(2)

    # Usage chart
    with ch1:
        try:
            if _local:
                import db_queries as dbq
                events_data = dbq.fetch_usage_events("1970-01-01", "9999-12-31", org_id=org_id)
            else:
                events = service.table("usage_events") \
                    .select("event_type, created_at") \
                    .eq("organization_id", org_id) \
                    .order("created_at") \
                    .execute()
                events_data = events.data or []
            if events_data:
                df = pd.DataFrame(events_data)
                df["created_at"] = pd.to_datetime(df["created_at"])
                # Bucket into hourly frames for hh:mm display
                df["time_frame_hhmm"] = df["created_at"].dt.floor("h")
                hourly = df.groupby(["time_frame_hhmm", "event_type"]).size().reset_index(
                    name="count"
                )

                fig = px.bar(
                    hourly,
                    x="time_frame_hhmm",
                    y="count",
                    color="event_type",
                    title="Usage Over Time",
                    labels={"time_frame_hhmm": "", "count": "Events", "event_type": ""},
                    color_discrete_sequence=_color_seq,
                )
                # Apply base layout, remove overlapping axis/legend titles, and format x-axis as hh:mm
                fig.update_layout(
                    **_plotly_layout,
                    height=340,
                    xaxis_title="",
                    legend_title_text="",
                )
                fig.update_xaxes(tickformat="%H:%M")
                fig.update_traces(marker_line_color="#0A2E42", marker_line_width=1)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No usage data yet for your organization.")
        except Exception as e:
            st.warning(f"Could not load usage chart: {e}")

    # Role distribution
    with ch2:
        try:
            if _local:
                import db_queries as dbq
                members_data = dbq.fetch_all_profiles(org_id=org_id, profile_status="approved")
            else:
                members = service.table("profiles") \
                    .select("role") \
                    .eq("organization_id", org_id) \
                    .eq("profile_status", "approved") \
                    .execute()
                members_data = members.data or []
            if members_data:
                roles_df = pd.DataFrame(members_data)
                roles_df["role"] = roles_df["role"].str.replace("_", " ").str.title()
                counts = roles_df["role"].value_counts().reset_index()
                counts.columns = ["Role", "Count"]
                fig2 = px.pie(
                    counts, values="Count", names="Role",
                    title="Role Distribution",
                    color_discrete_sequence=_color_seq,
                    hole=0.45,
                )
                fig2.update_layout(**_plotly_layout, height=340, showlegend=True)
                fig2.update_traces(textposition="inside", textinfo="label+percent", textfont_size=12)
                st.plotly_chart(fig2, use_container_width=True)
        except Exception:
            pass

    st.markdown("---")

    # ── Members Table ──
    st.markdown("### 👥 Member Breakdown")
    try:
        if _local:
            import db_queries as dbq
            all_members_data = dbq.fetch_all_profiles(org_id=org_id, order_desc=False)
            # Sort by name mostly
            all_members_data = sorted(all_members_data, key=lambda x: (x.get("full_name") or "").lower())
        else:
            all_members = service.table("profiles") \
                .select("full_name, email, role, is_active, profile_status, last_login") \
                .eq("organization_id", org_id) \
                .order("full_name") \
                .execute()
            all_members_data = all_members.data or []
        if all_members_data:
            member_rows = []
            for m in all_members_data:
                role_emoji = {"super_admin": "👑", "org_admin": "🛡️", "editor": "✏️", "viewer": "👁️"}.get(m.get("role", ""), "👤")
                status = "🟢 Active" if m.get("is_active") is not False else "🔴 Inactive"
                last_seen = _format_ist(m.get("last_login"))
                member_rows.append({
                    "Name": m.get("full_name") or "—",
                    "Email": m.get("email") or "—",
                    "Role": f"{role_emoji} {(m.get('role') or 'viewer').replace('_', ' ').title()}",
                    "Status": status,
                    "Last Login": last_seen,
                })
            st.dataframe(pd.DataFrame(member_rows), use_container_width=True, hide_index=True)
        else:
            st.info("No members found in your organization.")
    except Exception:
        st.info("Could not load member data.")


# ======================================================================
# TAB 2: MEMBERS (ORG-SCOPED)
# ======================================================================
def _render_members(org_id: str):
    """List and manage users within this organization."""
    service = _get_supabase_service()
    if not _use_local_auth() and not service:
        st.error("Server configuration error.")
        return

    st.markdown("### 👥 Organization Members")

    _local = _use_local_auth()
    if _local:
        import db_queries as dbq

    try:
        if _local:
            users = dbq.fetch_all_profiles(org_id=org_id)
        else:
            users_result = service.table("profiles") \
                .select("id, email, full_name, role, is_active, profile_status, created_at, last_login") \
                .eq("organization_id", org_id) \
                .order("created_at", desc=True) \
                .execute()
            users = users_result.data or []
    except Exception as e:
        st.error(f"Failed to load members: {e}")
        users = []

    if not users:
        st.info("No members in your organization yet.")
        return

    # Filter controls
    fc1, fc2 = st.columns(2)
    with fc1:
        filter_role = st.selectbox("Filter by role", ["All", "org_admin", "editor", "viewer"], key="org_filter_role")
    with fc2:
        filter_status = st.selectbox("Filter by status", ["All", "Active", "Inactive"], key="org_filter_status")

    current_user_id = st.session_state.get("user_id")

    for user in users:
        if filter_role != "All" and user.get("role") != filter_role:
            continue
        if filter_status == "Active" and user.get("is_active") is False:
            continue
        if filter_status == "Inactive" and user.get("is_active") is not False:
            continue

        status = "🟢 Active" if user.get("is_active") is not False else "🔴 Inactive"
        user_role = user.get("role") or "viewer"
        role_emoji = {"super_admin": "👑", "org_admin": "🛡️", "editor": "✏️", "viewer": "👁️"}.get(user_role, "👤")
        profile_status = user.get("profile_status") or "—"

        _esc_name = html.escape(user.get('full_name', 'Unknown'))
        _esc_email = html.escape(user.get('email', ''))
        with st.expander(f"{role_emoji} **{_esc_name}** — {_esc_email} {status} [{profile_status}]"):
            # Admins can only change roles within editor/viewer (not promote to admin/super_admin)
            ec1, ec2 = st.columns(2)
            with ec1:
                allowed_roles = ["viewer", "editor", "org_admin"]
                current_idx = allowed_roles.index(user_role) if user_role in allowed_roles else 0
                new_user_role = st.selectbox(
                    "Role",
                    allowed_roles,
                    index=current_idx,
                    key=f"org_role_{user['id']}",
                )
            with ec2:
                is_active = st.checkbox("Active", value=user.get("is_active") is not False, key=f"org_active_{user['id']}")

            # Prevent admin from editing themselves
            if user["id"] == current_user_id:
                st.caption("ℹ️ This is your account. Contact a Rotrix super admin to change your own role.")
            else:
                _save_col, _del_col = st.columns([1, 1])
                with _save_col:
                    if st.button("💾 Save Changes", key=f"org_save_{user['id']}"):
                        try:
                            update_data = {
                                "role": new_user_role,
                                "is_active": is_active,
                            }
                            if _local:
                                dbq.update_profile(user["id"], update_data)
                            else:
                                service.table("profiles").update(update_data).eq("id", user["id"]).execute()
                            st.toast(f"Updated {user.get('full_name', 'User')}", icon="✅")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to update: {e}")
                with _del_col:
                    _confirm_key = f"confirm_del_member_{user['id']}"
                    _confirmed = st.checkbox("Confirm removal", key=_confirm_key)
                    if st.button("🗑️ Remove Member", key=f"del_member_{user['id']}", disabled=not _confirmed, type="primary"):
                        try:
                            if _local:
                                dbq.delete_profile(user["id"])
                            else:
                                # Delete profile row
                                service.table("profiles").delete().eq("id", user["id"]).execute()
                                # Delete from Supabase Auth
                                try:
                                    service.auth.admin.delete_user(user["id"])
                                except Exception:
                                    pass  # Best-effort auth deletion
                            st.toast(f"Removed {user.get('full_name', 'User')}", icon="🗑️")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to remove member: {e}")


# ======================================================================
# TAB 3: ACTIVITY (ORG-SCOPED)
# ======================================================================
def _render_activity(org_id: str):
    """Org-scoped usage events and analytics."""
    service = _get_supabase_service()
    if not _use_local_auth() and not service:
        st.error("Server configuration error.")
        return

    st.markdown("### 📈 Organization Activity")

    _plotly_layout = dict(
        font=dict(family="Inter, -apple-system, sans-serif", size=15, color="#000000"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FAFCFE",
        margin=dict(t=48, b=24, l=48, r=24),
        title_font=dict(size=18, color="#000000", family="Inter, sans-serif"),
        legend=dict(font=dict(size=14, color="#000000"), orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
        xaxis=dict(gridcolor="#E2E8F0", linecolor="#000000", linewidth=1.5, showline=True, title_font=dict(size=14, color="#000000"), tickfont=dict(size=13, color="#000000")),
        yaxis=dict(gridcolor="#E2E8F0", linecolor="#000000", linewidth=1.5, showline=True, title_font=dict(size=14, color="#000000"), tickfont=dict(size=13, color="#000000")),
    )
    _color_seq = ["#1B6CA8", "#0A2E42", "#B8941F", "#10b981", "#8B5CF6", "#EC4899"]

    dc1, dc2 = st.columns(2)
    with dc1:
        start_date = st.date_input("From", value=datetime.now() - timedelta(days=30), key="org_act_from")
    with dc2:
        end_date = st.date_input("To", value=datetime.now(), key="org_act_to")

    start_iso = start_date.isoformat()
    end_iso = (datetime.combine(end_date, datetime.max.time())).isoformat()

    _local = _use_local_auth()
    if _local:
        import db_queries as dbq

    # --- 1. USAGE EVENTS ---
    st.markdown("#### User Activity")
    try:
        if _local:
            events = dbq.fetch_usage_events(start_iso, end_iso, org_id=org_id)
        else:
            events_result = service.table("usage_events") \
                .select("user_id, event_type, metadata, created_at") \
                .eq("organization_id", org_id) \
                .gte("created_at", start_iso) \
                .lte("created_at", end_iso) \
                .order("created_at") \
                .execute()
            events = events_result.data or []
    except Exception as e:
        st.error(f"Failed to load activity: {e}")
        events = []

    if events:
        df = pd.DataFrame(events)
        df["created_at"] = pd.to_datetime(df["created_at"])
        df = df[df["event_type"] != "plot_created"]

        if not df.empty:
            df["date"] = df["created_at"].dt.date

            st.markdown("##### Activity Timeline")
            df_time = df.copy()
            df_time["time_frame_hhmm"] = df_time["created_at"].dt.floor("h").dt.strftime("%Y-%m-%d %H:%M")
            time_counts = df_time.groupby(["time_frame_hhmm", "event_type"]).size().reset_index(name="Count")
            
            fig_time = px.bar(
                time_counts,
                x="time_frame_hhmm",
                y="Count",
                color="event_type",
                title="Activity Timeline (hh:mm)",
                color_discrete_sequence=_color_seq,
                labels={"time_frame_hhmm": "", "event_type": ""},
            )
            # Apply base layout then customize x-axis separately to avoid
            # passing xaxis twice (via **_plotly_layout and xaxis=...),
            # which causes a TypeError.
            fig_time.update_layout(
                **_plotly_layout,
                height=350,
                xaxis_title="",
                legend_title="",
            )
            fig_time.update_xaxes(tickformat="%H:%M", title_text="", dtick=None)
            fig_time.update_traces(marker_line_color="#0A2E42", marker_line_width=1)
            st.plotly_chart(fig_time, use_container_width=True)

            event_counts = df["event_type"].value_counts().reset_index()
            event_counts.columns = ["Event Type", "Count"]

            # Event breakdown dataframe only
            st.dataframe(event_counts, use_container_width=True, hide_index=True)
        else:
            event_counts = pd.DataFrame(columns=["Event Type", "Count"])
            st.info("No user activity events found (excluding legacy plot events).")

        st.markdown("---")
        # Most active members
        st.markdown("#### Most Active Members")
        try:
            user_events = df["user_id"].value_counts().head(10).reset_index()
            user_events.columns = ["user_id", "event_count"]

            user_ids = user_events["user_id"].tolist()
            if _local:
                profiles_data = dbq.fetch_profiles_by_ids(user_ids)
            else:
                profiles = service.table("profiles") \
                    .select("id, full_name, email") \
                    .in_("id", user_ids) \
                    .execute()
                profiles_data = profiles.data or []
            name_map = {p["id"]: f"{p.get('full_name', 'Unknown')} ({p.get('email', '')})" for p in profiles_data}
            user_events["Member"] = user_events["user_id"].map(name_map).fillna("Unknown")

            fig3 = px.bar(
                user_events, x="Member", y="event_count",
                title="Most Active Members",
                labels={"event_count": "Total Events"},
                color_discrete_sequence=_color_seq
            )
            fig3.update_layout(**_plotly_layout, height=350)
            st.plotly_chart(fig3, use_container_width=True)
        except Exception:
            st.caption("Could not load member activity data.")
    else:
        st.info("No activity found for the selected date range.")


    # --- 4. STORAGE USAGE (ALL TIME) ---
    st.markdown("---")
    st.markdown("#### Total Storage Usage")
    try:
        if _local:
            all_files_data = dbq.fetch_all_file_metadata(org_id=org_id)
        else:
            all_files = service.table("file_metadata") \
                .select("file_size") \
                .eq("organization_id", org_id) \
                .execute()
            all_files_data = all_files.data or []
        if all_files_data:
            total_bytes = sum((f.get("file_size", 0) or 0) for f in all_files_data)
            total_mb = total_bytes / (1024 * 1024)
            st.metric("📦 Total Organization Storage", f"{total_mb:.1f} MB")
        else:
            st.caption("No files stored yet.")
    except Exception:
        st.caption("Could not calculate storage usage.")

    # CSV export
    st.markdown("---")
    st.markdown("#### 📥 Export Data")
    if events:
        csv_data = df.to_csv(index=False)
        st.download_button(
            "📥 Download Activity CSV",
            data=csv_data,
            file_name=f"org_activity_{start_date}_{end_date}.csv",
            mime="text/csv",
            use_container_width=True,
        )


# ======================================================================
# TAB 4: REPORTS (ORG-SCOPED)
# ======================================================================
def _render_reports(org_id: str):
    """List reports generated by org members with download actions."""
    from storage import get_download_url

    service = _get_supabase_service()
    if not _use_local_auth() and not service:
        st.error("Server configuration error.")
        return

    st.markdown("### 📄 Organization Reports")

    _local = _use_local_auth()
    if _local:
        import db_queries as dbq

    # Fetch org reports
    try:
        if _local:
            reports = dbq.fetch_report_metadata("1970-01-01", "9999-12-31", org_id=org_id)
            reports = reports[:200]
        else:
            reports_result = service.table("report_metadata") \
                .select("id, report_name, user_id, pdf_storage_path, csv_storage_path, generated_at") \
                .eq("organization_id", org_id) \
                .order("generated_at", desc=True) \
                .limit(200) \
                .execute()
            reports = reports_result.data or []
    except Exception as e:
        st.error(f"Failed to load reports: {e}")
        reports = []

    if not reports:
        st.info("No reports generated yet by your organization.")
        return

    # Fetch user names for display
    user_ids = list(set(r.get("user_id") for r in reports if r.get("user_id")))
    name_map = {}
    try:
        if user_ids:
            if _local:
                profiles_data = dbq.fetch_profiles_by_ids(user_ids)
            else:
                profiles = service.table("profiles").select("id, full_name, email").in_("id", user_ids).execute()
                profiles_data = profiles.data or []
            name_map = {p["id"]: p.get("full_name") or p.get("email", "Unknown") for p in profiles_data}
    except Exception:
        pass

    # Select all / action bar
    col_sel, col_filter, col_dl = st.columns([1, 2, 1])

    with col_filter:
        options = ["All"] + list(name_map.values())
        selected_name = st.selectbox("Editor", options=options, key="org_admin_creator_filter", label_visibility="collapsed")
        selected_editor_id = "All"
        if selected_name != "All":
            selected_editor_id = next((k for k, v in name_map.items() if v == selected_name), "All")

    if selected_editor_id != "All":
        reports = [r for r in reports if r.get("user_id") == selected_editor_id]

    with col_sel:
        select_all = st.checkbox("Select all", key="org_reports_select_all")
        if select_all:
            for i in range(len(reports)):
                st.session_state[f"org_report_cb_{i}"] = True

    selected_indices = [
        i for i in range(len(reports))
        if st.session_state.get(f"org_report_cb_{i}", False)
    ]

    with col_dl:
        if selected_indices:
            selected_reports = [reports[i] for i in selected_indices]
            dl_items = []
            for r in selected_reports:
                pdf_path = r.get("pdf_storage_path")
                if pdf_path:
                    url = get_download_url(pdf_path)
                    if url:
                        dl_items.append((r.get("report_name", "report"), url))
            if len(dl_items) == 1:
                st.link_button("⬇️ Download PDF", dl_items[0][1], use_container_width=True)
            elif len(dl_items) > 1:
                st.caption(f"{len(dl_items)} report(s) selected")
                for name, url in dl_items:
                    st.link_button(f"⬇️ {name}", url)
        else:
            st.caption("Select reports to download")

    # Report list
    with st.container(height=500):
        for i, report in enumerate(reports):
            rname = report.get("report_name") or "Report"
            gen_at = report.get("generated_at") or ""
            if isinstance(gen_at, str) and len(gen_at) >= 10:
                gen_at = gen_at[:10]
            user_name = name_map.get(report.get("user_id"), "Unknown")
            label = f"{html.escape(rname)} — {html.escape(user_name)} · {gen_at}"
            st.checkbox(label, key=f"org_report_cb_{i}")

