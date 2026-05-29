#type: ignore
"""
Admin Dashboard for Rotrix Dashboard.

Accessible only to super_admin users. Provides:
  - Tab 0: Pending Approvals
  - Tab 1: Overview statistics (KPIs + charts)
  - Tab 2: User management (create, edit, deactivate)
  - Tab 3: Organization management
  - Tab 4: Analytics (deep dive + CSV export)
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


def _require_super_admin():
    """Check that the current user is a super_admin."""
    role = st.session_state.get("user_role", "")
    if role != "super_admin":
        st.error("🚫 Access denied. Only Rotrix administrators can access this page.")
        st.stop()


# ======================================================================
# MAIN RENDER FUNCTION
# ======================================================================
def render():
    """Render the admin dashboard."""
    _require_super_admin()

    # Close button to exit admin dashboard and return to front page
    _title_col, _close_col = st.columns([0.88, 0.12])
    with _title_col:
        st.markdown("## ⚙️ Admin Dashboard")
    with _close_col:
        if st.button("✕ Close", key="admin_close_btn", use_container_width=True):
            # Return to the page the user came from (analysis/upload),
            # not the front page.
            st.session_state.show_author_form = False
            st.session_state.show_upload_area = st.session_state.get("prev_page_show_upload_area", True)
            st.session_state.files_submitted = st.session_state.get("prev_page_files_submitted", False)
            st.rerun()
    # st.markdown("---")

    tab0, tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Overview", " Analytics", "👥 Users", "🏢 Organizations", "🔔 Pending Approvals"
    ])

    with tab0:
        _render_overview()
    with tab1:
        _render_analytics()
    with tab2:
        _render_user_management()
    with tab3:
        _render_org_management()
    with tab4:
        _render_pending_approvals()


# ======================================================================
# TAB 0: PENDING APPROVALS
# ======================================================================
def _render_pending_approvals():
    """Show profiles awaiting approval with approve/reject actions."""
    from auth import get_pending_profiles, approve_profile, reject_profile

    st.markdown("### 🔔 Pending Profile Approvals")
    st.markdown("Users who have submitted their profile and are waiting for your approval.")
    st.markdown("---")

    pending = get_pending_profiles()

    if not pending:
        st.success("✅ No pending approvals — all clear!")
        return

    st.info(f"**{len(pending)}** profile(s) awaiting your review.")

    for i, profile in enumerate(pending):
        org_data = profile.get("organizations")
        org_name = org_data.get("name", "—") if isinstance(org_data, dict) else "Not specified"
        name = profile.get("full_name", "Unknown")
        email = profile.get("email", "")
        role = profile.get("role", "viewer")
        role_display = role.replace("_", " ").title()
        safe_name = html.escape(str(name))
        safe_email = html.escape(str(email))
        safe_org_name = html.escape(str(org_name))
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
                <div style="font-size: 0.85rem; color: #64748b; margin-top: 2px;">🏢 {safe_org_name}</div>
                <div style="font-size: 0.85rem; color: #64748b; margin-top: 2px;">🔖 Requested role: {safe_role_display}</div>
            </div>
            """, unsafe_allow_html=True)

            ac1, ac2, _ = st.columns([1, 1, 3])
            with ac1:
                if st.button("✅ Approve", key=f"approve_{profile['id']}_{i}", type="primary", use_container_width=True):
                    success, msg = approve_profile(profile["id"])
                    if success:
                        st.toast(f"Approved {name}!", icon="✅")
                        st.rerun()
                    else:
                        st.error(msg)
            with ac2:
                if st.button("❌ Reject", key=f"reject_{profile['id']}_{i}", use_container_width=True):
                    success, msg = reject_profile(profile["id"])
                    if success:
                        st.toast(f"Rejected {name}.", icon="❌")
                        st.rerun()
                    else:
                        st.error(msg)

            st.markdown("---")


# ======================================================================
# TAB 1: OVERVIEW
# ======================================================================
def _render_overview():
    """KPI cards, trending stats, top orgs, recent signups, and system health."""
    # Use service-role client to bypass RLS and see all data across orgs
    supabase = _get_supabase_service() or _get_supabase()
    _local = _use_local_auth()

    # Shared Plotly layout for a professional look
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

    # ── Fetch all counts ──
    if _local:
        import db_queries as dbq
        total_users = dbq.count_all_profiles()
        total_files = dbq.count_events_where(event_type="file_uploaded")
        total_reports = dbq.count_all_reports()
        total_orgs = dbq.count_all_organizations()
        thirty_days_ago = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        active_users_30d = len(dbq.get_active_user_ids(thirty_days_ago))
        pending_count = dbq.count_profiles_where(profile_status="pending_approval")
        week_ago = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
        new_users_week = dbq.count_profiles_where(created_after=week_ago)
        reports_this_week = dbq.count_reports_where(generated_after=week_ago)
    else:
        try:
            users = supabase.table("profiles").select("id", count="exact").execute()
            total_users = users.count or 0
        except Exception:
            total_users = 0
        try:
            files = supabase.table("usage_events").select("id", count="exact").eq("event_type", "file_uploaded").execute()
            total_files = files.count or 0
        except Exception:
            total_files = 0
        try:
            reports = supabase.table("report_metadata").select("id", count="exact").execute()
            total_reports = reports.count or 0
        except Exception:
            total_reports = 0
        try:
            orgs = supabase.table("organizations").select("id", count="exact").execute()
            total_orgs = orgs.count or 0
        except Exception:
            total_orgs = 0
        try:
            thirty_days_ago = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
            active = supabase.table("usage_events") \
                .select("user_id") \
                .eq("event_type", "login") \
                .gte("created_at", thirty_days_ago) \
                .execute()
            active_users_30d = len(set(e["user_id"] for e in (active.data or [])))
        except Exception:
            active_users_30d = 0
        try:
            pending = supabase.table("profiles").select("id", count="exact") \
                .eq("profile_status", "pending_approval").execute()
            pending_count = pending.count or 0
        except Exception:
            pending_count = 0
        try:
            week_ago = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
            new_week = supabase.table("profiles").select("id", count="exact") \
                .gte("created_at", week_ago).execute()
            new_users_week = new_week.count or 0
        except Exception:
            new_users_week = 0
        try:
            reports_week = supabase.table("report_metadata").select("id", count="exact") \
                .gte("generated_at", week_ago).execute()
            reports_this_week = reports_week.count or 0
        except Exception:
            reports_this_week = 0

    # ── KPI Row 1: Primary metrics ──
    st.markdown("### 📊 Platform Overview")
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
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("👥 Total Users", total_users, delta=f"+{new_users_week} this week" if new_users_week else None)
    col2.metric("📁 Files Uploaded", total_files)
    col3.metric("📄 Reports", total_reports, delta=f"+{reports_this_week} this week" if reports_this_week else None)
    col4.metric("🏢 Organizations", total_orgs)
    col5.metric("🟢 Active (30d)", active_users_30d)

    # ── KPI Row 2: Status indicators ──
    s1, s2, s3 = st.columns(3)
    with s1:
        _badge_label = f"{pending_count} pending" if pending_count > 0 else "All clear"
        # Use a simple metric card so the Approval Queue KPI is not visually
        # nested inside an extra HTML container, matching the style of other KPIs.
        st.metric("🔔 Approval Queue", pending_count, delta=_badge_label)
    with s2:
        _adoption = round(active_users_30d / max(total_users, 1) * 100)
        _adopt_color = "#22c55e" if _adoption >= 50 else ("#f59e0b" if _adoption >= 25 else "#ef4444")
        st.markdown(f"""
        <div style="padding: 0.75rem 1rem; border-radius: 10px; border: 1px solid #e2e8f0;
                    background: linear-gradient(135deg, #ecfdf5, #d1fae5);
                    height: 95px; box-sizing: border-box; display: flex; flex-direction: column; justify-content: space-between;">
            <div style="font-size: 0.8rem; color: #065f46; font-weight: 600;">📈 Adoption Rate</div>
            <div style="font-size: 1.4rem; font-weight: 700; color: {_adopt_color};">{_adoption}%
                <span style="font-size: 0.75rem; color: #6b7280; font-weight: 400;">of users active</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with s3:
        _reports_per_user = round(total_reports / max(total_users, 1), 1)
        st.markdown(f"""
        <div style="padding: 0.75rem 1rem; border-radius: 10px; border: 1px solid #e2e8f0;
                    background: linear-gradient(135deg, #eff6ff, #dbeafe);
                    height: 95px; box-sizing: border-box; display: flex; flex-direction: column; justify-content: space-between;">
            <div style="font-size: 0.8rem; color: #1e40af; font-weight: 600;">📄 Reports Per User</div>
            <div style="font-size: 1.4rem; font-weight: 700; color: #1e3a5f;">{_reports_per_user}
                <span style="font-size: 0.75rem; color: #6b7280; font-weight: 400;">avg reports/user</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Charts Row ──
    ch1, ch2 = st.columns(2)
    try:
        if _local:
            members_data = dbq.fetch_all_profiles()
        else:
            members = supabase.table("profiles") \
                .select("organization_id, organizations(name), role") \
                .execute()
            members_data = members.data or []
        if members_data:
            org_names = []
            roles = []
            for m in members_data:
                org = m.get("organizations")
                org_names.append(org.get("name", "Unknown") if isinstance(org, dict) else "Unassigned")
                roles.append((m.get("role") or "viewer").replace("_", " ").title())

            with ch1:
                org_df = pd.DataFrame({"Organization": org_names})
                counts = org_df["Organization"].value_counts().reset_index()
                counts.columns = ["Organization", "Users"]
                fig2 = px.pie(
                    counts, values="Users", names="Organization",
                    title="Users by Organization",
                    color_discrete_sequence=_color_seq,
                    hole=0.45,
                )
                fig2.update_layout(**_plotly_layout, height=340, showlegend=True)
                fig2.update_traces(textposition="inside", textinfo="label+percent", textfont_size=12)
                st.plotly_chart(fig2, use_container_width=True)

            with ch2:
                role_df = pd.DataFrame({"Role": roles})
                role_counts = role_df["Role"].value_counts().reset_index()
                role_counts.columns = ["Role", "Count"]
                fig3 = px.bar(
                    role_counts, x="Role", y="Count",
                    title="Users by Role",
                    color="Role",
                    color_discrete_sequence=_color_seq,
                )
                fig3.update_layout(**_plotly_layout, height=340, showlegend=False)
                fig3.update_traces(marker_line_color="#0A2E42", marker_line_width=1)
                st.plotly_chart(fig3, use_container_width=True)
    except Exception:
        pass

    st.markdown("---")

    # ── Top Organizations Table ──
    st.markdown("### 🏆 Top Organizations")
    try:
        if _local:
            all_profiles_data = dbq.fetch_all_profiles(profile_status="approved")
        else:
            all_profiles = supabase.table("profiles") \
                .select("organization_id, organizations(name), role, is_active, created_at") \
                .eq("profile_status", "approved") \
                .execute()
            all_profiles_data = all_profiles.data or []
        org_stats: dict = {}
        for p in all_profiles_data:
            org_data = p.get("organizations")
            org_nm = org_data.get("name", "Unknown") if isinstance(org_data, dict) else "Unassigned"
            if org_nm not in org_stats:
                org_stats[org_nm] = {"members": 0, "active": 0, "admins": 0, "editors": 0, "viewers": 0}
            org_stats[org_nm]["members"] += 1
            if p.get("is_active") is not False:
                org_stats[org_nm]["active"] += 1
            role = p.get("role", "viewer")
            if role in ("super_admin", "org_admin"):
                org_stats[org_nm]["admins"] += 1
            elif role == "editor":
                org_stats[org_nm]["editors"] += 1
            else:
                org_stats[org_nm]["viewers"] += 1

        if org_stats:
            top_orgs_df = pd.DataFrame([
                {"Organization": k, "Members": v["members"], "Active": v["active"],
                 "Admins": v["admins"], "Editors": v["editors"], "Viewers": v["viewers"]}
                for k, v in sorted(org_stats.items(), key=lambda x: x[1]["members"], reverse=True)
            ])
            st.dataframe(top_orgs_df, use_container_width=True, hide_index=True)
    except Exception:
        st.info("Could not load organization stats.")

    # ── Recent Signups ──
    st.markdown("### 🆕 Recent Signups (Last 7 Days)")
    try:
        week_ago_str = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
        if _local:
            recent_data = dbq.fetch_all_profiles(created_after=week_ago_str, order_desc=True, limit=15)
        else:
            recent = supabase.table("profiles") \
                .select("full_name, email, role, profile_status, created_at, organizations(name)") \
                .gte("created_at", week_ago_str) \
                .order("created_at", desc=True) \
                .limit(15) \
                .execute()
            recent_data = recent.data or []
        if recent_data:
            for u in recent_data:
                org_d = u.get("organizations")
                org_label = org_d.get("name", "—") if isinstance(org_d, dict) else "—"
                status_badge = {"approved": "🟢", "pending_approval": "🟡", "rejected": "🔴", "pending_setup": "⚪"}.get(u.get("profile_status", ""), "⚪")
                role_emoji = {"super_admin": "👑", "org_admin": "🛡️", "editor": "✏️", "viewer": "👁️"}.get(u.get("role", ""), "👤")
                name = html.escape(u.get("full_name") or "—")
                email = html.escape(u.get("email") or "")
                created = _format_ist(u.get("created_at"))
                st.markdown(f"{status_badge} {role_emoji} **{name}** — {email} · {html.escape(org_label)} · {created}")
        else:
            st.info("No new signups this week.")
    except Exception:
        st.info("Could not load recent signups.")


# ======================================================================
# TAB 2: USER MANAGEMENT
# ======================================================================
def _render_user_management():
    """List, create, edit, deactivate users."""
    supabase = _get_supabase()
    _local = _use_local_auth()
    from report_quota import DEFAULT_DAILY_REPORT_QUOTA

    st.markdown("### 👥 User Management")

    # ---- Show confirmation if user was just created ----
    if st.session_state.pop("_user_created_msg", None):
        st.success(st.session_state.pop("_user_created_detail", "User created successfully!"))

    # ---- Create new user ----
    with st.expander("➕ Create New User", expanded=False):
        with st.form("create_user_form"):
            c1, c2 = st.columns(2)
            with c1:
                new_email = st.text_input("Email *")
                new_password = st.text_input("Password *", type="password")
            with c2:
                new_name = st.text_input("Full Name *")
                new_role = st.selectbox("Role", ["viewer", "editor", "org_admin", "super_admin"])

            # Fetch orgs for dropdown
            if _local:
                import db_queries as dbq
                _orgs_list = dbq.fetch_all_organizations()
                org_options = {o["name"]: o["id"] for o in _orgs_list}
            else:
                try:
                    orgs_result = supabase.table("organizations").select("id, name").execute()
                    org_options = {o["name"]: o["id"] for o in (orgs_result.data or [])}
                except Exception:
                    org_options = {}

            selected_org = st.selectbox(
                "Organization *",
                options=list(org_options.keys()) if org_options else ["No organizations found"],
            )

            submitted = st.form_submit_button("Create User", type="primary")

            if submitted:
                if not new_email or not new_password or not new_name:
                    st.warning("Please fill in all required fields.")
                elif selected_org not in org_options:
                    st.warning("Please select a valid organization.")
                else:
                    if _local:
                        success, msg = dbq.create_local_user(
                            new_email, new_password, new_name, new_role,
                            org_options[selected_org],
                        )
                        if success:
                            st.session_state["_user_created_msg"] = True
                            st.session_state["_user_created_detail"] = f"✅ User **{new_name}** ({new_email}) created as **{new_role}** in **{selected_org}**."
                            st.rerun()
                        else:
                            st.error(msg)
                    else:
                        try:
                            from supabase import create_client
                            import os
                            service_client = create_client(
                                os.getenv("SUPABASE_URL", ""),
                                os.getenv("SUPABASE_SERVICE_KEY", ""),
                            )
                            auth_response = service_client.auth.admin.create_user({
                                "email": new_email,
                                "password": new_password,
                                "email_confirm": True,
                                "user_metadata": {
                                    "full_name": new_name,
                                    "role": new_role,
                                },
                            })

                            if auth_response.user:
                                service_client.table("profiles").update({
                                    "organization_id": org_options[selected_org],
                                    "role": new_role,
                                    "full_name": new_name,
                                    "profile_status": "approved",
                                }).eq("id", auth_response.user.id).execute()

                                st.session_state["_user_created_msg"] = True
                                st.session_state["_user_created_detail"] = f"✅ User **{new_name}** ({new_email}) created as **{new_role}** in **{selected_org}**."
                                st.rerun()
                            else:
                                st.error("Failed to create user in auth system.")
                        except Exception as e:
                            st.error(f"Error creating user: {e}")

    # ---- User list ----
    st.markdown("### User List")

    if _local:
        if "dbq" not in dir():
            import db_queries as dbq
        users = dbq.fetch_all_profiles(order_desc=True)
    else:
        try:
            users_result = supabase.table("profiles") \
                .select("id, email, full_name, role, is_active, profile_status, created_at, last_login, organization_id, organizations(name), daily_report_quota") \
                .order("created_at", desc=True) \
                .execute()
            users = users_result.data or []
        except Exception as e:
            st.error(f"Failed to load users: {e}")
            users = []

    if not users:
        st.info("No users found.")
        return

    # Fetch organizations once for all user rows (avoid N+1)
    if _local:
        if "dbq" not in dir():
            import db_queries as dbq
        _orgs_list = dbq.fetch_all_organizations()
        org_opts = {o["name"]: o["id"] for o in _orgs_list}
        org_names_list = list(org_opts.keys())
    else:
        try:
            orgs_r = supabase.table("organizations").select("id, name").execute()
            org_opts = {o["name"]: o["id"] for o in (orgs_r.data or [])}
            org_names_list = list(org_opts.keys())
        except Exception:
            org_opts = {}
            org_names_list = []

    # Build org name lookup for display and filter
    org_id_to_name = {}
    for u in users:
        org_data = u.get("organizations")
        if isinstance(org_data, dict) and org_data.get("name"):
            org_id_to_name[u.get("organization_id", "")] = org_data["name"]
    unique_org_names = sorted(set(org_id_to_name.values()))

    # Filter controls
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        filter_role = st.selectbox("Filter by role", ["All", "super_admin", "org_admin", "editor", "viewer"])
    with fc2:
        filter_status = st.selectbox("Filter by status", ["All", "Active", "Inactive"])
    with fc3:
        filter_org = st.selectbox("Filter by organization", ["All"] + unique_org_names)

    current_user_id = st.session_state.get("user_id")

    for user in users:
        # Apply filters (treat NULL is_active as active for "Active" filter)
        if filter_role != "All" and user.get("role") != filter_role:
            continue
        if filter_status == "Active" and user.get("is_active") is False:
            continue
        if filter_status == "Inactive" and user.get("is_active") is not False:
            continue
        # Organization filter
        if filter_org != "All":
            user_org = user.get("organizations")
            user_org_name = user_org.get("name", "") if isinstance(user_org, dict) else ""
            if user_org_name != filter_org:
                continue

        org_data = user.get("organizations")
        org_name = org_data.get("name", "—") if isinstance(org_data, dict) else "—"
        status = "🟢 Active" if user.get("is_active") is not False else "🔴 Inactive"
        user_role = user.get("role") or "viewer"
        role_emoji = {"super_admin": "👑", "admin": "🛡️", "editor": "✏️", "viewer": "👁️"}.get(user_role, "👤")
        profile_status = user.get("profile_status") or "—"

        _esc_name = html.escape(user.get('full_name', 'Unknown'))
        _esc_email = html.escape(user.get('email', ''))
        _esc_org = html.escape(org_name)
        with st.expander(f"{role_emoji} **{_esc_name}** — {_esc_email} ({_esc_org}) {status} [{profile_status}]"):
            ec1, ec2, ec3 = st.columns(3)
            with ec1:
                # Temporarily hide org-admin role from editing UI as a selectable option.
                role_choices = ["viewer", "editor", "super_admin"]
                # If an existing user currently has role 'admin', keep it as-is but
                # do not offer 'admin' as a target role in the dropdown.
                if user_role not in role_choices and user_role != "admin":
                    role_choices.insert(0, user_role)
                new_user_role = st.selectbox(
                    "Role",
                    role_choices,
                    index=role_choices.index(user_role) if user_role in role_choices else 0,
                    key=f"role_{user['id']}",
                )
            with ec2:
                if not org_names_list:
                    st.caption("No organizations available.")
                    new_org = None
                else:
                    current_org_name = org_name if org_name != "—" else None
                    default_idx = org_names_list.index(current_org_name) if current_org_name in org_names_list else 0
                    new_org = st.selectbox("Organization", org_names_list, index=default_idx, key=f"org_{user['id']}")
            with ec3:
                is_active = st.checkbox("Active", value=user.get("is_active") is not False, key=f"active_{user['id']}")

            # Per-user daily report quota override
            _user_current_quota = user.get("daily_report_quota")
            _uq_col1, _uq_col2 = st.columns([2, 1])
            with _uq_col1:
                _user_quota_val = st.number_input(
                    "Daily Report Quota (per user override)",
                    min_value=0,
                    max_value=100000,
                    value=int(_user_current_quota) if _user_current_quota is not None else int(DEFAULT_DAILY_REPORT_QUOTA),
                    key=f"user_report_quota_{user['id']}",
                    help="Set to 0 to block this user from generating reports. Leave empty to use the system default.",
                )
            with _uq_col2:
                _clear_override = st.checkbox(
                    "Use system default",
                    value=(_user_current_quota is None),
                    key=f"user_quota_clear_{user['id']}",
                    help="When checked, user inherits the system-wide daily report quota.",
                )

            _save_col, _del_col = st.columns([1, 1])
            with _save_col:
                if st.button("💾 Save Changes", key=f"save_{user['id']}"):
                    # Prevent super_admin from demoting themselves
                    if user["id"] == current_user_id and new_user_role != "super_admin":
                        st.error("You cannot change your own role away from Super Admin. Ask another admin to change it.")
                    else:
                        try:
                            update_data = {
                                "role": new_user_role,
                                "is_active": is_active,
                            }
                            if new_org and new_org in org_opts:
                                update_data["organization_id"] = org_opts[new_org]

                            if _local:
                                if "dbq" not in dir():
                                    import db_queries as dbq
                                dbq.update_profile(user["id"], update_data)
                            else:
                                service = _get_supabase_service()
                                if not service:
                                    st.error("Server configuration error (service key not set).")
                                else:
                                    service.table("profiles").update(update_data).eq("id", user["id"]).execute()

                            # Update per-user report quota via helper
                            from report_quota import set_user_quota

                            _target_quota = None if _clear_override else _user_quota_val
                            success, msg = set_user_quota(user["id"], _target_quota)
                            if not success:
                                st.warning(msg)

                            st.toast(f"Updated {user.get('full_name', 'User')}", icon="✅")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to update: {e}")
            with _del_col:
                if user["id"] == current_user_id:
                    st.caption("Cannot remove yourself.")
                else:
                    _confirm_key = f"confirm_del_user_{user['id']}"
                    _confirmed = st.checkbox("Confirm removal", key=_confirm_key)
                    if st.button("🗑️ Remove User", key=f"del_user_{user['id']}", disabled=not _confirmed, type="primary"):
                        try:
                            if _local:
                                if "dbq" not in dir():
                                    import db_queries as dbq
                                dbq.delete_profile(user["id"])
                            else:
                                service = _get_supabase_service()
                                if not service:
                                    st.error("Server configuration error.")
                                else:
                                    service.table("profiles").delete().eq("id", user["id"]).execute()
                                    try:
                                        service.auth.admin.delete_user(user["id"])
                                    except Exception:
                                        pass
                            st.toast(f"Removed {user.get('full_name', 'User')}", icon="🗑️")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to remove user: {e}")


# ======================================================================
# TAB 3: ORGANIZATION MANAGEMENT
# ======================================================================
def _render_org_management():
    """List, create, edit organizations."""
    supabase = _get_supabase()
    _local = _use_local_auth()
    if _local:
        import db_queries as dbq

    st.markdown("### 🏢 Organization Management")

    # ---- Show confirmation if org was just created ----
    if st.session_state.pop("_org_created_msg", None):
        st.success(st.session_state.pop("_org_created_detail", "Organization created successfully!"))

    # ---- Create new org ----
    with st.expander("➕ Create New Organization", expanded=False):
        with st.form("create_org_form"):
            org_name = st.text_input("Organization Name *")
            create_clicked = st.form_submit_button("Create Organization", type="primary")

            if create_clicked:
                if not org_name.strip():
                    st.warning("Please enter an organization name.")
                else:
                    try:
                        if _local:
                            result = dbq.create_organization(org_name.strip())
                            if not result:
                                raise Exception("Failed to create (possibly duplicate name).")
                        else:
                            supabase.table("organizations").insert({
                                "name": org_name.strip(),
                            }).execute()
                        st.session_state["_org_created_msg"] = True
                        st.session_state["_org_created_detail"] = f"✅ Organization **{org_name.strip()}** created successfully."
                        st.rerun()
                    except Exception as e:
                        if "duplicate" in str(e).lower():
                            st.error("An organization with this name already exists.")
                        else:
                            st.error(f"Failed to create organization: {e}")

    # ---- Org list ----
    if _local:
        orgs = dbq.fetch_all_organizations()
    else:
        try:
            orgs_result = supabase.table("organizations") \
                .select("id, name, max_users, created_at") \
                .order("created_at") \
                .execute()
            orgs = orgs_result.data or []
        except Exception as e:
            st.error(f"Failed to load organizations: {e}")
            orgs = []

    if not orgs:
        st.info("No organizations found.")
        return

    for org in orgs:
        # Count members
        if _local:
            member_count = dbq.count_profiles_where(org_id=org["id"])
        else:
            try:
                members = supabase.table("profiles") \
                    .select("id", count="exact") \
                    .eq("organization_id", org["id"]) \
                    .execute()
                member_count = members.count or 0
            except Exception:
                member_count = "?"

        with st.expander(f"🏢 **{html.escape(org['name'])}** — {member_count} members"):
            # Edit org name and quota
            oc1, oc2 = st.columns([2, 1])
            with oc1:
                new_name = st.text_input("Organization Name", value=org["name"], key=f"orgname_{org['id']}")
            with oc2:
                current_max = org.get("max_users", 50) or 50
                new_max = st.number_input(
                    "Max Users Quota", min_value=1, max_value=10000, value=current_max,
                    key=f"orgmax_{org['id']}", help="Maximum number of approved users in this org"
                )

            # Quota usage display
            quota_pct = min(100, int((member_count if isinstance(member_count, int) else 0) / max(current_max, 1) * 100))
            gauge_color = "#22c55e" if quota_pct < 70 else ("#f59e0b" if quota_pct < 90 else "#ef4444")
            st.markdown(f"""
            <div style="margin-bottom: 0.75rem;">
                <div style="font-size: 0.85rem; font-weight: 500; margin-bottom: 4px;">Quota: {member_count} / {current_max} users</div>
                <div style="background: #e5e7eb; border-radius: 6px; height: 16px; overflow: hidden;">
                    <div style="background: {gauge_color}; height: 100%; width: {quota_pct}%; border-radius: 6px;
                                transition: width 0.5s ease;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            _org_save_col, _org_del_col = st.columns([1, 1])
            with _org_save_col:
                if st.button("💾 Save", key=f"orgsave_{org['id']}"):
                    changes_made = False
                    if new_name.strip() and new_name.strip() != org["name"]:
                        try:
                            if _local:
                                dbq.update_organization(org["id"], {"name": new_name.strip()})
                            else:
                                supabase.table("organizations").update({
                                    "name": new_name.strip(),
                                }).eq("id", org["id"]).execute()
                            changes_made = True
                        except Exception as e:
                            st.error(f"Failed to rename: {e}")
                    if new_max != current_max:
                        try:
                            if _local:
                                dbq.update_organization(org["id"], {"max_users": new_max})
                            else:
                                supabase.table("organizations").update({
                                    "max_users": new_max,
                                }).eq("id", org["id"]).execute()
                            changes_made = True
                        except Exception as e:
                            st.error(f"Failed to update quota: {e}")
                    if changes_made:
                        st.toast(f"Updated {new_name.strip() or org['name']}", icon="✅")
                        st.rerun()
            with _org_del_col:
                if isinstance(member_count, int) and member_count > 0:
                    st.caption(f"Cannot delete — {member_count} member(s) exist.")
                else:
                    _confirm_org_key = f"confirm_del_org_{org['id']}"
                    _org_confirmed = st.checkbox("Confirm deletion", key=_confirm_org_key)
                    if st.button("🗑️ Delete Organization", key=f"del_org_{org['id']}", disabled=not _org_confirmed, type="primary"):
                        try:
                            if _local:
                                dbq.delete_organization(org["id"])
                            else:
                                supabase.table("organizations").delete().eq("id", org["id"]).execute()
                            st.toast(f"Deleted {org['name']}", icon="🗑️")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to delete: {e}")

            # Show members
            st.markdown("**Members:**")
            try:
                if _local:
                    members_list = dbq.fetch_all_profiles(org_id=org["id"], order_desc=False)
                else:
                    members_data = supabase.table("profiles") \
                        .select("full_name, email, role, is_active") \
                        .eq("organization_id", org["id"]) \
                        .execute()
                    members_list = members_data.data or []
                if members_list:
                    for m in members_list:
                        status = "🟢" if m.get("is_active", True) else "🔴"
                        role_emoji = {"super_admin": "👑", "org_admin": "🛡️", "editor": "✏️", "viewer": "👁️"}.get(m.get("role", ""), "👤")
                        st.markdown(f"  {status} {role_emoji} **{html.escape(m.get('full_name', ''))}** ({html.escape(m.get('email', ''))}) — {html.escape(m.get('role', ''))}")
                else:
                    st.caption("No members yet.")
            except Exception:
                st.caption("Could not load members.")


# ======================================================================
# TAB 4: ANALYTICS
# ======================================================================
def _render_analytics():
    """Deep dive analytics with charts and CSV export."""
    # Use service-role client to bypass RLS and see all data across orgs
    supabase = _get_supabase_service() or _get_supabase()
    _local = _use_local_auth()
    if _local:
        import db_queries as dbq

    # Professional Plotly layout shared across all analytics charts
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

    st.markdown("### 📈 Analytics")

    # Fetch organizations for filter dropdown
    if _local:
        orgs = dbq.fetch_all_organizations()
    else:
        try:
            orgs_result = supabase.table("organizations").select("id, name").order("name").execute()
            orgs = orgs_result.data or []
        except Exception:
            orgs = []
    org_options = {"All Organizations": None}
    for o in orgs:
        org_options[o["name"]] = o["id"]

    # Filters
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        start_date = st.date_input("From", value=datetime.now() - timedelta(days=30))
    with fc2:
        end_date = st.date_input("To", value=datetime.now())
    with fc3:
        selected_org_name = st.selectbox("Organization Filter", list(org_options.keys()))
        filter_org_id = org_options[selected_org_name]

    # Convert dates to ISO for querying
    start_iso = start_date.isoformat()
    end_iso = (datetime.combine(end_date, datetime.max.time())).isoformat()

    # --- 1. USAGE EVENTS ---
    st.markdown("---")
    # st.markdown("#### User Activity")
    try:
        if _local:
            events = dbq.fetch_usage_events(start_iso, end_iso, org_id=filter_org_id)
        else:
            query = supabase.table("usage_events") \
                .select("user_id, organization_id, event_type, metadata, created_at") \
                .gte("created_at", start_iso) \
                .lte("created_at", end_iso) \
                .order("created_at")
            if filter_org_id:
                query = query.eq("organization_id", filter_org_id)
            events_result = query.execute()
            events = events_result.data or []
    except Exception as e:
        st.error(f"Failed to load user activity: {e}")
        events = []

    if events:
        df = pd.DataFrame(events)
        df["created_at"] = pd.to_datetime(df["created_at"])
        # Remove plot_created events from user activity as per request to focus on file uploads
        df = df[df["event_type"] != "plot_created"]
        
        if not df.empty:
            df["date"] = df["created_at"].dt.date
            
            # Additional Time Series chart grouped by hour:minute (hh:mm)
            # st.markdown("##### Activity Timeline")
            # Group by 1-hour intervals to show hh:mm time frame
            df_time = df.copy()
            df_time["time_frame_hhmm"] = df_time["created_at"].dt.floor("h").dt.strftime("%Y-%m-%d %H:%M")
            time_counts = df_time.groupby(["time_frame_hhmm", "event_type"]).size().reset_index(name="Count")
            
            fig_time = px.line(
                time_counts, x="time_frame_hhmm", y="Count", color="event_type",
                title="Activity Timeline (hh:mm)", markers=True, color_discrete_sequence=_color_seq,
                labels={"time_frame_hhmm": "", "event_type": "Event"}
            )
            fig_time.update_layout(**_plotly_layout, height=320)
            st.plotly_chart(fig_time, use_container_width=True)

            # Event breakdown
            event_counts = df["event_type"].value_counts().reset_index()
            event_counts.columns = ["Event Type", "Count"]

            ec1, ec2 = st.columns([1, 2])
            with ec1:
                st.dataframe(event_counts, use_container_width=True, hide_index=True)
            with ec2:
                fig = px.bar(
                    event_counts, x="Event Type", y="Count",
                    color="Event Type", title="Events by Type",
                    color_discrete_sequence=_color_seq,
                )
                fig.update_layout(**_plotly_layout, height=320, showlegend=False)
                fig.update_traces(marker_line_color="#0A2E42", marker_line_width=1)
                st.plotly_chart(fig, use_container_width=True)
        else:
            event_counts = pd.DataFrame(columns=["Event Type", "Count"])
            st.info("No user activity events found (excluding legacy plot events).")

        # Most active users
        try:
            user_events = df["user_id"].value_counts().head(10).reset_index()
            user_events.columns = ["user_id", "event_count"]

            # Fetch user names
            user_ids = user_events["user_id"].tolist()
            if _local:
                profiles_data = dbq.fetch_profiles_by_ids(user_ids)
            else:
                profiles = supabase.table("profiles") \
                    .select("id, full_name, email") \
                    .in_("id", user_ids) \
                    .execute()
                profiles_data = profiles.data or []
            name_map = {p["id"]: f"{p.get('full_name', 'Unknown')} ({p.get('email', '')})" for p in profiles_data}
            user_events["User"] = user_events["user_id"].map(name_map).fillna("Unknown")

            fig3 = px.bar(
                user_events, x="User", y="event_count",
                title="Top 10 Most Active Users",
                labels={"event_count": "Total Events"},
                color_discrete_sequence=_color_seq,
            )
            fig3.update_layout(**_plotly_layout, height=380)
            fig3.update_traces(marker_line_color="#0A2E42", marker_line_width=1)
            st.plotly_chart(fig3, use_container_width=True)
        except Exception:
            st.caption("Could not load user activity data.")
    else:
        st.info("No user activity events found for the selected filters.")

    # --- 2. PREPARE FILES DATA ---
    try:
        if _local:
            files = dbq.fetch_file_metadata(start_iso, end_iso, org_id=filter_org_id)
        else:
            f_query = supabase.table("file_metadata") \
                .select("id, uploaded_at, file_size, user_id, organization_id, original_filename") \
                .gte("uploaded_at", start_iso) \
                .lte("uploaded_at", end_iso)
            if filter_org_id:
                f_query = f_query.eq("organization_id", filter_org_id)
            files_result = f_query.execute()
            files = files_result.data or []

        if files:
            files_df = pd.DataFrame(files)
            files_df["uploaded_at"] = pd.to_datetime(files_df["uploaded_at"])
            files_df["date"] = files_df["uploaded_at"].dt.date
    except Exception as e:
        pass

    # --- 3. PREPARE REPORTS DATA ---
    try:
        if _local:
            reports = dbq.fetch_report_metadata(start_iso, end_iso, org_id=filter_org_id)
        else:
            r_query = supabase.table("report_metadata") \
                .select("id, generated_at, user_id, organization_id") \
                .gte("generated_at", start_iso) \
                .lte("generated_at", end_iso)
            if filter_org_id:
                r_query = r_query.eq("organization_id", filter_org_id)
            reports_result = r_query.execute()
            reports = reports_result.data or []
        
        if reports:
            reports_df = pd.DataFrame(reports)
            reports_df["generated_at"] = pd.to_datetime(reports_df["generated_at"])
            reports_df["date"] = reports_df["generated_at"].dt.date
    except Exception as e:
        pass

    # --- 3b. PER-USER SUMMARY (FILES + REPORTS) ---
    st.markdown("---")
    st.markdown("#### Per-User Summary")
    try:
        # Build aggregates from files and reports within the selected window
        user_file_counts = {}
        if files:
            files_df2 = files_df.copy()
            files_df2 = files_df2.dropna(subset=["user_id"])
            user_file_counts = (
                files_df2.groupby("user_id")["id"]
                .count()
                .to_dict()
            )

        user_report_counts = {}
        if reports:
            reports_df2 = reports_df.copy()
            reports_df2 = reports_df2.dropna(subset=["user_id"])
            user_report_counts = (
                reports_df2.groupby("user_id")["id"]
                .count()
                .to_dict()
            )

        # Last login within window (from usage_events)
        user_last_login = {}
        if events:
            login_df = df[df["event_type"] == "login"].copy()
            if not login_df.empty:
                login_df["created_at"] = pd.to_datetime(login_df["created_at"])
                user_last_login = (
                    login_df.groupby("user_id")["created_at"]
                    .max()
                    .to_dict()
                )

        # Union of all user_ids we have activity for
        user_ids_all = set(user_file_counts.keys()) | set(user_report_counts.keys()) | set(user_last_login.keys())

        if user_ids_all:
            # Fetch user profiles for display
            if _local:
                pdata = dbq.fetch_profiles_by_ids(list(user_ids_all))
            else:
                profiles = supabase.table("profiles") \
                    .select("id, full_name, email, organization_id, organizations(name), last_login") \
                    .in_("id", list(user_ids_all)) \
                    .execute()
                pdata = profiles.data or []

            rows = []
            for p in pdata:
                uid = p["id"]
                org = p.get("organizations")
                org_name = org.get("name", "—") if isinstance(org, dict) else "—"
                rows.append({
                    "User": f"{p.get('full_name', 'Unknown')} ({p.get('email', '')})",
                    "Organization": org_name,
                    "Last login (profile)": _format_ist(p.get("last_login")),
                    "Last login (events)": _format_ist(user_last_login.get(uid)),
                    "Files uploaded": user_file_counts.get(uid, 0),
                    "Reports generated": user_report_counts.get(uid, 0),
                })

            if rows:
                user_summary_df = pd.DataFrame(rows)
                # Sort by reports then files desc
                user_summary_df = user_summary_df.sort_values(
                    by=["Reports generated", "Files uploaded"],
                    ascending=[False, False],
                )
                st.dataframe(user_summary_df, use_container_width=True, hide_index=True)
            else:
                st.caption("No user activity in this period.")
        else:
            st.caption("No user activity in this period.")
    except Exception as e:
        st.caption(f"Could not build per-user summary. {e}")

    # --- 3c. PER-ORGANIZATION SUMMARY ---
    st.markdown("---")
    st.markdown("#### Per-Organization Summary")
    try:
        org_rows = []

        # Precompute lookups
        org_file_counts = {}
        if files:
            org_file_counts = (
                files_df.groupby("organization_id")["id"]
                .count()
                .to_dict()
            )
        org_report_counts = {}
        if reports:
            org_report_counts = (
                reports_df.groupby("organization_id")["id"]
                .count()
                .to_dict()
            )
        org_last_activity = {}
        if events:
            df["created_at"] = pd.to_datetime(df["created_at"])
            org_last_activity = (
                df.groupby("organization_id")["created_at"]
                .max()
                .to_dict()
            )

        # Active users per org (unique user_ids with login events)
        org_active_users = {}
        if events:
            login_df2 = df[df["event_type"] == "login"].copy()
            if not login_df2.empty:
                for (oid, uid), _ in login_df2.groupby(["organization_id", "user_id"]).groups.items():
                    if oid not in org_active_users:
                        org_active_users[oid] = set()
                    org_active_users[oid].add(uid)

        for o in orgs:
            oid = o["id"]
            if filter_org_id and oid != filter_org_id:
                continue
            org_rows.append({
                "Organization": o["name"],
                "Active users (logins)": len(org_active_users.get(oid, set())),
                "Files uploaded": org_file_counts.get(oid, 0),
                "Reports generated": org_report_counts.get(oid, 0),
                "Last activity": org_last_activity.get(oid),
            })

        if org_rows:
            org_df = pd.DataFrame(org_rows)
            org_df = org_df.sort_values(by=["Reports generated", "Files uploaded"], ascending=[False, False])
            st.dataframe(org_df, use_container_width=True, hide_index=True)
        else:
            st.caption("No organization activity in this period.")
    except Exception as e:
        st.caption(f"Could not build per-organization summary. {e}")

    # --- 4. STORAGE USAGE PIE (ALL TIME) ---
    st.markdown("---")
    st.markdown("#### Total Storage Usage (All Time)")
    try:
        if _local:
            all_files_data = dbq.fetch_all_file_metadata(org_id=filter_org_id)
        else:
            all_f_query = supabase.table("file_metadata").select("file_size, organization_id, organizations(name)")
            if filter_org_id:
                all_f_query = all_f_query.eq("organization_id", filter_org_id)
            all_files = all_f_query.execute()
            all_files_data = all_files.data or []
        if all_files_data:
            storage_data = []
            for f in all_files_data:
                org = f.get("organizations")
                org_name = org.get("name", "Unknown") if isinstance(org, dict) else "Unknown"
                storage_data.append({
                    "Organization": org_name,
                    "Size (MB)": (f.get("file_size", 0) or 0) / (1024 * 1024),
                })
            storage_df = pd.DataFrame(storage_data)
            org_storage = storage_df.groupby("Organization")["Size (MB)"].sum().reset_index()
            org_storage["Size (MB)"] = org_storage["Size (MB)"].round(2)

            sc1, sc2 = st.columns([1, 2])
            with sc1:
                total_mb = org_storage["Size (MB)"].sum()
                st.metric("Total Storage Used", f"{total_mb:.1f} MB")
                if not filter_org_id:
                    st.dataframe(org_storage, use_container_width=True, hide_index=True)
            with sc2:
                if not filter_org_id:
                    fig_storage = px.pie(
                        org_storage, values="Size (MB)", names="Organization",
                        title="Storage by Organization",
                        color_discrete_sequence=_color_seq,
                        hole=0.45,
                    )
                    fig_storage.update_layout(**_plotly_layout, height=320)
                    fig_storage.update_traces(textposition="inside", textinfo="label+percent", textfont_size=12)
                    st.plotly_chart(fig_storage, use_container_width=True)
        else:
            st.caption("No files stored yet.")
    except Exception:
        st.caption("Could not calculate storage usage.")

    # CSV export & raw event log
    st.markdown("---")
    st.markdown("#### 📥 Export & Event Log")

    export_col1, export_col2 = st.columns(2)
    with export_col1:
        if events:
            csv_data = df.to_csv(index=False)
            st.download_button(
                "📥 Download Events CSV",
                data=csv_data,
                file_name=f"usage_events_{start_date}_{end_date}.csv",
                mime="text/csv",
                use_container_width=True,
            )
    with export_col2:
        if events and not event_counts.empty:
            summary_csv = event_counts.to_csv(index=False)
            st.download_button(
                "📥 Download Summary CSV",
                data=summary_csv,
                file_name=f"event_summary_{start_date}_{end_date}.csv",
                mime="text/csv",
                use_container_width=True,
            )

    # Raw events table (latest N rows)
    if events:
        st.markdown("##### Recent Events")
        try:
            log_df = df.copy()
            log_df["created_at"] = pd.to_datetime(log_df["created_at"])

            # Map org_id and user_id to display names
            org_ids = list(set(log_df.get("organization_id", pd.Series(dtype=object)).dropna().tolist()))
            org_name_map = {}
            if org_ids:
                if _local:
                    _org_data = dbq.fetch_all_organizations()
                    org_name_map = {o["id"]: o["name"] for o in _org_data if o["id"] in org_ids}
                else:
                    org_res = supabase.table("organizations").select("id, name").in_("id", org_ids).execute()
                    org_name_map = {o["id"]: o["name"] for o in (org_res.data or [])}

            user_ids = list(set(log_df.get("user_id", pd.Series(dtype=object)).dropna().tolist()))
            user_name_map = {}
            if user_ids:
                if _local:
                    _prof_data = dbq.fetch_profiles_by_ids(user_ids)
                else:
                    prof_res = supabase.table("profiles").select("id, full_name, email").in_("id", user_ids).execute()
                    _prof_data = prof_res.data or []
                user_name_map = {
                    p["id"]: f"{p.get('full_name', 'Unknown')} ({p.get('email', '')})"
                    for p in _prof_data
                }

            log_df["Organization"] = log_df.get("organization_id").map(org_name_map).fillna("—")
            log_df["User"] = log_df.get("user_id").map(user_name_map).fillna("Unknown")

            # Flatten metadata for quick view
            def _meta_to_str(m):
                try:
                    if isinstance(m, dict):
                        # keep it short
                        return ", ".join(f"{k}={v}" for k, v in list(m.items())[:4])
                    return str(m)
                except Exception:
                    return ""

            log_df["Details"] = log_df.get("metadata", "").apply(_meta_to_str)

            display_cols = ["created_at", "Organization", "User", "event_type", "Details"]
            display_df = log_df[display_cols].sort_values("created_at", ascending=False).head(200)
            st.dataframe(display_df, use_container_width=True, hide_index=True)
        except Exception:
            st.caption("Could not render raw events table.")

