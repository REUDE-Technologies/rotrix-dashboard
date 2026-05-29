import os

#type: ignore
"""
Header bar component: REUDE logo (left) + profile icon with popover (right).

Renders the fixed top bar. The logo is a clickable button that returns to
the front page. The 👤 icon toggles a profile popover with user info,
admin dashboard link, and edit profile button.
"""
import html as html_mod
from urllib.parse import quote_plus

import streamlit as st

from auth import check_role, logout

from config import SHOW_CALCULATORS_BUTTON, _get_reude_logo_b64


def render_header_bar():
    """Render the fixed header bar with logo and profile icon/popover.

    Uses st.session_state for all shared state (show_front_page,
    show_author_form, show_profile_popover, etc.).
    """
    logo_b64 = _get_reude_logo_b64()
    if not logo_b64:
        return

    # Show profile icon only when:
    # 1. Not on front page
    # 2. Not on author form
    # 3. Not on login form
    # 4. Profile status is approved
    is_approved_user = st.session_state.get('profile_status') == 'approved'
    show_details_icon = (
        not st.session_state.show_front_page and
        not st.session_state.show_author_form and
        not st.session_state.get('show_login_form', False) and
        is_approved_user
    )

    # Upload icon only on analysis page (dashboard), next to the details icon
    on_analysis_page = st.session_state.files_submitted and not st.session_state.show_upload_area
    if show_details_icon:
        col_logo, col_mid, col_profile = st.columns([0.18, 0.64, 0.18])
    else:
        col_logo, _ = st.columns([0.2, 0.8])

    # ── Logo (clickable → home) ──
    with col_logo:
        st.markdown('<div class="logo-home-wrap">', unsafe_allow_html=True)
        if st.button(" ", key="logo_go_home", type="secondary"):
            st.session_state.show_front_page = True
            st.session_state.show_author_form = False
            st.session_state.show_report_history = False
            st.session_state.show_calculators = False
            st.session_state.show_profile_popover = False
            st.session_state["show_support_popover"] = False
            st.rerun()
        st.markdown(
            f"""
            <div class="logo-overlay">
            <div class="fixed-header logo-card" style="cursor:pointer;">
                <img src="data:image/png;base64,{logo_b64}" alt="REUDE" />
            </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Calculators shortcut (hidden when SHOW_CALCULATORS_BUTTON is False) ──
    if show_details_icon and SHOW_CALCULATORS_BUTTON:
        with col_mid:
            st.markdown('<div class="header-calc-wrap">', unsafe_allow_html=True)
            if st.button("Calculators", key="header_calculators_btn", type="secondary"):
                st.session_state.show_calculators = True
                st.session_state.show_front_page = False
                st.session_state.show_author_form = False
                st.session_state.show_report_history = False
                st.session_state.show_profile_editor = False
                st.session_state.show_upload_area = False
                st.session_state.show_profile_popover = False
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

    # ── Profile / upload icons ──
    if show_details_icon:
        with col_profile:
            st.markdown('<div class="details-icon-btn-wrap">', unsafe_allow_html=True)
            if on_analysis_page:
                col_upload, col_details = st.columns(2)
                with col_upload:
                    if st.button("➕", key="header_upload_btn", help="Add or upload files", use_container_width=True):
                        st.session_state.show_upload_area = True
                        st.session_state.show_calculators = False
                        st.rerun()
                with col_details:
                    if st.button("👤", key="details_icon_btn", help="View profile", use_container_width=True):
                        st.session_state["show_profile_popover"] = not st.session_state.get("show_profile_popover", False)
                        st.rerun()
            else:
                if st.button("👤", key="details_icon_btn", help="View profile", use_container_width=True):
                    st.session_state["show_profile_popover"] = not st.session_state.get("show_profile_popover", False)
                    st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

    # ── Profile action menu (shown when 👤 is toggled) ──
    # User details are now in the sidebar; this popover is a compact action menu.
    if st.session_state.get("show_profile_popover", False):
        _dummy, col_popover = st.columns([0.01, 0.99])
        with col_popover:
            st.markdown('<div class="profile-popover-anchor" style="display:none;"></div>', unsafe_allow_html=True)
            
            # Close button for the popover
            if st.button("✕", key="profile_popover_close_btn"):
                st.session_state["show_profile_popover"] = False
                st.rerun()
            
            _prof_name = (st.session_state.get("user_name") or st.session_state.get("author_name") or "User").strip()
            _prof_email = (st.session_state.get("user_email") or st.session_state.get("author_email") or "").strip()
            _prof_company = (st.session_state.get("organization_name") or st.session_state.get("author_company") or "").strip()
            _prof_initial = _prof_name[0].upper() if _prof_name else "U"
            _prof_role = (st.session_state.get("user_role") or "viewer").replace("_", " ").title()
            
            _role_emoji = {"Super Admin": "👑", "Org Admin": "🛡️", "Editor": "✏️", "Viewer": "👁️"}
            _r_emoji = _role_emoji.get(_prof_role, "👤")
            
            _popover_lines = [
                '<div class="user-profile-popover">',
                '<div class="profile-header" style="margin-bottom: 0.5rem; padding-bottom: 0.5rem; border-bottom: 1px solid #E2E8F0;">',
                f'<div class="profile-avatar">{html_mod.escape(_prof_initial)}</div>',
                '<div>',
                f'<p class="profile-name">{html_mod.escape(_prof_name)}</p>',
                f'<p class="profile-role" style="font-size: 0.78rem;">{_r_emoji} {html_mod.escape(_prof_role)}</p>',
                '</div></div>',
            ]
            if _prof_email:
                _popover_lines.append(f'<div class="profile-detail"><span class="profile-detail-icon">✉️</span> {html_mod.escape(_prof_email)}</div>')
            if _prof_company:
                _popover_lines.append(f'<div class="profile-detail"><span class="profile-detail-icon">🏢</span> {html_mod.escape(_prof_company)}</div>')
            _popover_lines.append('</div>')
            
            st.markdown("\n".join(_popover_lines), unsafe_allow_html=True)

            # Admin dashboard link (super_admin or org_admin)
            if check_role(["super_admin", "org_admin"]):
                if st.button("\u2699\ufe0f Admin Dashboard", key="profile_admin_btn", use_container_width=True):
                    st.session_state["show_profile_popover"] = False
                    st.session_state.prev_page_show_upload_area = st.session_state.show_upload_area
                    st.session_state.prev_page_files_submitted = st.session_state.files_submitted
                    st.session_state.show_author_form = True
                    st.session_state.show_front_page = False
                    st.session_state.show_upload_area = False
                    st.session_state.show_report_history = False
                    st.session_state.show_calculators = False
                    st.rerun()

            # Edit profile button
            if st.button("✏️ Edit Profile", key="profile_edit_btn", use_container_width=True):
                st.session_state["show_profile_popover"] = False
                st.session_state.prev_page_show_upload_area = st.session_state.show_upload_area
                st.session_state.prev_page_files_submitted = st.session_state.files_submitted
                st.session_state.show_profile_editor = True
                st.session_state.show_author_form = False
                st.session_state.show_front_page = False
                st.session_state.show_upload_area = False
                st.session_state.show_report_history = False
                st.session_state.show_calculators = False
                st.rerun()

            # Report History button (hidden while already on the history page)
            if not st.session_state.get("show_report_history", False):
                if st.button("📄 Report History", key="profile_report_history_btn", use_container_width=True):
                    st.session_state["show_profile_popover"] = False
                    st.session_state.prev_page_show_upload_area = st.session_state.get("show_upload_area", False)
                    st.session_state.prev_page_files_submitted = st.session_state.get("files_submitted", False)
                    st.session_state.show_report_history = True
                    st.session_state.show_front_page = False
                    st.session_state.show_author_form = False
                    st.session_state.show_upload_area = False
                    st.session_state.show_calculators = False
                    st.rerun()
                
            # Logout button
            if st.button("🚪 Logout", key="profile_logout_btn", use_container_width=True):
                st.session_state["show_profile_popover"] = False
                logout()
                st.rerun()
