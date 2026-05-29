#type: ignore
"""
Initialize all Streamlit session_state defaults in one place.
"""
import streamlit as st


def init_session_state():
    """Set up all session_state keys with their default values."""

    defaults = {
        'uploaded_files': [],
        'files_submitted': False,
        'show_upload_area': True,
        'show_front_page': True,
        'show_author_form': False,
        'show_report_history': False,
        'show_calculators': False,
        'author_name': "",
        'author_company': "",
        'author_data_source': "",
        'author_email': "",
        'author_details_completed': False,
        'prev_page_show_upload_area': True,
        'prev_page_files_submitted': False,
        'upload_opened_by_plus': False,
        'upload_source': "desktop",
        'show_file_preview': False,
        'file_rename_mode': {},
        'reports_exported_count': 0,
        'plots_generated_count': 0,
        'multi_param_file_insights': {},
        'uploader_key_counter': 0,       # Incremented on file remove to reset file_uploader widget

        # Authentication
        'authenticated': False,
        'user_id': None,
        'user_email': '',
        'user_name': '',
        'user_role': '',
        'organization_id': None,
        'organization_name': '',
        'show_profile_popover': False,
        'show_profile_editor': False,
        'profile_status': 'pending_setup',      # pending_setup | pending_approval | approved | rejected
        'show_login_form': False,                # True = show login/signup panel on right side
        'login_mode': 'signin',                  # signin | signup | forgot_password
        # Multi-Parameter Analysis
        'multi_param_file_selection': None,
        'multi_param_ulog_topic': None,
        'multi_param_x_axis': '',
        'multi_param_left_y_axes': [],
        'multi_param_right_y_axes': [],
        'multi_param_data_selected_cols': [],
        'multi_param_smoothing': False,
        'multi_param_smoothing_window': 5,
        'multi_param_saved_graphs': [],

        # Multi-File Comparison Analysis
        'multi_file_comparison_selections': [],
        'multi_file_comparison_x_axis': '',
        'multi_file_comparison_left_y_axes': [],
        'multi_file_comparison_right_y_axes': [],
        'multi_file_comparison_data_selected_cols': [],
        'multi_file_selected_benchmark': 'None',
        'multi_file_selected_targets': [],
        'multi_file_data_view_mode': 'Raw Data',
        'multi_file_plot_style_value': 'Line',
        # Fingerprint of the currently loaded file set (hash of sorted filenames)
        # — used to detect when cached multi-file data should be invalidated.
        '_multi_file_set_fingerprint': '',
        # Cached loaded DataFrames and metadata (populated by multi_file_view)
        '_multi_file_cached_file_data': None,
        '_multi_file_cached_file_extensions': None,
        '_multi_file_cached_throttle_cols': None,
        '_multi_file_cached_numeric_cols': None,

        # Report-related
        'report_graph_entries': {},
        'report_raw_data_df': None,
        'report_sorted_table_df': None,
        'report_file_info_text': "",
        'manual_file_info_text': "",
        'summary_stats_df': None,
        # When True, right side of Report tab shows Plot | Data instead of Single report | Multi report
        'report_multi_add_new_view': False,
    }

    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


def cleanup_stale_session_data():
    """
    Remove heavy or now-stale data from session when the active file set
    changes. This helps keep per-session memory usage bounded when users
    upload/remove many files over time.
    """
    # Drop cached multi-file figures and preview flags
    for key in list(st.session_state.keys()):
        if key.startswith("_multi_fig_") or key.startswith("preview_mode_"):
            try:
                del st.session_state[key]
            except Exception:
                pass

    # Clear multi-file cached data so it is reloaded with the new file set
    for mf_key in (
        "_multi_file_cached_file_data",
        "_multi_file_cached_file_extensions",
        "_multi_file_cached_throttle_cols",
        "_multi_file_cached_numeric_cols",
        "_multi_file_set_fingerprint",
    ):
        st.session_state.pop(mf_key, None)

    # Clean up stale multi-file widget keys that reference removed columns
    for key in list(st.session_state.keys()):
        if key.startswith("multi_file_multi_param_"):
            try:
                del st.session_state[key]
            except Exception:
                pass

    # Clear report-related heavyframes
    for heavy_key in (
        "report_graph_entries",
        "report_raw_data_df",
        "report_sorted_table_df",
        "summary_stats_df",
    ):
        st.session_state.pop(heavy_key, None)

