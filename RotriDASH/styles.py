#type: ignore
"""
Global CSS styles for the RotriDash Streamlit app.

All CSS is centralized here to keep app.py focused on logic.
Call inject_global_styles() once at the start of main() to apply them.
"""
import streamlit as st


def inject_global_styles():
    """Inject all global CSS styles into the Streamlit page.

    Includes:
      - Fixed header bar (logo + profile icon positioning)
      - Front page hero, cards, and use-case layout
      - File management section styles
      - Profile popover dropdown
      - Responsive breakpoints (1200px, 992px, 768px)
    """
    st.markdown("""
    <style>
    /* ── Professional font: Inter (Google Fonts) ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    html, body, [data-testid="stAppViewContainer"], [data-testid="stSidebar"],
    .stMarkdown, .stButton > button, input, select, textarea {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }
    /* Full-width fixed background bar */
    .app-top-bar-bg {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: clamp(80px, 10vh, 120px);
        background: #ffffff;
        box-shadow: 0 2px 8px rgba(10, 46, 66, 0.06);
        z-index: 900;
    }
    .fixed-header {
        z-index: 1001;
        background: #fff;
        border-radius: 14px;
        box-shadow: 0 4px 16px rgba(10, 46, 66, 0.10);
        padding: 10px 16px;
        display: flex;
        flex-direction: column;
        align-items: flex-start;
        min-width: 180px;
        max-width: 240px;
        border: 1px solid #E2E8F0;
    }
    .fixed-header h1 {
        color: #1B6CA8;
        margin: 0 0 2px 0;
        font-size: 1.7rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        line-height: 1.1;
        font-weight: 700;
    }
    .fixed-header .rocket-icon {
        font-size: 1.7rem;
        line-height: 1;
    }
    .fixed-header p {
        color: #64748B;
        margin: 0;
        font-size: 0.98rem;
        line-height: 1.2;
        font-weight: 400;
    }
    /* Logo column: merge button and logo into one clickable area (no data loss) */
    div[data-testid="column"]:has(.logo-home-wrap) {
        position: fixed !important;
        top: clamp(26px, 4vh, 46px) !important;
        left: clamp(16px, 2vw, 32px) !important;
        width: 180px !important; min-height: 64px !important;
        z-index: 1001 !important;
    }
    /* Logo layout: small inset from viewport; negative top to align logo higher */
    div[data-testid="column"]:has(.logo-home-wrap) .stButton {
        position: absolute !important; top: -10px !important; left: 6px !important;
        width: 168px !important; height: 52px !important; z-index: 0 !important;
    }
    div[data-testid="column"]:has(.logo-home-wrap) .stButton > button {
        width: 100% !important; height: 100% !important; min-height: 52px !important;
        border: none !important; outline: none !important; box-shadow: none !important;
        background: transparent !important; opacity: 0 !important; cursor: pointer !important;
    }
    div[data-testid="column"]:has(.logo-home-wrap) .logo-overlay {
        position: absolute !important; top: -30px !important; left: 6px !important;
        width: 168px !important; min-height: 52px !important; z-index: 1 !important;
        pointer-events: none !important;
    }
    /* Profile / header icon group: toolbar icons aligned to header center */
    div[data-testid="column"]:has(.details-icon-btn-wrap) {
        position: fixed !important;
        top: clamp(28px, 4vh, 46px) !important;
        right: clamp(16px, 2.5vw, 28px) !important;
        width: 280px !important;
        min-height: 72px !important;
        z-index: 1001 !important;
        display: flex !important;
        flex-direction: row !important;
        flex-wrap: nowrap !important;
        justify-content: flex-end !important;
        align-items: center !important;
        gap: 16px !important;
    }
    /* Each icon gets its own space like standard app toolbar icons */
    div[data-testid="column"]:has(.details-icon-btn-wrap) > div[data-testid="column"],
    div[data-testid="column"]:has(.details-icon-btn-wrap) > div > div[data-testid="column"] {
        flex: 0 0 auto !important;
        width: 112px !important;
        min-width: 112px !important;
        max-width: 112px !important;
    }
    div[data-testid="column"]:has(.details-icon-btn-wrap) .stButton,
    div[data-testid="column"]:has(.details-icon-btn-wrap) [data-testid="stVerticalBlock"] > .stButton {
        width: 112px !important;
        height: 56px !important;
        min-width: 112px !important;
        min-height: 56px !important;
        position: relative !important;
        top: -8px !important;
        z-index: 0 !important;
    }
    div[data-testid="column"]:has(.details-icon-btn-wrap) .stButton > button {
        width: 112px !important;
        height: 56px !important;
        min-width: 112px !important;
        min-height: 56px !important;
        border-radius: 14px !important;
        border: 2px solid #E2E8F0 !important;
        background: #fff !important;
        color: #1B6CA8 !important;
        font-size: 1.75rem !important;
        padding: 0 !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08) !important;
        transition: all 0.2s ease !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    div[data-testid="column"]:has(.details-icon-btn-wrap) .stButton > button:hover {
        background: #EFF6FF !important;
        border-color: #1B6CA8 !important;
        box-shadow: 0 4px 14px rgba(27, 108, 168, 0.2) !important;
        transform: translateY(-1px) scale(1.02) !important;
    }

    /* ── Report generation play / stop control ── */
    .report-gen-control-wrap .stButton > button[data-testid="stBaseButton-primary"],
    .report-gen-control-wrap .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #16a34a 0%, #22c55e 50%, #15803d 100%) !important; /* green */
        color: #ffffff !important;
        border: none !important;
        border-radius: 999px !important;
        font-size: 1.75rem !important;
        width: 72px !important;
        height: 72px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        box-shadow: 0 4px 14px rgba(22, 163, 74, 0.35) !important;
        padding: 0 !important;
    }
    .report-gen-control-wrap .stButton > button[data-testid="stBaseButton-primary"]:hover,
    .report-gen-control-wrap .stButton > button[kind="primary"]:hover {
        box-shadow: 0 6px 20px rgba(22, 163, 74, 0.5) !important;
        transform: translateY(-1px) scale(1.03) !important;
    }

    .report-gen-control-wrap .stButton > button:not([data-testid="stBaseButton-primary"]):not([kind="primary"]) {
        background: linear-gradient(135deg, #dc2626 0%, #ef4444 50%, #b91c1c 100%) !important; /* red */
        color: #ffffff !important;
        border: none !important;
        border-radius: 999px !important;
        font-size: 1.75rem !important;
        width: 72px !important;
        height: 72px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        box-shadow: 0 4px 14px rgba(220, 38, 38, 0.35) !important;
        padding: 0 !important;
    }
    .report-gen-control-wrap .stButton > button:not([data-testid="stBaseButton-primary"]):not([kind="primary"]):hover {
        box-shadow: 0 6px 20px rgba(220, 38, 38, 0.5) !important;
        transform: translateY(-1px) scale(1.03) !important;
    }
    /* Logo card: padding and sizing so the white box fits the image nicely */
    .logo-overlay .logo-card {
        padding: 10px 12px !important; min-width: auto !important; max-width: 100% !important;
        display: flex !important; align-items: center !important; justify-content: center !important;
    }
    .logo-overlay .logo-card img { max-width: 140px !important; height: auto !important; display: block !important; }
    /* Reduced top padding so content sits just below the fixed header (removes gap above Welcome / RotriDash) */
    .main .block-container {
        padding-top: clamp(70px, 9vh, 100px) !important;
    }
    /* File uploader inside columns: remove double-border look */
    [data-testid="stColumn"] [data-testid="stFileUploader"] {
        border: none !important;
        box-shadow: none !important;
        padding: 0 !important;
    }
    [data-testid="stColumn"] [data-testid="stFileUploader"] section {
        border: 2px dashed #CBD5E1 !important;
        border-radius: 12px !important;
        padding: 1rem !important;
        background: #F8FAFC !important;
    }
    [data-testid="stColumn"] [data-testid="stFileUploader"] section:hover {
        border-color: #1B6CA8 !important;
        background: #EFF6FF !important;
    }
    /* Make the first column of horizontal blocks (used by RotriDash report preview) scrollable for long report previews */
    div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:first-child {
        max-height: 95vh !important;
        overflow-y: auto !important;
    }
    /* Remove gap between carousel (KPI cards) and File Management section */
    div:has(iframe) {
        margin-bottom: 0 !important;
        padding-bottom: 0 !important;
    }
    div:has(iframe) + div {
        margin-top: 0.5rem !important;
        padding-top: 0 !important;
    }
    /* Tighten Streamlit vertical block spacing before File Management */
    div[data-testid="stVerticalBlock"] > div:has(.fm-section-title) {
        margin-top: 0 !important;
        padding-top: 0 !important;
    }
    /* Reduce gap inside column that contains File Management */
    div[data-testid="stVerticalBlock"]:has(.fm-section-title) {
        padding-top: 0 !important;
        margin-top: 0 !important;
    }
    /* Pull File Management section up (wrapper around title) - works regardless of Streamlit DOM */
    .fm-file-management-wrap {
        margin-top: 0 !important;
        padding-top: 0 !important;
    }
    /* Pull the file upload block up so it sits close to the File Management title */
    div:has(.fm-file-management-wrap) + div {
        margin-top: 0 !important;
        padding-top: 0 !important;
    }
    /* Prevent double scrollbars on Streamlit dataframes */
    /* Completely disable Streamlit's internal scrolling mechanism */
    div[data-testid="stDataFrame"] {
        overflow: visible !important;
        max-height: none !important;
    }
    /* Disable scrolling on all internal divs of the dataframe */
    div[data-testid="stDataFrame"] > div {
        overflow: visible !important;
        max-height: none !important;
    }
    div[data-testid="stDataFrame"] > div > div {
        overflow: visible !important;
        max-height: none !important;
    }
    /* Hide any scrollbars that Streamlit creates internally */
    div[data-testid="stDataFrame"] * {
        overflow-y: visible !important;
    }
    /* Target the specific scrolling container Streamlit creates */
    div[data-testid="stDataFrame"] div[style*="overflow"] {
        overflow: visible !important;
    }
    div[data-testid="stDataFrame"] div[style*="max-height"] {
        max-height: none !important;
        overflow: visible !important;
    }
    /* File uploader: remove gap below, compact drop zone, small elite button */
    div:has(> section[data-testid="stFileUploader"]) {
        margin-bottom: 0 !important;
        padding-bottom: 0 !important;
    }
    section[data-testid="stFileUploader"] {
        margin-bottom: -0.5rem !important;
        padding-bottom: 0 !important;
    }
    /* Pull KPI/summary block up next to file uploader */
    div:has(section[data-testid="stFileUploader"]) + div,
    div:has(> section[data-testid="stFileUploader"]) + div {
        margin-top: 0 !important;
        padding-top: 0.25rem !important;
    }
    section[data-testid="stFileUploader"] > div {
        min-height: 48px !important;
        height: auto !important;
        padding: 0.4rem 1rem !important;
        margin-bottom: 0 !important;
    }
    section[data-testid="stFileUploader"] button {
        font-size: 0.65rem !important;
        padding: 0.2rem 0.4rem !important;
        border-radius: 5px !important;
        min-height: unset !important;
        max-width: 5.5rem !important;
        width: auto !important;
    }
    /* User profile popover (dropdown from 👤 icon) */
    /* User profile popover (dropdown from 👤 icon) */
    div[data-testid="column"]:has(.profile-popover-anchor) {
        position: fixed !important;
        top: clamp(90px, 10vh, 115px) !important;
        right: clamp(12px, 2vw, 24px) !important;
        width: min(280px, calc(100vw - 48px)) !important;
        background: #ffffff !important;
        border-radius: 16px !important;
        border: 1px solid #E2E8F0 !important;
        box-shadow: 0 12px 40px rgba(10, 46, 66, 0.15), 0 4px 12px rgba(0,0,0,0.06) !important;
        z-index: 2000 !important;
        padding: 1.25rem !important;
        animation: profile-slide-down 0.2s ease-out !important;
    }
    .user-profile-popover {
        /* Inner wrapper - inherited or reset */
        width: 100%;
    }
    @keyframes profile-slide-down {
        from { opacity: 0; transform: translateY(-8px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .user-profile-popover .profile-header {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin-bottom: 1rem;
        padding-bottom: 0.75rem;
        border-bottom: 1px solid #E2E8F0;
    }
    .user-profile-popover .profile-avatar {
        width: 44px; height: 44px;
        border-radius: 50%;
        background: linear-gradient(135deg, #0A2E42, #1B6CA8);
        color: #fff;
        display: flex; align-items: center; justify-content: center;
        font-size: 1.25rem; font-weight: 700;
        flex-shrink: 0;
    }
    .user-profile-popover .profile-name {
        font-size: 1rem; font-weight: 600; color: #0F2B3D; margin: 0; line-height: 1.2;
    }
    .user-profile-popover .profile-role {
        font-size: 0.78rem; color: #64748B; margin: 2px 0 0 0;
    }
    .user-profile-popover .profile-detail {
        display: flex; align-items: center; gap: 0.5rem;
        font-size: 0.85rem; color: #334155;
        padding: 0.4rem 0;
    }
    .user-profile-popover .profile-detail-icon {
        font-size: 0.9rem; width: 20px; text-align: center; flex-shrink: 0;
    }
    /* Popover close button — float top-right (only the FIRST button = ✕) */
    div[data-testid="column"]:has(.profile-popover-anchor) > div[data-testid="stVerticalBlockBorderWrapper"] > div > div[data-testid="stVerticalBlock"] > div.stElementContainer:nth-child(2) button {
        position: absolute !important;
        top: 0.5rem !important;
        right: 0.5rem !important;
        width: 28px !important;
        height: 28px !important;
        min-height: 28px !important;
        padding: 0 !important;
        border-radius: 50% !important;
        border: none !important;
        background: transparent !important;
        color: #94a3b8 !important;
        font-size: 1rem !important;
        cursor: pointer !important;
        z-index: 10 !important;
        transition: all 0.15s ease !important;
    }
    div[data-testid="column"]:has(.profile-popover-anchor) > div[data-testid="stVerticalBlockBorderWrapper"] > div > div[data-testid="stVerticalBlock"] > div.stElementContainer:nth-child(2) button:hover {
        background: #f1f5f9 !important;
        color: #475569 !important;
    }
    /* ═══════════════════════════════════════════════════════════════
       GLOBAL STREAMLIT WIDGET POLISH
       Premium styling for all native Streamlit components
       ═══════════════════════════════════════════════════════════════ */

    /* ── Typography: headings, paragraphs, labels ── */
    .stMarkdown h1 {
        font-weight: 800 !important;
        color: #0A2E42 !important;
        letter-spacing: -0.03em !important;
        line-height: 1.15 !important;
        margin-bottom: 0.5rem !important;
    }
    .stMarkdown h2 {
        font-weight: 700 !important;
        color: #0F2B3D !important;
        letter-spacing: -0.02em !important;
        line-height: 1.2 !important;
        margin-bottom: 0.4rem !important;
    }
    .stMarkdown h3 {
        font-weight: 700 !important;
        color: #0F2B3D !important;
        letter-spacing: -0.015em !important;
        line-height: 1.25 !important;
    }
    .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
        font-weight: 600 !important;
        color: #1E3A50 !important;
        letter-spacing: -0.01em !important;
    }
    .stMarkdown p, .stMarkdown li {
        color: #334155 !important;
        line-height: 1.65 !important;
        letter-spacing: -0.005em !important;
    }
    /* Force hero card text white — must come AFTER generic headings to win specificity */
    .stMarkdown .fp-hero h2,
    .stMarkdown .fp-hero-wrap h2,
    .fp-hero h2 {
        color: #ffffff !important;
        text-shadow: 0 2px 8px rgba(0,0,0,0.2) !important;
    }
    .stMarkdown .fp-hero p,
    .stMarkdown .fp-hero-wrap p,
    .fp-hero p {
        color: rgba(255,255,255,0.96) !important;
        text-shadow: 0 1px 3px rgba(0,0,0,0.12) !important;
    }
    /* Label text across all widgets */
    .stTextInput label, .stSelectbox label, .stMultiSelect label,
    .stDateInput label, .stTimeInput label, .stNumberInput label,
    .stTextArea label, .stFileUploader label, .stColorPicker label,
    .stSlider label, .stCheckbox label, .stRadio label {
        font-weight: 600 !important;
        color: #0F2B3D !important;
        font-size: 0.88rem !important;
        letter-spacing: -0.01em !important;
    }

    /* ── Buttons: Primary ── */
    .stButton > button[kind="primary"], .stButton > button[data-testid="stBaseButton-primary"] {
        background: linear-gradient(135deg, #0A2E42 0%, #0F3D5C 40%, #1B6CA8 100%) !important;
        color: #fff !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        letter-spacing: -0.01em !important;
        padding: 0.55rem 1.4rem !important;
        box-shadow: 0 3px 12px rgba(10, 46, 66, 0.2) !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    .stButton > button[kind="primary"]:hover, .stButton > button[data-testid="stBaseButton-primary"]:hover {
        box-shadow: 0 6px 20px rgba(10, 46, 66, 0.3) !important;
        transform: translateY(-1px) !important;
    }
    /* ── Buttons: Secondary ── */
    .stButton > button[kind="secondary"], .stButton > button[data-testid="stBaseButton-secondary"] {
        background: #fff !important;
        color: #0F2B3D !important;
        border: 1.5px solid #CBD5E1 !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        letter-spacing: -0.01em !important;
        padding: 0.55rem 1.4rem !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04) !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    .stButton > button[kind="secondary"]:hover, .stButton > button[data-testid="stBaseButton-secondary"]:hover {
        border-color: #1B6CA8 !important;
        color: #1B6CA8 !important;
        background: #F0F7FF !important;
        box-shadow: 0 3px 10px rgba(27, 108, 168, 0.1) !important;
    }
    /* ── Download button ── */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #0A2E42 0%, #1B6CA8 100%) !important;
        color: #fff !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        padding: 0.5rem 1.2rem !important;
        box-shadow: 0 3px 10px rgba(10, 46, 66, 0.18) !important;
        transition: all 0.2s ease !important;
    }
    .stDownloadButton > button:hover {
        box-shadow: 0 5px 16px rgba(10, 46, 66, 0.28) !important;
        transform: translateY(-1px) !important;
    }

    /* ── Tabs: sleek underline style ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0 !important;
        border-bottom: 2px solid #E2E8F0 !important;
        background: transparent !important;
    }
    .stTabs [data-baseweb="tab"] {
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        color: #64748B !important;
        letter-spacing: -0.01em !important;
        padding: 0.65rem 1.25rem !important;
        border-radius: 8px 8px 0 0 !important;
        border: none !important;
        background: transparent !important;
        transition: color 0.2s ease, background 0.2s ease !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #0F2B3D !important;
        background: #F1F5F9 !important;
    }
    .stTabs [aria-selected="true"] {
        color: #1B6CA8 !important;
        border-bottom: 3px solid #1B6CA8 !important;
        background: transparent !important;
        font-weight: 700 !important;
    }
    .stTabs [data-baseweb="tab-highlight"] {
        background-color: #1B6CA8 !important;
    }
    .stTabs [data-baseweb="tab-border"] {
        background-color: #E2E8F0 !important;
    }

    /* ── Text inputs, number inputs, text areas ── */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stTextArea > div > div > textarea {
        border: 1.5px solid #CBD5E1 !important;
        border-radius: 10px !important;
        padding: 0.6rem 0.85rem !important;
        font-size: 0.9rem !important;
        color: #0F2B3D !important;
        background: #FAFCFE !important;
        transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
    }
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: #1B6CA8 !important;
        box-shadow: 0 0 0 3px rgba(27, 108, 168, 0.1) !important;
        outline: none !important;
    }

    /* ── Selectbox / multiselect ── */
    [data-baseweb="select"] > div {
        border: 1.5px solid #CBD5E1 !important;
        border-radius: 10px !important;
        background: #FAFCFE !important;
        transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
    }
    [data-baseweb="select"] > div:focus-within {
        border-color: #1B6CA8 !important;
        box-shadow: 0 0 0 3px rgba(27, 108, 168, 0.1) !important;
    }
    [data-baseweb="select"] [data-baseweb="tag"] {
        background: #EFF6FF !important;
        border: 1px solid #BFDBFE !important;
        border-radius: 6px !important;
        color: #1B6CA8 !important;
        font-weight: 500 !important;
    }

    /* ── Metrics: uniform-height cards ── */
    /* Force every metric card to the same fixed height so rows with mixed
       delta / no-delta cards look perfectly level. */
    [data-testid="stMetric"] {
        background: #F8FAFC !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 14px !important;
        padding: 1rem 1.25rem !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.03) !important;
        transition: box-shadow 0.2s ease, transform 0.2s ease !important;
        min-height: 130px !important;
        box-sizing: border-box !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: flex-start !important;
    }
    [data-testid="stMetric"]:hover {
        box-shadow: 0 6px 18px rgba(10, 46, 66, 0.08) !important;
        transform: translateY(-2px) !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        color: #64748B !important;
        text-transform: uppercase !important;
        letter-spacing: 0.06em !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.75rem !important;
        font-weight: 800 !important;
        color: #0A2E42 !important;
        letter-spacing: -0.03em !important;
    }
    /* Reserve delta row height even when there is no delta rendered,
       so the label+value block always sits at the same vertical position. */
    [data-testid="stMetricDelta"] {
        font-weight: 600 !important;
        font-size: 0.82rem !important;
        min-height: 1.6em !important;
        display: block !important;
    }
    /* The stElementContainer wrapping stMetric sometimes collapses; un-collapse it */
    [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] > div,
    [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] > div > div {
        height: 100% !important;
    }

    /* ── Expanders ── */
    .streamlit-expanderHeader {
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        color: #0F2B3D !important;
        background: #F8FAFC !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 12px !important;
        padding: 0.75rem 1rem !important;
        transition: background 0.2s ease !important;
    }
    .streamlit-expanderHeader:hover {
        background: #EFF6FF !important;
        border-color: #CBD5E1 !important;
    }
    details[data-testid="stExpander"] {
        border: 1px solid #E2E8F0 !important;
        border-radius: 12px !important;
        overflow: hidden !important;
    }
    details[data-testid="stExpander"] summary {
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        color: #0F2B3D !important;
        padding: 0.75rem 1rem !important;
    }

    /* ── Dataframes ── */
    [data-testid="stDataFrame"] th {
        background: #F1F5F9 !important;
        color: #0F2B3D !important;
        font-weight: 700 !important;
        font-size: 0.82rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.04em !important;
        border-bottom: 2px solid #CBD5E1 !important;
    }
    [data-testid="stDataFrame"] td {
        font-size: 0.88rem !important;
        color: #334155 !important;
        border-bottom: 1px solid #F1F5F9 !important;
    }

    /* ── Alerts / info / success / warning / error ── */
    [data-testid="stAlert"] {
        border-radius: 12px !important;
        font-size: 0.9rem !important;
        font-weight: 500 !important;
        letter-spacing: -0.005em !important;
    }

    /* ── Checkbox & Radio ── */
    .stCheckbox label span, .stRadio label span {
        font-size: 0.9rem !important;
        color: #334155 !important;
    }

    /* ── Slider ── */
    [data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
        background: #1B6CA8 !important;
        border-color: #1B6CA8 !important;
    }

    /* ── Progress bar ── */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #0A2E42, #1B6CA8) !important;
        border-radius: 6px !important;
    }
    .stProgress > div > div {
        background: #E2E8F0 !important;
        border-radius: 6px !important;
    }

    /* ── Sidebar polish ── */
    [data-testid="stSidebar"] {
        background: #F8FAFC !important;
        border-right: 1px solid #E2E8F0 !important;
    }
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        font-size: 1rem !important;
        font-weight: 700 !important;
    }

    /* ── Date input ── */
    [data-testid="stDateInput"] > div > div > input {
        border: 1.5px solid #CBD5E1 !important;
        border-radius: 10px !important;
        font-size: 0.9rem !important;
        color: #0F2B3D !important;
        padding: 0.55rem 0.85rem !important;
    }
    [data-testid="stDateInput"] > div > div > input:focus {
        border-color: #1B6CA8 !important;
        box-shadow: 0 0 0 3px rgba(27, 108, 168, 0.1) !important;
    }

    /* ── Link button ── */
    .stLinkButton > a {
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        color: #1B6CA8 !important;
        border: 1.5px solid #CBD5E1 !important;
        border-radius: 10px !important;
        padding: 0.5rem 1rem !important;
        transition: all 0.2s ease !important;
        text-decoration: none !important;
    }
    .stLinkButton > a:hover {
        background: #EFF6FF !important;
        border-color: #1B6CA8 !important;
        box-shadow: 0 2px 8px rgba(27, 108, 168, 0.1) !important;
    }

    /* ── Toast / notification ── */
    [data-testid="stToast"] {
        border-radius: 12px !important;
        font-size: 0.88rem !important;
        font-weight: 500 !important;
        box-shadow: 0 8px 24px rgba(0,0,0,0.12) !important;
    }

    /* ── Form submit button ── */
    .stFormSubmitButton > button {
        background: linear-gradient(135deg, #0A2E42 0%, #1B6CA8 100%) !important;
        color: #fff !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        padding: 0.55rem 1.4rem !important;
        box-shadow: 0 3px 12px rgba(10, 46, 66, 0.2) !important;
        transition: all 0.2s ease !important;
    }
    .stFormSubmitButton > button:hover {
        box-shadow: 0 6px 20px rgba(10, 46, 66, 0.3) !important;
        transform: translateY(-1px) !important;
    }

    /* ── Spinner / loading ── */
    .stSpinner > div {
        border-top-color: #1B6CA8 !important;
    }

    /* ===== RESPONSIVE BREAKPOINTS ===== */
    @media (max-width: 1200px) {
        .fp-hero { padding: 2.5rem 2rem 2.5rem 3rem !important; }
        .fp-hero h2 { font-size: 2.2rem !important; }
        .fp-cards-row, .fp-usecases-wrap { max-width: 95% !important; }
        .fp-hero-wrap { max-width: 95% !important; }
        .fm-uniform-wrap, .fm-kpi-card, .fm-carousel-wrap { max-width: 95% !important; }
        .fm-welcome { max-width: 95% !important; }
        div[data-testid="column"]:has(.fp-cta-btn-wrap) { padding-left: 3rem !important; }
        .report-quick-preview-sidebar { width: min(420px, 100vw) !important; }
    }
    @media (max-width: 992px) {
        .fp-hero { padding: 2rem 1.5rem !important; }
        .fp-hero h2 { font-size: 1.8rem !important; }
        .fp-hero p { font-size: 1.05rem !important; }
        .fp-cards-row { flex-wrap: nowrap !important; overflow-x: auto !important; }
        .fp-cards-row .fp-card { min-width: 220px !important; flex: 0 0 auto !important; }
        .fp-card { height: auto !important; min-height: 200px !important; }
        .fp-usecases-row { flex-wrap: nowrap !important; overflow-x: auto !important; }
        .fp-usecases-row .fp-usecase { min-width: 220px !important; flex: 0 0 auto !important; }
        .fp-usecase { min-height: 180px !important; }
        .fm-kpi-card { flex-direction: column !important; text-align: center !important; }
        .fm-kpi-row { flex-wrap: wrap !important; }
        .fm-welcome { font-size: 1.4rem !important; }
        .report-quick-preview-sidebar { width: min(380px, 100vw) !important; }
        div[data-testid="column"]:has(.fp-cta-btn-wrap) { padding-left: 1.5rem !important; }
        .fm-detail-title { font-size: 2rem !important; }
        .fm-detail-sub { font-size: 1.4rem !important; }
    }
    @media (max-width: 768px) {
        .app-top-bar-bg { height: 80px !important; }
        div[data-testid="column"]:has(.logo-home-wrap) { top: 20px !important; left: 12px !important; }
        div[data-testid="column"]:has(.logo-home-wrap) .logo-overlay { top: -24px !important; }
        div[data-testid="column"]:has(.details-icon-btn-wrap) { top: 24px !important; right: 8px !important; width: 130px !important; gap: 10px !important; }
        div[data-testid="column"]:has(.details-icon-btn-wrap) .stButton,
        div[data-testid="column"]:has(.details-icon-btn-wrap) [data-testid="stVerticalBlock"] > .stButton {
            width: 44px !important; height: 44px !important; min-width: 44px !important; min-height: 44px !important;
        }
        div[data-testid="column"]:has(.details-icon-btn-wrap) .stButton > button {
            width: 44px !important; height: 44px !important; min-width: 44px !important; min-height: 44px !important;
            font-size: 1.35rem !important; border-radius: 10px !important;
        }
        div[data-testid="column"]:has(.details-icon-btn-wrap) > div[data-testid="column"],
        div[data-testid="column"]:has(.details-icon-btn-wrap) > div > div[data-testid="column"] {
            width: 44px !important; min-width: 44px !important; max-width: 44px !important;
        }
        .main .block-container { padding-top: 50px !important; }
        .fp-hero h2 { font-size: 1.5rem !important; }
        .fp-hero p { font-size: 0.95rem !important; }
        .fp-cards-row .fp-card { flex: 1 1 100% !important; min-width: 0 !important; }
        .fp-card { height: auto !important; }
        .fp-usecases-row .fp-usecase { flex: 1 1 100% !important; min-width: 0 !important; min-height: 160px !important; }
        .fp-usecases-wrap { padding: 1.5rem 1.25rem !important; }
        .user-profile-popover { width: min(260px, calc(100vw - 32px)) !important; right: 8px !important; top: 85px !important; }
        .report-quick-preview-sidebar { width: 100vw !important; right: 0 !important; }
        .fm-detail-title { font-size: 1.5rem !important; }
        .fm-detail-sub { font-size: 1.1rem !important; }
        .fm-detail-icon { width: 80px !important; height: 80px !important; min-width: 80px !important; }
        .fm-file-management-wrap { margin-top: 0 !important; }
        div:has(.fm-file-management-wrap) + div { margin-top: 0 !important; }
        div[data-testid="column"]:has(.fp-cta-btn-wrap) { padding-left: 1rem !important; }
        .fm-kpi-card { padding: 1rem 1.25rem !important; }
        .fm-kpi-value { font-size: 1.35rem !important; }
        .fp-section-label { font-size: 1.2rem !important; }
        .fp-use-heading { font-size: 1.2rem !important; }
    }
    </style>
    """, unsafe_allow_html=True)

    # Background bar element for the fixed header row
    st.markdown('<div class="app-top-bar-bg"></div>', unsafe_allow_html=True)
