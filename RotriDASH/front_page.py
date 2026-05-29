#type: ignore
"""
Front page / landing page for RotriDash.

Renders the hero banner, About section, feature cards, use-case section, and CTA.
All CSS is inline (scoped to the front page only).
"""
import html as html_mod

import streamlit as st

from auth import is_authenticated, is_approved


def render_front_page():
    """Render the landing page hero, feature cards, use-cases, and CTA."""

    # ── Front-page scoped CSS ─────────────────────────────────────
    st.markdown("""
    <style>
    /* Premium front page: depth, motion, and polish */
    @keyframes fp-fade-up {
        from { opacity: 0; transform: translateY(12px); }
        to { opacity: 1; transform: translateY(0); }
    }
    [data-testid="stAppViewContainer"] .front-page-wrap {
        padding-top: 0 !important;
        margin-top: -1rem !important;
        padding-bottom: 2rem !important;
        background: linear-gradient(180deg, #E8EEF4 0%, #EDF2F7 8%, #F1F5F9 22%, #F8FAFC 40%, #ffffff 60%);
        min-height: auto;
    }
    /* Remove Streamlit's default block gap above hero */
    [data-testid="stAppViewContainer"] .front-page-wrap .fp-hero-wrap {
        margin-top: 0 !important;
    }
    .fp-hero-wrap {
        max-width: min(86%, 1400px);
        margin-left: auto;
        margin-right: auto;
        overflow: hidden;
        padding-bottom: 6px;
    }
    .fp-hero {
        width: 100%;
        box-sizing: border-box;
        background: linear-gradient(135deg, #0A2E42 0%, #0F3D5C 22%, #185F80 48%, #1B6CA8 75%, #4A9FD4 100%);
        border-radius: 24px;
        padding: clamp(2rem, 4vw, 3.75rem) clamp(1.5rem, 3vw, 2.5rem) clamp(2rem, 4vw, 3.75rem) clamp(2rem, 8vw, 10rem);
        margin-bottom: 0.5rem;
        color: #fff;
        box-shadow: 0 12px 40px rgba(10, 46, 66, 0.18), 0 4px 16px rgba(10, 46, 66, 0.06);
        position: relative;
        overflow: hidden;
        animation: fp-fade-up 0.6s ease-out;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: flex-start;
    }
    .fp-hero::before {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        background: radial-gradient(ellipse 90% 60% at 50% -10%, rgba(255,255,255,0.2) 0%, transparent 50%);
        pointer-events: none;
    }
    .fp-hero::after {
        content: "";
        position: absolute;
        bottom: 0; left: 50%; transform: translateX(-50%);
        width: 120px;
        height: 3px;
        background: linear-gradient(90deg, transparent, rgba(201, 162, 39, 0.6), transparent);
        border-radius: 2px;
        pointer-events: none;
    }
    .fp-hero h2 {
        margin: 0 0 1rem 0;
        font-size: clamp(1.5rem, 3.5vw, 2.75rem);
        font-weight: 700;
        line-height: 1.12;
        letter-spacing: -0.02em;
        position: relative;
        color: #ffffff;
        text-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }
    .fp-hero p {
        margin: 0;
        font-size: clamp(0.95rem, 1.5vw, 1.24rem);
        line-height: 1.5;
        color: rgba(255,255,255,0.96);
        max-width: 42em;
        margin-left: 0;
        margin-right: auto;
        position: relative;
        text-shadow: 0 1px 3px rgba(0,0,0,0.12);
        font-weight: 500;
        letter-spacing: -0.01em;
    }
    .fp-section-label {
        font-size: 1.5rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #0F2B3D;
        margin-bottom: 1.25rem;
        text-align: center;
        animation: fp-fade-up 0.6s ease-out 0.1s both;
    }
    .fp-section-label::after {
        content: "";
        display: block;
        width: 40px;
        height: 2px;
        background: linear-gradient(90deg, transparent, #0F2B3D, transparent);
        margin: 0.75rem auto 0;
        border-radius: 1px;
    }
    .fp-cards-row {
        display: flex;
        gap: 1.25rem;
        margin-bottom: 2.5rem;
        margin-left: auto;
        margin-right: auto;
        max-width: min(85%, 1400px);
        flex-wrap: wrap;
        animation: fp-fade-up 0.6s ease-out 0.15s both;
    }
    .fp-cards-row .fp-card {
        flex: 1 1 260px;
        min-width: 0;
        max-width: 100%;
    }
    .fp-card {
        width: 100%;
        height: auto;
        min-height: 0;
        display: flex;
        flex-direction: column;
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 18px;
        padding: 1.75rem 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.04), 0 1px 3px rgba(0,0,0,0.04);
        transition: transform 0.28s cubic-bezier(0.25, 0.46, 0.45, 0.94), box-shadow 0.28s ease, border-color 0.28s ease;
        box-sizing: border-box;
    }
    .fp-card:hover {
        transform: translateY(-6px);
        box-shadow: 0 20px 40px rgba(27, 108, 168, 0.12), 0 4px 12px rgba(0,0,0,0.06);
        border-color: #1B6CA8;
        background: #fff;
    }
    .fp-card-icon {
        width: 40px;
        height: 40px;
        border-radius: 10px;
        background: linear-gradient(135deg, #0F3D5C 0%, #1B6CA8 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 1rem;
    }
    .fp-card-icon svg {
        width: 22px;
        height: 22px;
        fill: none;
        stroke: #fff;
        stroke-width: 2;
        stroke-linecap: round;
        stroke-linejoin: round;
    }
    .fp-card h3 {
        margin: 0 0 0.6rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #1B6CA8;
        font-size: 1.18rem;
        font-weight: 700;
        color: #0F2B3D;
        line-height: 1.3;
        text-align: left;
    }
    .fp-card p {
        margin: 0;
        font-size: 1.02rem;
        line-height: 1.55;
        color: #334155;
        text-align: justify;
        flex: 1;
    }
    .fp-cta-wrap {
        text-align: left;
        margin-top: 0.5rem;
        margin-bottom: 1rem;
        padding: 1rem 0 0.75rem;
        animation: fp-fade-up 0.6s ease-out 0.3s both;
    }
    .fp-cta-divider {
        width: 80px;
        height: 2px;
        background: linear-gradient(90deg, #1B6CA8, transparent);
        margin: 0 0 0.75rem 0;
        border-radius: 1px;
        opacity: 0.8;
    }
    .fp-cta-tagline {
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #64748B;
        margin-bottom: 1rem;
    }
    .fp-below-cta {
        width: 100%;
        margin-top: -1.5rem; /* Reduced empty space above cards */
        position: relative;
        z-index: 5;
    }
    .fp-below-cta .fp-section-label {
        margin-top: 0;
    }
    .fp-about-wrap {
        max-width: min(88%, 900px);
        margin: 0 auto 2.25rem auto;
        padding: 0 clamp(1rem, 3vw, 2rem);
        animation: fp-fade-up 0.6s ease-out 0.12s both;
    }
    .fp-about {
        font-family: system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        background: #eef6fb;
        border: 1px solid #d8e8f2;
        border-radius: 16px;
        padding: clamp(1.75rem, 3.5vw, 2.5rem) clamp(1.5rem, 3.5vw, 2.75rem);
        box-shadow: 0 1px 3px rgba(26, 43, 60, 0.06), 0 8px 28px rgba(26, 43, 60, 0.04);
    }
    .fp-about h3 {
        margin: 0 0 1.125rem 0;
        padding-bottom: 0;
        font-size: clamp(1.2rem, 2vw, 1.35rem);
        font-weight: 700;
        color: #1a2b3c;
        letter-spacing: -0.025em;
        line-height: 1.3;
    }
    .fp-about p {
        margin: 0 0 1.25rem 0;
        font-size: clamp(0.98rem, 1.15vw, 1.0625rem);
        font-weight: 400;
        line-height: 1.72;
        color: #4a4a4a;
        text-align: justify;
        letter-spacing: 0.01em;
    }
    .fp-about p:last-child {
        margin-bottom: 0;
    }
    .fp-cta-sub {
        font-size: 0.88rem;
        color: #6c757d;
        margin-top: 0.75rem;
    }
    /* CTA button: align with hero text */
    div[data-testid="column"]:has(.fp-cta-btn-wrap) {
        padding-left: clamp(4rem, 12vw, 12rem) !important; /* Moved button to the right */
    }
    /* CTA button styling */
    div[data-testid="column"]:has(.fp-cta-btn-wrap) .stButton > button {
        background: linear-gradient(135deg, #0A2E42 0%, #0F3D5C 35%, #1B6CA8 100%) !important;
        color: #fff !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        padding: 0.6rem 1.25rem !important;
        letter-spacing: -0.01em !important;
        box-shadow: 0 4px 14px rgba(10, 46, 66, 0.25) !important;
        transition: box-shadow 0.2s ease, transform 0.2s ease !important;
        animation: fp-fade-up 0.6s ease-out 0.3s both !important;
    }
    div[data-testid="column"]:has(.fp-cta-btn-wrap) .stButton > button:hover {
        box-shadow: 0 6px 20px rgba(10, 46, 66, 0.35) !important;
    }
    .fp-use-heading {
        font-size: 1.5rem;
        font-weight: 700;
        color: #0F2B3D;
        text-align: center;
        margin: 2rem 0 1.25rem 0;
        text-transform: uppercase;
        animation: fp-fade-up 0.6s ease-out 0.2s both;
    }
    /* USE CASES: animations */
    @keyframes fp-usecase-reveal {
        from { opacity: 0; transform: translateY(20px) scale(0.95); }
        to { opacity: 1; transform: translateY(0) scale(1); }
    }
    @keyframes fp-usecase-num-pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    .fp-usecases-wrap {
        max-width: min(85%, 1400px);
        margin-left: auto;
        margin-right: auto;
        margin-bottom: 2.5rem;
        background: linear-gradient(180deg, #FDFCF8 0%, #FAF9F5 100%);
        border-radius: 22px;
        padding: 2.5rem 2.75rem;
        border: 1px solid #E8E3D8;
        box-shadow: 0 8px 32px rgba(184, 148, 31, 0.08), 0 2px 8px rgba(184, 148, 31, 0.04), 0 1px 0 rgba(255,255,255,0.9) inset;
        animation: fp-fade-up 0.6s ease-out 0.25s both;
        position: relative;
        overflow: hidden;
    }
    .fp-usecases-wrap::before {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, #B8941F, transparent);
        opacity: 0.3;
    }
    .fp-usecases-row {
        display: flex;
        gap: 1.75rem;
        flex-wrap: nowrap;
        position: relative;
        overflow-x: auto;
    }
    .fp-usecases-row .fp-usecase {
        flex: 1 1 240px;
        min-width: 240px;
    }
    .fp-usecase {
        width: 100%;
        display: flex;
        flex-direction: column;
        padding: 1.5rem 1.25rem 1.5rem 1.5rem;
        border-left: 4px solid #B8941F;
        min-height: 220px;
        box-sizing: border-box;
        background: rgba(255, 255, 255, 0.45);
        border-radius: 12px;
        transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
        animation: fp-usecase-reveal 0.6s ease-out both;
        cursor: default;
    }
    .fp-usecase:nth-child(1) { animation-delay: 0.3s; }
    .fp-usecase:nth-child(2) { animation-delay: 0.4s; }
    .fp-usecase:nth-child(3) { animation-delay: 0.5s; }
    .fp-usecase:nth-child(4) { animation-delay: 0.6s; }
    .fp-usecase::before {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        background: linear-gradient(135deg, rgba(184, 148, 31, 0.03) 0%, rgba(10, 46, 66, 0.02) 100%);
        opacity: 0;
        transition: opacity 0.35s ease;
        pointer-events: none;
    }
    .fp-usecase:hover {
        border-left-color: #0F3D5C;
        transform: translateY(-4px) scale(1.02);
        box-shadow: 0 12px 28px rgba(10, 46, 66, 0.12), 0 4px 12px rgba(184, 148, 31, 0.08);
        background: rgba(255, 255, 255, 0.7);
    }
    .fp-usecase:hover::before {
        opacity: 1;
    }
    .fp-usecase-num {
        font-size: 1.85rem;
        font-weight: 800;
        color: #B8941F;
        line-height: 1;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
        text-shadow: 0 2px 4px rgba(184, 148, 31, 0.2);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        display: inline-block;
    }
    .fp-usecase:hover .fp-usecase-num {
        color: #0F3D5C;
        transform: scale(1.1);
        text-shadow: 0 2px 6px rgba(15, 61, 92, 0.25);
    }
    .fp-usecase h4 {
        margin: 0 0 0.75rem 0;
        font-size: 1.15rem;
        font-weight: 700;
        color: #0F2B3D;
        line-height: 1.3;
        transition: color 0.3s ease;
    }
    .fp-usecase:hover h4 {
        color: #0F3D5C;
    }
    .fp-usecase .fp-usecase-p {
        margin: 0;
        font-size: 0.95rem;
        line-height: 1.6;
        color: #334155;
        text-align: justify;
        flex: 1;
        transition: color 0.3s ease;
    }
    .fp-usecase:hover .fp-usecase-p {
        color: #1E293B;
    }
    </style>
    """, unsafe_allow_html=True)

    # ── Feature card icons (SVG) ──────────────────────────────────
    fp_icons = [
        '<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path fill="none" stroke="#fff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" d="M3 17l4-6 4 4 5-8 4 6"/></svg>',
        '<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path fill="none" stroke="#fff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" d="M12 3a5 5 0 00-3.5 8.5L12 16l3.5-4.5A5 5 0 0012 3z"/><path fill="none" stroke="#fff" stroke-width="2" d="M9 18h6M9 14h6"/></svg>',
        '<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path fill="none" stroke="#fff" stroke-width="2" d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/><path fill="none" stroke="#fff" stroke-width="2" d="M14 2v6h6M16 13H8M16 17H8M10 9H8"/></svg>',
        '<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path fill="none" stroke="#fff" stroke-width="2" d="M22 19a2 2 0 01-2 2H4a2 2 0 01-2-2V5a2 2 0 012-2h5l2 3h9a2 2 0 012 2z"/><path fill="none" stroke="#fff" stroke-width="2" d="M12 11v6M9 14h6"/></svg>',
    ]
    key_points = [
        ("Smart Visualization", "Explore interactive plots with configurable X and dual Y-axes. Thrust, efficiency, vibration, torque\u2014in one view."),
        ("AI\u2011Assisted Intelligence", "Uncover performance peaks, key trends and statistics, reveal what\u2019s driving results."),
        ("Ready to Share Reports", "Generate sleek, branded reports with your chosen graphs \u2014 instant and presentation-ready."),
        ("Connected Testing Flow", "Bring ROTRIX or other bench exports into RotriDash and close the loop from test stand to charts and reports."),
    ]
    cards_html = "".join(
        f'<div class="fp-card"><div class="fp-card-icon">{fp_icons[i]}</div><h3>{html_mod.escape(t)}</h3><p>{html_mod.escape(d)}</p></div>'
        for i, (t, d) in enumerate(key_points)
    )

    # ── Use-case section ──────────────────────────────────────────
    use_cases = [
        ("UAV propulsion tuning", "Compare propulsion setups and throttle profiles to optimize endurance and flight stability."),
        ("R&D and Prototyping", "Iterate faster \u2014 analyze multiple test runs, refine motor efficiency, and benchmark evolving designs."),
        ("Quality Assurance", "Ensure performance integrity across every batch. Align units with certification and audit standards."),
        ("Flight Performance Prediction", "Transform test-stand data into real-world behaviour insights. Forecast performance for flight optimization."),
    ]
    use_cases_html = "".join(
        f'<div class="fp-usecase"><span class="fp-usecase-num">0{i+1}</span><h4>{html_mod.escape(t)}</h4><p class="fp-usecase-p">{html_mod.escape(d)}</p></div>'
        for i, (t, d) in enumerate(use_cases)
    )

    # ── Part 1: Hero banner ───────────────────────────────────────
    front_page_part1 = """
    <div class="front-page-wrap">
    <div class="fp-hero-wrap">
    <div class="fp-hero" role="banner">
        <h2>RotriDASH – Decision Analytics & Support Hub</h2>
        <p>No scripts, no code — just upload your ROTRIX or any motor test-bed data and uncover real performance in seconds with benchmark insights, anomaly flags, and engineering recommendations.</p>
    </div>
    </div>
    """
    st.markdown(front_page_part1, unsafe_allow_html=True)

    # ── CTA button ────────────────────────────────────────────────
    cta_col, _ = st.columns([1, 3])
    with cta_col:
        st.markdown('<div class="fp-cta-btn-wrap" style="height:0;overflow:hidden;margin:0;padding:0;"></div>', unsafe_allow_html=True)
        if is_authenticated() and is_approved():
            if st.button("Ready To Transform \u2192", type="primary", use_container_width=True, key="front_page_cta"):
                st.session_state.show_front_page = False
                st.session_state.author_details_completed = True
                # Viewers go to Report History (read-only); others go to upload
                from auth import check_role
                if check_role(["viewer"]):
                    st.session_state.show_report_history = True
                    st.session_state.show_calculators = False
                else:
                    st.session_state.show_upload_area = True
                    st.session_state.show_calculators = False
                st.rerun()
        else:
            if st.button("Ready To Transform \u2192", type="primary", use_container_width=True, key="front_page_cta"):
                st.session_state.show_front_page = False
                st.session_state.show_login_form = True
                st.rerun()

    # ── About RotriDash (temporarily hidden) ───────────────────────
    about_html = ""

    # ── Part 2: About + feature cards + use-cases ─────────────────
    front_page_part2 = f"""
    <div class="fp-below-cta">
    {about_html}
    <p class="fp-section-label">What Powers It</p>
    <div class="fp-cards-row">{cards_html}</div>
    <p class="fp-use-heading">Where It Excels</p>
    <div class="fp-usecases-wrap"><div class="fp-usecases-row">{use_cases_html}</div></div>
    </div>
    </div>
    """
    with st.container():
        st.markdown(front_page_part2, unsafe_allow_html=True)

