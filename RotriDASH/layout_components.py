"""
Shared layout components for the RotriDash dashboard.

Currently exposes a single footer component that can be reused across
all Streamlit pages in this app.
"""

from __future__ import annotations

import datetime

import streamlit as st


def render_footer() -> None:
    """Render a consistent footer at the bottom of the page.

    CSS is injected on every call because Streamlit rebuilds the page on each rerun.
    """
    st.markdown(
        """
        <style>
        /* Global footer bar — always at the bottom of the viewport */
        .mdv-footer {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            z-index: 100;
            padding: 1rem 0;
            background: #ffffff;
            border-top: 1px solid #E2E8F0;
            box-shadow: 0 -4px 12px rgba(10, 46, 66, 0.04);
            font-size: 0.86rem;
            color: #64748B;
            width: 100%;
        }
        /* Prevent content from being hidden behind the fixed footer */
        [data-testid="stAppViewContainer"] > .main {
            padding-bottom: 4rem !important;
        }
        .mdv-footer-inner {
            max-width: 1200px;
            margin: 0 auto;
            padding: 0 1.5rem;
            display: flex;
            flex-wrap: wrap;
            justify-content: space-between;
            align-items: center;
            gap: 0.75rem;
        }
        .mdv-footer-left {
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            gap: 0.4rem;
        }
        .mdv-footer-center {
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            gap: 0.35rem;
            font-size: 0.83rem;
            color: #64748B;
        }
        .mdv-footer-brand {
            font-weight: 600;
            color: #E77618;
        }
        .mdv-footer-copy {
            color: #64748B;
        }
        .mdv-footer-right {
            color: #475569;
        }
        .mdv-footer-sep {
            color: #CBD5E1;
        }
        .mdv-footer a {
            color: #1B6CA8;
            text-decoration: none;
        }
        .mdv-footer a:hover {
            text-decoration: underline;
        }
        /* ===== Footer responsive breakpoints ===== */
        @media (max-width: 768px) {
            .mdv-footer-inner {
                flex-direction: column;
                text-align: center;
                gap: 0.5rem;
            }
            .mdv-footer-left, .mdv-footer-center, .mdv-footer-right {
                justify-content: center;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    year = datetime.datetime.now().year

    st.markdown(
        f"""
        <div class="mdv-footer">
          <div class="mdv-footer-inner">
            <div class="mdv-footer-left">
              <span class="mdv-footer-copy">© {year} REUDE Technologies | All rights reserved.</span>
            </div>
            <div class="mdv-footer-center">
              <span class="mdv-footer-copy">
                Contact support:
                <a href="mailto:support.rotrix@redue.tech">support.rotrix@redue.tech</a>
              </span>
            </div>
            <div class="mdv-footer-right">
              Powered by <span class="mdv-footer-brand">REUDE</span>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

