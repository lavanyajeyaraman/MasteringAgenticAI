from __future__ import annotations

import streamlit as st


def apply_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #101418;
            --panel: #171d23;
            --panel-2: #1e2630;
            --line: #2d3845;
            --text: #edf2f7;
            --muted: #9aa7b5;
            --accent: #31d0aa;
            --accent-2: #f7b955;
            --danger: #ff6b6b;
        }
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(49, 208, 170, .09), transparent 32rem),
                linear-gradient(180deg, #0e1317 0%, #111820 100%);
            color: var(--text);
        }
        header[data-testid="stHeader"] { background: transparent; }
        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 7rem;
            max-width: 1220px;
        }
        h1, h2, h3 { letter-spacing: 0; }
        div[data-testid="stMetric"] {
            background: linear-gradient(180deg, var(--panel-2), var(--panel));
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 1rem;
            min-height: 118px;
        }
        div[data-testid="stMetricLabel"] p { color: var(--muted); }
        div[data-testid="stMetricValue"] { color: var(--text); }
        .smart-panel {
            background: rgba(23, 29, 35, .88);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 1rem;
        }
        .smart-top {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            padding: 1.05rem 1.15rem;
            border: 1px solid var(--line);
            border-radius: 8px;
            background:
                linear-gradient(135deg, rgba(49, 208, 170, .18), rgba(247, 185, 85, .08)),
                rgba(23, 29, 35, .86);
            margin-bottom: 1rem;
        }
        .brand {
            display: flex;
            align-items: center;
            gap: .75rem;
            font-size: 2.15rem;
            font-weight: 850;
            line-height: 1;
            color: #ffffff;
        }
        .brand-mark {
            width: 46px;
            height: 46px;
            display: inline-grid;
            place-items: center;
            border-radius: 8px;
            background: linear-gradient(135deg, #31d0aa, #f7b955);
            color: #101418;
            font-weight: 900;
            box-shadow: 0 10px 32px rgba(49, 208, 170, .22);
        }
        .brand-tagline {
            color: #c8d3df;
            font-size: .98rem;
            margin-top: .35rem;
        }
        .subtle { color: var(--muted); }
        .stars { color: var(--accent-2); font-size: 1.25rem; }
        .status-pill {
            display: inline-flex;
            align-items: center;
            padding: .25rem .55rem;
            border-radius: 999px;
            border: 1px solid var(--line);
            color: var(--muted);
            font-size: .85rem;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: .2rem;
            border-bottom: 1px solid var(--line);
        }
        .stTabs [data-baseweb="tab"] {
            background: transparent;
            border-radius: 8px 8px 0 0;
            color: var(--muted);
            padding: .75rem 1rem;
        }
        .stTabs [aria-selected="true"] {
            color: var(--text);
            background: rgba(49, 208, 170, .12);
        }
        .stProgress > div > div > div > div { background-color: var(--danger); }
        .st-key-floating_chat {
            position: fixed;
            right: 1.4rem;
            bottom: 1.2rem;
            z-index: 9999;
            width: min(380px, calc(100vw - 2rem));
            background: rgba(23, 29, 35, .96);
            border: 1px solid var(--line);
            border-radius: 8px;
            box-shadow: 0 18px 60px rgba(0, 0, 0, .36);
            padding: .75rem;
        }
        .st-key-floating_chat [data-testid="stVerticalBlock"] { gap: .45rem; }
        .stButton > button {
            border-radius: 8px;
            border: 1px solid var(--line);
        }
        @media (max-width: 760px) {
            .smart-top { align-items: flex-start; flex-direction: column; }
            .block-container { padding-left: .9rem; padding-right: .9rem; }
            .st-key-floating_chat { right: .75rem; bottom: .75rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
