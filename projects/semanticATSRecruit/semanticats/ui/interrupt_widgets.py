from __future__ import annotations

import streamlit as st


def render_interrupts(interrupts: list[dict]) -> None:
    if not interrupts:
        st.info("No HITL checkpoints were created in this run.")
        return
    st.warning(f"{len(interrupts)} HITL checkpoint(s) need recruiter review.")
    for item in interrupts:
        label = str(item.get("checkpoint", "checkpoint")).replace("_", " ").title()
        with st.expander(f"{label} - {item.get('status', 'review')}", expanded=True):
            st.write(item.get("message"))
            st.json(item.get("payload"))
