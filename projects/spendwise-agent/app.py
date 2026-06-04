from __future__ import annotations

import hashlib
import os

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

from src.ai import ask_openai, extract_transactions_from_pdf_text, generate_tips
from src.analytics import (
    ai_context,
    available_months,
    category_summary,
    daily_spending,
    detect_subscriptions,
    filter_month,
    overspending_alerts,
    pattern_observations,
    prepare_transactions,
    savings_opportunities,
    summary_metrics,
    unusual_merchants,
)
from src.ingestion import extract_pdf_text, merge_category_rules, parse_csv, parse_pdf_text, rules_from_reviewed_transactions
from src.llm import available_providers, get_provider, get_provider_spec, has_api_key, provider_label
from src.sample_data import load_sample_transactions
from src.styles import apply_styles


st.set_page_config(page_title="SpendWise Agent", page_icon="💳", layout="wide")
load_dotenv()
apply_styles()


def init_state() -> None:
    if "transactions" not in st.session_state:
        st.session_state.transactions = load_sample_transactions()
        st.session_state.using_sample = True
    if "parse_status" not in st.session_state:
        st.session_state.parse_status = ["Using sample transactions until you upload CSV or PDF data."]
    if "schema_mapping" not in st.session_state:
        st.session_state.schema_mapping = {}
    if "chat_open" not in st.session_state:
        st.session_state.chat_open = False
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "last_upload_signature" not in st.session_state:
        st.session_state.last_upload_signature = None


def money(value: float) -> str:
    return f"${value:,.0f}"


def month_label(month: str) -> str:
    if month == "All":
        return "All periods"
    return pd.Period(month).strftime("%B %Y")


def configure_ai_key() -> None:
    with st.sidebar:
        st.subheader("AI Provider")
        provider = st.selectbox(
            "Provider",
            available_providers(),
            index=available_providers().index(get_provider()),
            format_func=provider_label,
        )
        os.environ["AI_PROVIDER"] = provider
        spec = get_provider_spec(provider)
        key = st.text_input(
            "API key",
            type="password",
            placeholder=f"Paste key or set {spec.api_key_env} in .env",
            help="Stored only in this Streamlit session unless you add it to .env yourself.",
        )
        if key:
            os.environ[spec.api_key_env] = key.strip()
            st.success(f"{provider_label()} key loaded for this session.")
        model = st.text_input(
            "Model",
            value=os.getenv(spec.model_env, spec.default_model),
        )
        if model:
            os.environ[spec.model_env] = model.strip()
        if not has_api_key():
            st.info(f"Create a .env file from .env.example or paste a {provider_label()} key here.")
        st.divider()
        st.subheader("Data")
        if st.button("Reset uploaded data", help="Clear uploaded transactions and return to sample data."):
            st.session_state.transactions = load_sample_transactions()
            st.session_state.using_sample = True
            st.session_state.parse_status = ["Reset to sample transactions. Upload a CSV or PDF to replace them."]
            st.session_state.schema_mapping = {}
            st.session_state.last_upload_signature = None
            st.rerun()


def upload_signature(files) -> tuple[tuple[str, int], ...] | None:
    if not files:
        return None
    signature = []
    for uploaded in files:
        data = uploaded.getvalue()
        signature.append((uploaded.name, len(data), hashlib.sha256(data).hexdigest()))
    return tuple(signature)


def load_uploads(files, force: bool = False) -> bool:
    if not files:
        return False
    signature = upload_signature(files)
    if not force and signature == st.session_state.last_upload_signature:
        return False
    frames = []
    statuses = []
    mappings = {}
    for uploaded in files:
        uploaded.seek(0)
        name = uploaded.name.lower()
        if name.endswith(".csv"):
            result = parse_csv(uploaded)
            frames.append(result.transactions)
            statuses.append(f"{uploaded.name}: {result.status}")
            mappings[uploaded.name] = result.mapping
        elif name.endswith(".pdf"):
            text = extract_pdf_text(uploaded)
            statuses.append(f"{uploaded.name}: extracted {len(text):,} characters of PDF text.")
            result = parse_pdf_text(text)
            pdf_frames = [result.transactions]
            if result.transactions.empty:
                if not has_api_key():
                    statuses.append(
                        f"{uploaded.name}: no table-like transactions found. Add a {provider_label()} API key to enable AI PDF extraction fallback."
                    )
                else:
                    ai_df = extract_transactions_from_pdf_text(text)
                    if not ai_df.empty:
                        pdf_frames.append(ai_df)
                        statuses.append(f"{uploaded.name}: extracted {len(ai_df)} rows with {provider_label()} fallback.")
                    else:
                        statuses.append(f"{uploaded.name}: {provider_label()} fallback did not find transactions.")
            else:
                statuses.append(f"{uploaded.name}: {result.status}")
            frames.extend([frame for frame in pdf_frames if not frame.empty])
        else:
            statuses.append(f"{uploaded.name}: unsupported file type.")
    if frames:
        st.session_state.transactions = pd.concat(frames, ignore_index=True).sort_values("date")
        st.session_state.using_sample = False
    else:
        statuses.append("No upload rows were added. Keeping the current dashboard data.")
    st.session_state.parse_status = statuses
    st.session_state.schema_mapping = mappings
    st.session_state.last_upload_signature = signature
    return True


def top_bar(months: list[str]) -> str | None:
    st.markdown(
        """
        <div class="smart-top">
            <div>
                <div class="brand"><span class="brand-mark">$</span><span>SpendWise Agent</span></div>
                <div class="brand-tagline">Your statement-to-insight expense copilot</div>
            </div>
            <div class="subtle">Upload, review, and understand your spending</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    month_col, status_col = st.columns([1.1, 1.2], vertical_alignment="bottom")
    with month_col:
        options = ["All"] + (months or [])
        selected = st.selectbox(
            "Month",
            options,
            index=0 if options else None,
            format_func=month_label,
            placeholder="No months available",
        )
    with status_col:
        key_status = f"{provider_label()} ready" if has_api_key() else f"{provider_label()} key missing"
        st.markdown(f"<span class='status-pill'>{key_status}</span>", unsafe_allow_html=True)
    return None if selected == "All" else selected


def import_tab(df: pd.DataFrame) -> None:
    st.subheader("Import Statement")
    st.caption("Upload a bank PDF or CSV. SpendWise Agent will extract, clean, categorize, and review the transactions before analysis.")
    upload_col, summary_col = st.columns([1.15, 1], vertical_alignment="top")
    with upload_col:
        files = st.file_uploader("Statement file", type=["csv", "pdf"], accept_multiple_files=True)
        action_cols = st.columns([1, 1])
        force_reprocess = bool(files) and action_cols[0].button(
            "Reprocess",
            help="Run the current parser and category rules against the uploaded files again.",
            use_container_width=True,
        )
        if action_cols[1].button("Use sample", use_container_width=True):
            st.session_state.transactions = load_sample_transactions()
            st.session_state.using_sample = True
            st.session_state.parse_status = ["Using sample transactions."]
            st.session_state.schema_mapping = {}
            st.session_state.last_upload_signature = None
            st.rerun()
        if load_uploads(files, force=force_reprocess):
            st.rerun()

    prepared = prepare_transactions(st.session_state.transactions)
    metrics = summary_metrics(prepared)
    uncategorized = int(prepared["category"].eq("Uncategorized").sum()) if not prepared.empty else 0
    needs_review = int(prepared["category_source"].isin(["uncategorized"]).sum()) if "category_source" in prepared.columns else uncategorized
    with summary_col:
        st.markdown("<div class='smart-panel'>", unsafe_allow_html=True)
        st.metric("Transactions Ready", f"{len(prepared):,}")
        st.metric("Extracted Total", money(metrics["total"]))
        st.write(f"Needs review: **{needs_review}**")
        st.write(f"Uncategorized rows: **{uncategorized}**")
        st.write("Source: **Sample data**" if st.session_state.using_sample else "Source: **Uploaded data**")
        st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("Extraction notes", expanded=not st.session_state.using_sample):
        for status in st.session_state.parse_status:
            st.write(status)
        if st.session_state.schema_mapping:
            st.json(st.session_state.schema_mapping)

    st.subheader("Review Transactions")
    st.caption("Edit categories here if your bank statement needs a final human pass. The dashboard uses the reviewed rows below.")
    review_columns = ["date", "merchant", "category", "amount", "source"]
    if "category_source" in prepared.columns:
        review_columns.append("category_source")
    reviewed = st.data_editor(
        prepared[review_columns],
        use_container_width=True,
        hide_index=True,
        num_rows="dynamic",
        column_config={
            "date": st.column_config.DateColumn("Date"),
            "amount": st.column_config.NumberColumn("Amount", format="$%.2f"),
        },
        key="review_editor",
    )
    editor_cols = st.columns([1, 1, 2])
    if editor_cols[0].button("Use reviewed rows", type="primary", use_container_width=True):
        reviewed["date"] = pd.to_datetime(reviewed["date"], errors="coerce")
        reviewed["amount"] = pd.to_numeric(reviewed["amount"], errors="coerce").fillna(0.0)
        if "category_source" not in reviewed.columns:
            reviewed["category_source"] = "manual_review"
        reviewed.loc[reviewed["category"].ne("Uncategorized"), "category_source"] = "manual_review"
        st.session_state.transactions = reviewed.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        st.session_state.using_sample = False
        st.session_state.parse_status = ["Using reviewed transactions from the Import tab."]
        st.success("Reviewed transactions are now active.")
        st.rerun()
    if editor_cols[2].button("Save category rules", use_container_width=True):
        new_rules = rules_from_reviewed_transactions(reviewed)
        added = merge_category_rules(new_rules)
        st.success(f"Saved category rules. {added} new merchant pattern(s) added.")
    csv_bytes = prepared[review_columns].to_csv(index=False).encode("utf-8")
    editor_cols[1].download_button(
        "Download CSV",
        data=csv_bytes,
        file_name="spendwise_standardized_transactions.csv",
        mime="text/csv",
        use_container_width=True,
    )


def dashboard_tab(df: pd.DataFrame, selected_month: str | None) -> None:
    current = filter_month(df, selected_month)
    metrics = summary_metrics(current)
    st.subheader("Dashboard")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Spend", money(metrics["total"]))
    c2.metric("Avg/Day", f"${metrics['avg_day']:,.2f}")
    c3.metric("Top Category", str(metrics["top_category"]), money(metrics["top_category_amount"]))
    c4.metric("Finance Score", f"{metrics['score']}/100")

    left, right = st.columns([1, 1.25])
    with left:
        st.subheader("Spending by Category")
        categories = category_summary(current)
        fig = px.pie(categories, names="category", values="amount", hole=0.42, color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(template="plotly_dark", margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.subheader("Daily Spending")
        daily = daily_spending(current)
        fig = px.bar(daily, x="date", y="amount", color_discrete_sequence=["#31d0aa"])
        fig.update_layout(template="plotly_dark", margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top Recommendations")
    for item in savings_opportunities(current)[:3]:
        st.info(item)


def transactions_tab(df: pd.DataFrame) -> None:
    prepared = prepare_transactions(df)
    with st.expander("Upload and parsing status", expanded=True):
        for status in st.session_state.parse_status:
            st.write(status)
        if st.session_state.schema_mapping:
            st.json(st.session_state.schema_mapping)
        export_columns = ["date", "merchant", "category", "amount", "source"]
        if "category_source" in prepared.columns:
            export_columns.append("category_source")
        csv_bytes = prepared[export_columns].to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download standardized CSV",
            data=csv_bytes,
            file_name="spendwise_standardized_transactions.csv",
            mime="text/csv",
            help="Use this as the clean source of truth for review, edits, or re-upload.",
        )
    col1, col2, col3 = st.columns(3)
    month_options = ["All"] + available_months(prepared)
    month = col1.selectbox("Filter month", month_options, format_func=lambda value: "All" if value == "All" else month_label(value))
    categories = ["All"] + sorted(prepared["category"].unique().tolist())
    category = col2.selectbox("Filter category", categories)
    merchant_query = col3.text_input("Search merchant")
    filtered = prepared.copy()
    if month != "All":
        filtered = filtered[filtered["month"] == month]
    if category != "All":
        filtered = filtered[filtered["category"] == category]
    if merchant_query:
        filtered = filtered[filtered["merchant"].str.contains(merchant_query, case=False, na=False)]
    table_columns = ["date", "merchant", "category", "amount", "source"]
    if "category_source" in filtered.columns:
        table_columns.append("category_source")
    st.dataframe(filtered[table_columns], use_container_width=True, hide_index=True)


def insights_tab(df: pd.DataFrame, selected_month: str | None) -> None:
    current = filter_month(df, selected_month)
    metrics = summary_metrics(current)
    st.subheader("Insights")
    score_col, alerts_col = st.columns([0.85, 1.15])
    with score_col:
        st.metric("Spending Health", f"{metrics['score']}/100")
        st.markdown(f"<div class='stars'>{'★' * max(1, round(int(metrics['score']) / 20))}</div>", unsafe_allow_html=True)
        st.subheader("Savings")
        for item in savings_opportunities(current)[:4]:
            st.info(item)
    with alerts_col:
        st.subheader("Overspending")
        alerts = overspending_alerts(current)
        if not alerts:
            st.success("No categories are above current smart benchmarks.")
        else:
            context = ai_context(df, selected_month)
            ai_tips = generate_tips(alerts, context)
            for alert in alerts[:5]:
                st.write(f"**{alert['category']}** · {money(alert['amount'])} · +{alert['over_pct']}% over")
                st.progress(min(1.0, float(alert["amount"]) / max(float(alert["benchmark"]) * 1.8, 1.0)))
                st.caption(ai_tips.get(str(alert["category"]), str(alert["tip"])))

    sub_col, pattern_col = st.columns(2)
    with sub_col:
        st.subheader("Subscriptions")
        subscriptions = detect_subscriptions(df)
        if subscriptions.empty:
            st.info("No recurring charges detected yet.")
        else:
            st.dataframe(subscriptions, use_container_width=True, hide_index=True)
    with pattern_col:
        st.subheader("Patterns")
        merchants = unusual_merchants(df)
        if merchants:
            st.write("New merchants: " + ", ".join(merchants[:5]))
        else:
            st.caption("No new merchants detected across tracked months.")
        for item in pattern_observations(df):
            st.write(f"- {item}")


def floating_chat(df: pd.DataFrame, selected_month: str | None) -> None:
    with st.container(key="floating_chat"):
        top_cols = st.columns([1, 0.28])
        top_cols[0].markdown("**🤖 SpendWise Agent**")
        label = "−" if st.session_state.chat_open else "💬"
        if top_cols[1].button(label, key="toggle_chat", help="Open or collapse SpendWise Agent"):
            st.session_state.chat_open = not st.session_state.chat_open
            st.rerun()
        if not st.session_state.chat_open:
            st.caption("Ask SpendWise")
            return
        for message in st.session_state.chat_messages[-6:]:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        with st.form("smart_chat_form", clear_on_submit=True):
            question = st.text_input("Question", placeholder="Compare April vs May spending")
            submitted = st.form_submit_button("Send")
        if submitted and question:
            context = ai_context(df, selected_month)
            history = st.session_state.chat_messages[-8:]
            answer = ask_openai(question, context, history)
            st.session_state.chat_messages.append({"role": "user", "content": question})
            st.session_state.chat_messages.append({"role": "assistant", "content": answer})
            st.rerun()


def main() -> None:
    init_state()
    configure_ai_key()
    df = st.session_state.transactions
    months = available_months(df)
    selected_month = top_bar(months)
    st.caption("Demo data is active until you upload your own statement." if st.session_state.using_sample else "Using uploaded statement data.")

    import_view, dashboard, insights, transactions = st.tabs(
        ["Import", "Dashboard", "Insights", "Transactions"]
    )
    with import_view:
        import_tab(df)
    with dashboard:
        dashboard_tab(df, selected_month)
    with transactions:
        transactions_tab(df)
    with insights:
        insights_tab(df, selected_month)
    floating_chat(df, selected_month)


if __name__ == "__main__":
    main()
