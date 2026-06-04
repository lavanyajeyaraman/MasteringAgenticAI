from src.analytics import (
    ai_context,
    detect_subscriptions,
    overspending_alerts,
    summary_metrics,
)
from src.sample_data import load_sample_transactions
import pandas as pd


def test_summary_metrics_returns_expected_shape():
    df = load_sample_transactions()

    metrics = summary_metrics(df)

    assert metrics["total"] > 0
    assert metrics["avg_day"] > 0
    assert metrics["top_category"]
    assert 0 <= metrics["score"] <= 100


def test_detect_subscriptions_finds_recurring_services():
    df = load_sample_transactions()

    subscriptions = detect_subscriptions(df)

    services = set(subscriptions["service"])
    assert "Spotify" in services
    assert "Netflix" in services
    assert subscriptions["amount"].sum() > 0


def test_detect_subscriptions_handles_no_recurring_rows():
    df = load_sample_transactions().head(1)

    subscriptions = detect_subscriptions(df)

    assert subscriptions.empty
    assert list(subscriptions.columns) == ["service", "amount", "last_charge", "action", "confidence"]


def test_detect_subscriptions_excludes_repeated_groceries_and_shopping():
    df = pd.DataFrame(
        [
            ("2026-04-10", "Publix", "Groceries", 30.22, "test"),
            ("2026-04-16", "Publix", "Groceries", 30.22, "test"),
            ("2026-05-02", "Target", "Shopping", 48.67, "test"),
            ("2026-05-09", "Target", "Shopping", 48.67, "test"),
            ("2026-04-03", "Spotify", "Subscriptions", 9.99, "test"),
            ("2026-05-03", "Spotify", "Subscriptions", 9.99, "test"),
        ],
        columns=["date", "merchant", "category", "amount", "source"],
    )
    df["date"] = pd.to_datetime(df["date"])

    subscriptions = detect_subscriptions(df)

    assert subscriptions["service"].tolist() == ["Spotify"]


def test_overspending_alerts_flags_categories_above_benchmark():
    df = load_sample_transactions()
    may = df[(df["date"].dt.month == 5) & (df["date"].dt.year == 2026)]

    alerts = overspending_alerts(may)

    categories = {alert["category"] for alert in alerts}
    assert "Dining Out" in categories
    assert all(alert["over_pct"] > 0 for alert in alerts)


def test_ai_context_contains_selected_month_and_core_sections():
    df = load_sample_transactions()

    context = ai_context(df, "2026-05")

    assert "Selected month: 2026-05" in context
    assert "Top categories" in context
    assert "Subscriptions" in context
