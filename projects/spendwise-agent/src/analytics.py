from __future__ import annotations

import math

import pandas as pd


BENCHMARKS = {
    "Dining Out": 300,
    "Shopping": 250,
    "Ride Share": 150,
    "Subscriptions": 175,
    "Groceries": 550,
    "Entertainment": 160,
}

SUBSCRIPTION_MERCHANT_TERMS = {
    "adobe",
    "amazon prime",
    "apple",
    "audible",
    "chatgpt",
    "claude",
    "comcast",
    "disney",
    "dropbox",
    "google",
    "gym",
    "hbo",
    "hulu",
    "icloud",
    "linkedin",
    "microsoft",
    "netflix",
    "notion",
    "openai",
    "paramount",
    "peacock",
    "planet fitness",
    "rocket money",
    "slack",
    "spotify",
    "verizon",
    "xfinity",
    "youtube",
    "zoom",
}

NON_SUBSCRIPTION_CATEGORIES = {
    "Auto",
    "Dining Out",
    "Donations",
    "Education",
    "Entertainment",
    "Groceries",
    "Ride Share",
    "Services",
    "Shopping",
    "Travel",
}


def prepare_transactions(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    prepared["date"] = pd.to_datetime(prepared["date"], errors="coerce")
    prepared["amount"] = pd.to_numeric(prepared["amount"], errors="coerce").fillna(0.0)
    if "category_source" not in prepared.columns:
        prepared["category_source"] = "unknown"
    prepared = prepared.dropna(subset=["date"])
    prepared["month"] = prepared["date"].dt.to_period("M").astype(str)
    prepared["month_label"] = prepared["date"].dt.strftime("%b %Y")
    return prepared


def available_months(df: pd.DataFrame) -> list[str]:
    if df.empty:
        return []
    prepared = prepare_transactions(df)
    return sorted(prepared["month"].unique().tolist())


def filter_month(df: pd.DataFrame, month: str | None) -> pd.DataFrame:
    prepared = prepare_transactions(df)
    if month:
        prepared = prepared[prepared["month"] == month]
    return prepared


def summary_metrics(df: pd.DataFrame) -> dict[str, object]:
    if df.empty:
        return {"total": 0.0, "avg_day": 0.0, "top_category": "N/A", "top_category_amount": 0.0, "score": 100}
    total = float(df["amount"].sum())
    days = max(1, int((df["date"].max() - df["date"].min()).days) + 1)
    category_totals = df.groupby("category", as_index=False)["amount"].sum().sort_values("amount", ascending=False)
    top = category_totals.iloc[0] if not category_totals.empty else {"category": "N/A", "amount": 0.0}
    overages = overspending_alerts(df)
    penalty = min(45, sum(alert["over_pct"] for alert in overages) / 4)
    subscription_penalty = min(15, category_totals.loc[category_totals["category"].eq("Subscriptions"), "amount"].sum() / 30)
    score = max(0, min(100, round(88 - penalty - subscription_penalty + (5 if total < 1500 else 0))))
    return {
        "total": total,
        "avg_day": total / days,
        "top_category": str(top["category"]),
        "top_category_amount": float(top["amount"]),
        "score": int(score),
    }


def category_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["category", "amount"])
    return df.groupby("category", as_index=False)["amount"].sum().sort_values("amount", ascending=False)


def daily_spending(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["date", "amount"])
    return df.groupby("date", as_index=False)["amount"].sum().sort_values("date")


def monthly_category_spending(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["month", "category", "amount"])
    prepared = prepare_transactions(df)
    return prepared.groupby(["month", "category"], as_index=False)["amount"].sum()


def monthly_totals(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["month", "amount"])
    prepared = prepare_transactions(df)
    return prepared.groupby("month", as_index=False)["amount"].sum().sort_values("month")


def detect_subscriptions(df: pd.DataFrame) -> pd.DataFrame:
    columns = ["service", "amount", "last_charge", "action", "confidence"]
    if df.empty:
        return pd.DataFrame(columns=columns)
    prepared = prepare_transactions(df)
    rows = []
    for merchant, group in prepared.groupby("merchant"):
        group = group.sort_values("date")
        if len(group) < 2:
            continue
        rounded_amounts = group["amount"].round(2)
        repeated_amount = rounded_amounts.value_counts().iloc[0] >= 2
        monthly_count = group["month"].nunique()
        categories = set(group["category"].astype(str))
        is_subscription_category = "Subscriptions" in categories
        is_subscription_name = is_subscription_merchant(merchant)
        is_excluded_category = bool(categories & NON_SUBSCRIPTION_CATEGORIES) and not is_subscription_category
        if not is_subscription_name and not is_subscription_category:
            continue
        if is_excluded_category and not is_subscription_name:
            continue
        if is_subscription_category or (repeated_amount and monthly_count >= 2):
            median_amount = float(group["amount"].median())
            confidence = min(98, 45 + monthly_count * 12 + (20 if repeated_amount else 0) + (15 if is_subscription_category else 0))
            action = "Keep"
            if median_amount > 30 and confidence < 85:
                action = "Review"
            if median_amount > 75:
                action = "Cancel?"
            rows.append(
                {
                    "service": merchant,
                    "amount": median_amount,
                    "last_charge": group["date"].max().date().isoformat(),
                    "action": action,
                    "confidence": confidence,
                }
            )
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values(["amount", "service"], ascending=[False, True]).reset_index(drop=True)


def is_subscription_merchant(merchant: str) -> bool:
    text = str(merchant).lower()
    return any(term in text for term in SUBSCRIPTION_MERCHANT_TERMS)


def overspending_alerts(df: pd.DataFrame) -> list[dict[str, object]]:
    alerts = []
    totals = category_summary(df)
    for _, row in totals.iterrows():
        category = str(row["category"])
        benchmark = BENCHMARKS.get(category)
        amount = float(row["amount"])
        if benchmark and amount > benchmark:
            over_pct = round(((amount - benchmark) / benchmark) * 100)
            alerts.append(
                {
                    "category": category,
                    "amount": amount,
                    "benchmark": benchmark,
                    "over_pct": over_pct,
                    "tip": tip_for_category(category),
                }
            )
    return sorted(alerts, key=lambda item: item["over_pct"], reverse=True)


def tip_for_category(category: str) -> str:
    tips = {
        "Dining Out": "Try meal prepping two dinners and one lunch block this week.",
        "Shopping": "Add a 24-hour pause before non-essential purchases over $50.",
        "Ride Share": "Set a weekly ride-share ceiling and batch trips where possible.",
        "Subscriptions": "Cancel or pause subscriptions you used fewer than twice this month.",
        "Groceries": "Plan around pantry staples before the next grocery run.",
    }
    return tips.get(category, "Set a clear weekly cap and review this category every Friday.")


def savings_opportunities(df: pd.DataFrame) -> list[str]:
    opportunities = []
    subscriptions = detect_subscriptions(df)
    reviewable = subscriptions[subscriptions["action"].isin(["Review", "Cancel?"])] if not subscriptions.empty else pd.DataFrame()
    if not reviewable.empty:
        savings = reviewable["amount"].sum()
        opportunities.append(f"Review {len(reviewable)} subscription(s) to save about ${savings:,.0f}/mo.")
    alerts = overspending_alerts(df)
    for alert in alerts[:2]:
        reducible = max(0, float(alert["amount"]) - float(alert["benchmark"]))
        opportunities.append(f"Trim {alert['category']} toward benchmark to save about ${reducible:,.0f}.")
    if not opportunities:
        opportunities.append("You are close to current benchmarks. Keep monitoring new merchants and subscriptions.")
    return opportunities


def unusual_merchants(df: pd.DataFrame) -> list[str]:
    prepared = prepare_transactions(df)
    months = available_months(prepared)
    if len(months) < 2:
        return []
    current = prepared[prepared["month"] == months[-1]]
    previous = prepared[prepared["month"].isin(months[:-1])]
    new_merchants = sorted(set(current["merchant"]) - set(previous["merchant"]))
    return new_merchants[:6]


def pattern_observations(df: pd.DataFrame) -> list[str]:
    observations = []
    monthly = monthly_totals(df)
    if len(monthly) >= 2:
        last = float(monthly.iloc[-1]["amount"])
        previous = float(monthly.iloc[-2]["amount"])
        delta = last - previous
        direction = "up" if delta >= 0 else "down"
        observations.append(f"Latest month is {direction} ${abs(delta):,.0f} versus the prior month.")
    category = category_summary(df)
    if not category.empty:
        top = category.iloc[0]
        share = float(top["amount"]) / max(1.0, float(category["amount"].sum()))
        observations.append(f"{top['category']} represents {math.ceil(share * 100)}% of tracked spend.")
    return observations


def ai_context(df: pd.DataFrame, month: str | None) -> str:
    current = filter_month(df, month)
    metrics = summary_metrics(current)
    categories = category_summary(current).head(8).to_dict(orient="records")
    subscriptions = detect_subscriptions(df).head(8).to_dict(orient="records")
    alerts = overspending_alerts(current)
    return (
        f"Selected month: {month or 'all'}\n"
        f"Metrics: {metrics}\n"
        f"Top categories: {categories}\n"
        f"Subscriptions: {subscriptions}\n"
        f"Overspending alerts: {alerts}\n"
        "Answer as a concise personal finance assistant. Do not claim to be a financial adviser."
    )
