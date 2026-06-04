from __future__ import annotations

import pandas as pd


def load_sample_transactions() -> pd.DataFrame:
    rows = [
        ("2026-01-03", "Spotify", "Subscriptions", 9.99, "sample"),
        ("2026-01-07", "Netflix", "Subscriptions", 15.99, "sample"),
        ("2026-01-08", "Trader Joe's", "Groceries", 82.15, "sample"),
        ("2026-01-10", "Uber", "Ride Share", 38.2, "sample"),
        ("2026-01-12", "Sweetgreen", "Dining Out", 24.75, "sample"),
        ("2026-01-18", "Amazon", "Shopping", 116.88, "sample"),
        ("2026-01-21", "DoorDash", "Dining Out", 58.4, "sample"),
        ("2026-01-27", "Planet Fitness", "Subscriptions", 45.0, "sample"),
        ("2026-02-03", "Spotify", "Subscriptions", 9.99, "sample"),
        ("2026-02-07", "Netflix", "Subscriptions", 15.99, "sample"),
        ("2026-02-09", "Whole Foods", "Groceries", 122.42, "sample"),
        ("2026-02-13", "Lyft", "Ride Share", 32.5, "sample"),
        ("2026-02-16", "Zara", "Shopping", 88.0, "sample"),
        ("2026-02-19", "Chipotle", "Dining Out", 18.2, "sample"),
        ("2026-02-24", "DoorDash", "Dining Out", 64.1, "sample"),
        ("2026-02-27", "Planet Fitness", "Subscriptions", 45.0, "sample"),
        ("2026-03-03", "Spotify", "Subscriptions", 9.99, "sample"),
        ("2026-03-07", "Netflix", "Subscriptions", 15.99, "sample"),
        ("2026-03-08", "Rocket Money", "Subscriptions", 12.0, "sample"),
        ("2026-03-11", "Costco", "Groceries", 214.36, "sample"),
        ("2026-03-12", "Uber", "Ride Share", 45.0, "sample"),
        ("2026-03-17", "H&M", "Shopping", 72.3, "sample"),
        ("2026-03-22", "DoorDash", "Dining Out", 79.2, "sample"),
        ("2026-03-27", "Planet Fitness", "Subscriptions", 45.0, "sample"),
        ("2026-04-03", "Spotify", "Subscriptions", 9.99, "sample"),
        ("2026-04-07", "Netflix", "Subscriptions", 15.99, "sample"),
        ("2026-04-08", "Rocket Money", "Subscriptions", 12.0, "sample"),
        ("2026-04-10", "Whole Foods", "Groceries", 146.72, "sample"),
        ("2026-04-12", "Uber", "Ride Share", 52.4, "sample"),
        ("2026-04-15", "Sephora", "Shopping", 134.25, "sample"),
        ("2026-04-19", "Cava", "Dining Out", 27.9, "sample"),
        ("2026-04-23", "DoorDash", "Dining Out", 86.6, "sample"),
        ("2026-04-27", "Planet Fitness", "Subscriptions", 45.0, "sample"),
        ("2026-05-01", "Rent Cafe", "Housing", 1850.0, "sample"),
        ("2026-05-03", "Spotify", "Subscriptions", 9.99, "sample"),
        ("2026-05-05", "Instacart", "Groceries", 176.82, "sample"),
        ("2026-05-07", "Netflix", "Subscriptions", 15.99, "sample"),
        ("2026-05-08", "Rocket Money", "Subscriptions", 12.0, "sample"),
        ("2026-05-10", "Uber", "Ride Share", 61.8, "sample"),
        ("2026-05-12", "DoorDash", "Dining Out", 92.1, "sample"),
        ("2026-05-13", "Sweetgreen", "Dining Out", 48.9, "sample"),
        ("2026-05-20", "Uber Eats", "Dining Out", 73.6, "sample"),
        ("2026-05-15", "Nordstrom", "Shopping", 187.45, "sample"),
        ("2026-05-18", "Blue Bottle", "Dining Out", 22.4, "sample"),
        ("2026-05-22", "DoorDash", "Dining Out", 104.35, "sample"),
        ("2026-05-27", "Planet Fitness", "Subscriptions", 45.0, "sample"),
        ("2026-06-01", "Rent Cafe", "Housing", 1850.0, "sample"),
        ("2026-06-03", "Spotify", "Subscriptions", 9.99, "sample"),
        ("2026-06-05", "Whole Foods", "Groceries", 132.14, "sample"),
        ("2026-06-07", "Netflix", "Subscriptions", 15.99, "sample"),
        ("2026-06-08", "Rocket Money", "Subscriptions", 12.0, "sample"),
        ("2026-06-10", "Uber", "Ride Share", 42.0, "sample"),
        ("2026-06-12", "DoorDash", "Dining Out", 67.25, "sample"),
    ]
    df = pd.DataFrame(rows, columns=["date", "merchant", "category", "amount", "source"])
    df["date"] = pd.to_datetime(df["date"])
    df["amount"] = df["amount"].astype(float)
    df["category_source"] = "sample"
    return df
