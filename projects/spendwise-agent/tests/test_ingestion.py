import pandas as pd

from pathlib import Path

from src.ingestion import (
    cleanse_merchant,
    load_category_rules,
    merge_category_rules,
    normalize_transactions,
    parse_pdf_text,
    rules_from_reviewed_transactions,
)


def test_normalize_transactions_maps_common_csv_columns():
    raw = pd.DataFrame(
        {
            "Transaction Date": ["2026-05-01", "bad-date", "2026-05-03"],
            "Description": ["Coffee Shop", "Ignored", "Spotify"],
            "Type": ["Dining Out", "Dining Out", "Subscriptions"],
            "Debit": ["$4.50", "$10.00", "(9.99)"],
        }
    )

    result = normalize_transactions(raw, source="csv")

    assert list(result.transactions.columns) == ["date", "merchant", "category", "amount", "source", "category_source"]
    assert len(result.transactions) == 2
    assert result.transactions["merchant"].tolist() == ["Coffee Shop", "Spotify"]
    assert result.transactions["amount"].round(2).tolist() == [4.50, 9.99]
    assert result.mapping["date"] == "Transaction Date"


def test_parse_pdf_text_extracts_simple_statement_lines():
    text = """
    05/03 Spotify 9.99
    05/07 Netflix $15.99
    """

    result = parse_pdf_text(text)

    assert len(result.transactions) == 2
    assert set(result.transactions["merchant"]) == {"Spotify", "Netflix"}
    assert set(result.transactions["category"]) == {"Subscriptions"}


def test_parse_pdf_text_extracts_pipe_separated_table_lines():
    text = """
    Date | Description | Amount
    05/03/2026 | Spotify | $9.99
    05/07/2026 | Netflix | $15.99
    """

    result = parse_pdf_text(text)

    assert len(result.transactions) == 2
    assert result.transactions["amount"].round(2).tolist() == [9.99, 15.99]


def test_cleanse_merchant_standardizes_bank_noise():
    assert cleanse_merchant("POS DEBIT DOORDASH*1234 CA") == "DoorDash"
    assert cleanse_merchant("CHECKCARD UBER EATS 9988 US") == "Uber Eats"


def test_parse_pdf_text_filters_payments_and_categorizes_cleaned_merchants():
    text = """
    Statement Period 05/01/2026 - 05/31/2026
    05/03/2026 POS DEBIT DOORDASH*1234 CA $32.44
    05/04/2026 ONLINE PAYMENT THANK YOU $500.00
    05/07/2026 CHECKCARD TRADER JOE 9912 CA $84.12
    """

    result = parse_pdf_text(text)

    assert result.transactions["merchant"].tolist() == ["DoorDash", "Trader Joe's"]
    assert result.transactions["category"].tolist() == ["Dining Out", "Groceries"]


def test_parse_pdf_text_handles_amex_statement_noise():
    text = """
    Closing Date05/12/26
    05/05/26* MOBILE PAYMENT - THANK YOU -$2,300.00
    04/10/26 AplPay PUBLIX CUMMING GA $52.75
    04/14/26 Uber Trip help.uber.com CA $21.84
    05/12/26 Interest Charge on Purchases $209.20
    Purchases 10/31/2025 29.99% (v) $7,953.02 $209.20
    """

    result = parse_pdf_text(text)

    assert result.transactions["merchant"].tolist() == ["Publix", "Uber"]
    assert result.transactions["category"].tolist() == ["Groceries", "Ride Share"]
    assert result.transactions["date"].dt.strftime("%Y-%m-%d").tolist() == ["2026-04-10", "2026-04-14"]


def test_parse_pdf_text_categorizes_jan_feb_statement_merchants():
    text = """
    Closing Date02/09/26
    01/17/26 CHICK FIL A DORAVILLE GA $33.65
    01/17/26 FEVERUP BUBBLE PLANET FLORAL PARK NY $335.90
    01/21/26 HOLIDAY INN DFWTC TROPHY CLUB TX $178.54
    01/26/26 ALDI CUMMING GA $10.39
    01/29/26 DELTA AIR LINES ATLANTA GA $5.60
    02/04/26 HOMEWITHLOAN.COM THE COLONY TX $160.00
    02/06/26 MARTA N1A ATLANTA GA $40.00
    02/07/26 OLD NAVY CUMMING GA $32.09
    """

    result = parse_pdf_text(text)

    assert not result.transactions["category"].eq("Uncategorized").any()
    assert set(result.transactions["category"]) >= {"Dining Out", "Entertainment", "Travel", "Groceries", "Finance", "Ride Share", "Shopping"}


def test_parse_pdf_text_keeps_same_day_same_amount_real_charges():
    text = """
    Closing Date05/12/26
    04/25/26 FRONTIER AIRLINES AIRLINES DENVER CO $68.98
    04/25/26 FRONTIER AIRLINES AIRLINES DENVER CO $68.98
    """

    result = parse_pdf_text(text)

    assert len(result.transactions) == 2
    assert result.transactions["amount"].sum() == 137.96


def test_reviewed_transactions_create_persistent_rules(tmp_path):
    path = tmp_path / "category_rules.csv"
    reviewed = pd.DataFrame(
        [
            {
                "date": "2026-01-01",
                "merchant": "Local Bakery",
                "category": "Dining Out",
                "amount": 12.5,
                "source": "test",
            }
        ]
    )

    new_rules = rules_from_reviewed_transactions(reviewed)
    added = merge_category_rules(new_rules, path=path)
    rules = load_category_rules(path)

    assert added == 1
    assert Path(path).exists()
    assert rules.to_dict(orient="records") == [
        {"pattern": "local bakery", "merchant": "Local Bakery", "category": "Dining Out"}
    ]
