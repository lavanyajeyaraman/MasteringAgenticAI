from spendwise_rag.processing.transaction_parser import (
    CategoryRule,
    clean_merchant,
    extract_transactions_from_text_block,
    infer_category,
    load_golden_category_rules,
    merge_golden_category_rules,
)


def test_clean_merchant_removes_payment_noise():
    assert clean_merchant("LYFT 855-280-0278 CA") == "Lyft"
    assert clean_merchant("AplPay PUBLIX CUMMING GA") == "Publix"
    assert clean_merchant("Uber Trip help.uber.com CA") == "Uber"


def test_extract_transactions_from_text_block_splits_repeated_dates():
    transactions = extract_transactions_from_text_block(
        "01/30/26 AplPay FRESH ROTI Suwanee GA $9.22 squareup.com/receipts "
        "01/30/26 AplPay PATEL BROTHERS SUWANEE 000000001 SUWANEE GA $35.61 7707816557",
        2026,
    )

    assert [(txn.date, txn.merchant, txn.amount, txn.category) for txn in transactions] == [
        ("2026-01-30", "Fresh Roti", 9.22, "Dining"),
        ("2026-01-30", "Patel Brothers", 35.61, "Groceries"),
    ]


def test_infer_category_defaults_when_unknown():
    assert infer_category("Mystery Merchant") == "Uncategorized"


def test_golden_category_rules_are_loaded_read_only():
    rules = load_golden_category_rules()

    assert any(rule.pattern == "lyft" and rule.category == "Ride Share" for rule in rules)
    assert clean_merchant("LYFT 855-280-0278 CA") == "Lyft"
    assert infer_category("Lyft") == "Ride Share"


def test_merge_golden_category_rules_writes_deduped_rules(tmp_path):
    path = tmp_path / "rules.csv"
    added = merge_golden_category_rules(
        [{"pattern": "shell", "merchant": "Shell", "category": "Auto"}],
        path=str(path),
    )

    assert added == 1
    assert load_golden_category_rules(str(path)) == (CategoryRule("shell", "Shell", "Auto"),)
