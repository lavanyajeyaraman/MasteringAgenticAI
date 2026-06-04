from __future__ import annotations

import io
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd
import pdfplumber


CANONICAL_COLUMNS = ["date", "merchant", "category", "amount", "source"]
TRANSACTION_COLUMNS = CANONICAL_COLUMNS + ["category_source"]
RULE_COLUMNS = ["pattern", "merchant", "category"]
RULES_PATH = Path("data/category_rules.csv")

DATE_COLUMNS = {"date", "transaction date", "transaction_date", "posted date", "post date"}
MERCHANT_COLUMNS = {"merchant", "description", "name", "vendor", "payee", "details"}
CATEGORY_COLUMNS = {"category", "type", "group"}
AMOUNT_COLUMNS = {"amount", "debit", "charge", "withdrawal", "spent"}
CREDIT_COLUMNS = {"credit", "deposit", "payment"}

EXCLUDED_MERCHANT_TERMS = {
    "payment thank you",
    "online payment",
    "autopay payment",
    "transfer from",
    "transfer to",
    "deposit",
    "interest paid",
    "interest charge",
    "cashback",
    "refund",
}

MERCHANT_ALIASES = {
    "365 market": "365 Market",
    "aldi": "Aldi",
    "amazon": "Amazon",
    "amzn": "Amazon",
    "annapoorani": "Annapoorani Caterers",
    "apple": "Apple",
    "aqua bubble": "Aqua Bubble Car Wash",
    "blue bottle": "Blue Bottle",
    "bww": "Buffalo Wild Wings",
    "buffalo wild wngs": "Buffalo Wild Wings",
    "cava": "Cava",
    "chai bisket": "Chai Bisket",
    "chick fil a": "Chick-fil-A",
    "chipotle": "Chipotle",
    "come 2 chutneys": "Come 2 Chutneys",
    "costco": "Costco",
    "deccan spice": "Deccan Spice",
    "delta air lines": "Delta Air Lines",
    "desi chowrastha": "Desi Chowrastha",
    "doordash": "DoorDash",
    "door dash": "DoorDash",
    "domino": "Domino's",
    "dunkin": "Dunkin",
    "elite india": "Elite India",
    "fidelity 2d cafe": "Fidelity Cafe",
    "flik mobile": "Flik Mobile",
    "feverup": "Fever",
    "forsyth co ga parks": "Forsyth County Parks",
    "fresh roti": "Fresh Roti",
    "frontier airlines": "Frontier Airlines",
    "gopuff": "Gopuff",
    "gofndme": "GoFundMe",
    "groupon": "Groupon",
    "hashtag india": "Hashtag India",
    "holiday inn": "Holiday Inn",
    "hospitable": "Hospitable",
    "home depot": "Home Depot",
    "homewithloan": "HomeWithLoan",
    "hulu": "Hulu",
    "indi fresh": "Indi Fresh",
    "indifresh": "Indi Fresh",
    "instacart": "Instacart",
    "kakatiya": "Kakatiya Indian Kitchen",
    "kona ice": "Kona Ice",
    "lakshmi florist": "Lakshmi Florist",
    "lyft": "Lyft",
    "marshalls": "Marshalls",
    "marta": "MARTA",
    "mwj vending": "MWJ Vending",
    "netflix": "Netflix",
    "nordstrom": "Nordstrom",
    "planet fitness": "Planet Fitness",
    "panda express": "Panda Express",
    "paradies": "Paradies",
    "parking spot": "The Parking Spot",
    "theparkingspot": "The Parking Spot",
    "patel brothers": "Patel Brothers",
    "paypal paypal gats": "PayPal GATS",
    "ponce de leon music": "Ponce De Leon Music",
    "publix": "Publix",
    "qt": "QuikTrip",
    "rocket money": "Rocket Money",
    "sagar fresh market": "Sagar Fresh Market",
    "samplesdojo": "SamplesDojo",
    "sephora": "Sephora",
    "spotify": "Spotify",
    "starbucks": "Starbucks",
    "sri maha lakshmi temple": "Sri Maha Lakshmi Temple",
    "suvidha intl market": "Suvidha International Market",
    "sweetgreen": "Sweetgreen",
    "target": "Target",
    "thales ifec": "Thales IFEC",
    "the human bean": "The Human Bean",
    "the ups store": "The UPS Store",
    "trader joe": "Trader Joe's",
    "uber eats": "Uber Eats",
    "ubereats": "Uber Eats",
    "uber": "Uber",
    "whole foods": "Whole Foods",
    "old navy": "Old Navy",
    "zara": "Zara",
}


@dataclass(frozen=True)
class ParseResult:
    transactions: pd.DataFrame
    status: str
    mapping: dict[str, str]


def _clean_column(column: str) -> str:
    return re.sub(r"\s+", " ", str(column).strip().lower().replace("-", " ").replace("_", " "))


def _find_column(columns: Iterable[str], candidates: set[str]) -> str | None:
    cleaned = {_clean_column(col): col for col in columns}
    for candidate in candidates:
        if candidate in cleaned:
            return cleaned[candidate]
    for cleaned_col, original in cleaned.items():
        if any(candidate in cleaned_col for candidate in candidates):
            return original
    return None


def _parse_amount(value: object) -> float:
    if pd.isna(value):
        return 0.0
    text = str(value).strip()
    negative = text.startswith("(") and text.endswith(")")
    text = re.sub(r"[^0-9.\-]", "", text)
    if text in {"", "-", "."}:
        return 0.0
    amount = float(text)
    return abs(amount) if not negative else abs(amount)


def cleanse_merchant(value: object) -> str:
    text = str(value or "").strip()
    text = re.sub(
        r"\b(APLPAY|APPLEPAY|TST|POS|DEBIT|CARD|PURCHASE|AUTH|PENDING|RECURRING|CHECKCARD|VISA)\b",
        " ",
        text,
        flags=re.I,
    )
    text = re.sub(r"\b\d{4,}\b", " ", text)
    text = re.sub(
        r"\b(CUMMING|ALPHARETTA|SUWANEE|ROANOKE|HAPEVILLE|BOSTON|CHICAGO|DENVER|IRVINE|TROY|FAIRFAX|SAN FRANCISCO)\b",
        " ",
        text,
        flags=re.I,
    )
    text = re.sub(r"\s+(GA|CA|NY|TX|WA|FL|IL|MA|MI|CO|VA|US|USA)$", "", text, flags=re.I)
    text = re.sub(r"[*#:_\-]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip(" .")
    lowered = text.lower()
    for needle, label in MERCHANT_ALIASES.items():
        if needle in lowered:
            return label
    return text.title() if text else "Unknown Merchant"


def merchant_pattern(value: object) -> str:
    text = str(value or "").lower()
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_category_rules(path: Path = RULES_PATH) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=RULE_COLUMNS)
    rules = pd.read_csv(path).fillna("")
    for column in RULE_COLUMNS:
        if column not in rules.columns:
            rules[column] = ""
    rules["pattern"] = rules["pattern"].map(merchant_pattern)
    rules = rules[rules["pattern"].astype(bool)]
    return rules[RULE_COLUMNS].drop_duplicates(subset=["pattern"], keep="last").reset_index(drop=True)


def save_category_rules(rules: pd.DataFrame, path: Path = RULES_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cleaned = rules.copy()
    for column in RULE_COLUMNS:
        if column not in cleaned.columns:
            cleaned[column] = ""
    cleaned["pattern"] = cleaned["pattern"].map(merchant_pattern)
    cleaned["merchant"] = cleaned["merchant"].astype(str).str.strip()
    cleaned["category"] = cleaned["category"].astype(str).str.strip()
    cleaned = cleaned[cleaned["pattern"].astype(bool) & cleaned["category"].astype(bool)]
    cleaned = cleaned[RULE_COLUMNS].drop_duplicates(subset=["pattern"], keep="last").sort_values("pattern")
    cleaned.to_csv(path, index=False)


def apply_learned_rule(merchant: str, rules: pd.DataFrame | None = None) -> tuple[str | None, str | None]:
    rules = load_category_rules() if rules is None else rules
    if rules.empty:
        return None, None
    normalized = merchant_pattern(merchant)
    matches = [
        row
        for row in rules.to_dict(orient="records")
        if row["pattern"] and row["pattern"] in normalized
    ]
    if not matches:
        return None, None
    best = max(matches, key=lambda row: len(str(row["pattern"])))
    return str(best["merchant"] or merchant), str(best["category"])


def rules_from_reviewed_transactions(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=RULE_COLUMNS)
    rows = []
    for _, row in df.iterrows():
        merchant = str(row.get("merchant", "")).strip()
        category = str(row.get("category", "")).strip()
        if merchant and category and category != "Uncategorized":
            rows.append({"pattern": merchant_pattern(merchant), "merchant": merchant, "category": category})
    return pd.DataFrame(rows, columns=RULE_COLUMNS).drop_duplicates(subset=["pattern"], keep="last")


def merge_category_rules(new_rules: pd.DataFrame, path: Path = RULES_PATH) -> int:
    existing = load_category_rules(path)
    combined = pd.concat([existing, new_rules], ignore_index=True)
    before = len(existing)
    save_category_rules(combined, path)
    after = len(load_category_rules(path))
    return max(0, after - before)


def is_non_expense_merchant(merchant: str) -> bool:
    lowered = str(merchant).lower()
    return any(term in lowered for term in EXCLUDED_MERCHANT_TERMS)


def cleanse_category(value: object, merchant: str) -> tuple[str, str]:
    learned_merchant, learned_category = apply_learned_rule(merchant)
    if learned_category:
        return learned_category, "learned_rule"
    text = str(value or "").strip()
    if not text or text.lower() in {"nan", "none", "uncategorized", "other"}:
        inferred = infer_category(merchant)
        return inferred, "uncategorized" if inferred == "Uncategorized" else "built_in_rule"
    normalized = text.title()
    category_aliases = {
        "Food & Drink": "Dining Out",
        "Restaurants": "Dining Out",
        "Restaurant": "Dining Out",
        "Food": "Dining Out",
        "Travel": "Travel",
        "Transport": "Ride Share",
        "Transportation": "Ride Share",
        "Merchandise": "Shopping",
        "Shops": "Shopping",
        "Supermarkets": "Groceries",
        "Grocery": "Groceries",
        "Bills & Utilities": "Utilities",
    }
    category = category_aliases.get(normalized, normalized)
    return category, "source_category"


def standardize_transactions(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return _empty_transactions()
    standardized = df.copy()
    standardized["date"] = pd.to_datetime(standardized["date"], errors="coerce")
    standardized["merchant"] = standardized["merchant"].map(cleanse_merchant)
    standardized = standardized[~standardized["merchant"].map(is_non_expense_merchant)]
    category_results = [
        cleanse_category(category, merchant)
        for category, merchant in zip(standardized["category"], standardized["merchant"])
    ]
    standardized["category"] = [category for category, _ in category_results]
    standardized["category_source"] = [source for _, source in category_results]
    standardized["amount"] = pd.to_numeric(standardized["amount"], errors="coerce").fillna(0.0).abs()
    standardized = standardized.dropna(subset=["date"])
    standardized = standardized[standardized["amount"] > 0]
    standardized["source"] = standardized["source"].fillna("unknown").astype(str)
    return standardized[TRANSACTION_COLUMNS].sort_values(["date", "merchant"]).reset_index(drop=True)


def normalize_transactions(df: pd.DataFrame, source: str = "csv") -> ParseResult:
    if df.empty:
        return ParseResult(_empty_transactions(), "No transactions found.", {})

    date_col = _find_column(df.columns, DATE_COLUMNS)
    merchant_col = _find_column(df.columns, MERCHANT_COLUMNS)
    category_col = _find_column(df.columns, CATEGORY_COLUMNS)
    amount_col = _find_column(df.columns, AMOUNT_COLUMNS)
    credit_col = _find_column(df.columns, CREDIT_COLUMNS)
    mapping = {
        "date": date_col or "",
        "merchant": merchant_col or "",
        "category": category_col or "",
        "amount": amount_col or "",
    }

    normalized = pd.DataFrame()
    normalized["date"] = pd.to_datetime(df[date_col], errors="coerce") if date_col else pd.NaT
    normalized["merchant"] = df[merchant_col].astype(str).str.strip() if merchant_col else "Unknown Merchant"
    normalized["category"] = (
        df[category_col].astype(str).str.strip().replace({"": "Uncategorized"})
        if category_col
        else "Uncategorized"
    )
    if amount_col:
        normalized["amount"] = df[amount_col].map(_parse_amount)
    elif credit_col:
        normalized["amount"] = df[credit_col].map(_parse_amount)
        mapping["amount"] = credit_col
    else:
        normalized["amount"] = 0.0
    normalized["source"] = source
    normalized = standardize_transactions(normalized)
    status = f"Loaded {len(normalized)} transactions from {source.upper()}."
    if not date_col or not merchant_col or not mapping["amount"]:
        status = "Loaded partial data. Some required columns were inferred or missing."
    return ParseResult(normalized, status, mapping)


def parse_csv(uploaded_file) -> ParseResult:
    df = pd.read_csv(uploaded_file)
    return normalize_transactions(df, source="csv")


def extract_pdf_text(uploaded_file) -> str:
    data = uploaded_file.read()
    text_parts: list[str] = []
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            text_parts.append(page_text)
            for table in page.extract_tables() or []:
                for row in table:
                    text_parts.append(" | ".join(str(cell or "") for cell in row))
    return "\n".join(part for part in text_parts if part.strip())


def parse_pdf_text(text: str) -> ParseResult:
    rows = []
    statement_year = infer_statement_year(text)
    for line in text.splitlines():
        row = _parse_pdf_line(line, statement_year)
        if row:
            rows.append(row)
    if not rows:
        return ParseResult(_empty_transactions(), "No deterministic PDF transactions found.", {})
    return normalize_transactions(pd.DataFrame(rows), source="pdf")


def _pdf_match_to_row(match: re.Match[str], default_year: int | None = None) -> dict[str, object]:
    date_text = match.group("date")
    parsed_date = parse_statement_date(date_text, default_year)
    merchant = cleanse_merchant(match.group("merchant").strip(" |:-"))
    return {
        "date": parsed_date,
        "merchant": merchant,
        "category": infer_category(merchant),
        "amount": _parse_amount(match.group("amount")),
    }


def _parse_pdf_line(line: str, default_year: int | None = None) -> dict[str, object] | None:
    normalized = re.sub(r"\s*\|\s*", " ", line.strip())
    match = re.match(
        r"(?P<date>\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?)\s+"
        r"(?P<merchant>.*?)\s+"
        r"(?P<amount>\$?\(?-?\d{1,5}(?:,\d{3})*(?:\.\d{2})\)?)\s*$",
        normalized,
    )
    if not match:
        return None
    merchant = match.group("merchant").strip(" |:-")
    if not merchant or merchant.lower() in {"date", "description", "merchant"}:
        return None
    if "$" in merchant or "%" in merchant:
        return None
    if is_non_expense_merchant(merchant):
        return None
    return _pdf_match_to_row(match, default_year)


def parse_statement_date(date_text: str, default_year: int | None = None) -> pd.Timestamp:
    date_text = date_text.strip()
    for fmt in ("%m/%d/%Y", "%m-%d-%Y", "%m/%d/%y", "%m-%d-%y"):
        parsed = pd.to_datetime(date_text, format=fmt, errors="coerce")
        if not pd.isna(parsed):
            return parsed
    if re.fullmatch(r"\d{1,2}[/-]\d{1,2}", date_text):
        return pd.to_datetime(f"{date_text}/{default_year or datetime.now().year}", format="%m/%d/%Y", errors="coerce")
    return pd.to_datetime(date_text, errors="coerce")


def infer_statement_year(text: str) -> int | None:
    years = [int(year) for year in re.findall(r"\b(20\d{2})\b", text)]
    if not years:
        return None
    current_year = datetime.now().year
    reasonable = [year for year in years if 2000 <= year <= current_year + 1]
    return max(set(reasonable or years), key=(reasonable or years).count)


def infer_category(merchant: str) -> str:
    text = merchant.lower()
    category_terms = {
        "Subscriptions": ["spotify", "netflix", "hulu", "disney", "rocket", "apple", "google", "gym"],
        "Dining Out": [
            "365 market",
            "annapoorani",
            "bean",
            "buffalo wild wings",
            "bww",
            "cafe",
            "caterers",
            "chai bisket",
            "chowrastha",
            "coffee",
            "come 2 chutneys",
            "cava",
            "chipotle",
            "chick fil",
            "chick-fil",
            "deccan spice",
            "doordash",
            "domino",
            "dunkin",
            "elite india",
            "fresh roti",
            "hashtag india",
            "flik",
            "indi fresh",
            "kakatiya",
            "kona ice",
            "mwj vending",
            "panda express",
            "restaurant",
            "biryani",
            "starbucks",
            "sweetgreen",
            "uber eats",
        ],
        "Groceries": ["whole foods", "trader", "costco", "instacart", "market", "publix", "sagar", "suvidha", "aldi", "patel brothers"],
        "Ride Share": ["uber", "lyft", "taxi", "marta"],
        "Shopping": ["amazon", "target", "nordstrom", "zara", "sephora", "groupon", "walgreens", "florist", "home depot", "marshalls", "old navy"],
        "Housing": ["rent", "mortgage", "apartment"],
        "Travel": ["frontier", "airline", "air lines", "delta", "thales", "hospitable", "trawell", "holiday inn", "paradies"],
        "Auto": ["car wash", "quiktrip", "qt", "parking spot"],
        "Entertainment": ["music", "parks", "fever"],
        "Education": ["samplesdojo"],
        "Donations": ["gofundme", "temple", "paypal gats"],
        "Finance": ["homewithloan"],
        "Services": ["ups store"],
        "Utilities": ["electric", "water", "internet", "verizon", "comcast"],
    }
    for category, terms in category_terms.items():
        if any(term in text for term in terms):
            return category
    return "Uncategorized"


def _empty_transactions() -> pd.DataFrame:
    return pd.DataFrame(columns=TRANSACTION_COLUMNS)
