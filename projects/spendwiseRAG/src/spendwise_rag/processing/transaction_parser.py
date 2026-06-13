from __future__ import annotations

import re
import csv
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path


DATE_RE = re.compile(r"\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b")
AMOUNT_RE = re.compile(r"\$?\(?-?\d{1,6}(?:,\d{3})*(?:\.\d{2})\)?")
TEXT_TRANSACTION_RE = re.compile(
    r"(?P<date>\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?)\s+"
    r"(?P<merchant>.*?)\s+"
    r"(?P<amount>\$?\(?-?\d{1,6}(?:,\d{3})*(?:\.\d{2})\)?)"
)

NON_EXPENSE_TERMS = (
    "payment thank you",
    "online payment",
    "autopay payment",
    "interest charge",
    "balance transfer",
    "minimum payment",
    "new balance",
    "previous balance",
    "payments and credits",
)

US_STATES = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "IA",
    "ID", "IL", "IN", "KS", "KY", "LA", "MA", "MD", "ME", "MI", "MN", "MO",
    "MS", "MT", "NC", "ND", "NE", "NH", "NJ", "NM", "NV", "NY", "OH", "OK",
    "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VA", "VT", "WA", "WI",
    "WV", "WY", "US", "USA",
}

MERCHANT_ALIASES = {
    "fresh roti": "Fresh Roti",
    "om indian market": "Om Indian Market",
    "publix": "Publix",
    "patel brothers": "Patel Brothers",
    "target": "Target",
    "marshalls": "Marshalls",
    "flik mobile": "Flik Mobile",
    "kona ice": "Kona Ice",
    "uber": "Uber",
    "lyft": "Lyft",
    "spotify": "Spotify",
    "trader joe": "Trader Joe's",
    "whole foods": "Whole Foods",
    "costco": "Costco",
    "aldi": "Aldi",
}

CATEGORY_TERMS = {
    "Ride Share": ("uber", "lyft", "taxi", "marta"),
    "Groceries": ("publix", "patel brothers", "indian market", "trader joe", "whole foods", "costco", "aldi", "grocery"),
    "Dining Out": ("fresh roti", "restaurant", "cafe", "coffee", "kona ice", "flik mobile", "doordash", "ubereats"),
    "Shopping": ("target", "marshalls", "amazon", "walmart", "nordstrom", "sephora", "zara"),
    "Subscriptions": ("spotify", "netflix", "hulu", "apple", "google"),
}

GOLDEN_RULES_PATH = Path(__file__).resolve().parents[4] / "spendwise-agent" / "data" / "category_rules.csv"


@dataclass(frozen=True)
class ParsedTransaction:
    date: str
    merchant: str
    amount: float
    category: str


@dataclass(frozen=True)
class CategoryRule:
    pattern: str
    merchant: str
    category: str


def normalize_pattern(value: object) -> str:
    text = str(value or "").lower()
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


@lru_cache(maxsize=1)
def load_golden_category_rules(path: str | None = None) -> tuple[CategoryRule, ...]:
    rules_path = Path(path) if path else GOLDEN_RULES_PATH
    if not rules_path.exists():
        return ()

    rules: list[CategoryRule] = []
    with rules_path.open(newline="") as file:
        for row in csv.DictReader(file):
            pattern = normalize_pattern(row.get("pattern", ""))
            merchant = str(row.get("merchant", "")).strip()
            category = str(row.get("category", "")).strip()
            if pattern and merchant and category:
                rules.append(CategoryRule(pattern=pattern, merchant=merchant, category=category))
    return tuple(sorted(rules, key=lambda rule: len(rule.pattern), reverse=True))


def save_golden_category_rules(rules: list[CategoryRule], path: str | None = None) -> None:
    rules_path = Path(path) if path else GOLDEN_RULES_PATH
    rules_path.parent.mkdir(parents=True, exist_ok=True)
    deduped = {rule.pattern: rule for rule in rules if rule.pattern and rule.merchant and rule.category}
    ordered = sorted(deduped.values(), key=lambda rule: rule.pattern)
    with rules_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["pattern", "merchant", "category"])
        writer.writeheader()
        for rule in ordered:
            writer.writerow({"pattern": rule.pattern, "merchant": rule.merchant, "category": rule.category})
    load_golden_category_rules.cache_clear()


def merge_golden_category_rules(rows: list[dict[str, str]], path: str | None = None) -> int:
    existing = list(load_golden_category_rules(path))
    before = {rule.pattern for rule in existing}
    new_rules = []
    for row in rows:
        merchant = str(row.get("merchant", "")).strip()
        category = str(row.get("category", row.get("suggested_category", ""))).strip()
        pattern = normalize_pattern(row.get("pattern", "") or merchant)
        if merchant and pattern and category and category != "Uncategorized":
            new_rules.append(CategoryRule(pattern=pattern, merchant=merchant, category=category))
    save_golden_category_rules(existing + new_rules, path)
    after = {rule.pattern for rule in load_golden_category_rules(path)}
    return len(after - before)


def match_golden_category_rule(value: str) -> CategoryRule | None:
    normalized = normalize_pattern(value)
    for rule in load_golden_category_rules():
        if rule.pattern in normalized:
            return rule
    return None


def parse_amount(value: object) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    negative = text.startswith("(") and text.endswith(")")
    cleaned = re.sub(r"[^0-9.\-]", "", text)
    if cleaned in {"", "-", "."}:
        return None
    try:
        amount = abs(float(cleaned))
    except ValueError:
        return None
    return -amount if negative else amount


def parse_date(value: object, default_year: int) -> str | None:
    text = str(value or "").strip()
    for fmt in ("%m/%d/%Y", "%m/%d/%y", "%m-%d-%Y", "%m-%d-%y"):
        try:
            return datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            pass
    for fmt in ("%m/%d", "%m-%d"):
        try:
            parsed = datetime.strptime(text, fmt)
            return parsed.replace(year=default_year).date().isoformat()
        except ValueError:
            pass
    return None


def is_non_expense_text(text: str) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in NON_EXPENSE_TERMS)


def clean_merchant(value: str) -> str:
    merchant = re.sub(r"\s+", " ", value or "").strip(" |:-")
    golden_rule = match_golden_category_rule(merchant)
    if golden_rule:
        return golden_rule.merchant

    lowered_original = merchant.lower()
    for needle, label in MERCHANT_ALIASES.items():
        if needle in lowered_original:
            return label
    merchant = re.sub(r"\b(APLPAY|APPLEPAY|POS|DEBIT|CARD|PURCHASE|AUTH|CHECKCARD|VISA|TST)\b", " ", merchant, flags=re.I)
    merchant = re.sub(r"\b(?:\+?1?[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b", " ", merchant)
    merchant = re.sub(r"\b(?:squareup\.com/receipts|help\.uber\.com|[a-z0-9.-]+\.(?:com|net|org)(?:/\S*)?)\b", " ", merchant, flags=re.I)
    merchant = re.sub(r"\b\d{5,}\b", " ", merchant)
    merchant = re.sub(r"\b(?=[A-Z0-9]*\d)[A-Z0-9]{6,}\b", " ", merchant)
    merchant = re.sub(r"\s+", " ", merchant).strip(" |:-")

    parts = merchant.split()
    while parts and parts[-1].upper() in US_STATES:
        parts.pop()
    merchant = " ".join(parts)
    merchant = re.sub(r"\s+", " ", merchant).strip(" |:-")

    lowered = merchant.lower()
    for needle, label in MERCHANT_ALIASES.items():
        if needle in lowered:
            return label
    return merchant.title() if merchant else ""


def infer_category(merchant: str) -> str:
    golden_rule = match_golden_category_rule(merchant)
    if golden_rule:
        return golden_rule.category

    lowered = merchant.lower()
    for category, terms in CATEGORY_TERMS.items():
        if any(term in lowered for term in terms):
            return category
    return "Uncategorized"


def _segments_by_date(text: str) -> list[str]:
    matches = list(DATE_RE.finditer(text))
    segments: list[str] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        segment = text[match.start() : end].strip()
        if segment:
            segments.append(segment)
    return segments


def _parse_transaction_segment(segment: str, default_year: int) -> ParsedTransaction | None:
    normalized = re.sub(r"\s+", " ", segment).strip()
    if not normalized or is_non_expense_text(normalized):
        return None
    match = TEXT_TRANSACTION_RE.match(normalized)
    if not match:
        return None
    merchant = match.group("merchant").strip()
    if not merchant or "$" in merchant or "%" in merchant:
        return None
    date = parse_date(match.group("date"), default_year)
    amount = parse_amount(match.group("amount"))
    cleaned = clean_merchant(merchant)
    if not date or amount is None or not cleaned:
        return None
    return ParsedTransaction(date=date, merchant=cleaned, amount=amount, category=infer_category(cleaned))


def extract_transactions_from_text_block(text: str, default_year: int) -> list[ParsedTransaction]:
    transactions: list[ParsedTransaction] = []
    seen: set[tuple[str, str, float]] = set()
    normalized_block = re.sub(r"\s*\|\s*", " ", text)
    candidates = normalized_block.splitlines()
    if len(candidates) <= 1:
        candidates = _segments_by_date(normalized_block)

    for candidate in candidates:
        segments = _segments_by_date(candidate) if len(DATE_RE.findall(candidate)) > 1 else [candidate]
        for segment in segments:
            parsed = _parse_transaction_segment(segment, default_year)
            if not parsed:
                continue
            key = (parsed.date, parsed.merchant.lower(), round(parsed.amount, 2))
            if key in seen:
                continue
            seen.add(key)
            transactions.append(parsed)
    return transactions


def is_transaction_dense_text(text: str) -> bool:
    return len(DATE_RE.findall(text)) >= 3 and len(AMOUNT_RE.findall(text)) >= 3
