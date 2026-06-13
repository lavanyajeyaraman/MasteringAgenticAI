from __future__ import annotations

from .models import BankConfig


BANK_CONFIGS: dict[str, BankConfig] = {
    "chase": BankConfig(
        name="chase",
        amount_column_index=3,
        merchant_column_index=1,
        category_column_index=2,
    ),
    "amex": BankConfig(
        name="amex",
        date_format="%m/%d/%y",
        amount_column_index=2,
        merchant_column_index=1,
    ),
    "bofa": BankConfig(
        name="bofa",
        amount_column_index=3,
        merchant_column_index=1,
        category_column_index=2,
    ),
    "generic": BankConfig(name="generic", amount_column_index=2, merchant_column_index=1),
}


def get_bank_config(card_type: str) -> BankConfig:
    lowered = card_type.lower()
    for key, config in BANK_CONFIGS.items():
        if key in lowered:
            return config
    return BANK_CONFIGS["generic"]
