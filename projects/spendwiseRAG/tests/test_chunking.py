from spendwise_rag.core.models import ParsedStatement
from spendwise_rag.processing.chunking import chunk_node


def test_chunk_node_builds_transactions_and_removes_duplicate_artifacts():
    statement = ParsedStatement(
        card_type="chase",
        statement_month="May 2026",
        statement_year=2026,
        namespace="chase_2026",
        raw_tables=[
            ["Date", "Description", "Category", "Amount"],
            ["05/03/2026", "Spotify", "Subscriptions", "$9.99"],
            ["05/03/2026", "Spotify", "Subscriptions", "$9.99"],
            ["05/07/2026", "Trader Joe's", "Groceries", "$42.10"],
        ],
        raw_text=["Dining Out Total: $123.45. This statement contains 2 purchases."],
    )

    chunks = chunk_node(statement)

    transaction_chunks = [chunk for chunk in chunks if chunk.metadata["chunk_type"] == "transaction"]
    rollup_chunks = [chunk for chunk in chunks if chunk.metadata["chunk_type"] == "rollup"]

    assert len(transaction_chunks) == 2
    assert transaction_chunks[0].metadata["merchant"] == "Spotify"
    assert len(rollup_chunks) == 1
    assert "Total Dining Out" in rollup_chunks[0].text


def test_chunk_node_falls_back_to_text_transaction_lines():
    statement = ParsedStatement(
        card_type="amex",
        statement_month="May 2026",
        statement_year=2026,
        namespace="amex_2026",
        raw_tables=[],
        raw_text=[
            """
            Closing Date 05/31/2026
            05/03/26 Spotify $9.99
            05/04/26 ONLINE PAYMENT THANK YOU $500.00
            05/07/26 Trader Joe's $42.10
            """
        ],
    )

    chunks = chunk_node(statement)
    transaction_chunks = [chunk for chunk in chunks if chunk.metadata["chunk_type"] == "transaction"]

    assert len(transaction_chunks) == 2
    assert [chunk.metadata["merchant"] for chunk in transaction_chunks] == ["Spotify", "Trader Joe's"]
    assert [chunk.metadata["amount"] for chunk in transaction_chunks] == [9.99, 42.10]


def test_chunk_node_extracts_collapsed_amex_transaction_block_and_suppresses_summary_blob():
    block = (
        "#351179 Q35 351179 ALPHARETTA GA $4.48 678-393-5146 "
        "01/30/26 AplPay FRESH ROTI Suwanee GA $9.22 squareup.com/receipts "
        "01/30/26 AplPay OM INDIAN MARKET Cumming GA $15.54 squareup.com/receipts "
        "01/30/26 AplPay PUBLIX CUMMING GA $76.61 8636471171 "
        "02/03/26 Uber Trip help.uber.com CA $19.42 COZMIJVB 76262"
    )
    statement = ParsedStatement(
        card_type="amex",
        statement_month="May 2026",
        statement_year=2026,
        namespace="amex_2026",
        raw_tables=[],
        raw_text=[block],
    )

    chunks = chunk_node(statement)
    transaction_chunks = [chunk for chunk in chunks if chunk.metadata["chunk_type"] == "transaction"]
    summary_chunks = [chunk for chunk in chunks if chunk.metadata["chunk_type"] == "summary"]

    assert [chunk.metadata["merchant"] for chunk in transaction_chunks] == [
        "Fresh Roti",
        "Om Indian Market",
        "Publix",
        "Uber",
    ]
    assert [chunk.metadata["category"] for chunk in transaction_chunks] == [
        "Dining",
        "Groceries",
        "Groceries",
        "Ride Share",
    ]
    assert summary_chunks == []


def test_chunk_node_cleans_lyft_phone_and_state_noise():
    statement = ParsedStatement(
        card_type="amex",
        statement_month="May 2026",
        statement_year=2026,
        namespace="amex_2026",
        raw_tables=[],
        raw_text=["01/21/26 LYFT 855-280-0278 CA $8.95"],
    )

    chunks = chunk_node(statement)
    transaction = [chunk for chunk in chunks if chunk.metadata["chunk_type"] == "transaction"][0]

    assert transaction.metadata["merchant"] == "Lyft"
    assert transaction.metadata["category"] == "Ride Share"
    assert transaction.text == "2026-01-21 | Lyft | $8.95 | Ride Share"
