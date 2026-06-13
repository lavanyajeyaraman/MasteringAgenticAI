from spendwise_rag.graphs.ingestion_graph import build_ingestion_graph


def test_ingestion_graph_exposes_expected_nodes():
    graph = build_ingestion_graph()

    assert set(graph.nodes) >= {
        "upload_node",
        "parse_node",
        "chunk_node",
        "embed_node",
        "upsert_node",
    }

