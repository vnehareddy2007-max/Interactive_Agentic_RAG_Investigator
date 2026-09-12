"""
Builds a network graph from the candidates, entities, and relationships
extracted during ingestion. Candidates and entities become nodes; extracted
relationships become directed edges.

"""

import os
import sys
import networkx as nx

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from schemas import IngestResponse, GraphNode, GraphEdge, EvidenceGraphResponse


def build_graph(ingest_result: IngestResponse) -> nx.DiGraph:
    graph = nx.DiGraph()

    for candidate in ingest_result.candidates:
        graph.add_node(
            candidate.id,
            label=candidate.name,
            type="candidate",
            role=candidate.role,
        )

    for entity in ingest_result.entities:
        graph.add_node(
            entity.id,
            label=entity.name,
            type=entity.type,
            description=entity.description,
        )

    known_node_ids = set(graph.nodes)

    for rel in ingest_result.relationships:
        if rel.source_id not in known_node_ids or rel.target_id not in known_node_ids:
            continue
        graph.add_edge(
            rel.source_id,
            rel.target_id,
            relation=rel.relation,
            document_id=rel.document_id,
        )

    return graph


def graph_to_response(graph: nx.DiGraph) -> EvidenceGraphResponse:
    nodes = [
        GraphNode(id=node_id, label=data.get("label", node_id), type=data.get("type", "unknown"))
        for node_id, data in graph.nodes(data=True)
    ]
    edges = [
        GraphEdge(source=src, target=tgt, relation=data.get("relation", ""))
        for src, tgt, data in graph.edges(data=True)
    ]
    return EvidenceGraphResponse(nodes=nodes, edges=edges)


def get_neighbors(graph: nx.DiGraph, node_id: str) -> list:
    """Returns all nodes connected to the given node (both directions),
    useful for agents exploring what's linked to a candidate or piece of
    evidence."""
    if node_id not in graph:
        return []
    neighbors = set(graph.successors(node_id)) | set(graph.predecessors(node_id))
    return [
        {"id": n, "label": graph.nodes[n].get("label", n), "type": graph.nodes[n].get("type")}
        for n in neighbors
    ]


def get_candidate_subgraph(graph: nx.DiGraph, candidate_id: str, depth: int = 1) -> nx.DiGraph:
    """Returns the local neighborhood around a candidate, useful for showing
    a focused view instead of the entire graph."""
    if candidate_id not in graph:
        return nx.DiGraph()
    nodes = {candidate_id}
    frontier = {candidate_id}
    for _ in range(depth):
        next_frontier = set()
        for node in frontier:
            next_frontier |= set(graph.successors(node)) | set(graph.predecessors(node))
        nodes |= next_frontier
        frontier = next_frontier
    return graph.subgraph(nodes).copy()


if __name__ == "__main__":
    from ingestion.preprocess import ingest_corpus

    result = ingest_corpus(use_cache=True)
    graph = build_graph(result)

    print(f"Graph built: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    print("\nCandidate nodes:")
    for node_id, data in graph.nodes(data=True):
        if data.get("type") == "candidate":
            print(f"  - {node_id}: {data.get('label')} ({data.get('role')})")

    if result.candidates:
        first_candidate_id = result.candidates[0].id
        neighbors = get_neighbors(graph, first_candidate_id)
        print(f"\nNeighbors of {result.candidates[0].name} ({len(neighbors)} connections):")
        for n in neighbors[:10]:
            print(f"  - {n['label']} ({n['type']})")