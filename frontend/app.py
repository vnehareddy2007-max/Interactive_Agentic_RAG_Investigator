"""
Streamlit UI for the Interactive Agentic RAG Investigation app. Talks to
the FastAPI backend (must be running separately via `uvicorn main:app`).

"""

import requests
import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt

BACKEND_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Solve the Case", layout="wide")


def call_api(method: str, path: str, json: dict = None):
    try:
        response = requests.request(method, f"{BACKEND_URL}{path}", json=json, timeout=60)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API error calling {path}: {e}")
        return None


@st.cache_data(show_spinner="Loading case file...")
def load_ingest():
    result = call_api("POST", "/ingest")
    if result is None:
        st.cache_data.clear()
    return result


if "ingest_result" not in st.session_state:
    st.session_state.ingest_result = load_ingest()
if "investigator_result" not in st.session_state:
    st.session_state.investigator_result = None
if "fact_check_result" not in st.session_state:
    st.session_state.fact_check_result = None
if "user_prediction" not in st.session_state:
    st.session_state.user_prediction = None

ingest_result = st.session_state.ingest_result

st.title("🔍 Solve the Case: The Skyline Retreat Poisoning")

if not ingest_result:
    st.error("Could not load the case file. Is the backend running (`uvicorn main:app --reload`)?")
    st.stop()

candidates = ingest_result["candidates"]
candidate_names = [c["name"] for c in candidates]
name_to_id = {c["name"]: c["id"] for c in candidates}

tabs = st.tabs([
    "Case Overview", "Evidence Graph", "Interrogate", "Search Evidence",
    "Investigate & Fact-Check", "Submit Verdict",
])

with tabs[0]:
    st.subheader("Candidates / Suspects")
    for c in candidates:
        st.markdown(f"**{c['name']}** — {c.get('role', 'Unknown role')}")

    st.subheader("Case Summary")
    st.write(
        "CEO Aditya Rao was found dead at his company's retreat, poisoned with "
        "cyanide. Four people were near the bar during the window of poisoning. "
        "Use the tabs above to interrogate suspects, search evidence, run the "
        "Investigator and Fact-Checker agents, and submit your final verdict."
    )


with tabs[1]:
    st.subheader("Evidence Graph")
    graph_data = call_api("GET", "/graph")

    if graph_data:
        g = nx.DiGraph()
        for node in graph_data["nodes"]:
            g.add_node(node["id"], label=node["label"], type=node["type"])
        for edge in graph_data["edges"]:
            g.add_edge(edge["source"], edge["target"], relation=edge["relation"])

        fig, ax = plt.subplots(figsize=(12, 8))
        pos = nx.spring_layout(g, k=0.6, seed=42)

        color_map = {"candidate": "#e74c3c"}
        node_colors = [color_map.get(g.nodes[n]["type"], "#3498db") for n in g.nodes]
        labels = {n: g.nodes[n]["label"] for n in g.nodes}

        nx.draw_networkx_nodes(g, pos, node_color=node_colors, node_size=600, alpha=0.9, ax=ax)
        nx.draw_networkx_edges(g, pos, alpha=0.3, arrows=True, ax=ax)
        nx.draw_networkx_labels(g, pos, labels=labels, font_size=7, ax=ax)

        ax.axis("off")
        st.pyplot(fig)
        st.caption("Red nodes = candidates/suspects. Blue nodes = other entities (people, places, objects, events).")
    else:
        st.warning("Could not load the graph.")


with tabs[2]:
    st.subheader("Interrogate a Candidate")
    selected_name = st.selectbox("Choose a candidate", candidate_names)
    question = st.text_input("Your question")

    if st.button("Ask") and question:
        with st.spinner("Awaiting response..."):
            result = call_api("POST", "/interrogate", {
                "candidate_id": name_to_id[selected_name],
                "question": question,
            })
        if result:
            st.markdown(f"**{selected_name}:** {result['answer']}")
            if result.get("grounded_document_ids"):
                st.caption(f"Grounded in: {', '.join(result['grounded_document_ids'])}")


with tabs[3]:
    st.subheader("Search the Evidence Corpus")
    query = st.text_input("Search query", key="search_query")
    top_k = st.slider("Number of results", 1, 10, 5)

    if st.button("Search") and query:
        with st.spinner("Searching..."):
            result = call_api("POST", "/search_evidence", {"query": query, "top_k": top_k})
        if result:
            for r in result["results"]:
                with st.expander(f"{r['document_id']} (combined score: {r.get('combined_score', 0):.3f})"):
                    st.text(r["text"])


with tabs[4]:
    st.subheader("Make Your Prediction First")
    st.caption("Before seeing the AI agents' results, record your own guess.")

    with st.form("prediction_form"):
        predicted_suspect = st.selectbox("Who do you think did it?", candidate_names)
        predicted_reasoning = st.text_area("Why?")
        submitted = st.form_submit_button("Lock In My Prediction")
        if submitted:
            st.session_state.user_prediction = {
                "suspect": predicted_suspect,
                "reasoning": predicted_reasoning,
            }
            st.success("Prediction locked in. Now run the agents below.")

    st.divider()

    st.subheader("Investigator Agent")
    if st.button("Run Investigator"):
        with st.spinner("Investigating (this may take a moment)..."):
            result = call_api("POST", "/investigate", {})
        st.session_state.investigator_result = result

    if st.session_state.investigator_result:
        r = st.session_state.investigator_result
        st.markdown(f"**Theory:** {r['theory']}")
        st.markdown(f"**Suspect:** {r.get('suspect_id', 'unknown')} | **Confidence:** {r['confidence']:.2f} | **Retries used:** {r['retries_used']}")
        st.markdown("**Citations:**")
        for c in r["citations"]:
            st.markdown(f"- `[{c['document_id']}]` ({c['status']}) {c['claim']}")

    st.divider()

    st.subheader("Fact-Checker Agent")
    fact_check_disabled = st.session_state.investigator_result is None
    if st.button("Run Fact-Checker", disabled=fact_check_disabled):
        with st.spinner("Fact-checking the theory..."):
            inv = st.session_state.investigator_result
            result = call_api("POST", "/fact_check", {
                "theory": inv["theory"],
                "suspect_id": inv.get("suspect_id"),
            })
        st.session_state.fact_check_result = result

    if fact_check_disabled:
        st.info("Run the Investigator first.")

    if st.session_state.fact_check_result:
        fc = st.session_state.fact_check_result
        st.markdown(f"**Overall Assessment:** {fc['overall_assessment']}")
        if fc["contradictions"]:
            st.markdown("**Contradictions found:**")
            for c in fc["contradictions"]:
                st.markdown(f"- `[{c['conflicting_document_id']}]` {c['claim']} → {c['explanation']}")
        if fc["weaknesses"]:
            st.markdown("**Weaknesses:**")
            for w in fc["weaknesses"]:
                st.markdown(f"- {w}")


with tabs[5]:
    st.subheader("Submit Your Final Verdict")

    if st.session_state.user_prediction:
        st.caption(f"Your initial prediction was: **{st.session_state.user_prediction['suspect']}**")

    final_suspect = st.selectbox("Final suspect", candidate_names, key="final_suspect")
    final_reasoning = st.text_area("Your final reasoning")
    supporting_docs = st.multiselect(
        "Supporting document IDs",
        [f"doc_{i:02d}" for i in range(1, 20)],
    )

    if st.button("Submit Verdict"):
        with st.spinner("Evaluating..."):
            result = call_api("POST", "/submit_verdict", {
                "suspect_id": name_to_id[final_suspect],
                "reasoning": final_reasoning,
                "supporting_document_ids": supporting_docs,
            })
        if result:
            if result["correct"]:
                st.success(result["feedback"])
            else:
                st.error(result["feedback"])