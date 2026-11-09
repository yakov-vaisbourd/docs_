# Practical Engineering Concepts

## Hallucination

LLMs generate plausible but factually incorrect content. Mitigations include: RAG grounding (anchoring answers to retrieved documents reduces fabrication of unsupported facts), generating responses multiple times and flagging inconsistencies, constrained decoding (restricting output format to a known schema), chain-of-thought prompting (making reasoning steps explicit and checkable), and human-in-the-loop review for high-stakes outputs.

## Vector Databases

Databases optimised for approximate nearest-neighbour search over high-dimensional embeddings (e.g.\ FAISS, ChromaDB). Used in RAG systems to retrieve the most similar past examples at inference time. Retrieval complexity is sublinear via indexing structures (e.g.\ HNSW, IVF). The cosine similarity $S_c(\mathbf{s}_q, \mathbf{s}^{(i)}) = \frac{\mathbf{s}_q \cdot \mathbf{s}^{(i)}}{\|\mathbf{s}_q\|\|\mathbf{s}^{(i)}\|}$ is the standard retrieval metric.
