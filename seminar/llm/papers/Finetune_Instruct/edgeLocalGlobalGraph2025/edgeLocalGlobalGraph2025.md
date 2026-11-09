\title{
From Local to Global: A GraphRAG Approach to Query-Focused Summarization
}

\author{
Darren Edge ${ }^{\mathbf{1} \boldsymbol{\dagger}}$ Ha Trinh ${ }^{\mathbf{1} \boldsymbol{\dagger}}$ Newman Cheng ${ }^{\mathbf{2}}$ Joshua Bradley ${ }^{\mathbf{2}}$ Alex Chao ${ }^{\mathbf{3}}$ \\ Apurva Mody ${ }^{3}$ Steven Truitt ${ }^{2}$ Dasha Metropolitansky ${ }^{1}$ Robert Osazuwa Ness ${ }^{1}$ \\ Jonathan Larson ${ }^{1}$ \\ ${ }^{1}$ Microsoft Research \\ ${ }^{2}$ Microsoft Strategic Missions and Technologies \\ ${ }^{3}$ Microsoft Office of the CTO \\ \{daedge,trinhha,newmancheng,joshbradley,achao,moapurva, steventruitt,dasham,robertness,jolarso\}@microsoft.com \\ ${ }^{\dagger}$ These authors contributed equally to this work
}

\begin{abstract}
The use of retrieval-augmented generation (RAG) to retrieve relevant information from an external knowledge source enables large language models (LLMs) to answer questions over private and/or previously unseen document collections. However, RAG fails on global questions directed at an entire text corpus, such as "What are the main themes in the dataset?", since this is inherently a queryfocused summarization (QFS) task, rather than an explicit retrieval task. Prior QFS methods, meanwhile, do not scale to the quantities of text indexed by typical RAG systems. To combine the strengths of these contrasting methods, we propose GraphRAG, a graph-based approach to question answering over private text corpora that scales with both the generality of user questions and the quantity of source text. Our approach uses an LLM to build a graph index in two stages: first, to derive an entity knowledge graph from the source documents, then to pregenerate community summaries for all groups of closely related entities. Given a question, each community summary is used to generate a partial response, before all partial responses are again summarized in a final response to the user. For a class of global sensemaking questions over datasets in the 1 million token range, we show that GraphRAG leads to substantial improvements over a conventional RAG baseline for both the comprehensiveness and diversity of generated answers.
\end{abstract}

