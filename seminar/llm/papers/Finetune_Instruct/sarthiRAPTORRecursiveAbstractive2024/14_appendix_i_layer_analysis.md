# I Analysis of Different Layers on Raptor's Performance

# I. 1 How do different Layers impact performance ?

In this section, we present a detailed breakdown of RAPTOR's retrieval performance when querying different layers of the hierarchical tree structure for various stories. These tables validate the utility of RAPTOR's multi-layered structure for diverse query requirements.

\begin{table}

**Table 14: Performance of RAPTOR when querying different layers of the tree for Story 2.**
\begin{tabular}[t]{|l|l|l|l|}
\hline Layers Queried / Start Layer & Layer 0 (Leaf Nodes) & Layer 1 & Layer 2 \\
\hline 1 layer & 58.8 & 47.1 & 41.1 \\
\hline 2 layers & - & 64.7 & 52.9 \\
\hline 3 layers & - & - & 47.1 \\
\hline
\end{tabular}
\end{table}

![Figure](./sarthiRAPTORRecursiveAbstractive2024_images/4beeda7d-949d-4d45-a072-4861cfa71615-22_346_1048_285_518.jpg)

**Figure 7: Histogram showing the percentage of nodes retrieved from different layers of the RAPTOR tree across three datasets (NarrativeQA, Quality, and Qasper) using three retrievers (SBERT, BM25, and DPR). The data indicate that a substantial portion of the nodes contributing to the final retrieval comes from non-leaf layers, with a notable percentage from the first and second layers, highlighting the importance of RAPTOR's hierarchical summarization in the retrieval process.**

\begin{table}

**Table 15: Performance of RAPTOR when querying different layers of the tree for Story 3.**
\begin{tabular}[t]{|l|l|l|l|}
\hline Layers Queried / Start Layer & Layer 0 (Leaf Nodes) & Layer 1 & Layer 2 \\
\hline 1 layer & 66.6 & 61.1 & 61.1 \\
\hline 2 layers & - & 66.6 & 66.6 \\
\hline 3 layers & - & - & 83.3 \\
\hline
\end{tabular}
\end{table}

\begin{table}

**Table 16: Performance of RAPTOR when querying different layers of the tree for Story 4.**
\begin{tabular}[t]{|l|l|l|}
\hline Layers Queried / Start Layer & Layer 0 (Leaf Nodes) & Layer 1 \\
\hline 1 layer & 94.7 & 84.2 \\
\hline 2 layers & - & 89.4 \\
\hline
\end{tabular}
\end{table}

\begin{table}

**Table 17: Performance of RAPTOR when querying different layers of the tree for Story 5.**
\begin{tabular}[t]{|l|l|l|}
\hline Layers Queried / Start Layer & Layer 0 (Leaf Nodes) & Layer 1 \\
\hline 1 layer & 57.9 & 47.3 \\
\hline 2 layers & - & 68.4 \\
\hline
\end{tabular}
\end{table}

# I. 2 Which Layers do Retrieved Nodes come from ?

We further conduct an ablation study across all three datasets and across three different retrievers with RAPTOR with the collapsed tree retrieval to examine the layers from which the retrieved nodes originate. We observe that between 18.5\% to 57\% of the retrieved nodes come from non-leaf nodes. As illustrated in Figure 7, the retrieval pattern across layers reveals the importance of RAPTOR's multi-layered tree structure. Notably, a significant percentage of the nodes retrieved by RAPTOR using the DPR retriever for the NarrativeQA dataset come from the first and second layers of the tree, as opposed to the leaf nodes. This pattern is consistent across the other datasets and retrievers, albeit with varying percentages.

\begin{table}

**Table 18: Percentage of nodes from non-leaf nodes across different datasets and retrievers**
\begin{tabular}[t]{|l|l|l|l|}
\hline Dataset & DPR & SBERT & BM25 \\
\hline NarrativeQA & 57.36\% & 36.78\% & 34.96\% \\
\hline Quality & 32.28\% & 24.41\% & 32.36\% \\
\hline Qasper & 22.93\% & 18.49\% & 22.76\% \\
\hline
\end{tabular}
\end{table}

\begin{table}

**Table 19: Percentage of nodes from different layers with DPR as the retriever**
\begin{tabular}[t]{|l|l|l|l|}
\hline Layer & NarrativeQA & Quality & Qasper \\
\hline 0 & 42.64\% & 67.71\% & 77.07\% \\
\hline 1 & 45.00\% & 29.43\% & 21.88\% \\
\hline 2 & 10.57\% & 2.85\% & 1.05\% \\
\hline 3 & 1.78\% & - & - \\
\hline 4 & 0.003\% & - & - \\
\hline
\end{tabular}
\end{table}

\begin{table}

**Table 20: Percentage of nodes from different layers with SBERT as the retriever**
\begin{tabular}[t]{|l|l|l|l|}
\hline Layer & NarrativeQA & Quality & Qasper \\
\hline 0 & 63.22\% & 75.59\% & 81.51\% \\
\hline 1 & 31.51\% & 22.78\% & 17.84\% \\
\hline 2 & 4.85\% & 1.63\% & 0.65\% \\
\hline 3 & 0.42\% & - & - \\
\hline
\end{tabular}
\end{table}

\begin{table}

**Table 21: Percentage of nodes from different layers with BM25 as the retriever**
\begin{tabular}[t]{|l|l|l|l|}
\hline Layer & NarrativeQA & Quality & Qasper \\
\hline 0 & 65.04\% & 67.64\% & 77.24\% \\
\hline 1 & 28.79\% & 28.85\% & 21.57\% \\
\hline 2 & 5.36\% & 3.51\% & 1.19\% \\
\hline 3 & 0.81\% & - & - \\
\hline
\end{tabular}
\end{table}
