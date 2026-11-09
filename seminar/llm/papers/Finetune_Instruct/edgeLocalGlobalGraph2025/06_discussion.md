# 6 Discussion

## 6.1 Limitations of evaluation approach

Our evaluation to date has focused on sensemaking questions specific to two corpora each containing approximately 1 million tokens. More work is needed to understand how performance generalizes to datasets from various domains with different use cases. Comparison of fabrication rates, e.g., using approaches like SelfCheckGPT (Manakul et al., 2023), would also strengthen the current analysis.

\begin{table}

**Table 4: Average number of clusters across different distance thresholds, reported by condition and dataset type. Bolded values represent the highest score in each row.**
\begin{tabular}[t]{|l|l|l|l|l|l|l|l|}
\hline \multirow[t]{2}{*}{Dataset} & \multirow[t]{2}{*}{Distance Threshold} & \multicolumn{6}{|c|}{Average Number of Clusters} \\
\hline & & C0 & C1 & C2 & C3 & TS & SS \\
\hline \multirow{4}{*}{News Articles} & 0.5 & 23.42 & 21.85 & 21.90 & 22.13 & 21.80 & 17.92 \\
\hline & 0.6 & 21.65 & 20.38 & 20.30 & 20.52 & 20.13 & 16.78 \\
\hline & 0.7 & 20.19 & 19.06 & 19.03 & 19.13 & 18.62 & 15.80 \\
\hline & 0.8 & 18.86 & 17.78 & 17.82 & 17.79 & 17.30 & 14.80 \\
\hline \multirow{4}{*}{Podcast Transcripts} & 0.5 & 23.16 & 22.62 & 22.52 & 21.93 & 21.14 & 18.55 \\
\hline & 0.6 & 21.65 & 21.33 & 21.21 & 20.62 & 19.70 & 17.39 \\
\hline & 0.7 & 20.41 & 20.04 & 19.79 & 19.22 & 18.08 & 16.28 \\
\hline & 0.8 & 19.26 & 18.77 & 18.46 & 17.89 & 16.66 & 15.07 \\
\hline
\end{tabular}
\end{table}

## 6.2 Future work

The graph index, rich text annotations, and hierarchical community structure supporting the current GraphRAG approach offer many possibilities for refinement and adaptation. This includes RAG approaches that operate in a more local manner, via embedding-based matching of user queries and graph annotations. In particular, we see potential in hybrid RAG schemes that combine embeddingbased matching with just-in-time community report generation before employing our map-reduce summarization mechanisms. This "roll-up" approach could also be extended across multiple levels of the community hierarchy, as well as implemented as a more exploratory "drill down" mechanism that follows the information scent contained in higher-level community summaries.

Broader impacts. As a mechanism for question answering over large document collections, there are risks to downstream sensemaking and decision-making tasks if the generated answers do not accurately represent the source data. System use should be accompanied by clear disclosures of AI use and the potential for errors in outputs. Compared to vector RAG, however, GraphRAG shows promise as a way to mitigate these downstream risks for questions of a global nature, which might otherwise be answered by samples of retrieved facts falsely presented as global summaries.

