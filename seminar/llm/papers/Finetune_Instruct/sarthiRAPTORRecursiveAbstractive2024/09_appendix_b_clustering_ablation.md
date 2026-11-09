# B Ablation Study on Clustering Mechanism in RAPTOR

To assess the effectiveness of the clustering mechanism in our RAPTOR approach, we conducted an ablation study on the QuALITY dataset. This study compares RAPTOR's performance with a balanced tree-style encoding and summarization of contiguous chunks, in contrast to our standard clustering method.

# B. 1 Methodology

Both configurations in this ablation study utilized SBERT embeddings and UnifiedQA to maintain consistency in retrieval. For RAPTOR, we employed our typical clustering and summarization process. In contrast, the alternative setup involved creating a balanced tree by recursively encoding and summarizing contiguous text chunks. We determined the window size for this setup based on the average cluster size observed in RAPTOR, which is approximately 6.7 nodes. Hence, we chose a window size of 7 nodes. The collapsed tree approach was applied for retrieval in both models.

# B. 2 Results \& Discussion

The results of the ablation study are presented in table 9. The results from this ablation study clearly indicate an improvement in accuracy when employing RAPTOR's clustering mechanism over the recency-based tree approach. This finding substantiates our hypothesis that the clustering strategy in RAPTOR is more effective in capturing homogeneous content for summarization, thereby enhancing the overall retrieval performance.

\begin{table}

**Table 9: Ablation study results comparing RAPTOR with a recency-based tree approach**
\begin{tabular}[t]{|l|l|}
\hline Configuration & Accuracy \\
\hline RAPTOR + SBERT embeddings + UnifiedQA & 56.6\% \\
\hline Recency-based tree + SBERT embeddings + UnifiedQA & 55.8\% \\
\hline
\end{tabular}
\end{table}

