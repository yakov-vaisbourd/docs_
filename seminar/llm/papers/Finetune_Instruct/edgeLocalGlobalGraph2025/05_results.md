# 5 Results

## 5.1 Experiment 1

The indexing process resulted in a graph consisting of 8,564 nodes and 20,691 edges for the Podcast dataset, and a larger graph of 15,754 nodes and 19,520 edges for the News dataset. Table 2 shows the number of community summaries at different levels of each graph community hierarchy.

Global approaches vs. vector RAG. As shown in Figure 2 and Table 6, global approaches significantly outperformed conventional vector RAG (SS) in both comprehensiveness and diversity criteria across datasets. Specifically, global approaches achieved comprehensiveness win rates between 72-83\% (p<.001) for Podcast transcripts and 72-80\% (p<.001) for News articles, while diversity win rates ranged from 75-82\% (p<.001) and 62-71\% (p<.01) respectively. Our use of directness as a validity test confirmed that vector RAG produces the most direct responses across all comparisons.

Empowerment. Empowerment comparisons showed mixed results for both global approaches versus vector RAG (SS) and GraphRAG approaches versus source text summarization (TS). Using an LLM to analyze LLM reasoning for this measure indicated that the ability to provide specific exam-

![Figure](./edgeLocalGlobalGraph2025_images/61527f6e-2776-44d9-a50d-2dad02842bfe-10_903_1353_238_376.jpg)

**Figure 2: Head-to-head win rate percentages of (row condition) over (column condition) across two datasets, four metrics, and 125 questions per comparison (each repeated five times and averaged). The overall winner per dataset and metric is shown in bold. Self-win rates were not computed but are shown as the expected $50 \%$ for reference. All Graph RAG conditions outperformed naïve RAG on comprehensiveness and diversity. Conditions C1-C3 also showed slight improvements in answer comprehensiveness and diversity over TS (global text summarization without a graph index).**

\begin{table}

**Table 2: Number of context units (community summaries for C0-C3 and text chunks for TS), corresponding token counts, and percentage of the maximum token count. Map-reduce summarization of source texts is the most resource-intensive approach requiring the highest number of context tokens. Root-level community summaries (C0) require dramatically fewer tokens per query (9x-43x).**
\begin{tabular}[t]{|l|l|l|l|l|l|l|l|l|l|l|}
\hline & \multicolumn{4}{|l|}{Podcast Transcripts} & \multicolumn{6}{|c|}{News Articles} \\
\hline & C0 & C1 & C2 & C3 & TS & C0 & C1 & C2 & C3 & TS \\
\hline Units & 34 & 367 & 969 & 1310 & 1669 & 55 & 555 & 1797 & 2142 & 3197 \\
\hline Tokens & 26657 & 225756 & 565720 & 746100 & 1014611 & 39770 & 352641 & 980898 & 1140266 & 1707694 \\
\hline \% Max & 2.6 & 22.2 & 55.8 & 73.5 & 100 & 2.3 & 20.7 & 57.4 & 66.8 & 100 \\
\hline
\end{tabular}
\end{table}
ples, quotes, and citations was judged to be key to helping users reach an informed understanding. Tuning element extraction prompts may help to retain more of these details in the GraphRAG index.

Community summaries vs. source texts. When comparing community summaries to source texts using GraphRAG, community summaries generally provided a small but consistent improvement in answer comprehensiveness and diversity, except for root-level summaries. Intermediate-level summaries in the Podcast dataset and low-level community summaries in the News dataset achieved comprehensiveness win rates of \(57 \%(\mathrm{p}<.001)\) and \(64 \%(\mathrm{p}<.001)\), respectively. Diversity win rates were \(57 \%(\mathrm{p}=.036)\) for Podcast intermediate-level summaries and 60\% ( $\mathrm{p}<.001$ ) for News low-level community summaries. Table 2 also illustrates the scalability advantages of GraphRAG compared to source text summarization: for low-level community summaries (C3), GraphRAG required 26-33\% fewer context tokens, while for root-level community summaries (C0), it required over 97\% fewer tokens. For a modest drop in performance compared with other global methods, root-level GraphRAG offers a highly efficient method for the iterative question answering that characterizes sensemaking activity, while retaining advantages in comprehensiveness (72\% win rate) and diversity (62\% win rate) over vector RAG.

\begin{table}

**Table 3: Average number of extracted claims, reported by condition and dataset type. Bolded values represent the highest score in each column.**
\begin{tabular}[t]{|l|l|l|}
\hline \multirow[t]{2}{*}{Condition} & \multicolumn{2}{|c|}{Average Number of Claims} \\
\hline & News Articles & Podcast Transcripts \\
\hline C0 & 34.18 & 32.21 \\
\hline C1 & 32.50 & 32.20 \\
\hline C2 & 31.62 & 32.46 \\
\hline C3 & 33.14 & 32.28 \\
\hline TS & 32.89 & 31.39 \\
\hline SS & 25.23 & 26.50 \\
\hline
\end{tabular}
\end{table}

## 5.2 Experiment 2

Table 3 shows the results for the average number of extracted claims (i.e., the claim-based measure of comprehensiveness) per condition. For both the News and Podcast datasets, all global search conditions (C0-C3) and source text summarization (TS) had greater comprehensiveness than vector RAG (SS). The differences were statistically significant (p<.05) in all cases. These findings align with the LLM-based win rates from Experiment 1.

Table 4 contains the results for the average number of clusters, the claim-based measure of diversity. For the Podcast dataset, all global search conditions had significantly greater diversity than SS across all distance thresholds \((\mathrm{p}<.05)\), consistent with the win rates observed in Experiment 1. For the News dataset, however, only C0 significantly outperformed SS across all distance thresholds \((\mathrm{p}<.05)\). While C1-C3 also achieved higher average cluster counts than SS, the differences were statistically significant only at certain distance thresholds. In Experiment 1, all global search conditions significantly outperformed SS in the News dataset - not just C0. However, the differences in mean diversity scores between SS and the global search conditions were smaller for the News dataset than for the Podcast dataset, aligning directionally with the claim-based results.

For both comprehensiveness and diversity, across both datasets, there were no statistically significant differences observed among the global search conditions or between global search and TS.

Finally, for each pairwise comparison in Experiment 1, we tested whether the answer preferred by the LLM aligned with the winner based on the claim-based metrics. Since each pairwise comparison in Experiment 1 was performed five times, while the claim-based metrics provided only one outcome per comparison, we aggregated the Experiment 1 results into a single label using majority voting. For example, if C0 won over SS in three out of five judgments for comprehensiveness on a given question, C0 was labeled the winner and SS the loser. However, if C0 won twice, SS won once, and they tied twice, then there was no majority outcome, so the final label was a tie.

We found that exact ties were rare for the claim-based metrics. One possible solution is to define a tie based on a threshold (e.g., the absolute difference between the claim-based results for condition A and condition B must be less than or equal to $x$ ). However, we observed that the results were sensitive to the choice of threshold. As a result, we focused on cases where the aggregated LLM label was not a tie, representing 33\% and 39\% of pairwise comparisons for comprehensiveness and diversity, respectively. In these cases, the aggregated LLM label matched the claim-based label in 78\% of pairwise comparisons for comprehensiveness and 69-70\% for diversity (across all distance thresholds), indicating moderately strong alignment.

