# 6 Experimental Results

We first report overall results across 4 tasks/datasets and compare the performance of FLARE with all the baselines introduced in section 4. We then run ablation experiments to study the efficacy of various design choices of our method.

## 6.1 Comparison with Baselines

Overall results. The overall performance of FLARE and baseline across all tasks/datasets are reported in Figure 4. FLARE outperforms all baseline on all tasks/datasets, indicating that FLARE is a generic method that can effectively retrieve additional information throughout the generation.

Among various tasks, multihop QA shows the most significant improvement. This is largely due to the task's clear definition and specific objective of producing the final answer through a 2-hop reasoning process, which makes it easier for LMs to generate on-topic output. In contrast, ASQA and WikiAsp are more open-ended, which increases the difficulty of both generation and evaluation. The improvement on ASQA-hint is larger than that of ASQA because identifying ambiguous aspects is challenging even for humans in many cases, and providing a generic hint helps LMs to stay on topic.

Thorough comparisons with baselines. The performance of all baselines on 2WikiMultihopQA are reported in Table 1. FLARE outperforms all baselines by a large margin, which confirms that forward-looking active retrieval is highly effective. Most multi-time retrieval augmented approaches outperform single-time retrieval but with different

\begin{table}
\begin{tabular}[t]{|l|l|l|l|l|}
\hline Methods & EM & $\mathbf{F}_{1}$ & Prec. & Rec. \\
\hline No retrieval & 28.2 & 36.8 & 36.5 & 38.6 \\
\hline Single-time retrieval & 39.4 & 48.8 & 48.6 & 51.5 \\
\hline \multicolumn{5}{|c|}{Multi-time retrieval} \\
\hline Previous-window & 43.2 & 52.3 & 51.7 & 54.5 \\
\hline Previous-sentence & 39.0 & 49.2 & 48.9 & 51.8 \\
\hline Question decomposition & 47.8 & 56.4 & 56.1 & 58.6 \\
\hline FLARE ${ }_{\text {instruct }}$ (ours) & 42.4 & 49.8 & 49.1 & 52.5 \\
\hline $\operatorname{FLARE}_{\text {direct }}$ (ours) & 51.0 & 59.7 & 59.1 & 62.6 \\
\hline
\end{tabular}

**Table 1: FLARE and baselines on 2WikiMultihopQA. Previous-window (Borgeaud et al., 2022; Ram et al., 2023), previous-sentence (Trivedi et al., 2022), and question decomposition (Press et al., 2022; Yao et al., 2022) methods are reimplemented for fair comparisons.**
\end{table}
margins. The improvement of retrieving using the previous sentence is relatively small which we hypothesize is mainly because the previous sentence often describes entities or relations different from those in the next sentence in 2WikiMultihopQA. While the previous-window approach might use the first half of a sentence to retrieve information potentially helpful for generating the second half. Among all baselines, the question decomposition approach (Press et al., 2022) achieves the best performance. which is not surprising since the incontext exemplars manually annotated with decomposed sub-questions (Prompt D.2) guide LMs to generate sub-questions that align with the topic/intent of future generations. FLARE outperforms this baseline, indicating that manual exemplar annotation is not necessary for effective future-aware retrieval. The gap between FLARE $_{\text {instruct }}$ and question decomposition is large, indicating that teaching LMs to generate search queries using task-generic retrieval instructions and exemplars is challenging.

We report all metrics for the other datasets in Table 2. FLARE outperforms baselines with respect to all metrics. Retrieval using the previ-

\begin{table}
\begin{tabular}[t]{|l|l|l|l|l|l|l|l|l|l|l|l|l|}
\hline \multirow{2}{*}{Datasets Metrics} & \multirow{2}{*}{StrategyQA EM} & \multicolumn{4}{|c|}{ASQA} & \multicolumn{4}{|c|}{ASQA-hint} & \multicolumn{3}{|c|}{WikiAsp} \\
\hline & & EM & $\mathbf{D}-\mathbf{F}_{1}$ & R-L & DR & EM & $\mathbf{D}-\mathbf{F}_{1}$ & R-L & DR & UniEval & $\mathbf{E}-\mathbf{F}_{1}$ & R-L \\
\hline No retrieval & 72.9 & 33.8 & 24.2 & 33.3 & 28.4 & 40.1 & 32.5 & 36.4 & 34.4 & 47.1 & 14.1 & 26.4 \\
\hline Single-time retrieval & 68.6 & 40.0 & 27.1 & 34.0 & 30.4 & 43.2 & 34.8 & 37.4 & 36.0 & 52.4 & 17.4 & 26.9 \\
\hline & & \multicolumn{11}{|c|}{Multi-time retrieval} \\
\hline Previous-window & 71.2 & 39.9 & 27.0 & 34.3 & 30.4 & 43.7 & 35.7 & 37.5 & 36.6 & 51.8 & 18.1 & 27.3 \\
\hline Previous-sentence & 71.0 & 39.9 & 27.9 & 34.3 & 30.9 & 44.7 & 35.9 & 37.5 & 36.7 & 52.6 & 17.8 & 27.2 \\
\hline FLARE (ours) & 77.3 & 41.3 & 28.2 & 34.3 & 31.1 & 46.2 & 36.7 & 37.7 & 37.2 & 53.4 & 18.9 & 27.6 \\
\hline
\end{tabular}

**Table 2: Comparison between FLARE and baselines on StrategyQA, ASQA, ASQA-hint, and WikiAsp. D- $\mathrm{F**_{1}$ is Disambig- $\mathrm{F}_{1}$, R-L is ROUGE-L, and E- $\mathrm{F}_{1}$ is named entity-based $\mathrm{F}_{1}$.}
\end{table}

\begin{table}
\begin{tabular}[t]{|l|l|l|l|l|l|l|l|l|}
\hline \multirow{2}{*}{} & \multicolumn{4}{|r|}{2WikiMultihopQA} & \multicolumn{4}{|c|}{ASQA-hint} \\
\hline & EM & $\mathbf{F}_{1}$ & Prec. & Rec. & EM & $\mathbf{D}-\mathbf{F}_{1}$ & R-L & DR \\
\hline Previous & 39.0 & 49.2 & 48.9 & 51.8 & 42.5 & 34.1 & 36.9 & 35.5 \\
\hline Next & 48.8 & 57.6 & 57.1 & 60.5 & 45.9 & 35.7 & 37.5 & 36.6 \\
\hline
\end{tabular}

**Table 3: A head-to-head comparison between using the previous sentence and the next sentence for retrieval.**
\end{table}

\begin{table}
\begin{tabular}[t]{|l|l|l|l|l|}
\hline \#Tokens & EM & $\mathbf{F}_{1}$ & Prec. & Rec. \\
\hline 16 & 43.2 & 52.3 & 51.7 & 54.5 \\
\hline 32 & 43.6 & 52.4 & 52.0 & 55.0 \\
\hline 48 & 40.0 & 49.3 & 49.0 & 52.0 \\
\hline All & 39.0 & 48.5 & 48.2 & 51.1 \\
\hline
\end{tabular}

**Table 4: Previous-window approaches using different numbers of tokens as queries.**
\end{table}
ous window underperforms single-time retrieval on ASQA, which we hypothesize is because the previous window does not accurately reflect future intent. Since we focus on evaluating factuality, metrics with an emphasis on factual content (such as EM, Disambig- $\mathrm{F}_{1}$, UniEval) are more reliable than metrics computed over all tokens (ROUGE-L).

## 6.2 Ablation Study

Importance of forward-looking retrieval. We first validate that forward-looking retrieval is more effective than past-context-based retrieval. We run ablation experiments on 2WikiMultihopQA and ASQA-hint comparing retrieval using the previous versus the next sentence. Specifically, both methods retrieve every sentence and directly use the complete previous/next sentence as queries. As shown in Table 3, using the next sentence to retrieve is clearly better than using the previous sentence, confirming our hypothesis.

We also run previous-window approaches using different numbers of past tokens as queries. As shown in Table 4, using too many tokens (> 32) in

![Figure](./jiangActiveRetrievalAugmented2023_images/db4f0e65-0f0e-44fb-a560-94088cacb1ee-08_318_765_870_1060.jpg)

**Figure 5: Performance (EM) of FLARE with respect to the percentage of steps/sentences with retrieval on 2WikiMultihopQA and StrategyQA.**

the past hurts the performance, further confirming our hypothesis that previous context might not be relevant to intent of future generations.

Importance of active retrieval. Next, we investigate how active retrieval threshold $\theta$ affects performance. To alter our method from not retrieving to retrieving every sentence, we adjust the confidence threshold $\theta$ that determines when to trigger retrieval from 0 to 1. We then calculate the proportion of steps/sentences where retrieval is activated, and present the performance based on it. As shown in Figure 5, on 2WikiMultihopQA, the performance plateaus when the retrieval percentage exceeds 60\%, indicating that retrieval when LMs are confident is not necessary. On StrategyQA, the performance drops when the retrieval percentage exceeds 50\%, indicating that unnecessary retrieval can introduce noise and impede the original generation process. We found triggering retrieval for 40\%-80\% of sentences usually leads to a good performance across tasks/datasets.

Effectiveness of different query formulation methods We study implicit query formation by masking and explicit query formulation through question generation. In Table 5, we compare the performance of FLARE with different masking

\begin{table}
\begin{tabular}[t]{|l|l|l|l|l|}
\hline $\beta$ & EM & $\mathbf{F}_{1}$ & Prec. & Rec. \\
\hline 0.0 & 0.488 & 0.576 & 0.571 & 0.605 \\
\hline 0.2 & 0.498 & 0.588 & 0.582 & 0.616 \\
\hline 0.4 & 0.510 & 0.597 & 0.591 & 0.627 \\
\hline 0.6 & 0.506 & 0.593 & 0.586 & 0.622 \\
\hline
\end{tabular}

**Table 5: Performance of FLARE with respect to the masking threshold $\beta$ on 2WikiMultihopQA.**
\end{table}

\begin{table}
\begin{tabular}[t]{|l|l|l|l|l|l|l|l|}
\hline \multirow{2}{*}{} & \multicolumn{4}{|c|}{ASQA-hint} & \multicolumn{3}{|c|}{WikiAsp} \\
\hline & EM & $\mathbf{D}-\mathbf{F}_{1}$ & R-L & DR & UniEval & $\mathbf{E}-\mathbf{F}_{1}$ & R-L \\
\hline Implicit & 45.7 & 36.9 & 37.7 & 37.3 & 53.4 & 18.8 & 27.7 \\
\hline Explicit & 46.2 & 36.7 & 37.7 & 37.2 & 53.4 & 18.9 & 27.6 \\
\hline
\end{tabular}

**Table 6: A comparison between implicit and explicit query formulation methods in FLARE.**
\end{table}
thresholds $\beta$. Retrieving directly with the complete sentence \((\beta=0)\) is worse than masking tokens with low probabilities, confirming our hypothesis that low-confidence erroneous tokens can distract retrievers. We compare implicit and explicit query formulation methods in Table 6. Performances of both methods are similar, indicating that both methods can effectively reflect information needs.

