# 3 Scaling to 540B parameters and 1.8K tasks

We first examine the effect of scaling in terms of (1) the size of model and (2) the number of finetuning tasks on performance on held-out tasks. We scale the model size by performing experiments on three PaLM model sizes: 8B, 62B, and 540B. To scale the number of tasks, we sequentially add task mixtures starting from the mixture with the fewest tasks to the mixture with the most tasks: CoT, Muffin, T0-SF, and NIV2.

Figure 4 shows the joint effect of scaling these two variables on the normalized average of held-out benchmarks. Individual benchmark results are reported in Table 3. First, we see that for all three model sizes shown, multi-task instruction finetuning improves performance by a large margin compared to no finetuning. The performance gain ranges from 9.4\% to 15.5\%.

Second, increasing the number of finetuning tasks improves performance, although the majority of the improvement comes from using up to 282 tasks. There are two potential explanations for the small gain after 282 tasks. One is that the additional tasks are not particularly diverse, and so they are not providing the model with new knowledge. Another explanation is that most of the gains from multi-task instruction finetuning come from the model learning to better express knowledge that it already knows from pretraining, and more than 282 tasks does not help too much. This second explanation could make sense since the pre-training data consists of 780B tokens, while instruction finetuning only uses 1.4B tokens (0.2\% of the pre-training tokens).

Finally, we see that increasing model scale by an order of magnitude (i.e., 8B → 62B or 62B → 540B) improves performance substantially for both finetuned and non-finetuned models. Note that it could be complicated to determine whether instruction finetuning improves small models or large models more (compared to the baseline of no finetuning). For example, although the absolute gain was larger for the 8B model than the 540B model (15.5\% for 8B vs. 9.4\% for 540B), the relative reduction in error rate was larger for the 540B model ( $18.4 \%$ for 540B vs. $16.6 \%$ for 8B).

Plotting such scaling curves provides insights into how scaling the model size and the number of tasks even further might improve performance. Scaling model size by another order of magnitude (though challenging) is expected to provide substantial performance gain. Scaling number of finetuning tasks should also improve performance, although likely only incrementally. Overall, the scaling curves plotted indicate that future work should continue scaling instruction finetuning.

![Figure](./chungScalingInstructionFinetunedLanguage2022_images/5b970e92-029a-4684-b206-dafd7ac86e22-06_620_1468_1563_333.jpg)

**Figure 4: Scaling behavior of multi-task instruction finetuning with respect to model size (\# parameters) and number of finetuning tasks. The $x$-axes are log scale. The benchmark suites are MMLU (57 tasks), BBH (23 tasks), TyDiQA (8 languages), and MGSM (10 languages). The evaluation metric on all four benchmark suites is few-shot prompted accuracy (exact match), where we take an unweighted average over all tasks. As an aggregate metric we report the normalized average of MMLU-direct, MMLU-CoT, BBH-direct, BBH-CoT, TyDiQA, and MGSM. These evaluation benchmarks are held-out (not included in the finetuning data).**

\begin{table}
\begin{tabular}[t]{|l|l|l|l|l|l|l|l|l|l|}
\hline \multirow[b]{2}{*}{Model} & \multirow[b]{2}{*}{Finetuning Mixtures} & \multirow[b]{2}{*}{Tasks} & \multirow[b]{2}{*}{Norm. avg.} & \multicolumn{2}{|c|}{MMLU} & \multicolumn{2}{|c|}{BBH} & TyDiQA & MGSM \\
\hline & & & & Direct & CoT & Direct & CoT & Direct & CoT \\
\hline \multirow[t]{5}{*}{8B} & None (no finetuning) & 0 & 6.4 & 24.3 & 24.1 & 30.8 & 30.1 & 25.0 & 3.4 \\
\hline & CoT & 9 & 8.3 (+1.9) & 26.3 & 32.1 & 19.8 & 26.6 & 39.3 & 10.4 \\
\hline & CoT, Muffin & 89 & 14.8 (+8.4) & 37.6 & 38.4 & 31.0 & 30.9 & 32.4 & 8.4 \\
\hline & CoT, Muffin, T0-SF & 282 & 20.5 (+14.1) & 47.7 & 39.7 & 33.1 & 30.9 & $\underline{49.0}$ & 8.5 \\
\hline & CoT, Muffin, T0-SF, NIV2 & 1,836 & \(\underline{\mathbf{2 1 . 9}}(+\mathbf{1 5 . 5})\) & $\underline{49.3}$ & $\underline{41.3}$ & $\underline{36.4}$ & $\underline{31.1}$ & 47.5 & 8.2 \\
\hline \multirow[t]{5}{*}{62B} & None (no finetuning) & 0 & 28.4 & 55.1 & 49.0 & 37.4 & 43.0 & 40.5 & 18.2 \\
\hline & CoT & 9 & 29.0 (+0.4) & 48.5 & 48.7 & 34.5 & 39.5 & 48.8 & $\underline{32.6}$ \\
\hline & CoT, Muffin & 89 & 33.4 (+6.0) & 55.3 & 51.4 & 42.8 & 40.2 & 53.0 & 23.9 \\
\hline & CoT, Muffin, T0-SF & 282 & 37.9 (+9.5) & $\underline{60.0}$ & 56.0 & 44.7 & 43.8 & 58.2 & 30.0 \\
\hline & CoT, Muffin, T0-SF, NIV2 & 1,836 & \(\underline{38.8 \text { (+10.4) }}\) & 59.6 & $\underline{56.9}$ & $\underline{47.5}$ & $\underline{44.9}$ & $\underline{58.7}$ & 28.5 \\
\hline \multirow[t]{5}{*}{540B} & None (no finetuning) & 0 & 49.1 & 71.3 & 62.9 & 49.1 & 63.7 & 52.9 & 45.9 \\
\hline & CoT & 9 & 52.6 (+3.5) & 68.8 & 64.8 & 50.5 & 61.1 & 61.2 & 59.4 \\
\hline & CoT, Muffin & 89 & 57.0 (+7.9) & 71.8 & 66.7 & 56.7 & 64.0 & 65.3 & $\underline{63.0}$ \\
\hline & CoT, Muffin, T0-SF & 282 & 57.5 (+8.4) & 72.9 & $\underline{68.2}$ & 57.3 & 64.0 & 65.8 & 61.6 \\
\hline & CoT, Muffin, T0-SF, NIV2 & 1,836 & $\underline{58.5}$ (+9.4) & 73.2 & 68.1 & $\underline{58.8}$ & $\underline{65.6}$ & $\underline{67.4}$ & 61.3 \\
\hline
\end{tabular}

**Table 3: Increasing the number of tasks in the finetuning data improves performance of Flan-PaLM on most evaluation benchmarks. The benchmark suites are MMLU (57 tasks), BBH (23 tasks), TyDiQA (8 languages), and MGSM (10 languages). The evaluation metric on all four benchmark suites is few-shot prompted accuracy (exact match), where we take an unweighted average over all tasks. As an aggregate metric we report the normalized average of MMLU-direct, MMLU-CoT, BBH-direct, BBH-CoT, TyDiQA, and MGSM. These evaluation benchmarks are held-out (not included in the finetuning data).**
\end{table}

