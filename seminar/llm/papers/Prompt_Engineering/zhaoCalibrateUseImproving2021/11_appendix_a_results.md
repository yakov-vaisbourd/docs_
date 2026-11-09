# A. Additional Results on Variance and Calibration

\begin{table}

**Table 2 shows an example of the sensitivity to ordering.**
\begin{tabular}[t]{|l|l|}
\hline Prompt (test input not shown) & Acc. \\
\hline Review: the whole thing 's fairly lame, making it par for the course for disney sequels . Answer: Negative Review: this quiet , introspective and entertaining independent is worth seeking . Answer: Positive & 88.5\% \\
\hline Review: this quiet , introspective and entertaining independent is worth seeking . Answer: Positive Review: the whole thing 's fairly lame, making it par for the course for disney sequels . Answer: Negative & 51.3\% \\
\hline
\end{tabular}
\end{table}

Table 2. Top: a prompt consisting of two training examples (the test input is not shown) that leads to good test accuracy for GPT-3 2.7B (88.5\%). Bottom: simply reversing the order of the two examples causes the accuracy to drop to near random chance (51.3\%).

\begin{table}

**Table 3 demonstrates that the choice of content-free input does affect accuracy, however, many good choices exist.**
\begin{tabular}[t]{|l|l|l|}
\hline Content-free Input & SST-2 & AGNews \\
\hline Uncalibrated Baseline & 66.5 & 48.5 \\
\hline N/A & 74.2 & 64.5 \\
\hline [MASK] & 74.5 & 63.8 \\
\hline " & 72.9 & 64.7 \\
\hline N/A, [MASK], '' & 79.0 & 66.5 \\
\hline the & 69.1 & 59.0 \\
\hline abc & 77.5 & 57.3 \\
\hline the man. & 79.4 & 62.0 \\
\hline dasjhasjkdhjskdhds & 79.3 & 64.5 \\
\hline nfjkhdvy84tr9bpuirvwe & 78.4 & 65.5 \\
\hline
\end{tabular}
\end{table}

Table 3. We show the accuracy for 1-shot SST-2 and 0-shot AG-News over different choices for the content-free input. The choice of content-free input matters, however, many good choices exist. The token " indicates the empty string. Recall that in our experiments, we ensemble over N/A, [MASK], and the empty string.

Figure 9 shows how GPT-3 accuracy changes as the prompt format is varied for LAMA, with and without calibration.

Table 4 shows the effect of calibration for GPT-2.

