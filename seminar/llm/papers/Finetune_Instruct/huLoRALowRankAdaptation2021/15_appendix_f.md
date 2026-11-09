# F Additional Empirical Experiments

# F. 1 Additional Experiments on GPT-2

We also repeat our experiment on DART (Nan et al., 2020) and WebNLG (Gardent et al., 2017) following the setup of Li \& Liang (2021). The result is shown in Table 13 Similar to our result on E2E NLG Challenge, reported in Section 5, LoRA performs better than or at least on-par with prefix-based approaches given the same number of trainable parameters.

\begin{table}
\begin{tabular}[t]{|l|l|l|l|l|}
\hline \multirow[t]{2}{*}{Method} & \# Trainable & \multicolumn{3}{|c|}{DART} \\
\hline & Parameters & BLEU↑ & MET↑ & TER↓ \\
\hline \multicolumn{5}{|c|}{GPT-2 Medium} \\
\hline Fine-Tune & 354M & 46.2 & 0.39 & 0.46 \\
\hline Adapter ${ }^{\text {L }}$ & 0.37M & 42.4 & 0.36 & 0.48 \\
\hline Adapter ${ }^{\text {L }}$ & 11M & 45.2 & 0.38 & 0.46 \\
\hline $\mathrm{FT}^{\mathrm{Top} 2}$ & 24M & 41.0 & 0.34 & 0.56 \\
\hline PrefLayer & 0.35M & 46.4 & 0.38 & 0.46 \\
\hline LoRA & 0.35M & $\mathbf{4 7 . 1}_{\text {±. } 2}$ & 0.39 & 0.46 \\
\hline \multicolumn{5}{|c|}{GPT-2 Large} \\
\hline Fine-Tune & 774M & 47.0 & 0.39 & 0.46 \\
\hline Adapter ${ }^{\text {L }}$ & 0.88M & 45.7 ${ }_{\text {±. } 1}$ & 0.38 & 0.46 \\
\hline Adapter ${ }^{\text {L }}$ & 23M & $47.1_{ \pm .1}$ & 0.39 & 0.45 \\
\hline PrefLayer & 0.77M & 46.7 & 0.38 & 0.45 \\
\hline LoRA & 0.77M & $\mathbf{4 7 . 5}_{\text {±. } 1}$ & 0.39 & 0.45 \\
\hline
\end{tabular}

**Table 13: GPT-2 with different adaptation methods on DART. The variances of MET and TER are less than 0.01 for all adaption approaches.**
\end{table}

\begin{table}
\begin{tabular}[t]{|l|l|l|l|l|l|l|l|l|l|}
\hline \multirow[t]{3}{*}{Method} & \multicolumn{9}{|c|}{WebNLG} \\
\hline & \multirow[b]{2}{*}{U} & \multirow[t]{2}{*}{BLEU↑} & & \multicolumn{3}{|c|}{MET↑} & \multicolumn{3}{|c|}{TER↓} \\
\hline & & & A & U & S & A & U & S & A \\
\hline \multicolumn{10}{|c|}{GPT-2 Medium} \\
\hline Fine-Tune (354M) & 27.7 & 64.2 & 46.5 & . 30 & . 45 & . 38 & . 76 & . 33 & . 53 \\
\hline Adapter ${ }^{\text {L }}$ (0.37M) & 45.1 & 54.5 & 50.2 & . 36 & . 39 & . 38 & . 46 & . 40 & . 43 \\
\hline Adapter ${ }^{\text {L }}$ (11M) & 48.3 & 60.4 & 54.9 & . 38 & . 43 & . 41 & . 45 & . 35 & . 39 \\
\hline FT ${ }^{\text {Top } 2}$ (24M) & 18.9 & 53.6 & 36.0 & . 23 & . 38 & . 31 & . 99 & . 49 & . 72 \\
\hline Prefix (0.35M) & 45.6 & 62.9 & 55.1 & . 38 & . 44 & . 41 & . 49 & . 35 & . 40 \\
\hline LoRA (0.35M) & $46.7_{ \pm .4}$ & $62.1_{ \pm .2}$ & $\mathbf{5 5 . 3}_{ \pm .2}$ & . 38 & . 44 & . 41 & . 46 & . 33 & . 39 \\
\hline & \multicolumn{9}{|c|}{GPT-2 Large} \\
\hline Fine-Tune (774M) & 43.1 & 65.3 & 55.5 & . 38 & . 46 & . 42 & . 53 & . 33 & . 42 \\
\hline Adapter ${ }^{\text {L }}$ (0.88M) & $\mathbf{4 9 . 8}_{ \pm .0}$ & $61.1_{ \pm .0}$ & $56.0_{ \pm .0}$ & . 38 & . 43 & . 41 & . 44 & . 35 & . 39 \\
\hline Adapter ${ }^{\text {L }}$ (23M) & 49.2±.1 & $64.7_{ \pm .2}$ & $\mathbf{5 7 . 7}_{\text {±. } 1}$ & . 39 & . 46 & . 43 & . 46 & . 33 & . 39 \\
\hline Prefix (0.77M) & 47.7 & 63.4 & 56.3 & . 39 & . 45 & . 42 & . 48 & . 34 & . 40 \\
\hline LoRA (0.77M) & $48.4_{ \pm .3}$ & $64.0_{ \pm .3}$ & $57.0_{ \pm .1}$ & . 39 & . 45 & . 42 & . 45 & . 32 & . 38 \\
\hline
\end{tabular}

**Table 14: GPT-2 with different adaptation methods on WebNLG. The variances of MET and TER are less than 0.01 for all the experiments we ran. "U" indicates unseen categories, "S" indicates seen categories, and "A" indicates all categories in the test set of WebNLG.**
\end{table}

# F. 2 Additional Experiments on GPT-3

We present additional runs on GPT-3 with different adaptation methods in Table 15. The focus is on identifying the trade-off between performance and the number of trainable parameters.

# F. 3 Low-Data Regime

To evaluate the performance of different adaptation approaches in the low-data regime. we randomly sample 100, 1k and 10k training examples from the full training set of MNLI to form the low-data MNLI- $n$ tasks. In Table 16, we show the performance of different adaptation approaches on MNLI-$n$. To our surprise, PrefixEmbed and PrefixLayer performs very poorly on MNLI-100 dataset, with PrefixEmbed performing only slightly better than random chance (37.6\% vs. 33.3\%). PrefixLayer performs better than PrefixEmbed but is still significantly worse than Fine-Tune or LoRA on MNLI-100. The gap between prefix-based approaches and LoRA/Fine-tuning becomes smaller as we increase the number of training examples, which might suggest that prefix-based approaches are not suitable for low-data tasks in GPT-3. LoRA achieves better performance than fine-tuning on both MNLI-100 and MNLI-Full, and comparable results on MNLI-1k and MNLI-10K considering the (±0.3) variance due to random seeds.

The training hyperparameters of different adaptation approaches on MNLI-n are reported in Ta-ble 17. We use a smaller learning rate for PrefixLayer on the MNLI-100 set, as the training loss does not decrease with a larger learning rate.

