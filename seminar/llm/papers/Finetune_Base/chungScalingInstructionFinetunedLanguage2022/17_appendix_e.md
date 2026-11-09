# E Finetuning details

We found that learning rate, batch size and the dropout were the most important hyperparameters for instruction finetuning. Table 22 lists these values for all finetuned models studied in this paper. The reported batch size is the global batch size (not per-device batch size). Because we use packing (Raffel et al., 2020), the effective batch size is larger.

\begin{table}
\begin{tabular}[t]{|l|l|l|l|l|l|}
\hline Params & Model & Batch size & Dropout & LR & Steps \\
\hline 80M & Flan-T5-Small & 64 & 0.05 & 5e-4 & 98k \\
\hline 250M & Flan-T5-Base & 64 & 0.05 & 5e-4 & 84k \\
\hline 780M & Flan-T5-Large & 64 & 0.05 & 5e-4 & 64k \\
\hline 3B & Flan-T5-XL & 64 & 0.05 & 5e-4 & 38k \\
\hline 11B & Flan-T5-XXL & 64 & 0.05 & $5 \mathrm{e}-4$ & 14k \\
\hline 8B & Flan-PaLM & 32 & 0.05 & $3 \mathrm{e}-3$ & 40k \\
\hline 62B & Flan-PaLM & 32 & 0.05 & $3 \mathrm{e}-3$ & 40k \\
\hline 540B & Flan-PaLM & 32 & 0.1 & $1 \mathrm{e}-3$ & 21k \\
\hline 62B & Flan-cont-PaLM & 32 & 0.05 & $3 \mathrm{e}-3$ & 60k \\
\hline 540B & Flan-U-PaLM & 32 & 0.1 & $1 \mathrm{e}-3$ & 30k \\
\hline
\end{tabular}

**Table 22: Hyperparameters used for all finetuned models.**
\end{table}

Another important hyperparameter for instruction finetuning is the sampling rates for each tasks. Within the four mixtures (Muffin, T0-SF, NIV2, and CoT defined in Figure 2), we use the number of examples as the weight of each task. We apply the maximum cap for each task because there are tasks that are much larger than others in the same mixture, which can dominate the sampling. For example, some WMT translation datasets have millions of examples compared to BoolQ, which has 9k examples. We apply different maximum cap for each of the four mixtures summarized in Table 23.

\begin{table}
\begin{tabular}[t]{|l|l|l|l|}
\hline Mixture & Maximum cap & Proportion (A) & Proportion (B) \\
\hline Muffin & 30,000 & 52\% & 46.0\% \\
\hline T0-SF & 20,000 & 15\% & 27.9\% \\
\hline CoT & 100,000 & 3\% & 1.8\% \\
\hline NIV2 & 5,000 & 30\% & 24.2\% \\
\hline
\end{tabular}

**Table 23: Maximum example cap and proportion rates applied to the underlying mixtures. Proportion A was used in scaling and the ablation sections in Section 3 and Section 4. The rest of the experiments used Proportion B, since Proportion A signaled that the T0 mixture was good for performance.**
\end{table}

For balancing the four mixtures, we use the relative weights ensuring that none of the underlying tasks is repeated more than once. For the scaling and ablation experiments in Section 3 and Section 4, respectively, we used the mixture proportions in Table 23, Proportion A. Based on these experiments (specifically, strong gains from T0-SF), we updated the mixture proportions to the Proportion B values in Table 23 and finetuned the rest of the models.

