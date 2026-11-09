# A Large Language Models Still Need Parameter Updates

Few-shot learning, or prompt engineering, is very advantageous when we only have a handful of training samples. However, in practice, we can often afford to curate a few thousand or more training examples for performance-sensitive applications. As shown in Table 8, fine-tuning improves the model performance drastically compared to few-shot learning on datasets large and small. We take the GPT-3 few-shot result on RTE from the GPT-3 paper (Brown et al., 2020). For MNLI-matched, we use two demonstrations per class and six in-context examples in total.

\begin{table}
\begin{tabular}[t]{|l|l|l|}
\hline Method & MNLI-m (Val. Acc./\%) & RTE (Val. Acc./\%) \\
\hline GPT-3 Few-Shot & 40.6 & 69.0 \\
\hline GPT-3 Fine-Tuned & 89.5 & 85.4 \\
\hline
\end{tabular}

**Table 8: Fine-tuning significantly outperforms few-shot learning on GPT-3 (Brown et al., 2020).**
\end{table}

