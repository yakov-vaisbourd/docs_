# E Combining LoRA with Prefix Tuning

LoRA can be naturally combined with existing prefix-based approaches. In this section, we evaluate two combinations of LoRA and variants of prefix-tuning on WikiSQL and MNLI.

LoRA+PrefixEmbed (LoRA+PE) combines LoRA with prefix-embedding tuning, where we insert $l_{p}+l_{i}$ special tokens whose embeddings are treated as trainable parameters. For more on prefixembedding tuning, see Section 5.1

LoRA+PrefixLayer (LoRA+PL) combines LoRA with prefix-layer tuning. We also insert $l_{p}+l_{i}$ special tokens; however, instead of letting the hidden representations of these tokens evolve natu-

\begin{table}
\begin{tabular}[t]{|l|l|l|l|l|l|l|}
\hline Hyperparameters & Fine-Tune & PreEmbed & PreLayer & BitFit & Adapter ${ }^{\text {H }}$ & LoRA \\
\hline Optimizer & \multicolumn{6}{|c|}{AdamW} \\
\hline Batch Size & \multicolumn{6}{|c|}{128} \\
\hline \# Epoch & \multicolumn{6}{|c|}{2} \\
\hline Warmup Tokens & \multicolumn{6}{|c|}{250,000} \\
\hline LR Schedule & \multicolumn{6}{|c|}{Linear} \\
\hline Learning Rate & $5.00 \mathrm{E}-06$ & $5.00 \mathrm{E}-04$ & $1.00 \mathrm{E}-04$ & $1.6 \mathrm{E}-03$ & $1.00 \mathrm{E}-04$ & $2.00 \mathrm{E}-04$ \\
\hline
\end{tabular}

**Table 12: The training hyperparameters used for different GPT-3 adaption methods. We use the same hyperparameters for all datasets after tuning learning rate.**
\end{table}
rally, we replace them after every Transformer block with an input agnostic vector. Thus, both the embeddings and subsequent Transformer block activations are treated as trainable parameters. For more on prefix-layer tuning, see Section 5.1

In Table 15, we show the evaluation results of LoRA+PE and LoRA+PL on WikiSQL and MultiNLI. First of all, LoRA+PE significantly outperforms both LoRA and prefix-embedding tuning on WikiSQL, which indicates that LoRA is somewhat orthogonal to prefix-embedding tuning. On MultiNLI, the combination of LoRA+PE doesn't perform better than LoRA, possibly because LoRA on its own already achieves performance comparable to the human baseline. Secondly, we notice that LoRA+PL performs slightly worse than LoRA even with more trainable parameters. We attribute this to the fact that prefix-layer tuning is very sensitive to the choice of learning rate and thus makes the optimization of LoRA weights more difficult in LoRA+PL.

