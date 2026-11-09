# 7 Intrinsic Evaluation

We compare different variants of prefix-tuning. §7.1 studies the impact of the prefix length. §7.2 studies tuning only the embedding layer, which is more akin to tuning a discrete prompt. §7.3 compares prefixing and infixing, which inserts trainable activations between $x$ and $y$. §7.4 studies the impact of various prefix initialization strategies.

## 7.1 Prefix Length

A longer prefix means more trainable parameters, and therefore more expressive power. Figure 4 shows that performance increases as the prefix length increases up to a threshold (200 for summarization, 10 for table-to-text) and then a slight performance drop occurs. ${ }^{14}$

Empirically, longer prefixes have a negligible impact on inference speed, because attention computation over the entire prefix is parallellized on GPUs.

## 7.2 Full vs Embedding-only

Recall in §4.1, we discuss the option of optimizing the continuous embeddings of the "virtual tokens." We instantiate that idea and call it embedding-only ablation. The word embeddings are free parameters, and the upper activation layers are computed by the Transformer. Table 4 (top) shows that the performance drops significantly, suggesting that tuning only the embedding layer is not sufficiently expressive.

The embedding-only ablation upper bounds the performance of discrete prompt optimization (Shin

\footnotetext{
${ }^{14}$ Prefixes longer than the threshold lead to lower training loss, but slightly worse test performance, suggesting that they tend to overfit the training data.
}

\begin{table}
\begin{tabular}[t]{|l|l|l|l|l|l|}
\hline \multirow{2}{*}{} & \multirow[b]{2}{*}{BLEU} & \multicolumn{4}{|c|}{E2E} \\
\hline & & NIST & MET & ROUGE & CIDEr \\
\hline Prefix & 69.7 & 8.81 & 46.1 & 71.4 & 2.49 \\
\hline \multicolumn{6}{|c|}{Embedding-only: Emb-\{PrefixLength\}} \\
\hline Emb-1 & 48.1 & 3.33 & 32.1 & 60.2 & 1.10 \\
\hline Emb-10 & 62.2 & 6.70 & 38.6 & 66.4 & 1.75 \\
\hline Emb-20 & 61.9 & 7.11 & 39.3 & 65.6 & 1.85 \\
\hline \multicolumn{6}{|c|}{Infix-tuning: Infix-\{PrefixLength\}} \\
\hline Infix-1 & 67.9 & 8.63 & 45.8 & 69.4 & 2.42 \\
\hline Infix-10 & 67.2 & 8.48 & 45.8 & 69.9 & 2.40 \\
\hline Infix-20 & 66.7 & 8.47 & 45.8 & 70.0 & 2.42 \\
\hline
\end{tabular}

**Table 4: Intrinsic evaluation of Embedding-only (§7.2) and Infixing (§7.3). Both Embedding-only ablation and Infix-tuning underperforms full prefix-tuning.**
\end{table}

![Figure](./liPrefixTuningOptimizingContinuous2021_images/83a5514b-cc14-4b9b-9dd4-6bbee02942d4-08_420_784_895_1069.jpg)

**Figure 5: Initializing the prefix with activations of real words significantly outperforms random initialization, in low-data settings.**

et al., 2020), because discrete prompt restricts the embedding layer to exactly match the embedding of a real word. Consequently, we have this chain of increasing expressive power: discrete prompting < embedding-only ablation < prefix-tuning.

## 7.3 Prefixing vs Infixing

We also investigate how the trainable activations' position in the sequence affects performance. In prefix-tuning, we place them at the beginning $[$ Prefix $; x ; y]$. We can also place the trainable activations between $x$ and $y$ (i.e. $[x ; \operatorname{Infix} ; y]$ ) and call this infix-tuning. Table 4 (bottom) shows that infix-tuning slightly underperforms prefix-tuning. We believe this is because prefix-tuning can affect the activations of $x$ and $y$ whereas infix-tuning can only influence the activations of $y$.

## 7.4 Initialization

We find that how the prefix is initialized has a large impact in low-data settings. Random initialization leads to low performance with high variance. Initializing the prefix with activations of real words
significantly improves generation, as shown in Figure 5. In particular, initializing with task relevant words such as "summarization" and "table-to-text" obtains slightly better performance than task irrelevant words such as "elephant" and "divide", but using real words is still better than random.

Since we initialize the prefix with activations of real words computed by the LM, this initialization strategy is concordant with preserving the pretrained LM as much as possible.

