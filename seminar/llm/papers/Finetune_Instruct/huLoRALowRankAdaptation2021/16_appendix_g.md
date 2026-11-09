# G Measuring Similarity Between Subspaces

In this paper we use the measure \(\phi(A, B, i, j)=\psi\left(U_{A}^{i}, U_{B}^{j}\right)=\frac{\left\|U_{A}^{i \top} U_{B}\right\|_{F}^{2}}{\min \{i, j\}}\) to measure the subspace similarity between two column orthonormal matrices $U_{A}^{i} \in \mathbb{R}^{d \times i}$ and $U_{B}^{j} \in \mathbb{R}^{d \times j}$, obtained by taking columns of the left singular matrices of $A$ and $B$. We point out that this similarity is simply a reverse of the standard Projection Metric that measures distance between subspaces Ham \& Lee (2008).

\begin{table}
\begin{tabular}[t]{|l|l|l|l|l|}
\hline Method & Hyperparameters & \# Trainable Parameters & WikiSQL & MNLI-m \\
\hline Fine-Tune & - & 175B & 73.8 & 89.5 \\
\hline \multirow{5}{*}{PrefixEmbed} & $l_{p}=32, l_{i}=8$ & 0.4 M & 55.9 & 84.9 \\
\hline & $l_{p}=64, l_{i}=8$ & 0.9 M & 58.7 & 88.1 \\
\hline & $l_{p}=128, l_{i}=8$ & 1.7 M & 60.6 & 88.0 \\
\hline & $l_{p}=256, l_{i}=8$ & 3.2 M & 63.1 & 88.6 \\
\hline & $l_{p}=512, l_{i}=8$ & 6.4 M & 55.9 & 85.8 \\
\hline \multirow{5}{*}{PrefixLayer} & $l_{p}=2, l_{i}=2$ & 5.1 M & 68.5 & 89.2 \\
\hline & $l_{p}=8, l_{i}=0$ & 10.1 M & 69.8 & 88.2 \\
\hline & $l_{p}=8, l_{i}=8$ & 20.2 M & 70.1 & 89.5 \\
\hline & $l_{p}=32, l_{i}=4$ & 44.1 M & 66.4 & 89.6 \\
\hline & $l_{p}=64, l_{i}=0$ & 76.1 M & 64.9 & 87.9 \\
\hline \multirow{5}{*}{Adapter ${ }^{\text {H }}$} & $r=1$ & 7.1 M & 71.9 & 89.8 \\
\hline & $r=4$ & 21.2 M & 73.2 & 91.0 \\
\hline & $r=8$ & 40.1 M & 73.2 & 91.5 \\
\hline & $r=16$ & 77.9 M & 73.2 & 91.5 \\
\hline & $r=64$ & 304.4 M & 72.6 & 91.5 \\
\hline \multirow{10}{*}{LoRA} & $r_{v}=2$ & 4.7 M & 73.4 & 91.7 \\
\hline & $r_{q}=r_{v}=1$ & 4.7 M & 73.4 & 91.3 \\
\hline & $r_{q}=r_{v}=2$ & 9.4 M & 73.3 & 91.4 \\
\hline & $r_{q}=r_{k}=r_{v}=r_{o}=1$ & 9.4 M & 74.1 & 91.2 \\
\hline & $r_{q}=r_{v}=4$ & 18.8 M & 73.7 & 91.3 \\
\hline & $r_{q}=r_{k}=r_{v}=r_{o}=2$ & 18.8 M & 73.7 & 91.7 \\
\hline & $r_{q}=r_{v}=8$ & 37.7 M & 73.8 & 91.6 \\
\hline & $r_{q}=r_{k}=r_{v}=r_{o}=4$ & 37.7 M & 74.0 & 91.7 \\
\hline & $r_{q}=r_{v}=64$ & 301.9 M & 73.6 & 91.4 \\
\hline & $r_{q}=r_{k}=r_{v}=r_{o}=64$ & 603.8 M & 73.9 & 91.4 \\
\hline \multirow{3}{*}{LoRA+PE} & $r_{q}=r_{v}=8, l_{p}=8, l_{i}=4$ & 37.8 M & 75.0 & 91.4 \\
\hline & $r_{q}=r_{v}=32, l_{p}=8, l_{i}=4$ & 151.1 M & 75.9 & 91.1 \\
\hline & $r_{q}=r_{v}=64, l_{p}=8, l_{i}=4$ & 302.1 M & 76.2 & 91.3 \\
\hline LoRA+PL & $r_{q}=r_{v}=8, l_{p}=8, l_{i}=4$ & 52.8 M & 72.9 & 90.2 \\
\hline
\end{tabular}

**Table 15: Hyperparameter analysis of different adaptation approaches on WikiSQL and MNLI. Both prefix-embedding tuning (PrefixEmbed) and prefix-layer tuning (PrefixLayer) perform worse as we increase the number of trainable parameters, while LoRA's performance stabilizes. Performance is measured in validation accuracy.**
\end{table}

\begin{table}
\begin{tabular}[t]{|l|l|l|l|l|}
\hline Method & MNLI(m)-100 & MNLI(m)- 1 k & MNLI(m)- 10 k & MNLI(m)- 392 K \\
\hline GPT-3 (Fine-Tune) & 60.2 & 85.8 & 88.9 & 89.5 \\
\hline GPT-3 (PrefixEmbed) & 37.6 & 75.2 & 79.5 & 88.6 \\
\hline GPT-3 (PrefixLayer) & 48.3 & 82.5 & 85.9 & 89.6 \\
\hline GPT-3 (LoRA) & 63.8 & 85.6 & 89.2 & 91.7 \\
\hline
\end{tabular}

**Table 16: Validation accuracy of different methods on subsets of MNLI using GPT-3 175B. MNLI$n$ describes a subset with $n$ training examples. We evaluate with the full validation set. LoRA performs exhibits favorable sample-efficiency compared to other methods, including fine-tuning.**
\end{table}
To be concrete, let the singular values of $U_{A}^{i \top} U_{B}^{j}$ to be $\sigma_{1}, \sigma_{2}, \cdots, \sigma_{p}$ where $p=\min \{i, j\}$. We know that the Projection Metric Ham \& Lee (2008) is defined as:
$$
d\left(U_{A}^{i}, U_{B}^{j}\right)=\sqrt{p-\sum_{i=1}^{p} \sigma_{i}^{2}} \in[0, \sqrt{p}]
$$

\begin{table}
\begin{tabular}[t]{|l|l|l|l|l|l|}
\hline Hyperparameters & Adaptation & MNLI-100 & MNLI-1k & MNLI-10K & MNLI-392K \\
\hline Optimizer & - & \multicolumn{4}{|c|}{AdamW} \\
\hline Warmup Tokens & - & \multicolumn{4}{|c|}{250,000} \\
\hline LR Schedule & - & \multicolumn{4}{|c|}{Linear} \\
\hline Batch Size & - & 20 & 20 & 100 & 128 \\
\hline \# Epoch & - & 40 & 40 & 4 & 2 \\
\hline \multirow{4}{*}{Learning Rate} & FineTune & \multicolumn{4}{|c|}{$5.00 \mathrm{E}-6$} \\
\hline & PrefixEmbed & $2.00 \mathrm{E}-04$ & $2.00 \mathrm{E}-04$ & $4.00 \mathrm{E}-04$ & $5.00 \mathrm{E}-04$ \\
\hline & PrefixLayer & $5.00 \mathrm{E}-05$ & 5.00E-05 & $5.00 \mathrm{E}-05$ & $1.00 \mathrm{E}-04$ \\
\hline & LoRA & \multicolumn{4}{|c|}{$2.00 \mathrm{E}-4$} \\
\hline \multirow{4}{*}{AdaptationSpecific} & PrefixEmbed $l_{p}$ & \multirow[t]{4}{*}{16} & \multicolumn{2}{|c|}{\begin{tabular}[t]{l}
32 \\
64
\end{tabular}} & 256 \\
\hline & PrefixEmbed $l_{i}$ & & \multicolumn{3}{|c|}{8} \\
\hline & PrefixTune & \multicolumn{4}{|c|}{$l_{p}=l_{i}=8$} \\
\hline & LoRA & \multicolumn{4}{|c|}{$r_{q}=r_{v}=8$} \\
\hline
\end{tabular}

**Table 17: The hyperparameters used for different GPT-3 adaptation methods on MNLI(m)- $n$.**
\end{table}
where our similarity is defined as:
$$
\phi(A, B, i, j)=\psi\left(U_{A}^{i}, U_{B}^{j}\right)=\frac{\sum_{i=1}^{p} \sigma_{i}^{2}}{p}=\frac{1}{p}\left(1-d\left(U_{A}^{i}, U_{B}^{j}\right)^{2}\right)
$$
This similarity satisfies that if $U_{A}^{i}$ and $U_{B}^{j}$ share the same column span, then \(\phi(A, B, i, j)=1\). If they are completely orthogonal, then \(\phi(A, B, i, j)=0\). Otherwise, \(\phi(A, B, i, j) \in(0,1)\).

