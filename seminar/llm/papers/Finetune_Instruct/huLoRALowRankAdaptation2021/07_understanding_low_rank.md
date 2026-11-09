# 7 Understanding the Low-Rank Updates

Given the empirical advantage of LoRA, we hope to further explain the properties of the low-rank adaptation learned from downstream tasks. Note that the low-rank structure not only lowers the hardware barrier to entry which allows us to run multiple experiments in parallel, but also gives better interpretability of how the update weights are correlated with the pre-trained weights. We focus our study on GPT-3 175B, where we achieved the largest reduction of trainable parameters (up to 10,000×) without adversely affecting task performances.

We perform a sequence of empirical studies to answer the following questions: 1) Given a parameter budget constraint, which subset of weight matrices in a pre-trained Transformer should we adapt
to maximize downstream performance? 2) Is the "optimal" adaptation matrix $\Delta W$ really rankdeficient? If so, what is a good rank to use in practice? 3) What is the connection between $\Delta W$ and $W$ ? Does $\Delta W$ highly correlate with $W$ ? How large is $\Delta W$ comparing to $W$ ?

We believe that our answers to question (2) and (3) shed light on the fundamental principles of using pre-trained language models for downstream tasks, which is a critical topic in NLP.

## 7.1 Which Weight Matrices in Transformer Should We Apply LoRA to?

Given a limited parameter budget, which types of weights should we adapt with LoRA to obtain the best performance on downstream tasks? As mentioned in Section 4.2, we only consider weight matrices in the self-attention module. We set a parameter budget of 18 M (roughly 35 MB if stored in FP16) on GPT-3 175B, which corresponds to $r=8$ if we adapt one type of attention weights or $r=4$ if we adapt two types, for all 96 layers. The result is presented in Table 5

\begin{table}
\begin{tabular}[t]{|l|l|l|l|l|l|l|l|}
\hline & \multicolumn{7}{|c|}{\# of Trainable Parameters = 18M} \\
\hline Weight Type & $W_{q}$ & $W_{k}$ & $W_{v}$ & $W_{o}$ & $W_{q}, W_{k}$ & $W_{q}, W_{v}$ & $W_{q}, W_{k}, W_{v}, W_{o}$ \\
\hline Rank $r$ & 8 & 8 & 8 & 8 & 4 & 4 & 2 \\
\hline WikiSQL (±0.5\%) & 70.4 & 70.0 & 73.0 & 73.2 & 71.4 & 73.7 & 73.7 \\
\hline MultiNLI (±0.1\%) & 91.0 & 90.8 & 91.0 & 91.3 & 91.3 & 91.3 & 91.7 \\
\hline
\end{tabular}

**Table 5: Validation accuracy on WikiSQL and MultiNLI after applying LoRA to different types of attention weights in GPT-3, given the same number of trainable parameters. Adapting both $W_{q**$ and $W_{v}$ gives the best performance overall. We find the standard deviation across random seeds to be consistent for a given dataset, which we report in the first column.}
\end{table}

Note that putting all the parameters in $\Delta W_{q}$ or $\Delta W_{k}$ results in significantly lower performance, while adapting both $W_{q}$ and $W_{v}$ yields the best result. This suggests that even a rank of four captures enough information in $\Delta W$ such that it is preferable to adapt more weight matrices than adapting a single type of weights with a larger rank.

## 7.2 What is the Optimal Rank $r$ for LoRA?

We turn our attention to the effect of rank $r$ on model performance. We adapt $\left\{W_{q}, W_{v}\right\}$, $\left\{W_{q}, W_{k}, W_{v}, W_{c}\right\}$, and just $W_{q}$ for a comparison.

\begin{table}
\begin{tabular}[t]{|l|l|l|l|l|l|l|}
\hline & Weight Type & $r=1$ & $r=2$ & $r=4$ & $r=8$ & $r=64$ \\
\hline \multirow[t]{3}{*}{WikiSQL(±0.5\%)} & $W_{q}$ & 68.8 & 69.6 & 70.5 & 70.4 & 70.0 \\
\hline & $W_{q}, W_{v}$ & 73.4 & 73.3 & 73.7 & 73.8 & 73.5 \\
\hline & $W_{q}, W_{k}, W_{v}, W_{o}$ & 74.1 & 73.7 & 74.0 & 74.0 & 73.9 \\
\hline \multirow{3}{*}{MultiNLI ( $\pm 0.1 \%$ )} & $W_{q}$ & 90.7 & 90.9 & 91.1 & 90.7 & 90.7 \\
\hline & $W_{q}, W_{v}$ & 91.3 & 91.4 & 91.3 & 91.6 & 91.4 \\
\hline & $W_{q}, W_{k}, W_{v}, W_{o}$ & 91.2 & 91.7 & 91.7 & 91.5 & 91.4 \\
\hline
\end{tabular}

**Table 6: Validation accuracy on WikiSQL and MultiNLI with different rank $r$. To our surprise, a rank as small as one suffices for adapting both $W_{q**$ and $W_{v}$ on these datasets while training $W_{q}$ alone needs a larger $r$. We conduct a similar experiment on GPT-2 in Section H. 2}
\end{table}

Table 6 shows that, surprisingly, LoRA already performs competitively with a very small $r$ (more so for $\left\{W_{q}, W_{v}\right\}$ than just $W_{q}$. This suggests the update matrix $\Delta W$ could have a very small "intrinsic rank" ${ }^{6}$ To further support this finding, we check the overlap of the subspaces learned by different choices of $r$ and by different random seeds. We argue that increasing $r$ does not cover a more meaningful subspace, which suggests that a low-rank adaptation matrix is sufficient.

\footnotetext{
${ }^{6}$ However, we do not expect a small $r$ to work for every task or dataset. Consider the following thought experiment: if the downstream task were in a different language than the one used for pre-training, retraining the entire model (similar to LoRA with $r=d_{\text {model }}$ ) could certainly outperform LoRA with a small $r$.
}

Subspace similarity between different $\boldsymbol{r}$. Given $A_{r=8}$ and $A_{r=64}$ which are the learned adaptation matrices with rank $r=8$ and 64 using the same pre-trained model, we perform singular value decomposition and obtain the right-singular unitary matrices $U_{A_{r=8}}$ and $U_{A_{r=64}} .{ }^{7}$ We hope to answer: how much of the subspace spanned by the top $i$ singular vectors in $U_{A_{r=8}}$ (for $1 \leq i \leq 8$ ) is contained in the subspace spanned by top $j$ singular vectors of $U_{A_{r=64}}$ (for $1 \leq j \leq 64$ )? We measure this quantity with a normalized subspace similarity based on the Grassmann distance (See Ap-pendix G for a more formal discussion)
$$
\phi\left(A_{r=8}, A_{r=64}, i, j\right)=\frac{\left\|U_{A_{r=8}}^{i \top} U_{A_{r=64}}^{j}\right\|_{F}^{2}}{\min (i, j)} \in[0,1]
$$
where $U_{A_{r=8}}^{i}$ represents the columns of $U_{A_{r=8}}$ corresponding to the top- $i$ singular vectors.
\(\phi(\cdot)\) has a range of [0, 1], where 1 represents a complete overlap of subspaces and 0 a complete separation. See Figure 3 for how $\phi$ changes as we vary $i$ and $j$. We only look at the 48th layer (out of 96) due to space constraint, but the conclusion holds for other layers as well, as shown in Section H. 1.
$$
\phi\left(A_{r=64}, A_{r=8}, i, j\right)
$$

![Figure](./huLoRALowRankAdaptation2021_images/2f8c4be1-43fa-4600-99ef-0945e0605540-11_361_1349_997_387.jpg)

**Figure 3: Subspace similarity between column vectors of $A_{r=8**$ and $A_{r=64}$ for both $\Delta W_{q}$ and $\Delta W_{v}$. The third and the fourth figures zoom in on the lower-left triangle in the first two figures. The top directions in $r=8$ are included in $r=64$, and vice versa.}

We make an important observation from Figure 3
Directions corresponding to the top singular vector overlap significantly between $A_{r=8}$ and $A_{r=64}$, while others do not. Specifically, $\Delta W_{v}$ (resp. $\Delta W_{q}$ ) of $A_{r=8}$ and $\Delta W_{v}$ (resp. $\Delta W_{q}$ ) of $A_{r=64}$ share a subspace of dimension 1 with normalized similarity $>0.5$, providing an explanation of why $r=1$ performs quite well in our downstream tasks for GPT-3.

Since both $A_{r=8}$ and $A_{r=64}$ are learned using the same pre-trained model, Figure 3 indicates that the top singular-vector directions of $A_{r=8}$ and $A_{r=64}$ are the most useful, while other directions potentially contain mostly random noises accumulated during training. Hence, the adaptation matrix can indeed have a very low rank.

Subspace similarity between different random seeds. We further confirm this by plotting the normalized subspace similarity between two randomly seeded runs with $r=64$, shown in Figure 4 $\Delta W_{q}$ appears to have a higher "intrinsic rank" than $\Delta W_{v}$, since more common singular value directions are learned by both runs for $\Delta W_{q}$, which is in line with our empirical observation in Table 6 As a comparison, we also plot two random Gaussian matrices, which do not share any common singular value directions with each other.

## 7.3 How Does the Adaptation Matrix $\Delta W$ Compare to $W$ ?

We further investigate the relationship between $\Delta W$ and $W$. In particular, does $\Delta W$ highly correlate with $W$ ? (Or mathematically, is $\Delta W$ mostly contained in the top singular directions of $W$ ?) Also,

\footnotetext{
${ }^{7}$ Note that a similar analysis can be carried out with $B$ and the left-singular unitary matrices - we stick with $A$ for our experiments.
}

![Figure](./huLoRALowRankAdaptation2021_images/2f8c4be1-43fa-4600-99ef-0945e0605540-12_457_1278_276_423.jpg)

**Figure 4: Left and Middle: Normalized subspace similarity between the column vectors of $A_{r=64**$ from two random seeds, for both $\Delta W_{q}$ and $\Delta W_{v}$ in the 48-th layer. Right: the same heat-map between the column vectors of two random Gaussian matrices. See Section H. 1 for other layers.}

how "large" is $\Delta W$ comparing to its corresponding directions in $W$ ? This can shed light on the underlying mechanism for adapting pre-trained language models.

To answer these questions, we project $W$ onto the $r$-dimensional subspace of $\Delta W$ by computing $U^{\top} W V^{\top}$, with $U / V$ being the left/right singular-vector matrix of $\Delta W$. Then, we compare the Frobenius norm between $\left\|U^{\top} W V^{\top}\right\|_{F}$ and $\|W\|_{F}$. As a comparison, we also compute $\left\|U^{\top} W V^{\top}\right\|_{F}$ by replacing $U, V$ with the top $r$ singular vectors of $W$ or a random matrix.

\begin{table}
\begin{tabular}[t]{|l|l|l|l|l|l|l|}
\hline \multirow{2}{*}{} & \multicolumn{3}{|c|}{$r=4$} & \multicolumn{3}{|c|}{$r=64$} \\
\hline & $\Delta W_{q}$ & $W_{q}$ & Random & $\Delta W_{q}$ & $W_{q}$ & Random \\
\hline $\left\|U^{\top} W_{q} V^{\top}\right\|_{F}=$ & 0.32 & 21.67 & 0.02 & 1.90 & 37.71 & 0.33 \\
\hline $\left\|W_{q}\right\|_{F}=61.95$ & \multicolumn{3}{|c|}{$\left\|\Delta W_{q}\right\|_{F}=6.91$} & \multicolumn{3}{|c|}{$\left\|\Delta W_{q}\right\|_{F}=3.57$} \\
\hline
\end{tabular}

**Table 7: The Frobenius norm of $U^{\top** W_{q} V^{\top}$ where $U$ and $V$ are the left/right top $r$ singular vector directions of either (1) $\Delta W_{q}$, (2) $W_{q}$, or (3) a random matrix. The weight matrices are taken from the 48th layer of GPT-3.}
\end{table}
We draw several conclusions from Table 7. First, $\Delta W$ has a stronger correlation with $W$ compared to a random matrix, indicating that $\Delta W$ amplifies some features that are already in $W$. Second, instead of repeating the top singular directions of $W, \Delta W$ only amplifies directions that are not emphasized in $W$. Third, the amplification factor is rather huge: $21.5 \approx 6.91 / 0.32$ for $r=4$. See Section H. 4 for why $r=64$ has a smaller amplification factor. We also provide a visualization in Section H. 3 for how the correlation changes as we include more top singular directions from $W_{q}$. This suggests that the low-rank adaptation matrix potentially amplifies the important features for specific downstream tasks that were learned but not emphasized in the general pre-training model.

