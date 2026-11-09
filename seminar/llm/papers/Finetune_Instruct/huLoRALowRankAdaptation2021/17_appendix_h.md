# H Additional Experiments on Low-Rank Matrices

We present additional results from our investigation into the low-rank update matrices.

# H. 1 Correlation between LoRA Modules

See Figure 6 and Figure 7 for how the results presented in Figure 3 and Figure 4 generalize to other layers.

# H. 2 Effect of $r$ on GPT-2

We repeat our experiment on the effect of $r$ (Section 7.2) in GPT-2. Using the E2E NLG Challenge dataset as an example, we report the validation loss and test metrics achieved by different choices of $r$ after training for 26,000 steps. We present our result in Table 18. The optimal rank for GPT-2 Medium is between 4 and 16 depending on the metric used, which is similar to that for GPT-3 175B. Note that the relationship between model size and the optimal rank for adaptation is still an open question.

# H. 3 Correlation between $W$ and $\Delta W$

See Figure 8 for the normalized subspace similarity between $W$ and $\Delta W$ with varying $r$.
Note again that $\Delta W$ does not contain the top singular directions of $W$, since the similarity between the top 4 directions in $\Delta W$ and the top-10\% of those in $W$ barely exceeds 0.2 . This gives evidence that $\Delta W$ contains those "task-specific" directions that are otherwise not emphasized in $W$.

An interesting next question to answer, is how "strong" do we need to amplify those task-specific directions, in order for the model adaptation to work well?
\(\phi\left(A_{r=8}, A_{r=64}, i, j\right)\)

![Figure](./huLoRALowRankAdaptation2021_images/2f8c4be1-43fa-4600-99ef-0945e0605540-25_1212_1379_331_370.jpg)

**Figure 6: Normalized subspace similarity between the column vectors of $A_{r=8**$ and $A_{r=64}$ for both $\Delta W_{q}$ and $\Delta W_{v}$ from the 1st, 32nd, 64th, and 96th layers in a 96-layer Transformer.}

# H. 4 Amplification Factor

One can naturally consider a feature amplification factor as the ratio $\frac{\|\Delta W\|_{F}}{\left\|U^{\top} W V^{\top}\right\|_{F}}$, where $U$ and $V$ are the left- and right-singular matrices of the SVD decomposition of $\Delta W$. (Recall $U U^{\top} W V^{\top} V$ gives the "projection" of $W$ onto the subspace spanned by $\Delta W$.)

Intuitively, when $\Delta W$ mostly contains task-specific directions, this quantity measures how much of them are amplified by $\Delta W$. As shown in Section 7.3, for $r=4$, this amplification factor is as large as 20. In other words, there are (generally speaking) four feature directions in each layer (out of the entire feature space from the pre-trained model $W$ ), that need to be amplified by a very large factor 20, in order to achieve our reported accuracy for the downstream specific task. And, one should expect a very different set of feature directions to be amplified for each different downstream task.

One may notice, however, for $r=64$, this amplification factor is only around 2, meaning that most directions learned in $\Delta W$ with $r=64$ are not being amplified by much. This should not be surprising, and in fact gives evidence (once again) that the intrinsic rank needed to represent the "task-specific directions" (thus for model adaptation) is low. In contrast, those directions in the rank-4 version of $\Delta W$ (corresponding to $r=4$ ) are amplified by a much larger factor 20.

![Figure](./huLoRALowRankAdaptation2021_images/2f8c4be1-43fa-4600-99ef-0945e0605540-26_627_1380_278_371.jpg)

**Figure 7: Normalized subspace similarity between the column vectors of $A_{r=64**$ from two randomly seeded runs, for both $\Delta W_{q}$ and $\Delta W_{v}$ from the 1st, 32nd, 64th, and 96th layers in a 96-layer Transformer.}

\begin{table}
\begin{tabular}[t]{|l|l|l|l|l|l|l|}
\hline Rank $r$ & val_loss & BLEU & NIST & METEOR & ROUGE L & CIDEr \\
\hline 1 & 1.23 & 68.72 & 8.7215 & 0.4565 & 0.7052 & 2.4329 \\
\hline 2 & 1.21 & 69.17 & 8.7413 & 0.4590 & 0.7052 & 2.4639 \\
\hline 4 & 1.18 & 70.38 & 8.8439 & 0.4689 & 0.7186 & 2.5349 \\
\hline 8 & 1.17 & 69.57 & 8.7457 & 0.4636 & 0.7196 & 2.5196 \\
\hline 16 & 1.16 & 69.61 & 8.7483 & 0.4629 & 0.7177 & 2.4985 \\
\hline 32 & 1.16 & 69.33 & 8.7736 & 0.4642 & 0.7105 & 2.5255 \\
\hline 64 & 1.16 & 69.24 & 8.7174 & 0.4651 & 0.7180 & 2.5070 \\
\hline 128 & 1.16 & 68.73 & 8.6718 & 0.4628 & 0.7127 & 2.5030 \\
\hline 256 & 1.16 & 68.92 & 8.6982 & 0.4629 & 0.7128 & 2.5012 \\
\hline 512 & 1.16 & 68.78 & 8.6857 & 0.4637 & 0.7128 & 2.5025 \\
\hline 1024 & 1.17 & 69.37 & 8.7495 & 0.4659 & 0.7149 & 2.5090 \\
\hline
\end{tabular}

**Table 18: Validation loss and test set metrics on E2E NLG Challenge achieved by LoRA with different rank $r$ using GPT-2 Medium. Unlike on GPT-3 where $r=1$ suffices for many tasks, here the performance peaks at $r=16$ for validation loss and $r=4$ for BLEU, suggesting the GPT-2 Medium has a similar intrinsic rank for adaptation compared to GPT-3 175B. Note that some of our hyperparameters are tuned on $r=4$, which matches the parameter count of another baseline, and thus might not be optimal for other choices of $r$.**
\end{table}

![Figure](./huLoRALowRankAdaptation2021_images/2f8c4be1-43fa-4600-99ef-0945e0605540-26_480_1373_1869_376.jpg)

**Figure 8: Normalized subspace similarity between the singular directions of $W_{q**$ and those of $\Delta W_{q}$ with varying $r$ and a random baseline. $\Delta W_{q}$ amplifies directions that are important but not emphasized in $W . \Delta W$ with a larger $r$ tends to pick up more directions that are already emphasized in $W$.}

