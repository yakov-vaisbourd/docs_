# 4 Language Model Experiments

The results in Section 3 suggest that TailSFT can preserve reward-bearing responses more effectively than standard SFT. We now evaluate whether this behavior carries over to pretrained language models. We first compare standard SFT and TailSFT on math and code tasks. The variation across these results then allows us to study when preserving information from the base model is most useful. Finally, we follow matched SFT checkpoints through GRPO to test whether higher coverage before RL translates into stronger post-RL performance. Throughout, we use pass@16 as an empirical measure of coverage.

\begin{table}
\begin{tabular}[t]{|l|l|l|l|l|l|l|l|}
\hline \multirow[b]{2}{*}{SFT data} & \multirow[b]{2}{*}{Benchmark} & \multicolumn{3}{|c|}{pass@1} & \multicolumn{3}{|c|}{pass@16} \\
\hline & & Standard & TailSFT & $\Delta$ & Standard & TailSFT & $\Delta$ \\
\hline \multirow[t]{3}{*}{OMI} & AIME 2022-2025 ( $n=120$ ) & $3.65_{ \pm 0.37}$ & $4.03_{ \pm 0.02}$ & +0.37 & $15.24_{ \pm 2.02}$ & $18.31_{ \pm 0.57}$ & +3.07 \\
\hline & MATH-500 Level \(5(n=134)\) & $24.80_{ \pm 0.40}$ & $25.16_{ \pm 0.40}$ & +0.36 & $66.42_{ \pm 0.00}$ & $69.15_{ \pm 0.86}$ & +2.74 \\
\hline & OMEGA-500 ( $n=500$ ) & $6.58_{ \pm 0.50}$ & $6.51_{ \pm 0.26}$ & -0.06 & $32.80_{ \pm 2.25}$ & $32.60_{ \pm 1.56}$ & -0.20 \\
\hline \multirow[t]{5}{*}{BigCode} & MBPP+ \((n=378)\) & $53.04_{ \pm 0.93}$ & $51.36_{ \pm 0.14}$ & -1.68 & $74.69_{ \pm 1.10}$ & $78.84_{ \pm 1.06}$ & +4.14 \\
\hline & HumanEval+ \((n=164)\) & $45.06_{ \pm 0.57}$ & $46.33_{ \pm 1.07}$ & +1.27 & $76.63_{ \pm 0.70}$ & $80.08 \pm 0.93$ & +3.46 \\
\hline & CruxEval-I ( $n=800$ ) & $29.79_{ \pm 0.48}$ & $30.43_{ \pm 0.15}$ & +0.64 & $61.71_{ \pm 2.27}$ & $65.79_{ \pm 0.26}$ & +4.08 \\
\hline & CruxEval-O \((n=800)\) & $4.16_{ \pm 0.44}$ & $13.18_{ \pm 0.30}$ & +9.02 & $24.21_{ \pm 2.38}$ & $41.00_{ \pm 1.35}$ & +16.79 \\
\hline & LiveCodeBench ( $n=612$ ) & $10.74_{ \pm 0.46}$ & $11.49_{ \pm 0.38}$ & +0.75 & $30.39_{ \pm 0.86}$ & $33.12_{ \pm 0.34}$ & +2.72 \\
\hline \multirow[t]{5}{*}{Magicoder} & MBPP \(+(n=378)\) & $55.35 \pm 0.47$ & $54.60_{ \pm 0.55}$ & -0.75 & $75.31_{ \pm 0.93}$ & $78.66_{ \pm 1.25}$ & +3.35 \\
\hline & HumanEval+ \((n=164)\) & $48.49_{ \pm 0.58}$ & $47.69_{ \pm 0.19}$ & -0.80 & 77.85 ±0.70 & $78.66_{ \pm 1.83}$ & +0.81 \\
\hline & CruxEval-I ( $n=800$ ) & $28.50_{ \pm 0.13}$ & $26.48 \pm 0.45$ & -2.02 & $59.75 \pm 0.62$ & $68.08 \pm 0.94$ & +8.33 \\
\hline & CruxEval-O ( $n=800$ ) & $12.86_{ \pm 0.19}$ & $16.16_{ \pm 0.53}$ & +3.30 & $38.08 \pm 0.94$ & $47.92_{ \pm 1.66}$ & +9.83 \\
\hline & LiveCodeBench ( $n=612$ ) & $14.84_{ \pm 0.11}$ & $14.32_{ \pm 0.38}$ & -0.51 & $31.81_{ \pm 0.34}$ & $33.22_{ \pm 0.34}$ & +1.42 \\
\hline \multirow[t]{5}{*}{OCI} & MBPP+ \((n=378)\) & $58.26_{ \pm 0.88}$ & $55.81_{ \pm 0.06}$ & -2.44 & $81.22_{ \pm 0.75}$ & $82.36_{ \pm 0.81}$ & +1.15 \\
\hline & HumanEval+ \((n=164)\) & $54.76_{ \pm 1.16}$ & $51.94_{ \pm 1.83}$ & -2.82 & $85.98_{ \pm 1.61}$ & $83.23_{ \pm 2.16}$ & -2.74 \\
\hline & CruxEval-I ( $n=800$ ) & $29.46_{ \pm 0.46}$ & $30.17_{ \pm 0.64}$ & +0.71 & $68.08 \pm 1.77$ & $71.58_{ \pm 1.91}$ & +3.50 \\
\hline & CruxEval-O \((n=800)\) & $9.30_{ \pm 0.49}$ & $10.84_{ \pm 0.37}$ & +1.54 & $40.12_{ \pm 0.00}$ & $44.46_{ \pm 0.47}$ & +4.33 \\
\hline & LiveCodeBench ( $n=612$ ) & $12.24_{ \pm 0.36}$ & $11.49_{ \pm 0.84}$ & -0.75 & $34.59_{ \pm 0.90}$ & $33.50_{ \pm 0.16}$ & -1.09 \\
\hline
\end{tabular}

**Table 1: TailSFT improves coverage over standard SFT. Across 18 model-benchmark pairs, TailSFT often improves pass@16, while changes in pass@1 are mixed. Values are percentages reported as mean ± standard deviation over three seeds; $\Delta$ denotes TailSFT minus Standard SFT in percentage points, shown in green when positive, red when negative, and yellow when within one standard deviation. We later establish a sufficient condition (Definition 4.1) that accurately predicts the failure modes of TailSFT exhibited in this table (Figure 4). MATH-500 Level 5 refers to the subset of the benchmark with the highest difficulty.**
\end{table}

## 4.1 SFT Results

Setup. We fine-tune the OLMo-3 7B base model (Olmo, 2026) separately on domain-specific math and code instruction data, giving 18 dataset-benchmark pairs. For math, we train on a 350k-example subset of OpenMathInstruct-2 (OMI, Toshniwal et al. (2024)), decontaminated against MATH-500. We evaluate with the OLMES protocol (Gu et al., 2025) on the 2022-2025 AIME problems, the hardest level of MATH-500, and OMEGA-500. For code, we train separately on Magicoder (Wei et al., 2024b), BigCode Self-OSS-Instruct (Wei et al., 2024a), and OpenCodeInstruct (OCI, Ahmad et al. (2025)). We evaluate on MBPP+, HumanEval+, CruxEval-I/O, and LiveCodeBench. We focus on benchmarks where performance is not saturated.

We report pass@16 using the task-specific OLMES sampling protocol and average results over three seed sets. Results are reported in Table 1 and Appendix D. 1 contains experimental details.

TAILSFT improves coverage in most settings. Across the 18 dataset-benchmark pairs in Table 1, TAILSFT improves pass@16 in 15 cases, while changes in pass@1 are mixed. OMEGA-500 is essentially unchanged, with an absolute difference of -0.20\%. The largest gain occurs on CruxEval-O after SFT on BigCode, where pass@16 increases by an absolute 16.79\%. With Magicoder, pass@16 increases by 8.33\% on CruxEval-I and 9.83\% on CruxEval-O. On AIME, it increases by 3.07\%. On MBPP+, absolute gains are 4.14\%, 3.35\%, and 1.15\% for BigCode, Magicoder, and OCI. Overall, improvements are substantially more consistent at large $K$ than at $K=1$, in line with the coverage motivation for TailSFT.

## 4.2 Coverage Ratio Diagnostic

The variation across these settings motivates a closer look at when TAILSFT should perform well. In the graph-navigation task, standard SFT increased probability on the majority shard while reducing coverage of a rewarded path that was already represented by the initial policy. Based on this, we ask whether SFT in the

![Figure](./malladiTailSFTFilteredFineTuning2026_images/b0c2e9de-14c4-464c-9bb3-a4879161b9bf-08_729_1489_240_312.jpg)

**Figure 4: When standard SFT loses more coverage than it gains, TAILSFT preserves or improves coverage. The vertical line marks $\rho_{16**=1$, where the estimated coverage lost and gained by standard SFT are equal. Of the 11 evaluated settings with $\rho_{16}>1,10$ have positive coverage gains from TailSFT and one is essentially unchanged. Several settings with $\rho_{16}<1$ also improve. Coverage gain and ratio are computed on the base-reachable set (Definition 4.1).}

language model setting exhibits a measurable loss of coverage relative to the base model, and whether this coverage loss is predictive of the benefit of TailSFT.

We measure how standard SFT changes coverage on problems that are already reachable from the base model. Restricting attention to problems that are neither effectively unsolved nor already saturated under the base model, we use pass@1 to estimate the pass@16 that would result from independent sampling. We then aggregate the decreases and increases induced by standard SFT-their ratio measures whether standard SFT has lost more base-reachable coverage than it has gained.

Definition 4.1 (Coverage ratio). For problem $i$, model $\pi$, and $K \in\{1,16\}$, let \(P_{i, K}(\pi)\) denote the empirical pass@K, averaged over seeds. We convert pass@1 to an estimated pass@16 value using \(f_{16}(p):=1-(1-p)^{16}\). The base-reachable set is \(\mathcal{R}_{0}:=\left\{i: 0.05<P_{i, 16}\left(\pi_{0}\right)<0.95\right\}\). The coverage lost and gained by standard SFT on this set are
$$
\begin{aligned}
L & :=\sum_{i \in \mathcal{R}_{0}}\left[f_{16}\left(P_{i, 1}\left(\pi_{0}\right)\right)-f_{16}\left(P_{i, 1}\left(\pi_{\mathrm{SFT}}\right)\right)\right]_{+}, \\
G & :=\sum_{i \in \mathcal{R}_{0}}\left[f_{16}\left(P_{i, 1}\left(\pi_{\mathrm{SFT}}\right)\right)-f_{16}\left(P_{i, 1}\left(\pi_{0}\right)\right)\right]_{+} .
\end{aligned}
$$
The coverage ratio is $\rho_{16}:=L / G$. If $G=0$, we set $\rho_{16}:=\infty$.
Computing $\rho_{16}$ requires only the base model and a single standard SFT run. A value $\rho_{16}>1$ means that, on the base-reachable set, the estimated coverage lost during standard SFT exceeds the estimated coverage gained. This is the setting most directly targeted by TailSFT, which reduces training on responses that have already improved and thereby limits probability shifting away from responses represented by the base model.

The coverage ratio identifies a regime where TailSFT consistently preserves or improves coverage. In Figure 4, all 11 dataset-benchmark pairs satisfying $\rho_{16}>1$ have nonnegative coverage gains (improvement in pass@16 on the base-reachable set $\mathcal{R}_{0}$ ) from TailSFT. Ten improve and one is essentially unchanged, with gains reaching an absolute 28.69\% for BigCode evaluated on CruxEval-O. The condition is not necessary for improvement: several settings with $\rho_{16}<1$ also benefit from TailSFT. Thus, the coverage ratio provides a conservative way to identify settings in which TailSFT preserves or improves coverage. Appendix D. 3 contains further details.

\begin{table}
\begin{tabular}[t]{|l|l|l|l|l|l|l|l|}
\hline \multirow[b]{2}{*}{SFT data} & \multirow[b]{2}{*}{Benchmark} & \multicolumn{3}{|c|}{pass@1} & \multicolumn{3}{|c|}{pass@16} \\
\hline & & Standard & TailSFT & $\Delta$ & Standard & TailSFT & $\Delta$ \\
\hline \multicolumn{8}{|l|}{GRPO on MATH} \\
\hline \multirow[t]{2}{*}{OMI} & MATH-500 Level \(5(n=134)\) & $57.70_{ \pm 0.44}$ & $60.26_{ \pm 1.02}$ & +2.56 & $83.83_{ \pm 1.72}$ & $87.06_{ \pm 0.86}$ & +3.23 \\
\hline & AIME 2022-2025 ( $n=120$ ) & $14.40_{ \pm 0.26}$ & $15.61_{ \pm 0.44}$ & +1.21 & $32.76_{ \pm 0.88}$ & $36.06_{ \pm 0.98}$ & +3.30 \\
\hline \multicolumn{8}{|l|}{GRPO on MBPP+} \\
\hline BigCode & MBPP+ \((n=378)\) & $69.57_{ \pm 0.29}$ & $73.50_{ \pm 0.70}$ & +3.93 & $75.93_{ \pm 0.00}$ & $78.66_{ \pm 0.15}$ & +2.73 \\
\hline Magicoder & MBPP+ \((n=378)\) & $70.52_{ \pm 0.25}$ & $73.24_{ \pm 0.09}$ & +2.72 & $78.22 \pm 0.61$ & $80.60_{ \pm 0.55}$ & +2.38 \\
\hline OCI & MBPP+ \((n=378)\) & $74.67_{ \pm 0.08}$ & $76.30_{ \pm 0.05}$ & +1.62 & $84.22_{ \pm 0.40}$ & $84.04_{ \pm 0.15}$ & -0.18 \\
\hline
\end{tabular}

**Table 2: Initializing from the TailSFT model improves post-GRPO performance across math and code. GRPO is initialized from the Standard SFT and TailSFT checkpoints evaluated in Table 1, and rows are grouped by the corpus used for GRPO training: the MATH train split, excluding MATH-500, for the math rows, and the MBPP+ train split for the code rows. The SFT data column indicates the corpus used to train the initialization for each GRPO run. Values are percentages reported as mean ± standard deviation over three seeds, and $\Delta$ is improvement from TailSFT. Experiment details are in Appendix D.2.**
\end{table}

## 4.3 GRPO Results

The coverage principle predicts that a higher-coverage initialization should expose RL to rewarding responses more often (Chen et al., 2026a). This section therefore tests whether the additional coverage resulting from TAILSFT translates into better post-RL performance.

Setup. We initialize GRPO from the standard SFT and TAILSFT checkpoints evaluated in Table 1. For math reasoning, we train on the MATH training split with MATH-500 held out and evaluate on MATH-500 Level 5 and AIME. For code, we train on the MBPP+ training split and evaluate on its test split. The GRPO procedure and hyperparameters are held fixed within each comparison. The main comparisons use 4 rollouts per prompt and an actor learning rate of $2 \times 10^{-5}$. Evaluation follows the OLMES protocol over three seed sets.

Improved coverage yields better post-RL performance. Initializing from TailSFT improves post-RL pass@1 in every matched comparison in Table 2, with absolute improvements ranging between 1.21\% and 3.93\%. Our coding experiments make the role of initialization especially salient-each TailSFT checkpoint has lower pass@1 than its standard SFT counterpart before RL, but has higher pass@1 afterward. Broader initial coverage indeed results in higher single-sample accuracy after GRPO. These results are consistent with the relationship between large- $K$ performance and post-RL outcomes observed by Kang et al. (2026).

Improved coverage drives faster learning. Figure 8 shows that the difference between the two initializations appears early in training, with TailSFT runs beginning with lower reward but the gap quickly closing. In some settings, early reward increases up to $2.5 \times$ faster than for the corresponding standard SFT run. After RL, TailSFT retains higher pass@16 in four of the five matched comparisons, and is approximately tied in the fifth. The coverage preserved during SFT is therefore available to RL early in training and ultimately translates into higher pass@1.

