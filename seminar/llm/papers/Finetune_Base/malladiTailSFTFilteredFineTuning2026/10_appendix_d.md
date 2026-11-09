# D Details for Language Modeling Experiments

# D. 1 SFT

The language modeling experiments use the same TailSFT algorithm (Algorithm 1) for both math and code but differ in data, prompt format, hyperparameter grid, and evaluation. We describe the shared procedure here; the code (Appendix D.1.1) and math (Appendix D.1.2) subsections give the domain-specific details.

Supervised fine-tuning. Every run starts from the OLMo-3 7B base model and minimizes the standard supervised cross-entropy loss on chat-formatted instruction/response pairs. We apply the OLMo-3 chat template and mask every prompt token, so only the final assistant turn and its terminating end-of-turn token count toward the loss. All runs use AdamW with \(\left(\beta_{1}, \beta_{2}\right)=(0.9,0.95)\) and $\epsilon=10^{-8}$, gradient-norm clipping at 1.0, 8-way data parallelism, and a learning rate held constant after a linear warmup over the first 3\% of steps. Sequence length, weight decay, microbatch size, effective batch, epoch count, and the learning-rate grid are set per domain and given in each subsection. The per-step loss is length-normalized as in OLMo-3: we sum the target-token cross-entropy over the batch and divide by the number of unmasked target tokens, so every target token is weighted equally rather than every sequence.

Example filtering (TailSFT). TailSFT drops a fraction of the already-fit examples from the loss at each step. Before training we score every example $x$ once with the base model, using the same tokenization and assistant-only mask as SFT, to get its initialization loss \(\ell_{0}(x)\), the mean cross-entropy over the target tokens. During training we compare the current per-example loss \(\ell_{t}(x)\) against this baseline through the signed margin
$$
m_{t}(x)=\ell_{t}(x)-\ell_{0}(x) .
$$
A very negative margin means the loss on $x$ has already dropped well below its starting value. We filter out the examples with the smallest (most negative) margins and train on the rest. Both $\ell_{0}$ and $\ell_{t}$ are length-normalized per example - the mean cross-entropy over that example's target tokens - so the filtering decision does not favor longer or shorter responses.

The filtering decision runs on the selection batch, the $W \times b$ examples in a single forward pass across the $W=8$ data-parallel ranks with per-rank microbatch $b$. The per-example losses are all-gathered so every rank computes the same mask. We do not filter over the full gradient-accumulated optimizer batch or over one rank's local microbatch; each accumulation step is filtered on its own selection batch. Within a selection batch we rank the examples by $m_{t}$ and zero out the $\lfloor W b f\rceil$ with the smallest margins, then train the survivors with the usual token-averaged cross-entropy. The code runs use $b=1$ (an 8-example selection batch) and the math runs use $b=2$ (a 16-example selection batch). Reproducing the method needs only the base-model losses $\ell_{0}$, the filter fraction $f$, and its schedule; everything else is ordinary distributed SFT.

Filtering notation. A run is specified by its filter fraction $f \in[0,1]$, the fraction of each selection batch that is dropped ( $\lfloor W b f\rceil$ examples), and a schedule for how $f$ moves over training. A static schedule holds $f$

\begin{table}

**Table 3: Per-benchmark pass@1 for the OLMo-3 7B base model (no SFT), Standard SFT \((f=0)\), and TAILSFT, at the settings used in the main results table. $\Delta_{\text {Std-Base **}$ is Standard SFT minus base; $\Delta_{\text {Tail-Base }}$ is TailSFT minus base. Emdashes indicate missing evaluations.}
\begin{tabular}[t]{|l|l|l|l|l|l|l|}
\hline SFT data & Benchmark & Base & Standard SFT & TailSFT & $\Delta_{\text {Std-Base }}$ & $\Delta_{\text {Tail-Base }}$ \\
\hline \multirow[t]{3}{*}{OMI} & AIME & 5.16 & $3.65_{ \pm 0.37}$ & $4.03_{ \pm 0.02}$ & -1.51 & -1.13 \\
\hline & MATH-500 L5 & - & $24.80_{ \pm 0.40}$ & $25.16_{ \pm 0.40}$ & - & - \\
\hline & OMEGA-500 & $3.54_{ \pm 0.30}$ & $6.58_{ \pm 0.50}$ & $6.51_{ \pm 0.26}$ & +3.04 & +2.97 \\
\hline \multirow[t]{5}{*}{BigCode} & MBPP+ & $36.82_{ \pm 0.56}$ & $53.04_{ \pm 0.93}$ & $51.36_{ \pm 0.14}$ & +16.22 & +14.54 \\
\hline & HumanEval+ & $32.94_{ \pm 1.11}$ & $45.06_{ \pm 0.57}$ & $46.33_{ \pm 1.07}$ & +12.12 & +13.39 \\
\hline & CruxEval-I & $25.67_{ \pm 0.62}$ & $29.79_{ \pm 0.48}$ & $30.43_{ \pm 0.15}$ & +4.12 & +4.76 \\
\hline & CruxEval-O & $4.22_{ \pm 0.74}$ & $4.16_{ \pm 0.44}$ & $13.18_{ \pm 0.30}$ & -0.06 & +8.96 \\
\hline & LiveCodeBench & $5.50_{ \pm 0.42}$ & $10.74_{ \pm 0.46}$ & $11.49_{ \pm 0.38}$ & +5.24 & +5.99 \\
\hline \multirow[t]{5}{*}{Magicoder} & MBPP+ & $36.82_{ \pm 0.56}$ & $55.35_{ \pm 0.47}$ & $54.60_{ \pm 0.55}$ & +18.53 & +17.78 \\
\hline & HumanEval+ & $32.94_{ \pm 1.11}$ & $48.49_{ \pm 0.58}$ & $47.69_{ \pm 0.19}$ & +15.55 & +14.75 \\
\hline & CruxEval-I & $25.67_{ \pm 0.62}$ & $28.50_{ \pm 0.13}$ & $26.48_{ \pm 0.45}$ & +2.83 & +0.81 \\
\hline & CruxEval-O & $4.22_{ \pm 0.74}$ & $12.86_{ \pm 0.19}$ & $16.16_{ \pm 0.53}$ & +8.64 & +11.94 \\
\hline & LiveCodeBench & $5.50_{ \pm 0.42}$ & $14.84_{ \pm 0.11}$ & $14.32_{ \pm 0.38}$ & +9.34 & +8.82 \\
\hline \multirow[t]{5}{*}{OCI} & MBPP+ & $36.82_{ \pm 0.56}$ & $58.26_{ \pm 0.88}$ & $55.81_{ \pm 0.06}$ & +21.44 & +18.99 \\
\hline & HumanEval+ & $32.94_{ \pm 1.11}$ & $54.76_{ \pm 1.16}$ & $51.94_{ \pm 1.83}$ & +21.82 & +19.00 \\
\hline & CruxEval-I & $25.67_{ \pm 0.62}$ & $29.46_{ \pm 0.46}$ & $30.17_{ \pm 0.64}$ & +3.79 & +4.50 \\
\hline & CruxEval-O & $4.22_{ \pm 0.74}$ & $9.30_{ \pm 0.49}$ & $10.84_{ \pm 0.37}$ & +5.08 & +6.62 \\
\hline & LiveCodeBench & $5.50_{ \pm 0.42}$ & $12.24_{ \pm 0.36}$ & $11.49_{ \pm 0.84}$ & +6.74 & +5.99 \\
\hline
\end{tabular}
\end{table}
fixed. A ramp schedule written $0 \rightarrow f$ raises the fraction linearly from 0 at the first step to $f$ at the last, using every example early and filtering hardest late. Standard SFT is $f=0$. We describe every run by its filter fraction $f$ and schedule in the below results.

Filtering rarely hurts. The consistent finding across both domains is that filtering preserves pass@16 coverage. At matched settings it almost always matches or improves the no-filter pass@16, the gains are sometimes large, and the only cost is a small pass@1 decrease from deprioritizing examples the model already fits. Each subsection reports the full per-benchmark grids and tallies how rarely filtering regresses coverage. Tables 3 and 4 collect the base model, Standard SFT, and TailSFT side by side at the main-table settings for pass@1 and pass@16. In several cases where standard SFT marginally lowers pass@16 below the base model, TAILSFT rescues the coverage, recovering it back above the base level. If we instead select the best pass@16 checkpoint for each of Standard SFT and TailSFT, TailSFT still matches or improves coverage on most benchmarks.

The gains are broad, not concentrated. Table 5 decomposes each pass@16 gain into per-problem paired changes, showing that the improvements are spread across many problems rather than driven by a handful.

# D.1.1 Code

Model and optimization. Code SFT trains at sequence length 2048, which effectively never truncates training targets. Runs use no weight decay and a per-rank microbatch of 1 with 4 gradient-accumulation steps, for an effective batch of 32 and an 8-example selection batch. The learning rate is searched over $\{1,2,3\} \times 10^{-5}$. Each dataset uses a fixed epoch count.

Magicoder. We keep the Python subset of ise-uiuc/Magicoder-OSS-Instruct-75K (38,284 of 75,197 rows). We drop non-Python rows, rows without extractable code, rows matching an MBPP+ or HumanEval+ signature, and rows containing unit tests, and discard solutions that fail to parse as an abstract syntax tree. The assistant target starts with the prefix Here is the completed function:\n\n"'python\n, followed by the extracted code and a closing fence, matching the MBPP+ and HumanEval+ answer format. One further row is removed because it does not fit the 2048-token training tokenization, leaving 33,817 train rows.

\begin{table}

**Table 4: Per-benchmark pass@16 (coverage) for the OLMo-3 7B base model (no SFT), Standard SFT \((f=0)\), and TailSFT, at the settings used in the main results table. Columns as in Table 3.**
\begin{tabular}[t]{|l|l|l|l|l|l|l|}
\hline SFT data & Benchmark & Base & Standard SFT & TailSFT & $\Delta_{\text {Std-Base }}$ & $\Delta_{\text {Tail-Base }}$ \\
\hline \multirow[t]{3}{*}{OMI} & AIME & 20.91 & $15.24_{ \pm 2.02}$ & $18.31_{ \pm 0.57}$ & -5.67 & -2.60 \\
\hline & MATH-500 L5 & - & $66.42_{ \pm 0.00}$ & $69.15_{ \pm 0.86}$ & - & - \\
\hline & OMEGA-500 & $26.13_{ \pm 1.42}$ & $32.80_{ \pm 2.25}$ & $32.60_{ \pm 1.56}$ & +6.67 & +6.47 \\
\hline \multirow[t]{5}{*}{BigCode} & MBPP+ & $75.93_{ \pm 1.06}$ & $74.69_{ \pm 1.10}$ & $78.84_{ \pm 1.06}$ & -1.24 & +2.91 \\
\hline & HumanEval+ & $79.27_{ \pm 0.61}$ & $76.63_{ \pm 0.70}$ & $80.08_{ \pm 0.93}$ & -2.64 & +0.81 \\
\hline & CruxEval-I & $70.50_{ \pm 1.54}$ & $61.71_{ \pm 2.27}$ & $65.79_{ \pm 0.26}$ & -8.79 & -4.71 \\
\hline & CruxEval-O & $32.08_{ \pm 1.98}$ & $24.21_{ \pm 2.38}$ & $41.00_{ \pm 1.35}$ & -7.87 & +8.92 \\
\hline & LiveCodeBench & $32.30_{ \pm 0.93}$ & $30.39_{ \pm 0.86}$ & $33.12_{ \pm 0.34}$ & -1.91 & +0.82 \\
\hline \multirow[t]{5}{*}{Magicoder} & MBPP+ & $75.93_{ \pm 1.06}$ & $75.31_{ \pm 0.93}$ & $78.66_{ \pm 1.25}$ & -0.62 & +2.73 \\
\hline & HumanEval+ & $79.27_{ \pm 0.61}$ & $77.85_{ \pm 0.70}$ & $78.66_{ \pm 1.83}$ & -1.42 & -0.61 \\
\hline & CruxEval-I & $70.50_{ \pm 1.54}$ & $59.75_{ \pm 0.62}$ & $68.08_{ \pm 0.94}$ & -10.75 & -2.42 \\
\hline & CruxEval-O & $32.08_{ \pm 1.98}$ & $38.08_{ \pm 0.94}$ & $47.92_{ \pm 1.66}$ & +6.00 & +15.84 \\
\hline & LiveCodeBench & $32.30_{ \pm 0.93}$ & $31.81_{ \pm 0.34}$ & $33.22_{ \pm 0.34}$ & -0.49 & +0.92 \\
\hline \multirow[t]{5}{*}{OCI} & MBPP+ & $75.93_{ \pm 1.06}$ & $81.22_{ \pm 0.75}$ & $82.36_{ \pm 0.81}$ & +5.29 & +6.43 \\
\hline & HumanEval+ & $79.27_{ \pm 0.61}$ & $85.98_{ \pm 1.61}$ & $83.23_{ \pm 2.16}$ & +6.71 & +3.96 \\
\hline & CruxEval-I & $70.50_{ \pm 1.54}$ & $68.08_{ \pm 1.77}$ & $71.58_{ \pm 1.91}$ & -2.42 & +1.08 \\
\hline & CruxEval-O & $32.08_{ \pm 1.98}$ & $40.12_{ \pm 0.00}$ & $44.46_{ \pm 0.47}$ & +8.04 & +12.38 \\
\hline & LiveCodeBench & $32.30_{ \pm 0.93}$ & $34.59_{ \pm 0.90}$ & $33.50_{ \pm 0.16}$ & +2.29 & +1.20 \\
\hline
\end{tabular}
\end{table}

BigCode. For bigcode/self-oss-instruct-sc2-exec-filter-50k (50,661 raw rows) the user message is the instruction and the target uses the same evalplus-aligned prefix and fenced Python format as Magicoder. We remove exact and near MBPP+ and HumanEval+ contamination, benchmark-signature contamination, rows containing tests, and non-Python rows, leaving 49,260 train rows.

OCI. For nvidia/OpenCodeInstruct we load five evenly spaced shards and keep rows with average_test_score= 1.0. We remove MBPP+ contamination by exact prompt match, by test-assertion substring match against inputs, outputs, and unit tests, and by MBPP+ function-name definitions appearing in outputs, then shuffle with seed 42 and hold out a 0.5\% validation split. OCI keeps its native format, with the user message the OCI input and the target the OCI output, both verbatim.

Hyperparameter configurations. We vary the learning rate and filtering configuration as summarized in Table 7. The configurations used in Table 1 and the additional evaluated configurations are reported across several tables. Tables 8 and 9 give MBPP+ pass@1 and pass@16 for every completed BigCode and Magicoder configuration, with the change against the no-filter run at the same learning rate.

For OCI, Table 10 reports the no-filter Standard configuration and the TailSFT ramp configuration used in Table 1. MBPP+ coverage improves under filtering (+1.15 pp), while HumanEval+ loses coverage (-2.74 pp). Accordingly, OCI HumanEval+ has $\rho_{16}=0.23$, outside the $\rho_{16}>1$ regime identified by the coverage-ratio diagnostic.

Coverage holds across benchmarks and can repair coverage collapse. Tables 11-14 report every filtered configuration we evaluated on HumanEval+, CruxEval-I, CruxEval-O, and LiveCodeBench, against the no-filter run for the same dataset. The largest coverage gains land where standard SFT had driven coverage below the base model: on CruxEval-I standard SFT falls 8-11 pp under the base ( $70.5 \rightarrow 61.7$ for BigCode, 59.8 for Magicoder), so much of TailSFT's advantage is repairing a coverage collapse that ordinary SFT introduced.

Evaluation protocol. We evaluate with the OLMES (Gu et al., 2025) EvalPlus configuration used across the OLMo-3 code suite, so the appendix numbers are directly comparable to that standard. Every code result is sampled at temperature 1.0 , top- $p=1.0$, with 16 generations per problem for each of three independent evaluation seed sets; throughout the appendix, reported means and standard deviations are computed over these three seed sets, which we refer to simply as three seeds. MBPP+ uses the 378-problem EvalPlus test

\begin{table}
\begin{tabular}[t]{|l|l|l|l|l|l|l|l|}
\hline SFT data & Benchmark & \#gained & \#lost & \#unchanged & median $\Delta$ (pp) & mean \(\Delta(\mathrm{pp})\) & 95\% CI \\
\hline OMI & AIME & 31 & 14 & 75 & +0.00 & +3.07 & [+0.77, +5.59] \\
\hline OMI & MATH-500 L5 & 25 & 15 & 94 & +0.00 & +2.74 & [-1.24, +6.47] \\
\hline OMI & OMEGA-500 & 75 & 77 & 348 & +0.00 & -0.20 & [-2.33, +1.80] \\
\hline BigCode & MBPP+ & 39 & 19 & 320 & +0.00 & +4.14 & [+2.12, +6.35] \\
\hline BigCode & HumanEval+ & 21 & 11 & 132 & +0.00 & +3.46 & [+0.00, +6.91] \\
\hline BigCode & CruxEval-I & 157 & 89 & 554 & +0.00 & +4.08 & [+2.12, +6.12] \\
\hline BigCode & CruxEval-O & 274 & 44 & 482 & +0.00 & +16.79 & [+14.54, +19.04] \\
\hline BigCode & LiveCodeBench & 72 & 41 & 499 & +0.00 & +2.72 & [+1.09, +4.41] \\
\hline Magicoder & MBPP+ & 42 & 19 & 317 & +0.00 & +3.35 & [+1.06, +5.73] \\
\hline Magicoder & HumanEval+ & 13 & 10 & 141 & +0.00 & +0.81 & [-2.03, +3.86] \\
\hline Magicoder & CruxEval-I & 196 & 82 & 522 & +0.00 & +8.33 & [+6.17, +10.46] \\
\hline Magicoder & CruxEval-O & 183 & 48 & 569 & +0.00 & +9.83 & [+7.92, +11.83] \\
\hline Magicoder & LiveCodeBench & 56 & 41 & 515 & +0.00 & +1.42 & [-0.22, +3.05] \\
\hline OCI & MBPP+ & 22 & 15 & 341 & +0.00 & +1.15 & [-0.35, +2.60] \\
\hline OCI & HumanEval+ & 8 & 17 & 139 & +0.00 & -2.74 & [-5.18, -0.51] \\
\hline OCI & CruxEval-I & 134 & 69 & 597 & +0.00 & +3.50 & [+2.04, +4.92] \\
\hline OCI & CruxEval-O & 132 & 57 & 611 & +0.00 & +4.33 & [+2.83, +5.79] \\
\hline OCI & LiveCodeBench & 45 & 65 & 502 & +0.00 & -1.09 & [-2.61, +0.44] \\
\hline
\end{tabular}

**Table 5: Per-problem paired changes in pass@16 for TailSFT versus Standard SFT at the main-table settings. A problem counts as gained or lost only when its paired pass@16 strictly changes; unchanged includes ceiling and floor ties. The final column is a paired bootstrap 95\% interval for the mean per-problem change.**
\end{table}

\begin{table}
\begin{tabular}[t]{|l|l|l|}
\hline SFT data & Source & Train / val \\
\hline Magicoder & ise-uiuc/Magicoder-OSS-Instruct-75K & 33,817 / 342 \\
\hline BigCode & bigcode/self-oss-instruct-sc2-exec-filter-50k & 49,260 / 498 \\
\hline OCI & nvidia/OpenCodeInstruct & 72,635 / 365 \\
\hline
\end{tabular}

**Table 6: Code SFT datasets. Row counts are after decontamination, format conversion, and initialization-loss annotation.**
\end{table}
split with the assistant prefix Here is the completed function: \n\n" ' python\n, maximum generation length 2048, context length 4096, and stop strings " ' , newline-triple-quote, newline-assert, and newline-comment. HumanEval+ uses the 164-problem EvalPlus split with the same prefix, generation length 1024, and context 4096. CruxEval-I and CruxEval-O use 800 instances each with OLMo-3 multiturn few-shot prompts (two shots for input prediction, one for output), generation length 512, and context 4096. LiveCodeBench uses 612 code-generation problems with an expert-Python system prompt, generation length 8192, and context 32768. We report pass@ $k$ with the standard unbiased estimator implemented in OLMES, pass@ $k=1-\binom{n-c}{k} /\binom{n}{k}$ for a document with $c$ correct completions out of $n$ samples, averaged over documents rather than as a best-of- $k$ maximum. With $n=16$ samples this makes pass@1 the mean per-document pass rate and pass@16 the fraction of documents with at least one correct completion. Each generation length sits above the lengths these tasks actually decode to, so completions are not cut off.

The coverage advantage tends to widen with $k$. Table 16 reports the TailSFT minus Standard SFT pass@ $k$ gap at $k \in\{1,2,4,8,16\}$ for every benchmark. On most cells the gap starts near zero (or slightly negative) at $k=1$ and tends to increase with $k$-most strikingly CruxEval-O, where BigCode grows from +9.02 to +16.79-so the advantage is a coverage effect rather than a pass@1 shift. The two OCI transfer benchmarks (HumanEval+ and LiveCodeBench) are the main exception, where TailSFT trails at every $k$.

# D.1.2 Math

Model and optimization. Math SFT trains at sequence length 4096 with weight decay 0.1, a per-rank microbatch of 2 with 4 gradient-accumulation steps, for an effective batch of 64 and a 16-example selection batch. The learning rate is fixed at $3 \times 10^{-5}$. We train for 2 epochs and evaluate the final checkpoint. Two passes fit the ~ 350k-example subset without the overfitting seen at higher counts, and the count is held identical across every arm so differences come only from filtering. Each supervised example is a two-turn

\begin{table}
\begin{tabular}[t]{|l|l|}
\hline Hyperparameter & Values \\
\hline Base model & OLMo-3 7B base \\
\hline Learning rate & $\{1,2,3\} \times 10^{-5}$ \\
\hline Filter schedule & none; static; ramp $0 \rightarrow$. \\
\hline Filter fraction $f$ & 0 (none); static \{0.125, 0.25, 0.5\}; ramp 0 → 0.5 \\
\hline Epochs & fixed per dataset (4 / 5 / 3 for BigCode / Magicoder / OCI) \\
\hline Sequence length & 2048 \\
\hline Effective batch & 32 (8 × 1 × 4) \\
\hline Warmup / schedule & 3\% linear warmup, then constant \\
\hline Optimizer & AdamW (0.9, 0.95), wd 0, clip 1.0 \\
\hline
\end{tabular}

**Table 7: Code SFT hyperparameter grid over learning rates and filtering configurations.**
\end{table}

\begin{table}
\begin{tabular}[t]{|l|l|l|l|l|l|}
\hline Filter & LR & pass@1 & $\Delta$ & pass@16 & $\Delta$ \\
\hline base (no SFT) & - & $36.82 \pm 0.56$ & - & $75.93 \pm 1.06$ & - \\
\hline no filter & $1 \times 10^{-5}$ & 47.45 ± 0.82 & - & 79.45 ± 0.81 & - \\
\hline static $f=0.125$ & $1 \times 10^{-5}$ & 47.00 ± 0.60 & -0.45 & 79.81 ± 0.81 & +0.36 \\
\hline ramp 0→0.5 & $1 \times 10^{-5}$ & 45.62 ± 0.57 & -1.83 & 79.72 ± 0.85 & +0.27 \\
\hline no filter & $2 \times 10^{-5}$ & $52.58 \pm 0.11$ & - & 76.72 ± 1.06 & - \\
\hline static $f=0.125$ & $2 \times 10^{-5}$ & 52.07 ± 0.70 & -0.51 & $77.51 \pm 0.46$ & +0.79 \\
\hline static $f=0.25$ & $2 \times 10^{-5}$ & 51.36 ± 0.14 & -1.22 & 78.84 ± 1.06 & +2.12 \\
\hline static $f=0.5$ & $2 \times 10^{-5}$ & 47.22 ± 0.54 & -5.36 & 77.95 ± 0.76 & +1.23 \\
\hline ramp 0→0.5 & $2 \times 10^{-5}$ & 51.65 ± 0.18 & -0.93 & 78.66 ± 0.40 & +1.94 \\
\hline no filter & $3 \times 10^{-5}$ & 53.04 ± 0.93 & - & 74.69 ± 1.10 & - \\
\hline static $f=0.125$ & $3 \times 10^{-5}$ & $51.90 \pm 0.59$ & -1.14 & 75.40 ± 1.06 & +0.71 \\
\hline ramp 0→0.5 & $3 \times 10^{-5}$ & 50.67 ± 0.26 & -2.37 & 75.93 ± 0.46 & +1.24 \\
\hline
\end{tabular}

**Table 8: BigCode MBPP+ \((n=378)\) SFT filtering grid, 4 epochs. Values are percent, mean ± sample standard deviation over three seeds; $\Delta$ is the change against the no-filter run at the same learning rate. The base (no \(S F T)\) row is the untuned OLMo-3 7B base under the same MBPP+ protocol (format-robust EvalPlus stops).**
\end{table}
chat. The user message is the problem followed by the instruction "Present the answer in LaTex format: \boxed\{Your answer\}", and the assistant target is the provided chain-of-thought solution.

Data. The OpenMathInstruct-2 (OMI) subset is built from the OMI train_1M shards by keeping problem_source in \{math, augmented_math\}, exact-match decontaminating against MATH-500 problem strings, dropping rows whose problem plus solution exceeds 2048 tokens under the OLMo-3 tokenizer, and sampling a 350k-example stratified subset with a 95/5 train/validation split. No MATH-500 rows were found during decontamination. Since rows over 2048 tokens were removed during data construction, the 4096-token training sequence length does not truncate supervised targets. The final SFT parquets add init_ce, the base-model per-example cross-entropy used by the filtering rule.

Hyperparameter search and selection. For math we fixed the learning rate at $3 \times 10^{-5}$ and swept the filtering configuration over no filtering, static $f \in\{0.0625,0.125,0.1875,0.25,0.5\}$, and the linear ramps $0 \rightarrow f$ for the same endpoints. The optimizer, sequence length, batch size, and 2-epoch schedule are held fixed as shown in Table 18. Table 1 then reports the no-filter Standard checkpoint against the selected TailSFT checkpoint. We use the ramp 0 → 0.5 arm.

Per-benchmark grids. Tables 19-22 report every completed math SFT filtering evaluation in-domain math benchmarks. The broadest grid is MATH-500. Its 500-problem aggregate (Table 19) sits near ceiling, so Tables 20 and 21 break the same sweep out by the five MATH difficulty levels. No-filter pass@16 is 100\% at Level 1 and stays above 96\% through Level 3, so the easy levels dominate the aggregate and hide where filtering acts. The unreached coverage, and the gains from filtering, concentrate at the hardest levels: at Level 5 filtering matches or improves no-filter pass@16 in 8 of 10 arms and strictly improves it in 7, while the

\begin{table}
\begin{tabular}[t]{|l|l|l|l|l|l|}
\hline Filter & LR & pass@1 & $\Delta$ & pass@16 & $\Delta$ \\
\hline base (no SFT) & - & 36.82 ± 0.56 & - & $75.93 \pm 1.06$ & - \\
\hline no filter & $1 \times 10^{-5}$ & $53.14 \pm 0.52$ & - & 80.69 ± 0.26 & - \\
\hline static $f=0.125$ & $1 \times 10^{-5}$ & 52.19 ± 0.25 & -0.95 & 79.98 ± 0.61 & -0.71 \\
\hline static $f=0.25$ & $1 \times 10^{-5}$ & $51.51 \pm 0.04$ & -1.63 & 80.42 ± 1.40 & -0.27 \\
\hline static $f=0.5$ & $1 \times 10^{-5}$ & $50.00 \pm 0.59$ & -3.14 & 79.45 ± 1.55 & -1.24 \\
\hline no filter & $2 \times 10^{-5}$ & $55.01 \pm 0.36$ & - & $75.13 \pm 1.15$ & - \\
\hline static $f=0.125$ & $2 \times 10^{-5}$ & $54.83 \pm 0.52$ & -0.18 & 77.51 ± 0.53 & +2.38 \\
\hline static $f=0.25$ & $2 \times 10^{-5}$ & $54.60 \pm 0.55$ & -0.41 & 78.66 ± 1.25 & +3.53 \\
\hline static $f=0.5$ & $2 \times 10^{-5}$ & $51.92 \pm 0.41$ & -3.09 & 79.72 ± 0.31 & +4.59 \\
\hline ramp 0→0.5 & $2 \times 10^{-5}$ & 54.73 ± 0.73 & -0.28 & 78.31 ± 1.47 & +3.18 \\
\hline no filter & $3 \times 10^{-5}$ & $55.35 \pm 0.47$ & - & $75.31 \pm 0.93$ & - \\
\hline static $f=0.125$ & $3 \times 10^{-5}$ & $54.75 \pm 0.58$ & -0.60 & 76.54 ± 0.15 & +1.23 \\
\hline static $f=0.25$ & $3 \times 10^{-5}$ & 52.94 ± 0.67 & -2.41 & 75.93 ± 1.06 & +0.62 \\
\hline static $f=0.5$ & $3 \times 10^{-5}$ & 51.14 ± 0.47 & -4.21 & 77.95 ± 1.36 & +2.64 \\
\hline
\end{tabular}

**Table 9: Magicoder MBPP+ \((n=378)\) SFT filtering grid, 5 epochs. Values are percent, mean ± sample standard deviation over three seeds; $\Delta$ is the change against the no-filter run at the same learning rate. The base (no \(S F T)\) row is the same untuned base as in Table 8.**
\end{table}

\begin{table}
\begin{tabular}[t]{|l|l|l|l|l|l|}
\hline Benchmark & Filter & pass@1 & $\Delta$ & pass@16 & $\Delta$ \\
\hline MBPP+ \((n=378)\) & base (no SFT) & 36.82 ± 0.56 & - & $75.93 \pm 1.06$ & - \\
\hline MBPP+ \((n=378)\) & no filter & 58.26 ± 0.88 & - & 81.22 ± 0.75 & - \\
\hline MBPP+ \((n=378)\) & ramp 0→0.5 & $55.81 \pm 0.06$ & -2.44 & 82.36 ± 0.81 & +1.15 \\
\hline HumanEval+ \((n=164)\) & base (no SFT) & 32.94 ± 1.11 & - & 79.27 ± 0.61 & - \\
\hline HumanEval+ \((n=164)\) & no filter & $54.76 \pm 1.16$ & - & 85.98 ± 1.61 & - \\
\hline HumanEval+ \((n=164)\) & ramp 0→0.5 & 51.94 ± 1.83 & -2.82 & 83.23 ± 2.16 & -2.74 \\
\hline
\end{tabular}

**Table 10: OCI MBPP+ and HumanEval+ SFT grid, 3 epochs, learning rate $2 \times 10^{-5**$; Standard is no filtering and TAILSFT is the ramp $0 \rightarrow 0.5$. Values are percent, mean ± sample standard deviation; $\Delta$ is the change against no filtering. The base (no SFT) rows are the untuned base under the same per-benchmark protocol.}
\end{table}
near-saturated levels move little in either direction. No MATH-500 pass@16 cell drops by more than 1.8 pp, inside the seed noise. The completed AIME rows all improve pass@16, while the frozen OMEGA-500 arm is essentially tied with no filtering.

Reading the Level-4 numbers. Level 4 is the one subset where filtering neither clearly helps nor hurts, and this is an estimator artifact rather than a real regression. pass@16 is scored as a binary hit per seed and averaged over three seeds, so every per-problem value is quantized to $\left\{0, \frac{1}{3}, \frac{2}{3}, 1\right\}$ and a problem's filtered-minus-no-filter difference can only be a multiple of 33 pp . The Level-4 mean is then set by a few knife's-edge problems that the base model already solves on two of three seeds, where a single seed flip moves the subset mean by a point or two. Two such problems account for the whole gap: under the arms that most help Level 5, the Level-4 pass@16 change moves from +0.0 pp on the full subset to about +0.8 pp once those two problems are excluded, in the same direction as Level 5. Level 5 shows a larger and steadier gain because it contains many more genuinely contested problems, so this per-problem quantization averages out.

Evaluation protocol. We evaluate with the standard OLMES (Gu et al., 2025) math configuration used across the OLMo-3 math suite. AIME contains 30 problems from each year 2022-2025 \((n=120)\), chat prompting with the boxed-answer instruction, temperature 0.6 , top- $p=0.95$, maximum generation length and model context 8192, and 32 generations per problem for each of three seeds. MATH-500 uses the 500-problem test split; the main table reports the Level-5 subset \((n=134)\), since the full benchmark is near ceiling, with temperature 1.0 , top- $p=1.0$, maximum generation length 8192, model context 16384, and 16 generations per problem for each of three seeds. OMEGA-500 (the saumyamalik/omega-500 500-problem set) uses temperature

\begin{table}
\begin{tabular}[t]{|l|l|l|l|l|l|}
\hline Filter & LR & pass@1 & $\Delta$ & pass@16 & $\Delta$ \\
\hline base (no SFT) & - & 32.94 ± 1.11 & - & 79.27 ± 0.61 & - \\
\hline \multicolumn{6}{|l|}{BigCode} \\
\hline no filter & $3 \times 10^{-5}$ & 45.06 ± 0.57 & - & 76.63 ± 0.70 & - \\
\hline static $f=0.125$ & $2 \times 10^{-5}$ & 48.08 ± 0.12 & +3.02 & 79.47 ± 1.76 & +2.85 \\
\hline static $f=0.25$ & $2 \times 10^{-5}$ & $46.33 \pm 1.07$ & +1.27 & 80.08 ± 0.93 & +3.46 \\
\hline static $f=0.5$ & $2 \times 10^{-5}$ & $44.08 \pm 0.48$ & -0.98 & 80.69 ± 1.27 & +4.07 \\
\hline \multicolumn{6}{|l|}{Magicoder} \\
\hline no filter & $3 \times 10^{-5}$ & 48.49 ± 0.58 & - & 77.85 ± 0.70 & - \\
\hline static $f=0.25$ & $2 \times 10^{-5}$ & 47.69 ± 0.19 & -0.80 & $78.66 \pm 1.83$ & +0.81 \\
\hline static $f=0.5$ & $2 \times 10^{-5}$ & $46.60 \pm 0.66$ & -1.89 & 80.49 ± 1.61 & +2.64 \\
\hline
\end{tabular}

**Table 11: HumanEval+ \((n=164)\) SFT filtering grid. Epochs are 4 (BigCode) and 5 (Magicoder); OCI HumanEval+ appears in Table 10. Values are percent, mean ± sample standard deviation over three seeds; $\Delta$ is the change against the no-filter run for the same dataset; the base (no SFT) row is the untuned OLMo-3 7B base under the same protocol.**
\end{table}

\begin{table}
\begin{tabular}[t]{|l|l|l|l|l|l|}
\hline Filter & LR & pass@1 & $\Delta$ & pass@16 & $\Delta$ \\
\hline base (no SFT) & - & 25.67 ± 0.62 & - & $70.50 \pm 1.54$ & - \\
\hline \multicolumn{6}{|l|}{BigCode} \\
\hline no filter & $3 \times 10^{-5}$ & $29.79 \pm 0.48$ & - & $61.71 \pm 2.27$ & - \\
\hline static $f=0.125$ & $2 \times 10^{-5}$ & $30.65 \pm 0.26$ & +0.86 & 63.54 ± 0.76 & +1.83 \\
\hline static $f=0.25$ & $2 \times 10^{-5}$ & $30.43 \pm 0.15$ & +0.64 & 65.79 ± 0.26 & +4.08 \\
\hline static $f=0.5$ & $2 \times 10^{-5}$ & $29.76 \pm 0.35$ & -0.03 & $69.29 \pm 1.92$ & +7.58 \\
\hline \multicolumn{6}{|l|}{Magicoder} \\
\hline no filter & $3 \times 10^{-5}$ & $28.50 \pm 0.13$ & - & $59.75 \pm 0.62$ & - \\
\hline static $f=0.25$ & $2 \times 10^{-5}$ & $26.48 \pm 0.45$ & -2.02 & 68.08 ± 0.94 & +8.33 \\
\hline static $f=0.5$ & $2 \times 10^{-5}$ & $25.63 \pm 0.56$ & -2.88 & 69.79 ± 0.38 & +10.04 \\
\hline \multicolumn{6}{|l|}{OCI} \\
\hline no filter & $2 \times 10^{-5}$ & $29.46 \pm 0.46$ & - & $68.08 \pm 1.77$ & - \\
\hline ramp 0→0.5 & $2 \times 10^{-5}$ & $30.17 \pm 0.64$ & +0.71 & $71.58 \pm 1.91$ & +3.50 \\
\hline
\end{tabular}

**Table 12: CruxEval-I \((n=800)\) SFT filtering grid. Epochs are 4 / 5 / 3 for BigCode / Magicoder / OCI. Values are percent, mean ± sample standard deviation over three seeds; $\Delta$ is the change against the no-filter run for the same dataset; the base (no SFT) row is the untuned OLMo-3 7B base under the same protocol.**
\end{table}
1.0 , top- $p=1.0$, maximum generation length 4096, model context 8192, and 16 generations per problem for each of three seeds. For AIME and MATH-500 we use the stored per-document pass@k metrics. For OMEGA-500, whose aggregate files do not store pass@k, we recompute pass@1 and pass@16 from the 16 completions using the extracted flexible boxed answer, falling back to the standard extracted answer, and match it against the label. In every case pass@ $k$ is the same unbiased estimator implemented in OLMES (pass@ $k=1-\binom{n-c}{k} /\binom{n}{k}$ for $c$ correct of $n$ samples), so AIME's 32 samples are combined at $k=16$ rather than by best-of- $k$. The sweep of Table 19 uses MATH-500 full \((n=500)\) with 4096-token generations.

The coverage advantage tends to widen with $k$. Table 24 reports the TAILSFT minus Standard SFT pass@ $k$ gap at $k \in\{1,2,4,8,16\}$. On AIME and MATH-500 the gap is small at $k=1$ and tends to grow with $k$ (AIME +0.37 → +3.07; MATH-500 Level \(5+0.36 \rightarrow+2.74)\), while OMEGA-500, where the two models are already matched, stays flat. The math gains therefore come from broader coverage that surfaces at larger $k$ rather than from a shift in greedy accuracy.

# D. 2 GRPO

The in-domain math and code GRPO runs share one training procedure and differ only in data, reward, hyperparameter grid, and evaluation. We describe the shared procedure here; the math (Appendix D.2.1)

\begin{table}
\begin{tabular}[t]{|l|l|l|l|l|l|}
\hline Filter & LR & pass@1 & $\Delta$ & pass@16 & $\Delta$ \\
\hline base (no SFT) & - & 4.22 ± 0.74 & - & $32.08 \pm 1.98$ & - \\
\hline \multicolumn{6}{|l|}{BigCode} \\
\hline no filter & $3 \times 10^{-5}$ & 4.16 ± 0.44 & - & $24.21 \pm 2.38$ & - \\
\hline static $f=0.125$ & $2 \times 10^{-5}$ & 12.02 ± 0.11 & +7.86 & 39.75 ± 0.76 & +15.54 \\
\hline static $f=0.25$ & $2 \times 10^{-5}$ & 13.18 ± 0.30 & +9.02 & 41.00 ± 1.35 & +16.79 \\
\hline static $f=0.5$ & $2 \times 10^{-5}$ & 7.35 ± 0.37 & +3.20 & 36.54 ± 0.14 & +12.33 \\
\hline \multicolumn{6}{|l|}{Magicoder} \\
\hline no filter & $3 \times 10^{-5}$ & $12.86 \pm 0.19$ & - & $38.08 \pm 0.94$ & - \\
\hline static $f=0.25$ & $2 \times 10^{-5}$ & 16.16 ± 0.53 & +3.30 & $47.92 \pm 1.66$ & +9.83 \\
\hline static $f=0.5$ & $2 \times 10^{-5}$ & 10.12 ± 0.37 & -2.74 & 41.67 ± 0.63 & +3.58 \\
\hline \multicolumn{6}{|l|}{OCI} \\
\hline no filter & $2 \times 10^{-5}$ & 9.30 ± 0.49 & - & 40.12 ± 0.00 & - \\
\hline ramp 0→0.5 & $2 \times 10^{-5}$ & $10.84 \pm 0.37$ & +1.54 & 44.46 ± 0.47 & +4.33 \\
\hline
\end{tabular}

**Table 13: CruxEval-O \((n=800)\) SFT filtering grid. Epochs are 4 / 5 / 3 for BigCode / Magicoder / OCI. Values are percent, mean ± sample standard deviation over three seeds; $\Delta$ is the change against the no-filter run for the same dataset; the base (no SFT) row is the untuned OLMo-3 7B base under the same protocol.**
\end{table}

\begin{table}
\begin{tabular}[t]{|l|l|l|l|l|l|}
\hline Filter & LR & pass@1 & $\Delta$ & pass@16 & $\Delta$ \\
\hline base (no SFT) & - & 5.50 ± 0.42 & - & $32.30 \pm 0.93$ & - \\
\hline \multicolumn{6}{|l|}{BigCode} \\
\hline no filter & $3 \times 10^{-5}$ & $10.74 \pm 0.46$ & - & $30.39 \pm 0.86$ & - \\
\hline static $f=0.125$ & $2 \times 10^{-5}$ & 11.77 ± 0.44 & +1.03 & 31.54 ± 0.71 & +1.14 \\
\hline static $f=0.25$ & $2 \times 10^{-5}$ & $11.49 \pm 0.38$ & +0.75 & 33.12 ± 0.34 & +2.72 \\
\hline static $f=0.5$ & $2 \times 10^{-5}$ & 11.45 ± 0.42 & +0.71 & 33.39 ± 0.25 & +3.00 \\
\hline \multicolumn{6}{|l|}{Magicoder} \\
\hline no filter & $3 \times 10^{-5}$ & $14.84 \pm 0.11$ & - & $31.81 \pm 0.34$ & - \\
\hline static $f=0.25$ & $2 \times 10^{-5}$ & 14.32 ± 0.38 & -0.51 & $33.22 \pm 0.34$ & +1.42 \\
\hline static $f=0.5$ & $2 \times 10^{-5}$ & $14.81 \pm 0.22$ & -0.02 & $34.48 \pm 1.28$ & +2.67 \\
\hline \multicolumn{6}{|l|}{OCI} \\
\hline no filter & $2 \times 10^{-5}$ & $12.24 \pm 0.36$ & - & $34.59 \pm 0.90$ & - \\
\hline ramp 0→0.5 & $2 \times 10^{-5}$ & $11.49 \pm 0.84$ & -0.75 & $33.50 \pm 0.16$ & -1.09 \\
\hline
\end{tabular}

**Table 14: LiveCodeBench \((n=612)\) SFT filtering grid. Epochs are 4 / 5 / 3 for BigCode / Magicoder / OCI. Values are percent, mean ± sample standard deviation over three seeds; $\Delta$ is the change against the no-filter run for the same dataset; the base (no SFT) row is the untuned OLMo-3 7B base under the same protocol.**
\end{table}
and code (Appendix D.2.2) subsections give the domain-specific details. Every run is initialized from one of the matched SFT checkpoints evaluated in Table 1: the Standard-SFT run supplies the no-filter start and the paired TailSFT run supplies the filtered start, so the two policies differ only in how their initialization was trained. Table 2 reports a single tuned configuration per domain; the per-domain grids below give the tuned-configuration results in full.

Reinforcement learning with GRPO. Each run optimizes the initialized policy against a verifiable reward with GRPO (Shao et al., 2024). For every prompt we sample a group of $n$ responses at temperature 1.0, score each response with the domain reward, and use the group-relative advantage-each response's reward centered and scaled within its group-as the policy-gradient signal. We use no KL regularization and no learned reward model. Optimization uses a batch of 128 prompts, a mini-batch of 128, and 8-way data parallelism, with the rollout group size $n$ and the actor learning rate searched per domain and all other settings held fixed within a domain. Checkpoints are exported periodically during training and the reported checkpoint is chosen on a held-out validation split, as described per domain. Post-GRPO evaluation uses OLMES (Gu et al., 2025) and reuses the same per-benchmark coverage protocols as Appendix D.1, with pass@k estimated over three seeds; the exact per-benchmark sampling settings are given below.

\begin{table}
\begin{tabular}[t]{|l|l|l|l|l|l|}
\hline SFT data & Benchmark & Standard p@1 & TailSFT p@1 & Standard p@16 & TailSFT p@16 \\
\hline BigCode & MBPP+ \((n=378)\) & $53.04_{ \pm 0.93}$ & $51.36_{ \pm 0.14}$ & $74.69_{ \pm 1.10}$ & $78.84_{ \pm 1.06}$ \\
\hline BigCode & HumanEval+ \((n=164)\) & $45.06_{ \pm 0.57}$ & $46.33_{ \pm 1.07}$ & $76.63_{ \pm 0.70}$ & $80.08_{ \pm 0.93}$ \\
\hline BigCode & CruxEval-I ( $n=800$ ) & $29.79_{ \pm 0.48}$ & $30.43_{ \pm 0.15}$ & $61.71_{ \pm 2.27}$ & $65.79_{ \pm 0.26}$ \\
\hline BigCode & CruxEval-O \((n=800)\) & $4.16_{ \pm 0.44}$ & $13.18_{ \pm 0.30}$ & $24.21_{ \pm 2.38}$ & $41.00_{ \pm 1.35}$ \\
\hline BigCode & LiveCodeBench ( $n=612$ ) & $10.74_{ \pm 0.46}$ & $11.49_{ \pm 0.38}$ & $30.39_{ \pm 0.86}$ & $33.12_{ \pm 0.34}$ \\
\hline Magicoder & MBPP+ \((n=378)\) & $55.35_{ \pm 0.47}$ & $54.60_{ \pm 0.55}$ & $75.31_{ \pm 0.93}$ & $78.66_{ \pm 1.25}$ \\
\hline Magicoder & HumanEval+ \((n=164)\) & $48.49_{ \pm 0.58}$ & $47.69_{ \pm 0.19}$ & $77.85_{ \pm 0.70}$ & $78.66_{ \pm 1.83}$ \\
\hline Magicoder & CruxEval-I ( $n=800$ ) & $28.50_{ \pm 0.13}$ & $26.48_{ \pm 0.45}$ & $59.75_{ \pm 0.62}$ & $68.08_{ \pm 0.94}$ \\
\hline Magicoder & CruxEval-O \((n=800)\) & $12.86_{ \pm 0.19}$ & $16.16_{ \pm 0.53}$ & $38.08_{ \pm 0.94}$ & $47.92_{ \pm 1.66}$ \\
\hline Magicoder & LiveCodeBench \((n=612)\) & $14.84_{ \pm 0.11}$ & $14.32_{ \pm 0.38}$ & $31.81_{ \pm 0.34}$ & $33.22_{ \pm 0.34}$ \\
\hline OCI & CruxEval-I ( $n=800$ ) & $29.46_{ \pm 0.46}$ & $30.17_{ \pm 0.64}$ & $68.08_{ \pm 1.77}$ & $71.58_{ \pm 1.91}$ \\
\hline OCI & CruxEval-O \((n=800)\) & $9.30_{ \pm 0.49}$ & $10.84_{ \pm 0.37}$ & $40.12_{ \pm 0.00}$ & $44.46_{ \pm 0.47}$ \\
\hline OCI & LiveCodeBench ( $n=612$ ) & $12.24_{ \pm 0.36}$ & $11.49_{ \pm 0.84}$ & $34.59_{ \pm 0.90}$ & $33.50_{ \pm 0.16}$ \\
\hline
\end{tabular}

**Table 15: Post-SFT code results, matching Table 1. Values are percentages, mean ± sample standard deviation over three seeds.**
\end{table}

\begin{table}

**Table 16: SFT pass@k deltas for code benchmarks, reported as TailSFT minus Standard SFT in percentage points at the main-table settings. Deltas are computed with the unbiased OLMES (Gu et al., 2025) pass@k estimator.**
\begin{tabular}[t]{|l|l|l|l|l|l|l|}
\hline SFT data & Benchmark & $\Delta \mathrm{p} @ 1$ & $\Delta \mathrm{p}$ @ 2 & $\Delta \mathrm{p} @ 4$ & $\Delta \mathrm{p}$ @ 8 & $\Delta \mathrm{p} @ 16$ \\
\hline \multirow[t]{5}{*}{BigCode} & MBPP \(+(n=378)\) & -1.68 & +0.64 & +2.19 & +3.19 & +4.14 \\
\hline & HumanEval+ \((n=164)\) & +1.27 & +3.37 & +4.34 & +4.44 & +3.46 \\
\hline & CruxEval-I ( $n=800$ ) & +0.64 & +1.61 & +2.57 & +3.25 & +4.08 \\
\hline & CruxEval-O ( $n=800$ ) & +9.02 & +12.58 & +15.45 & +16.95 & +16.79 \\
\hline & LiveCodeBench ( $n=612$ ) & +0.75 & +1.39 & +1.86 & +2.21 & +2.72 \\
\hline \multirow[t]{5}{*}{Magicoder} & MBPP \(+(n=378)\) & -0.75 & +1.73 & +2.95 & +3.25 & +3.35 \\
\hline & HumanEval+ \((n=164)\) & -0.80 & +0.81 & +1.55 & +1.64 & +0.81 \\
\hline & CruxEval-I ( $n=800$ ) & -2.02 & -0.10 & +2.36 & +5.28 & +8.33 \\
\hline & CruxEval-O ( $n=800$ ) & +3.30 & +4.97 & +6.70 & +8.38 & +9.83 \\
\hline & LiveCodeBench ( $n=612$ ) & -0.51 & +0.30 & +0.80 & +1.08 & +1.42 \\
\hline \multirow[t]{5}{*}{OCI} & MBPP+ \((n=378)\) & -2.44 & -1.21 & -0.17 & +0.65 & +1.15 \\
\hline & HumanEval+ \((n=164)\) & -2.82 & -2.38 & -1.89 & -2.02 & -2.74 \\
\hline & CruxEval-I ( $n=800$ ) & +0.71 & +1.54 & +2.46 & +3.19 & +3.50 \\
\hline & CruxEval-O ( $n=800$ ) & +1.54 & +2.33 & +3.14 & +3.86 & +4.33 \\
\hline & LiveCodeBench ( $n=612$ ) & -0.75 & -0.86 & -0.91 & -0.91 & -1.09 \\
\hline
\end{tabular}
\end{table}

# D.2.1 Math

Initialization. The Standard start is the no-filter OMI SFT checkpoint and the TailSFT start is the matched ramp 0 → 0.5 checkpoint (Appendix D.1.2); these are the two OMI checkpoints reported in Table 1.

Data and reward. GRPO trains on the MATH training split (the MATH-lighteval release) with the MATH-500 test problems removed by exact-match decontamination, leaving 11,396 training and 600 validation prompts, so the evaluation set is never seen during RL. Each prompt uses the same boxed-answer instruction as math SFT, and the reward is 1 when a response's boxed final answer matches the reference under the verifiable math grader and 0 otherwise.

Optimization. We use a maximum prompt length of 512, a maximum response length of 4096, and train for 3 epochs. At rollout group size $n=4$ we search the actor learning rate; Table 26 reports the tuned configuration (actor learning rate $2 \times 10^{-5}$ ). Doubling the rollout group size to $n=8$ leaves the ordering unchanged-the TAILSFT initialization retains a post-GRPO pass@1 advantage of about +2 points on MATH-500 Level 5-so the gain is not an artifact of the group size. Table 25 summarizes the configuration.

Checkpoint selection and evaluation. We select the mean-best checkpoint-the one with the highest mean validation reward over the sampled group-which averages over the small held-out validation split to

\begin{table}
\begin{tabular}[t]{|l|l|l|l|}
\hline Dataset & Source/filter & Train rows & Validation rows \\
\hline OMI math subset & OpenMathInstruct-2 math/augmented_math, MATH-500 decontaminated & 332,500 & 17,500 \\
\hline
\end{tabular}

**Table 17: In-domain math SFT data. The annotated training split contains 326,725 augmented_math and 5,775 math rows; the validation split contains 17,192 and 308, respectively.**
\end{table}

\begin{table}
\begin{tabular}[t]{|l|l|}
\hline Hyperparameter & Values \\
\hline Base model & OLMo-3 7B base \\
\hline Learning rate & fixed at $3 \times 10^{-5}$ \\
\hline Filter schedule & none; static; ramp 0→. \\
\hline Filter fraction $f$ & 0 (none); static \{0.0625, 0.125, 0.1875, 0.25, 0.5\}; ramp 0 → \{0.0625, 0.125, 0.1875, 0.25, 0.5\} \\
\hline Epochs & 2 \\
\hline Sequence length & 4096 \\
\hline Effective batch & \(64(8 \times 2 \times 4)\) \\
\hline Warmup schedule & 3\% linear warmup, then constant \\
\hline Optimizer & AdamW (0.9, 0.95), wd 0.1, clip 1.0 \\
\hline
\end{tabular}

**Table 18: Math SFT hyperparameter grid.**
\end{table}
reduce selection noise. We evaluate on the MATH-500 Level-5 subset \((n=134)\) and AIME 2022-2025 \((n=120)\). Because these are long-form reasoning tasks we use a generation budget of 8192 tokens and report only these long-generation evaluations, since shorter budgets truncate reasoning traces and understate pass rates; at 8192 tokens truncation is not a concern, as no AIME generation and fewer than 4\% of MATH-500 generations reach the limit. Matching the math SFT protocol (Appendix D.1.2), MATH-500 Level 5 is sampled at temperature 1.0 , top- $p=1.0$ with 16 samples per problem, and AIME at temperature 0.6 , top- $p=0.95$ with 32 samples per problem, each over three seeds. At the tuned configuration the TailSFT initialization improves both pass@1 and pass@16 on both benchmarks (Table 26).

# D.2.2 Code

Initialization. For each dataset the Standard start is the no-filter SFT checkpoint and the TailSFT start is the matched filtered checkpoint from Appendix D.1.1 (static $f=0.25$ for BigCode and Magicoder, the ramp $0 \rightarrow 0.5$ for OCI); these are the checkpoints reported in Table 1.

Data and reward. GRPO trains on the MBPP+ training split with a held-out 100-problem validation subset. The reward is 1 when a sampled program passes the full MBPP+ unit-test suite for its problem and 0 otherwise.

Optimization. We use a maximum prompt and response length of 1024 each and train for 10 epochs. We search the rollout group size $n \in\{2,4\}$ and the actor learning rate $\in\left\{5 \times 10^{-6}, 1 \times 10^{-5}, 2 \times 10^{-5}\right\}$; Table 28 reports the tuned configuration for each dataset. Table 27 summarizes the configuration.

Checkpoint selection and evaluation. We select the checkpoint with the highest validation pass rate over the small 100-problem held-out split, applying the same rule to both initializations. Evaluation uses EvalPlus MBPP+ (378-problem test split) at temperature 1.0, top- $p=1.0$, with 16 samples per problem over three seeds and a 2048-token generation budget; MBPP+ solutions are short relative to this budget, so generations terminate well within the limit and truncation does not affect the reported rates. At the tuned configurations the TailSFT initialization improves post-GRPO pass@1 while generally retaining its pass@16 coverage advantage (Table 28).

Training efficiency. Figure 8 tracks the mean training reward per GRPO step for the Standard and TailSFT initializations. The TailSFT reward rises at least as fast as Standard early in training, and in some settings the

\begin{table}
\begin{tabular}[t]{|l|l|l|l|l|}
\hline Filter & pass@1 & $\Delta$ & pass@16 & $\Delta$ \\
\hline no filter & $53.55 \pm 0.49$ & - & 86.27 ± 0.61 & - \\
\hline static $f=0.0625$ & 53.66 ± 0.28 & +0.11 & 87.00 ± 0.87 & +0.73 \\
\hline static $f=0.125$ & $53.60 \pm 0.39$ & +0.05 & 87.20 ± 0.40 & +0.93 \\
\hline static $f=0.1875$ & $53.15 \pm 0.33$ & -0.41 & 86.20 ± 0.69 & -0.07 \\
\hline static $f=0.25$ & $53.51 \pm 0.41$ & -0.04 & 86.33 ± 1.29 & +0.07 \\
\hline static $f=0.5$ & 52.02 ± 0.13 & -1.54 & 85.60 ± 0.72 & -0.67 \\
\hline ramp 0→0.0625 & $53.80 \pm 0.29$ & +0.25 & 85.87 ± 0.31 & -0.40 \\
\hline ramp 0→0.125 & 53.37 ± 0.73 & -0.19 & 86.47 ± 0.50 & +0.20 \\
\hline ramp 0→0.1875 & 54.13 ± 0.50 & +0.57 & 87.27 ± 0.95 & +1.00 \\
\hline ramp 0→0.25 & $53.91 \pm 0.19$ & +0.35 & 86.13 ± 0.46 & -0.13 \\
\hline ramp 0→0.5 & 53.59 ± 0.41 & +0.03 & 86.27 ± 1.01 & +0.00 \\
\hline
\end{tabular}

**Table 19: MATH-500 full \((n=500)\) SFT filtering grid, 4096-token generations. Values are percent, mean ± sample standard deviation over three generation seeds; $\Delta$ is the change against no filtering.**
\end{table}

\begin{table}
\begin{tabular}[t]{|l|l|l|l|l|l|}
\hline Filter & Level \(1(n=43)\) & Level \(2(n=90)\) & Level \(3(n=105)\) & Level \(4(n=128)\) & Level \(5(n=134)\) \\
\hline no filter & 100.00 ± 0.00 & 97.78 ± 1.11 & $96.51 \pm 1.98$ & $85.68 \pm 1.63$ & 66.67 ± 1.88 \\
\hline static $f=0.0625$ & $100.00 \pm 0.00$ & 98.15 ± 0.64 & 96.19 ± 0.95 & 85.68 ± 0.90 & 69.40 ± 3.25 \\
\hline static $f=0.125$ & $100.00 \pm 0.00$ & 97.41 ± 0.64 & 96.83 ± 1.10 & 85.68 ± 2.51 & 70.15 ± 1.29 \\
\hline static $f=0.1875$ & $100.00 \pm 0.00$ & $97.78 \pm 0.00$ & $95.56 \pm 1.46$ & 84.38 ± 0.78 & $68.41 \pm 1.88$ \\
\hline static $f=0.25$ & $100.00 \pm 0.00$ & 97.41 ± 0.64 & $96.51 \pm 0.55$ & 85.94 ± 3.40 & 66.92 ± 2.62 \\
\hline static $f=0.5$ & $100.00 \pm 0.00$ & $96.30 \pm 1.70$ & $95.56 \pm 1.98$ & $85.68 \pm 3.25$ & 65.92 ± 4.24 \\
\hline ramp 0→0.0625 & 99.22 ± 1.34 & 97.04 ± 0.64 & 97.14 ± 0.95 & 84.38 ± 0.78 & 66.67 ± 1.55 \\
\hline ramp 0→0.125 & $100.00 \pm 0.00$ & 97.04 ± 1.70 & 96.83 ± 1.46 & $83.85 \pm 1.19$ & 69.40 ± 2.69 \\
\hline ramp 0→0.1875 & $100.00 \pm 0.00$ & $97.41 \pm 0.64$ & $96.51 \pm 0.55$ & $85.68 \pm 1.63$ & 70.65 ± 4.97 \\
\hline ramp 0→0.25 & $100.00 \pm 0.00$ & 97.78 ± 1.11 & $97.78 \pm 1.46$ & 84.90 ± 0.45 & 65.92 ± 1.14 \\
\hline ramp 0→0.5 & $100.00 \pm 0.00$ & $96.67 \pm 0.00$ & $97.14 \pm 0.00$ & $85.42 \pm 1.97$ & $67.16 \pm 1.97$ \\
\hline
\end{tabular}

**Table 20: MATH-500 SFT pass@16 by difficulty level, from the same 4096-token SFT filtering sweep as Table 19. Columns are the five MATH difficulty levels, with the number of problems in each header. Values are percent, mean ± sample standard deviation over three seeds. pass@16 is at or near ceiling through Level 3, so the aggregate is dominated by the easy levels; the gains from filtering concentrate at Levels 4-5. The frozen main table reports the selected ramp arm on Level 5, re-evaluated with 8192-token generations.**
\end{table}
early reward increases up to roughly 2.5× faster.

# D. 3 Coverage ratio diagnostic: computation

This subsection records exactly how each point in Figure 4 was produced from the evaluation runs, filling in the aggregation and run-selection choices that Definition 4.1 leaves implicit. The generation settings (temperature, top- $p$, generation length, and sample count) for each benchmark are exactly those documented in Appendices D.1.1 and D.1.2; we do not re-specify them here.

Per-problem scoring and seed aggregation. For each problem $i$ and model $\pi$ we draw at least 16 samples per evaluation seed, compute the empirical pass@1 and pass@16 within each seed, and average those per-problem values across the available seeds to form \(P_{i, 1}(\pi)\) and \(P_{i, 16}(\pi)\); this is the \(P_{i, K}(\pi)\) of Definition 4.1. Every run uses three seeds for base, standard, and TailSFT on every benchmark.

Horizontal axis \(\left(\rho_{16}\right)\). The coverage ratio is computed exactly as in Definition 4.1 from $\pi_{0}$ and $\pi_{\text {SFT }}$ : the base-reachable set \(\mathcal{R}_{0}=\left\{i: 0.05<P_{i, 16}\left(\pi_{0}\right)<0.95\right\}\) uses the empirical base pass@16, while $L$ and $G$ use the plug-in \(f_{16}\left(P_{i, 1}(\cdot)\right)\) of the empirical pass@1. Standard SFT is the no-filter \((f=0)\) run. The size $\left|\mathcal{R}_{0}\right|$ of the base-reachable set varies from 16 (AIME) to 237 (CruxEval-O) and is listed per point in Table 29.

\begin{table}
\begin{tabular}[t]{|l|l|l|l|l|l|}
\hline Filter & Level \(1(n=43)\) & Level \(2(n=90)\) & Level \(3(n=105)\) & Level \(4(n=128)\) & Level \(5(n=134)\) \\
\hline no filter & 85.27 ± 1.46 & 76.25 ± 0.43 & $65.95 \pm 1.59$ & 47.02 ± 0.65 & $24.66 \pm 0.63$ \\
\hline static $f=0.0625$ & $85.66 \pm 1.05$ & 76.44 ± 1.15 & $64.98 \pm 1.93$ & 46.92 ± 0.21 & 25.67 ± 0.72 \\
\hline static $f=0.125$ & 84.98 ± 0.89 & 76.69 ± 0.70 & 64.46 ± 0.77 & 47.28 ± 0.47 & 25.56 ± 0.73 \\
\hline static $f=0.1875$ & 84.79 ± 0.75 & 75.83 ± 1.05 & 65.06 ± 0.51 & 46.14 ± 1.65 & 25.11 ± 0.66 \\
\hline static $f=0.25$ & 85.08 ± 0.69 & 75.62 ± 0.77 & 65.08 ± 0.34 & 47.18 ± 0.86 & $25.51 \pm 0.33$ \\
\hline static $f=0.5$ & $83.58 \pm 0.58$ & 74.31 ± 0.90 & 62.38 ± 1.11 & 46.48 ± 0.13 & 24.08 ± 0.03 \\
\hline ramp 0→0.0625 & $85.61 \pm 1.54$ & 76.06 ± 0.51 & 66.03 ± 0.72 & 47.27 ± 0.30 & $25.30 \pm 0.56$ \\
\hline ramp 0→0.125 & $84.20 \pm 0.67$ & 76.41 ± 1.28 & 65.24 ± 1.17 & 47.20 ± 1.10 & $24.58 \pm 0.82$ \\
\hline ramp 0→0.1875 & 85.90 ± 0.52 & 76.39 ± 0.84 & 65.85 ± 0.30 & 48.00 ± 0.09 & 25.65 ± 1.10 \\
\hline ramp 0→0.25 & 85.85 ± 1.24 & 76.44 ± 0.26 & 65.66 ± 0.53 & 47.58 ± 0.31 & 25.37 ± 0.69 \\
\hline ramp 0→0.5 & 85.27 ± 0.22 & 75.93 ± 0.54 & $65.28 \pm 0.61$ & 47.84 ± 0.79 & 24.75 ± 1.02 \\
\hline
\end{tabular}

**Table 21: MATH-500 SFT pass@1 by difficulty level, from the same sweep as Table 20. Values are percent, mean ± sample standard deviation over three seeds.**
\end{table}

\begin{table}
\begin{tabular}[t]{|l|l|l|l|l|l|}
\hline Eval family & Filter & pass@1 & $\Delta$ & pass@16 & $\Delta$ \\
\hline base (no SFT) ${ }^{\dagger}$ & & 5.16 & - & 20.91 & - \\
\hline ramp sweep & no filter & 3.65 ± 0.37 & - & $15.24 \pm 2.02$ & - \\
\hline ramp sweep & ramp 0→0.1875 & 3.82 ± 0.20 & +0.16 & $16.06 \pm 1.42$ & +0.82 \\
\hline ramp sweep & ramp 0→0.25 & 3.78 ± 0.20 & +0.13 & 15.37 ± 0.23 & +0.13 \\
\hline ramp sweep & ramp 0→0.5 & $4.03 \pm 0.02$ & +0.37 & 18.31 ± 0.57 & +3.07 \\
\hline static & no filter & 3.65 ± 0.34 & - & $15.52 \pm 2.00$ & - \\
\hline static & static $f=0.125$ & $3.75 \pm 0.18$ & +0.10 & $16.14 \pm 1.73$ & +0.63 \\
\hline
\end{tabular}

**Table 22: AIME 2022-2025 \((n=120)\) completed SFT filtering evaluations. The ramp sweep and the static pair were evaluated in separate runs, each against its own matched no-filter baseline. Values are percent, mean ± sample standard deviation over three seeds; $\Delta$ is the change against the matching no-filter row. ${ **^{\dagger}$ The base (no \(S F T)\) row is the untuned OLMo-3 7B base under the same AIME protocol.}
\end{table}

Vertical axis (coverage gain $\Delta$ ). The plotted gain is the mean over $\mathcal{R}_{0}$ of the empirical pass@16 difference \(P_{i, 16}\left(\pi_{\text {Tail }}\right)-P_{i, 16}\left(\pi_{\text {SFT }}\right)\), in percentage points. The TAILSFT and standard runs use the same clip settings surfaced in Table 1: static $f=0.25$ for BigCode and Magicoder, ramp $0 \rightarrow 0.5$ for OCI, ramp $0 \rightarrow 0.5$ for AIME, OMEGA-500, and MATH-500 Level 5.

Special cases. Two points require extra care. For MATH-500 Level 5, the base model was only evaluated at a 4096-token generation length, so to keep $\pi_{0}$ and $\pi_{\mathrm{SFT}}$ length-matched inside $\rho_{16}$ we compute $\mathcal{R}_{0}, L$, and $G$ from a 4096-token base run and a 4096-token no-filter run, restricted to the 134 Level-5 problems; the plotted gain $\Delta$ still uses the 8192-token standard and TailSFT runs of Table 1. Recomputing $\rho_{16}$ from the 8192-token no-filter run instead leaves it essentially unchanged (2.28 vs. 2.04, both > 1), so the point's placement is robust to this choice. For $O M E G A-500$, per-problem pass@1 and pass@16 are recomputed from the stored generations with the same exact-match (flexible) grader used for Table 1, rather than read from cached metric files, so the diagnostic and the main table score the runs identically. Each point is thus a triple \(\left(\rho_{16}, \Delta,\left|\mathcal{R}_{0}\right|\right)\); Table 29 lists all eighteen.

\begin{table}
\begin{tabular}[t]{|l|l|l|l|l|l|}
\hline SFT data & Benchmark & Standard p@1 & TailSFT p@1 & Standard p@16 & TailSFT p@16 \\
\hline OMI & AIME 2022-2025 \((n=120)\) & $3.65_{ \pm 0.37}$ & $4.03_{ \pm 0.02}$ & $15.24_{ \pm 2.02}$ & $18.31_{ \pm 0.57}$ \\
\hline OMI & MATH-500 Level \(5(n=134)\) & $24.80_{ \pm 0.40}$ & $25.16_{ \pm 0.40}$ & $66.42_{ \pm 0.00}$ & $69.15_{ \pm 0.86}$ \\
\hline OMI & OMEGA-500 ( $n=500$ ) & $6.58_{ \pm 0.50}$ & $6.51_{ \pm 0.26}$ & $32.80_{ \pm 2.25}$ & $32.60_{ \pm 1.56}$ \\
\hline
\end{tabular}

**Table 23: Post-SFT math results, matching Table 1. Values are percentages, mean ± sample standard deviation over three seeds.**
\end{table}

\begin{table}

**Table 24: SFT pass@k deltas for math benchmarks, reported as TailSFT minus Standard SFT in percentage points at the main-table settings. Deltas are computed with the unbiased OLMES (Gu et al., 2025) pass@k estimator.**
\begin{tabular}[t]{|l|l|l|l|l|l|}
\hline Benchmark & $\Delta \mathrm{p} @ 1$ & $\Delta \mathrm{p} @ 2$ & $\Delta \mathrm{p} @ 4$ & $\Delta \mathrm{p}$ @ 8 & $\Delta \mathrm{p} @ 16$ \\
\hline AIME 2022-2025 ( $n=120$ ) & +0.37 & +0.72 & +1.20 & +1.93 & +3.07 \\
\hline MATH-500 Level \(5(n=134)\) & +0.36 & +0.71 & +0.83 & +0.87 & +2.74 \\
\hline OMEGA-500 ( $n=500$ ) & -0.06 & -0.09 & -0.12 & -0.23 & -0.20 \\
\hline
\end{tabular}
\end{table}

\begin{table}
\begin{tabular}[t]{|l|l|}
\hline Setting & Value \\
\hline Initialization & matched Standard / TailSFT OMI SFT checkpoints (Table 1) \\
\hline Algorithm & GRPO, group-relative advantage, no KL \\
\hline Training prompts & MATH-lighteval train, MATH-500 removed (11,396 train / 600 val) \\
\hline Reward & verifiable boxed-answer exact match (1/0) \\
\hline Rollout group size $n$ & 4 \\
\hline Actor learning rate & searched $\{1,2\} \times 10^{-5}$; reported $2 \times 10^{-5}$ \\
\hline Max prompt / response length & 512 / 4096 \\
\hline Epochs & 3 \\
\hline Prompt batch / mini-batch & 128 / 128 \\
\hline Checkpoint selection & mean-best (highest mean validation reward) \\
\hline Evaluation & MATH-500 L5 (temp 1.0, 16 samples) \& AIME (temp 0.6, 32 samples); 8192-token, 3 seeds \\
\hline
\end{tabular}

**Table 25: Math GRPO configuration. All runs share the fixed settings; only the actor learning rate is searched, and Table 26 reports the tuned configuration.**
\end{table}

\begin{table}
\begin{tabular}[t]{|l|l|l|l|l|l|l|}
\hline \multirow[b]{2}{*}{Benchmark} & \multicolumn{3}{|c|}{pass@1} & \multicolumn{3}{|c|}{pass@16} \\
\hline & Standard & TailSFT & $\Delta$ & Standard & TailSFT & $\Delta$ \\
\hline MATH-500 Level \(5(n=134)\) & 57.70 ± 0.44 & 60.26 ± 1.02 & +2.56 & $83.83 \pm 1.72$ & 87.06 ± 0.86 & +3.23 \\
\hline AIME 2022-2025 \((n=120)\) & $14.40 \pm 0.26$ & $15.61 \pm 0.44$ & +1.21 & $32.76 \pm 0.88$ & $36.06 \pm 0.98$ & +3.30 \\
\hline
\end{tabular}

**Table 26: Math GRPO at the tuned configuration (rollout $n=4$, actor learning rate $2 \times 10^{-5**$, mean-best checkpoint, 8192-token generation), evaluated from the matched Standard and TailSFT initializations of Table 1. Values are percent, mean ± standard deviation over three seeds; $\Delta$ is the TailSFT improvement.}
\end{table}

\begin{table}
\begin{tabular}[t]{|l|l|}
\hline Setting & Value \\
\hline Initialization & matched Standard / TailSFT SFT checkpoints per dataset (Table 1) \\
\hline Algorithm & GRPO, group-relative advantage, no KL \\
\hline Training prompts & MBPP+ train (100-problem held-out validation) \\
\hline Reward & full MBPP+ unit-test pass (1/0) \\
\hline Rollout group size $n$ & searched \{2, 4\}; reported 4 \\
\hline Actor learning rate & searched $\left\{5 \times 10^{-6}, 1 \times 10^{-5}, 2 \times 10^{-5}\right\}$; reported $2 \times 10^{-5}$ \\
\hline Max prompt / response length & 1024 / 1024 \\
\hline Epochs & 10 \\
\hline Prompt batch / mini-batch & 128 / 128 \\
\hline Checkpoint selection & highest validation pass rate \\
\hline Evaluation & EvalPlus MBPP+ \((n=378)\); 2048-token, temp 1.0, top- $p 1.0$, 16 samples × 3 seeds \\
\hline
\end{tabular}

**Table 27: Code GRPO configuration. All runs share the fixed settings; the rollout group size and actor learning rate are searched, and Table 28 reports the tuned configuration for each dataset.**
\end{table}

\begin{table}
\begin{tabular}[t]{|l|l|l|l|l|l|l|l|l|}
\hline \multirow[b]{2}{*}{SFT data (TailSFT filter)} & \multirow[b]{2}{*}{$n$} & \multirow[b]{2}{*}{LR} & \multicolumn{3}{|c|}{pass@1} & \multicolumn{3}{|c|}{pass@16} \\
\hline & & & Standard & TailSFT & $\Delta$ & Standard & TailSFT & $\Delta$ \\
\hline BigCode (static $f=0.25$ ) & 4 & $2 \times 10^{-5}$ & 69.57 ± 0.29 & 73.50 ± 0.70 & +3.93 & 75.93 ± 0.00 & 78.66 ± 0.15 & +2.73 \\
\hline Magicoder (static $f=0.25$ ) & 2 & $1 \times 10^{-5}$ & 64.08 ± 0.32 & 66.02 ± 0.55 & +1.94 & 75.84 ± 0.31 & 79.28 ± 0.55 & +3.44 \\
\hline Magicoder (static $f=0.25$ ) & 4 & $2 \times 10^{-5}$ & 70.52 ± 0.25 & 73.24 ± 0.09 & +2.72 & 78.22 ± 0.61 & 80.60 ± 0.55 & +2.38 \\
\hline OCI (ramp 0→0.5) & 4 & $2 \times 10^{-5}$ & 74.67 ± 0.08 & 76.30 ± 0.05 & +1.62 & 84.22 ± 0.40 & 84.04 ± 0.15 & -0.18 \\
\hline
\end{tabular}

**Table 28: Code GRPO (MBPP+, $n=378$, 2048-token generation) from the matched Standard and TailSFT initializations of Table 1, at tuned configurations. $n$ is the GRPO rollout group size and LR the actor learning rate. Values are percent, mean ± standard deviation over three seeds; $\Delta$ is the TailSFT improvement, with yellow marking a change within one standard deviation.**
\end{table}

\begin{table}
\begin{tabular}[t]{|l|l|l|l|l|}
\hline SFT dataset & Benchmark & $\left|\mathcal{R}_{0}\right|$ & $\rho_{16}$ & \(\Delta(\mathrm{pp})\) \\
\hline \multirow[t]{3}{*}{OMI} & AIME & 16 & 7.60 & +9.92 \\
\hline & MATH-500 L5 & 31 & 2.28 & +3.23 \\
\hline & OMEGA-500 & 140 & 0.90 & -0.95 \\
\hline \multirow[t]{5}{*}{BigCode} & CruxEval-I & 199 & 1.63 & +3.02 \\
\hline & CruxEval-O & 237 & 2.87 & +28.69 \\
\hline & HumanEval+ & 29 & 1.39 & +9.20 \\
\hline & LiveCodeBench & 105 & 1.37 & +3.81 \\
\hline & MBPP+ & 50 & 1.30 & +14.00 \\
\hline \multirow[t]{5}{*}{Magicoder} & CruxEval-I & 199 & 2.26 & +12.40 \\
\hline & CruxEval-O & 237 & 0.73 & +18.42 \\
\hline & HumanEval+ & 29 & 1.04 & -0.00 \\
\hline & LiveCodeBench & 105 & 1.22 & +1.59 \\
\hline & MBPP+ & 50 & 0.77 & +11.33 \\
\hline \multirow[t]{5}{*}{OCI} & CruxEval-I & 199 & 1.35 & +6.37 \\
\hline & CruxEval-O & 237 & 0.55 & +9.00 \\
\hline & HumanEval+ & 29 & 0.23 & -6.32 \\
\hline & LiveCodeBench & 105 & 0.97 & -1.27 \\
\hline & MBPP+ & 50 & 0.54 & +2.00 \\
\hline
\end{tabular}

**Table 29: Exact values plotted in Figure 4: the base-reachable set size $\left|\mathcal{R**_{0}\right|$, the coverage ratio $\rho_{16}$ (Definition 4.1), and the coverage gain $\Delta$ (mean empirical pass@16 difference of TAILSFT over standard SFT on $\mathcal{R}_{0}$, in percentage points), for each (SFT dataset, benchmark) pair.}
\end{table}

![Figure](./malladiTailSFTFilteredFineTuning2026_images/b0c2e9de-14c4-464c-9bb3-a4879161b9bf-39_1093_1620_708_253.jpg)

**Figure 8: GRPO training reward from Standard and TailSFT initializations. Mean training reward (critic/rewards/mean) versus GRPO step for the four main-table settings. The horizontal dashed line marks the Standard initialization's first-step reward. TailSFT begins below Standard but recovers Standard's starting reward within a small fraction of training and rises at least as fast thereafter.**

