# 3 TailSFT: Supervised Fine-Tuning for Coverage

In this section, we develop TAILSFT, a sequence-level filtering method designed to direct SFT updates toward improving coverage. The method combines two ideas: stopping updates on responses that are already sufficiently
\begin{lstlisting}[mathescape=true]
Algorithm 1 TailSFT
Require: Initial policy $\pi_{0}$, SFT data $\mathcal{D}$, filtering schedule $\left\{\gamma_{t}\right\}$
Record \(\ell_{i}^{0} \leftarrow \ell_{i}\left(\pi_{0}\right)\) for every \(\left(x_{i}, y_{i}\right) \in \mathcal{D}\)
for each training step $t$ do
Sample a batch $\mathcal{B}_{t}$
Compute \(\ell_{i}^{t} \leftarrow \ell_{i}\left(\pi_{t}\right)\) for each \(\left(x_{i}, y_{i}\right) \in \mathcal{B}_{t}\)
Let $\mathcal{F}_{t} \subseteq \mathcal{B}_{t}$ be the $\gamma_{t}$ fraction of sequences with the smallest $\ell_{i}^{t}-\ell_{i}^{0}$
Set $\pi_{t+1}$ by taking a gradient step from $\pi_{t}$ on
\(\mathcal{L}_{t}(\pi)=\frac{\sum_{i \in \mathcal{B}_{t} \backslash \mathcal{F}_{t}}-\log \pi\left(y_{i} \mid x_{i}\right)}{\sum_{i \in \mathcal{B}_{t} \backslash \mathcal{F}_{t}}\left|y_{i}\right|}\)
\end{lstlisting}

well modeled via filtering, and determining which responses to filter relative to the initial policy. We first introduce the algorithm, then isolate the effect of filtering in a controlled graph-navigation task, and finally analyze why the initial policy provides a useful reference for filtering.

## 3.1 The TailSFT Algorithm

Coverage at scale $N$ depends on whether useful responses receive enough probability to be reached via repeated sampling. Once a response is already well covered at that scale, increasing its likelihood further does not reduce $\operatorname{Cov}_{N}$. Improving coverage instead requires increasing the probability of responses that remain under-modeled. Standard SFT makes no such distinction and continues to increase the likelihood of every training response.

TailSFT reduces training on responses that have already improved substantially and concentrates subsequent updates on the remaining tail. Since \(\pi_{\mathrm{D}}(y \mid x)\) is unknown, we cannot determine directly which responses are already covered. Instead, we compare each response's current loss to its loss under the initial policy and filter the responses whose losses have decreased the most.

As described in Algorithm 1, TailSFT redirects training away from responses whose losses have decreased the most and toward responses that remain under-modeled. Filtering is applied at the sequence level. The length-normalized loss is used only to determine which sequences are filtered; optimization uses the standard token-averaged cross-entropy over target tokens in the retained sequences. The filtering fraction $\gamma_{t}$ may be fixed or vary over training.

Filtering itself does not require using the initial policy as a reference. We therefore consider two alternatives alongside TailSFT: absolute filtering, which stops training on an example once its loss falls below a fixed threshold, and quantile filtering, which filters the lowest-loss fraction of each batch. These alternatives allow us to separate the benefit of filtering itself from the benefit of measuring progress relative to the initial policy $\pi_{0}$.

## 3.2 Warm-Up: Graph Navigation

We study whether halting updates on an example once it is fit by the model can improve coverage, even at the cost of the standard SFT objective and pass@1. To do so, this section considers a controlled setting in which the distribution of rewarding responses is known exactly. Following Chen et al. (2026a), each prompt $x$ specifies a layered directed graph with source $s$, target $t$, and exactly eight valid $s \rightarrow t$ paths. Pretraining pairs each graph with a uniformly sampled valid path, while the SFT data deterministically selects and rewards a single path (see Figure 2). Thus, pretraining teaches generic graph navigation, while SFT teaches the task-specific path-selection rule. Since the target policy is deterministic, pass@ $K$ is exactly the probability that the rewarded path is contained in $K$ samples, yielding a finite-sample estimate of coverage. Full details of the graph construction and data-generating process appear in Appendix C.

Using a GPT-2-style architecture and a shared pretraining setup, we vary only the SFT objective. We compare standard SFT with absolute filtering, quantile filtering, and TailSFT, which uses offset quantile filtering (as in Algorithm 1). For each filtering variant, we sweep its single filtering hyperparameter and report the setting with the highest final pass@8; complete training details and sweeps appear in Appendix C.

![Figure](./malladiTailSFTFilteredFineTuning2026_images/b0c2e9de-14c4-464c-9bb3-a4879161b9bf-05_484_1593_235_260.jpg)

**Figure 2: TailSFT trades single-sample accuracy for better coverage under repeated sampling. The SFT distribution contains majority and minority shards with different rewarding paths. Standard SFT over-anchors on the majority shard, achieving high pass@1 but no improvement as $K$ grows. A filtered objective, illustrated here by TailSFT, retains probability on both paths, yielding lower pass@1 but substantially better pass@K scaling.**

![Figure](./malladiTailSFTFilteredFineTuning2026_images/b0c2e9de-14c4-464c-9bb3-a4879161b9bf-05_455_1656_962_233.jpg)

**Figure 3: Filtering improves pass@ $K$ at the expense of cross entropy and pass@1. Results on the graph-navigation task. Standard SFT achieves the lowest cross-entropy and highest pass@1, while all three filtering methods achieve substantially higher pass@8. See Appendix C for details.**

The results in Figure 3 reveal a clear divergence between the standard SFT objective and coverage. Standard SFT achieves the best cross-entropy and pass@1, yet all three filtering methods achieve substantially better pass@8, which we use as a surrogate for coverage. Thus, improving cross-entropy and single-sample accuracy can come at the expense of the probability that a rewarded response appears within a finite rollout budget. Filtering mitigates this failure mode by reducing further updates on responses that are already well modeled.

The graph task therefore provides clean evidence for the first design choice underlying TailSFT: filtering already-fit examples can improve coverage. It is, however, too simple a setting to determine whether the filtering criterion should depend on the initial policy. In fact, TailSFT is slightly weaker than other filtering variants in this setting, due to homogeneity across prompts. Under the ideal pretrained policy, every prompt has the same possible loss reduction (i.e., log 8), so the reference loss provides no information for distinguishing which examples have more room to improve. Imperfections in pretraining therefore introduce noise into the offset score without providing useful signal. We discuss this effect further in Appendix C.

## 3.3 Theoretical Analysis

We next turn to a setting where the initial policy carries information about the response distribution, and ask whether filtering rules can preserve it. To study this question, we turn to theoretical analysis. Let $S$ denote the set of rewarding responses, and suppose the target policy is obtained by discarding all responses outside $S$ and renormalizing the initial policy:
$$
\pi^{\star}(y) \propto \pi_{0}(y) \mathbf{1}\{y \in S\} .
$$

Thus, among rewarding responses, the target preserves the relative preferences already present in $\pi_{0}$. The initial model may place too much total probability on unrewarding responses, but it still contains information about how probability should be distributed among the rewarding ones.

We compare two ways to decide when an example should stop contributing to the SFT objective. Let \(\ell_{\pi}(x, y)=-\log \pi(y \mid x)\) denote the sequence-level SFT loss. With an absolute filtering constant $\alpha$, we use
$$
\ell_{\mathrm{abs}}(\pi ; x, y)=\left[\ell_{\pi}(x, y)+\log (\alpha)\right]_{+} .
$$
An example stops contributing once \(\pi(y \mid x) \geq \alpha\). Because the same threshold is applied to every response, this criterion ignores how likely the response was under the initial policy.

TailSFT instead measures progress relative to the initial policy $\pi_{0}$. The corresponding offset loss is
$$
\ell_{\mathrm{off}}(\pi ; x, y)=\left[\ell_{\pi}(x, y)-\ell_{\pi_{0}}(x, y)+\log \beta\right]_{+},
$$
which stops updating a response once
$$
\pi(y \mid x) \geq \beta \pi_{0}(y \mid x) .
$$
The stopping point therefore adapts to the probability that $\pi_{0}$ already assigns to each response rather than imposing a common final threshold.

Theorem 3.1 (Informal). Let $\pi_{\mathrm{ERM}}, \pi_{\mathrm{ABS}, \alpha}$, and $\pi_{\mathrm{OFF}, \beta}$ denote the policies obtained using standard SFT, absolute filtering, and offset filtering, respectively. For any initial policy, rewarding set, and SFT sample, offset filtering can be tuned to achieve coverage no worse than standard SFT or the best absolute threshold:
$$
\inf _{\beta} \operatorname{Cov}_{N}\left(\pi^{\star} \| \pi_{\mathrm{OFF}, \beta}\right) \leq \min \left\{\operatorname{Cov}_{N}\left(\pi^{\star} \| \pi_{\mathrm{ERM}}\right), \inf _{\alpha} \operatorname{Cov}_{N}\left(\pi^{\star} \| \pi_{\mathrm{ABS}, \alpha}\right)\right\} .
$$
The inequality can be strict, and absolute filtering can be worse than standard SFT.
The theorem formalizes the role of the initial policy as more than merely an initialization. In this setting, $\pi_{0}$ contains useful information about how probability should be distributed among rewarding responses. Standard SFT can overwrite this structure by fitting the empirical frequencies of a finite SFT sample, while absolute filtering pushes observed responses toward a common probability threshold regardless of where they started. Offset filtering instead uses each response's initial probability to determine when further training is unnecessary. By preserving more of the useful structure already present in $\pi_{0}$, it can achieve strictly better coverage. The full statement and proof appear in Appendix B.

Connections to prior work. TailSFT is most closely related to direct coverage optimization and loss-based data filtering. Chen et al. (2025) introduce the direct coverage optimization loss
$$
\ell_{\mathrm{DCO}}(\pi ; x, y)=-\log \left(1-(1-\pi(y \mid x))^{K}\right),
$$
which directly optimizes pass@ $K$ when $y$ is the final solution. For large $K$, its gradient becomes small once \(\pi(y \mid x)\) is sufficiently large, making it a soft analogue for filtering without reference to the initial policy. Theorem 3.1 shows that, in our stylized setting, offset filtering can achieve coverage no worse than the best absolute threshold and can be strictly better. TailSFT is also related to RHO-LOSS and subsequent variants, though these methods were designed and tested primarily to improve pretraining efficiency (Mindermann et al., 2022; Thirukovalluru et al., 2024; Lin et al., 2024; Zhao et al., 2026b). These approaches prioritize examples or tokens according to their potential loss reduction. TailSFT, by contrast, is designed to improve coverage for a subsequent RL stage and applies filtering at the sequence level. See Appendix A for further discussion.

