# 2 Background

The original RLHF pipeline for LLMs proposed in (Ziegler et al., 2020) consists of three stages:
(1) SFT Stage: A pre-trained LM is instruction-tuned using a dataset consisting of a given instruction prompt, and (typically) a human-written completion. The LM/policy is trained with a cross-entropy loss over the completion only. Often, the SFT model, denoted as $\pi^{\text {sft }}$ is used to initialize both the reward model and the RLHF policy.
$$
\mathcal{L}_{R M}=-\log \sigma\left(r_{\phi}\left(x, y_{+}\right)-r_{\phi}\left(x, y_{-}\right)\right)
$$
where $\sigma$ denotes the logistic function.
(3) RL Stage: In this stage, the reward model is used to provide online feedback in the optimization of the policy with the following objective:
$$
\begin{aligned}
& \max _{\pi_{\theta}} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_{\theta}(. \mid x)}\left[r_{\phi}(x, y)-\beta p_{K L}\right], \\
& \text { with } p_{\mathrm{KL}}=D_{\mathrm{KL}} \pi_{\theta}(. \mid x) \| \pi_{\mathrm{ref}}(. \mid x)
\end{aligned}
$$
where $\beta$ is meant to control the distance from the initial policy, $\pi_{\text {ref }}$ during the optimization of \(r_{\theta}(x, y)\) as proposed in (Stiennon et al., 2022). The KL-penalty $p_{\mathrm{KL}}$ is crucial as penalty-free optimization of the reward model leads to degradation in the coherence of the model. Optimizing this objective is equivalent to maximizing the following KL-shaped reward in expectation:
$$
R(x, y)=r_{\phi}(x, y)-\beta \log \frac{\pi_{\theta}(y \mid x)}{\pi_{\mathrm{ref}}(y \mid x)}
$$

While reinforcement learning approaches share the components above, techniques differ in the formulation of the reward. To understand these differences, we introduce PPO and distinct alternatives such as REINFORCE and REINFORCE Leave-One-Out in the following sections.

## 2.1 PPO

When using PPO in the RL stage, the initial state is determined by the prompt, each generated token is modeled as an action, and partial sequences are seen as states, with a discount factor \((\gamma \in[0,1])\) of 1 used. In this framework, only generating the <EOS> token carries a reward as output by the reward model which is combined with KL penalty, while for all other tokens in the vocabulary, only the KL component is non-zero:
$$
R(x, y)=\sum_{t=1}^{T} R_{t}\left(x, y_{t}\right)
$$
where $y_{t}$ denotes the $t$-th token of $y, T$ the number of tokens in the trajectory, and $R_{i}$ the correspondingly shaped reward.

In practice, the following token-level clipped objective is used in PPO:
$$
\begin{aligned}
& \min \left(f\left(y_{t} \mid s_{t}\right) \hat{A}_{\lambda}\left(y_{t}, s_{t}\right),\right. \\
& \left.\qquad \operatorname{clip}_{1-\epsilon}^{1+\epsilon}\left(f\left(y_{t} \mid s_{t}\right)\right) \hat{A}_{\lambda}\left(y_{t}, s_{t}\right)\right) \\
& \text { with } f\left(y_{t} \mid s_{t}\right)=\frac{\pi_{\theta}\left(y_{t} \mid s_{t}\right)}{\pi_{\text {old }}\left(y_{t} \mid s_{t}\right)},
\end{aligned}
$$
where $s_{t}=\left\{y_{<t}, x\right\}$ represents the state i.e. context at generation step $t$ that is composed of the history of generated tokens $y_{<t}$ and the given prompt $x, \pi_{\text {old }}$ is an older policy (not the same as $\pi_{\text {ref }}$ ), and \(\hat{A}\left(y_{t}, s_{t}\right)\) is the estimated advantage function, and $\epsilon$ is the clipping ratio. The advantage function \(\hat{A}\left(y_{t}, s_{t}\right)\) is estimated using Generalized Advantage Estimation (GAE) (Schulman et al., 2018), of generating token (action) $y_{t}$, at partial completion (state) at token $t-1$ of the generation.

## 2.2 REINFORCE

Given that in LLM applications, \(r(x, y)\) is only obtained at the end of the full sequence, it may be more appropriate to model the entire generation as a single action, as opposed to each token. Although it has not been explored in the context of LLM alignment, modeling the full completion as a single action, as in the bandit formulation, allows using the REINFORCE estimator (Kreutzer et al., 2017; Nguyen et al., 2017a; Williams, 1992). This allows for back-propagating through the discrete action (generation) space, and directly optimize the KL-shaped reward objective for the entire sequence.
$$
\mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_{\theta}(. \mid x)}\left[R(y, x) \nabla_{\theta} \log \pi_{\theta}(y \mid x)\right]
$$

To improve learning, one can reduce the variance of the estimator in Eq. 7, while keeping it unbiased, by subtracting a baseline $b$ that has high covariance with the stochastic gradient estimate of Eq. 7 (Williams, 1992; Mnih and Gregor, 2014):
$$
\mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_{\theta}(. \mid x)}\left[(R(y, x)-b) \nabla_{\theta} \log \pi_{\theta}(y \mid x)\right]
$$
With a strong parameter-free choice for the baseline being the moving average of all rewards throughout training (Williams, 1992):
$$
b_{\mathrm{MA}}=\frac{1}{S} \sum_{s} R\left(x^{s}, y^{s}\right)
$$
Where $S$ is the number of training steps, and \(\left(x^{s}, y^{s}\right)\) is the prompt-completion pair at the step $s$.

## 2.3 REINFORCE Leave-One-Out (RLOO)

The baseline in Eq. 9 is simple to implement and computationally cheap. However, it can be improved upon if we have access to multiple online samples, that can be used for further unbiased variance reduction: (1) The rewards for each sample can serve all other samples as a baseline. (2) Policy updates can be done on an average of gradient estimates for each sample, resulting in a variance-reduced multi-sample Monte-Carlo (MC) estimate. This is the intuition behind the REIN-FORCE Leave-One-Out (RLOO) estimator, proposed by (Kool et al., 2019) ${ }^{1}$ :
$$
\begin{aligned}
& \frac{1}{k} \sum_{i=1}^{k}\left[R\left(y_{(i)}, x\right)-\frac{1}{k-1} \sum_{j \neq i} R\left(y_{(j)}, x\right)\right] \\
& \nabla \log \pi\left(y_{(i)} \mid x\right) \text { for } y_{(1)}, \ldots, y_{(k)} \stackrel{\text { i.i.d }}{\sim} \pi_{\theta}(. \mid x)
\end{aligned}
$$
$\mathrm{RLOO}_{k}$ considers each \(y_{(i)}\) individually and uses the remaining $k-1$ samples to create an unbiased estimate of the expected return for the prompt, akin to a parameter-free value-function, but estimated at each training step. This is a much more effective baseline (as our experiments show) than $b_{M A}$ since it's created on-the-fly for each sample and at each training step, but comes at a cost of increased sampling time during training.

## 2.4 Alternatives to RL in Preference Training

Iterative Fine-tuning Iterative fine-tuning methods use the trained reward model to rank completions of online or offline sampled prompts, and then iteratively fine-tune the policy on a selected subset (Gulcehre et al., 2023; Dong et al., 2023). We benchmark Reward rAnked FineTuning (RAFT; Dong et al., 2023), a simple cross-entropy loss is used on the best-ranked completion out of $k$ online samples, based on \(R(x, y)\) or \(r(x, y)\). We note that RAFT does not make full use of all samples because it only optimizes using filtered top-ranked samples. In contrast, RLOO fully leverages constructing a baseline and a multi-sample MC estimate for the policy gradient.

Direct Preference Optimization (DPO) Unlike other methods, DPO (Rafailov et al., 2023) skips the reward modeling stage in the traditional RLHF pipeline and uses preference pairs to directly optimize the policy with the following loss:
$$
-\log \sigma\left(\beta \log \frac{\pi_{\theta}\left(y_{+} \mid x\right)}{\pi_{\text {ref }}\left(y_{+} \mid x\right)}-\beta \log \frac{\pi_{\theta}\left(y_{-} \mid x\right)}{\pi_{\text {ref }}\left(y_{-} \mid x\right)}\right)
$$

