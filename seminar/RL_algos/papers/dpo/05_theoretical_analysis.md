# 5 Theoretical Analysis of DPO

In this section, we give further interpretation of the DPO method, provide theoretical backing, and relate advantages of DPO to issues with actor critic algorithms used for RLHF (such as PPO [39]).

## 5.1 Your Language Model Is Secretly a Reward Model

DPO is able to bypass both fitting an explicit reward and performing RL to learn the policy using a single maximum likelihood objective. Note the optimization objective Eq. 5 is equivalent to a Bradley-Terry model with a reward parameterization \(r^{*}(x, y)=\beta \log \frac{\pi_{\theta}^{*}(y \mid x)}{\pi_{\text {ref }}(y \mid x)}\) and we optimize our parametric model $\pi_{\theta}$, equivalently to the reward model optimization in Eq. 2 under the change of variables. In this section we will build the theory behind this reparameterization, show that it does not constrain the class of learned reward models, and allows for the exact recovery of the optimal policy. We begin with by defining an equivalence relation between reward functions.
Definition 1. We say that two reward functions \(r(x, y)\) and \(r^{\prime}(x, y)\) are equivalent iff \(r(x, y)-r^{\prime}(x, y)=f(x)\) for some function $f$.

It is easy to see that this is indeed an equivalence relation, which partitions the set of reward functions into classes. We can state the following two lemmas:
Lemma 1. Under the Plackett-Luce, and in particular the Bradley-Terry, preference framework, two reward functions from the same class induce the same preference distribution.
Lemma 2. Two reward functions from the same equivalence class induce the same optimal policy under the constrained RL problem.

The proofs are straightforward and we defer them to Appendix A.5. The first lemma is a well-known under-specification issue with the Plackett-Luce family of models [32]. Due to this under-specification,
we usually have to impose additional identifiability constraints to achieve any guarantees on the MLE estimates from Eq. 2 [4]. The second lemma states that all reward functions from the same class yield the same optimal policy, hence for our final objective, we are only interested in recovering an arbitrary reward function from the optimal class. We prove the following Theorem in Appendix A.6:
Theorem 1. Under mild assumptions, all reward classes consistent with the Plackett-Luce (and Bradley-Terry in particular) models can be represented with the reparameterization \(r(x, y)=\beta \log \frac{\pi(y \mid x)}{\pi_{\text {ref }}(y \mid x)}\) for some model \(\pi(y \mid x)\) and a given reference model \(\pi_{\text {ref }}(y \mid x)\).

Proof Sketch. Consider any reward function \(r(x, y)\), which induces a corresponding optimal model \(\pi_{r}(y \mid x)\), specified by Eq. 4. We will show that a reward function from the equivalence class of $r$ can be represented using the reparameterization given above. We define the projection $f$ as
$$
f\left(r ; \pi_{\mathrm{ref}}, \beta\right)(x, y)=r(x, y)-\beta \log \sum_{y} \pi_{\mathrm{ref}}(y \mid x) \exp \left(\frac{1}{\beta} r(x, y)\right)
$$
The operator $f$ simply normalizes the reward function with the logarithm of the partition function of $\pi_{r}$. Since the added normalization term is only a function of the prefix \(x, f\left(r ; \pi_{\text {ref }}, \beta\right)(x, y)\) is a reward function in the equivalence class of \(r(x, y)\). Finally, replacing $r$ with the RHS of Eq. 5 (which holds for any reward function), we have \(f\left(r ; \pi_{\text {ref }}, \beta\right)(x, y)=\beta \log \frac{\pi_{r}(y \mid x)}{\pi_{\text {ref }}(y \mid x)}\). That is, the projection $f$ produces a member of the equivalence class of $r$ with the desired form, and we do not lose any generality in our reward model from the proposed reparameterization. $\square$

We can alternatively view Theorem 1 as specifying exactly which reward function within each equivalence class the DPO reparameterization selects, that is, the reward function satisfying:
$$
\sum_{y} \underbrace{\pi_{\text {ref }}(y \mid x) \exp \left(\frac{1}{\beta} r(x, y)\right)}_{=\pi(y \mid x), \text { using Thm. } 1 \text { reparam. }}=1,
$$
i.e., \(\pi(y \mid x)\) is a valid distribution (probabilities are positive and sum to 1). However, following Eq. 4, we can see that Eq. 9 is the partition function of the optimal policy induced by the reward function \(r(x, y)\). The key insight of the DPO algorithm is that we can impose certain constraints on the under-constrained Plackett-Luce (and Bradley-Terry in particular) family of preference models, such that we preserve the class of representable reward models, but explicitly make the optimal policy in Eq. 4 analytically tractable for all prompts $x$.

## 5.2 Instability of Actor-Critic Algorithms

We can also use our framework to diagnose instabilities with standard actor-critic algorithms used for the RLHF, such as PPO. We follow the RLHF pipeline and focus on the RL fine-tuning step outlined in Section 3. We can draw connections to the control as inference framework [22] for the constrained RL problem outlined in 3. We assume a parameterized model \(\pi_{\theta}(y \mid x)\) and minimize \(\mathbb{D}_{\mathrm{KL}}\left[\pi_{\theta}(y \mid x) \| \pi^{*}(y \mid x)\right]\) where $\pi^{*}$ is the optimal policy from Eq. 7 induced by the reward function \(r_{\phi}(y, x)\). With some algebra this leads to the optimization objective:
$$
\max _{\pi_{\theta}} \mathbb{E}_{\pi_{\theta}(y \mid x)}[\underbrace{r_{\phi}(x, y)-\beta \log \sum_{y} \pi_{\text {ref }}(y \mid x) \exp \left(\frac{1}{\beta} r_{\phi}(x, y)\right)}_{f\left(r_{\phi}, \pi_{\text {ref }}, \beta\right)}-\underbrace{\beta \log \frac{\pi_{\theta}(y \mid x)}{\pi_{\text {ref }}(y \mid x)}}_{\mathrm{KL}}]
$$
This is the same objective optimized in prior works [51, 40, 1, 28] using the DPO-equivalent reward for the reward class of $r_{\phi}$. In this setting, we can interpret the normalization term in \(f\left(r_{\phi}, \pi_{\text {ref }}, \beta\right)\) as the soft value function of the reference policy $\pi_{\text {ref }}$. While this term does not affect the optimal solution, without it, the policy gradient of the objective could have high variance, making learning unstable. We can accommodate for the normalization term using a learned value function, but that can also be difficult to optimize. Alternatively, prior works have normalized rewards using a human completion baseline, essentially a single sample Monte-Carlo estimate of the normalizing term. In contrast the DPO reparameterization yields a reward function that does not require any baselines.

![Figure](./dpo_images/6686fa43-2a2f-476b-bd78-4a8860263ebd-07_501_1360_256_382.jpg)

**Figure 2: Left. The frontier of expected reward vs KL to the reference policy. DPO provides the highest expected reward for all KL values, demonstrating the quality of the optimization. Right. TL;DR summarization win rates vs. human-written summaries, using GPT-4 as evaluator. DPO exceeds PPO's best-case performance on summarization, while being more robust to changes in the sampling temperature.**

