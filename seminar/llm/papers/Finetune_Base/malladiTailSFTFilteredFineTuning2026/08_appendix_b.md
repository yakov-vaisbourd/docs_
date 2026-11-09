# B Theoretical Analysis: TailSFT in the expert conditioning setting

We consider a stylized setup for supervised fine-tuning called expert conditioning where the SFT data distribution and optimal policy for the downstream task are defined by conditioning the pretrained model output distribution on some event. For theoretical analysis, we focus on a simplified setting where there is no context or prompt, and so all policies are elementary distributions over responses $y \in \mathcal{Y}$. Let \(\pi_{\text {ref }} \in \Delta(\mathcal{Y})\) be the pretrained model and let $S \subset \mathcal{Y}$ be some subset of responses. In expert conditioning, we define the expert policy as \(\pi^{\star}(y) \propto \pi_{\mathrm{ref}}(y) \cdot \mathbf{1}\{y \in S\}\). Note that this setup is closely related to formulations of RLHF where the expert is defined as \(\pi_{\text {ref }}(y) \cdot \exp \left(R^{\star}(y) / \beta\right)\) for some reward function $R^{\star}$; formally, expert conditioning is equivalent to this formulation in the limit where $\beta \rightarrow 0$.

We are given $n$ samples $y_{1}, \ldots, y_{n} \sim \pi^{\star}$ and obtain a policy by solving a certain optimization problem defined in terms of a loss function over the sample. The three loss functions are
$$
\begin{array}{ll}
\mathrm{ERM}: & L_{\mathrm{ERM}}(\pi):=\frac{1}{n} \sum_{i=1}^{n}-\log \left(\pi\left(y_{i}\right)\right) \\
\mathrm{ABS}: & L_{\mathrm{ABS}, \alpha}(\pi):=\frac{1}{n} \sum_{i=1}^{n} \max \left(-\log \left(\pi\left(y_{i}\right)\right)+\log (\alpha), 0\right) \\
\mathrm{OFF}: & L_{\mathrm{OFF}, \beta}(\pi):=\frac{1}{n} \sum_{i=1}^{n} \max \left(-\log \left(\pi\left(y_{i}\right)\right)+\log \left(\beta \cdot \pi_{\mathrm{ref}}\left(y_{i}\right)\right), 0\right)
\end{array}
$$
The first objective is standard empirical risk minimization on the cross entropy loss. The second and third are TailSFT variants with absolute and relative clipping respectively. Indeed for absolute clipping, we ignore sample $y_{i}$ if \(\pi\left(y_{i}\right) \geq \alpha\) and for relative clipping we ignore $y_{i}$ if \(\pi\left(y_{i}\right) \geq \beta \pi_{\text {ref }}\left(y_{i}\right)\). The formal optimization problem for a particular loss function is:
$$
\underset{\pi \in \Delta(\mathcal{Y})}{\operatorname{minimize}} \operatorname{KL}\left(\pi\left\|\| \pi_{\mathrm{ref}}\right) \text { subject to } L(\pi)=\min _{\pi^{\prime}} L\left(\pi^{\prime}\right)\right.
$$
Thus we minimize the forward Kullback-Leibler (KL) divergence between the optimization variable $\pi$ and the reference policy $\pi_{\text {ref }}$ subject to $\pi$ being among the minimizers of the loss $L$. We use forward KL as simplification of the implicit bias of optimization, and use forward KL primarily due to its mode-covering behavior, so that coverage is preserved to the extent possible subject to minimizing the loss. We take $L$ to be $L_{\text {ERM }}, L_{\text {ABS }, \alpha}, L_{\text {OFF }, \beta}$ accordingly.

Formally, fixing the dataset, define $\pi_{\mathrm{ERM}}, \pi_{\mathrm{ABS}, \alpha}, \pi_{\mathrm{OFF}, \beta}$ to be the optimizers of Eq. (1) with loss functions $L_{\mathrm{ERM}}, L_{\mathrm{ABS}, \alpha}, L_{\mathrm{OFF}, \beta}$ respectively. We are interested in qualitatively understanding how well these optimizers
cover the expert policy $\pi^{\star}$, where recall that we define coverage as:
$$
\operatorname{Cov}_{N}(\pi):=\operatorname{Pr}_{y \sim \pi^{\star}}\left[\frac{\pi^{\star}(y)}{\pi(y)} \geq N\right]
$$
Our main result is as follows:
Theorem B.1. We have the following comparisons:
\begin{itemize}
\item[1.] Offset clipping is always preferred: For all $\pi_{\mathrm{ref}}, S$ and datasets:
$$
\inf _{\beta} \operatorname{Cov}_{N}\left(\pi_{\mathrm{OFF}, \beta}\right) \leq \inf _{\beta: \min _{\pi} L_{\mathrm{OFF}, \beta}(\pi)=0} \operatorname{Cov}_{N}\left(\pi_{\mathrm{OFF}, \beta}\right) \leq\left(\inf _{\alpha} \operatorname{Cov}_{N}\left(\pi_{\mathrm{ABS}, \alpha}\right)\right) \wedge \operatorname{Cov}_{N}\left(\pi_{\mathrm{ERM}}\right)
$$
\item[2.] Offset clipping can dominate: For $|\mathcal{Y}|$ sufficiently large, there exists a reference policy $\pi_{\mathrm{ref}}$ and expert subset $S$ such that with probability at least \(1-\operatorname{poly}\left(|\mathcal{Y}|^{-1}\right)\)
$$
\inf _{\beta} \operatorname{Cov}_{N}\left(\pi_{\mathrm{OFF}, \beta}\right)<\left(\inf _{\alpha} \operatorname{Cov}_{N}\left(\pi_{\mathrm{ABS}, \alpha}\right)\right) \wedge \operatorname{Cov}_{N}\left(\pi_{\mathrm{ERM}}\right)
$$
\item[3.] Absolute clipping can be worse than ERM: There exists a reference policy $\pi_{\mathrm{ref}}$ and expert subset $S$ such that with high probability
$$
\operatorname{Cov}_{N}\left(\pi_{\mathrm{ERM}}\right)<\inf _{\alpha: \min _{\pi}} \operatorname{Lov}_{\mathrm{ABS}, \alpha}(\pi)=0 \operatorname{Cov}_{N}\left(\pi_{\mathrm{ABS}, \alpha}\right)
$$
That is, in the regime where zero loss is achievable for absolute clipping, we can have that empirical risk minimization strictly dominates absolute loss clipping.
\end{itemize}

Note that the first claim implies that offset clipping-even when restricted to the parameter regime where zero loss is achievable-is never worse than empirical risk minimization.

# B. 1 Proof of Theorem B. 1

First we derive a structural characterization of the solutions of Eq. (1). Next, we use this characterization to understand the coverage of the policies $\pi_{\mathrm{ERM}}, \pi_{\mathrm{ABS}, \alpha}, \pi_{\mathrm{OFF}, \beta}$. Finally we establish the comparisons in the theorem statement.

Lemma B. 1 (Structure of KL projections). Let $\left\{f_{y}: y \in \mathcal{Y}\right\} \subset[0,1]$ satisfy $\sum_{y} f_{y} \leq 1$. Consider the optimization problem
$$
\widehat{\pi} \leftarrow \underset{\pi \in \Delta(\mathcal{Y})}{\arg \min } \mathrm{KL}\left(\pi\left\|\| \pi_{\mathrm{ref}}\right) \text { subject to } \forall y: \pi(y) \geq f_{y} .\right.
$$
Then the minimizer $\widehat{\pi}$ is unique and given by
$$
\widehat{\pi}(y)=\max \left(f_{y}, \lambda \pi_{\mathrm{ref}}(y)\right) \quad \text { where } \quad \sum_{y} \max \left(f_{y}, \lambda \pi_{\mathrm{ref}}(y)\right)=1 .
$$

Proof of Lemma B.1. Observe that the optimization problem is strictly convex and feasible under the condition that $\sum_{y} f_{y} \leq 1$. We write the Lagrangian of the optimization problem as
$$
\sum_{y} \pi(y) \log \frac{\pi(y)}{\pi_{\mathrm{ref}}(y)}+\lambda\left(\sum_{y} \pi(y)-1\right)+\sum_{y} \mu_{y}(f(y)-\pi(y)) \quad \mu_{y} \geq 0 .
$$
The stationary conditions are
$$
\forall y: \log \frac{\pi(y)}{\pi_{\mathrm{ref}}(y)}+1+\lambda-\mu_{y}=0 .
$$

If \(\pi(y)>f_{y}\), so the constraint is inactive, then by complementary slackness we have $\mu_{y}=0$, so \(\pi(y)=\) \(\pi_{\mathrm{ref}}(y) \cdot \exp (-1-\lambda)\) and this must be larger than $f_{y}$. Otherwise, we must have \(\pi(y)=f_{y}\) solving the stationary condition for $\mu_{y}$ gives
$$
\mu_{y}=\log \left(\pi(y) / \pi_{\mathrm{ref}}(y)\right)+1+\lambda=\log \frac{f_{y}}{\pi_{\mathrm{ref}}(y) \cdot \exp (-1-\lambda)}
$$
The constraint that $\mu_{y} \geq 0$ thus implies that \(f_{y} \geq \pi_{\text {ref }}(y) \cdot \exp (-1-\lambda)\). Taken together, this gives \(\pi(y)=\max \left(f_{y}, \pi_{\text {ref }}(y) \exp (-1-\lambda)\right)\) and $\lambda$ is chosen to normalize the distribution. $\square$

Lemma B. 2 (Structure of clipped minimizer). Let $\left\{f_{y}: y \in \mathcal{Y}\right\}$ let \(p \in \Delta(Y)\) and let \(\sum_{y} \mathbf{1}\{p(y)>0\} f_{y}>1\). Then the minimizer of
$$
\sum_{y \in \mathcal{Y}} p(y) \max \left(-\log \pi(y)+\log \left(f_{y}\right), 0\right)
$$
is unique and given by
$$
\hat{\pi}(y)=\min \left(f_{y}, \lambda p(y)\right) \quad \text { where } \quad \sum_{y} \min \left(f_{y}, \lambda p(y)\right)=1 .
$$

Proof. First observe that for any $y$ such that \(p(y)=0\) we will have \(\pi(y)=0\) as well, because the conditions on $f_{y}$ imply that we cannot achieve an objective value of zero, and we can always shift mass from actions with \(p(y)=0\) to those with \(p(y) \neq 0\) to reduce the loss. Thus, without loss of generality we can assume that \(p(y) \neq 0\) which also implies \(\pi(y) \neq 0\). Next, we write the Lagrangian:
$$
\sum_{y} p(y) \max \left(-\log \pi(y)+\log \left(f_{y}\right), 0\right)+\lambda\left(\sum_{y} \pi(y)-1\right)
$$
The stationary condition is that \(\frac{-p(y)}{\pi(y)}+\lambda=0\) if \(\pi(y)<f_{y}\). If \(\pi(y)=f_{y}\) ) the sub-differential for the loss term is in \([-p(y) / \pi(y), 0]\) and if \(\pi(y)>0\) we must have $\lambda=0$ to satisfy stationarity. However, since at least one $y$ must have \(\pi(y)<f_{y}\) we get that $\lambda>0$. Therefore no action can have \(\pi(y)>f_{y}\) and we have that \(\pi(y)=\min \left(f_{y}, \lambda p(y)\right)\). $\square$

Next we characterize the structure and the coverage of the $\pi_{\mathrm{OFF}, \beta}$.
Lemma B. 3 (Coverage of $\pi_{\mathrm{OFF}, \beta}$ ). For any dataset, let $\hat{S} \subseteq S$ be the actions observed in the dataset. We have
$$
\inf _{\beta \geq 0} \operatorname{Cov}_{N}\left(\pi_{\mathrm{OFF}, \beta}\right) \leq \inf _{\beta: \min _{\pi} \operatorname{LOFF}_{\mathrm{L}, \beta}(\pi)=0} \operatorname{Cov}_{N}\left(\pi_{\mathrm{OFF}, \beta}\right)= \begin{cases}0, & N \geq 1 / \pi_{\mathrm{ref}}(S) \\ \frac{\pi_{\mathrm{ref}}(S \backslash \hat{S})}{\pi_{\mathrm{ref}}(S)}, & 1<N<1 / \pi_{\mathrm{ref}}(S)\end{cases}
$$

Proof of Lemma B.3. If \(N \geq 1 / \pi_{\text {ref }}(S)\) then take $\beta=1$ so that $\pi_{\text {ref }}$ itself is a minimizer of $L_{\mathrm{OFF}, \beta}$, i.e., \(L_{\mathrm{OFF}, \beta}\left(\pi_{\mathrm{ref}}\right)=0\). With this choice of $\beta$, clearly $\pi_{\mathrm{ref}}$ is the solution to Eq. (1). To compute the coverage, observe that for every action \(y, \pi^{\star}(y) / \pi_{\text {ref }}(y)=1 / \pi_{\text {ref }}(S) \leq N\). Thus, in this case \(\operatorname{Cov}\left(\pi_{\text {ref }}\right)=0\) and with optimally tuned $\beta$, the KL projection onto the set of offset-loss minimizers preserves this coverage.

If \(1<N<1 / \pi_{\text {ref }}(S)\) we first translate the offset loss optimization problem into the form in Lemma B. 1 and then compute the normalizing constant $\lambda$ in the distribution. Observe that the offset loss only involves observed actions $y \in \hat{S}$. If $\beta$ is such that 0 offset loss is feasible, then we have
$$
L_{\mathrm{OFF}, \beta}(\pi)=0 \Leftrightarrow \forall y \in \hat{S}: \pi(y) \geq \beta \pi_{\mathrm{ref}}(y) \text { and } \forall y \notin \hat{S}: \pi(y) \geq 0
$$
Here the first condition arises from achieving zero loss, while the second condition arises because $\pi$ must be a distribution. Thus the offset loss version of Eq. (1) corresponds to taking \(f_{y}=\beta \pi_{\text {ref }}(y)\) if $y \in \hat{S}$ and $f_{y}=0$ otherwise in Lemma B.1. Next, observing that every action in the minimizing distribution has mass at least
\(\lambda \pi_{\text {ref }}(y)\) we get that $\lambda \leq 1$. Since $f_{y}=0$ for actions $y \notin \hat{S}$, we have that \(\pi_{\mathrm{OFF}, \beta}(y)=\lambda \pi_{\text {ref }}(y)\) for all $y \notin \hat{S}$. Such actions that are further supported by $\pi^{\star}$ are uncovered, since
$$
\frac{\pi^{\star}(y)}{\pi_{\mathrm{OFF}, \beta}(y)} \geq \frac{\pi^{\star}(y)}{\pi_{\mathrm{ref}}(y)}=\frac{1}{\pi_{\mathrm{ref}}(S)}>N
$$
The first inequality uses that $\lambda \leq 1$ and the second inequality is by our assumed regime for $N$. Thus we obtain for any feasible $\beta$ we obtain a coverage lower bound
$$
\operatorname{Cov}_{N}\left(\pi_{\mathrm{OFF}, \beta}\right) \geq \frac{\pi_{\mathrm{ref}}(S \backslash \hat{S})}{\pi_{\mathrm{ref}}(S)} .
$$
To show that this is achievable take \(\beta=\left(\pi_{\text {ref }}(S) N\right)^{-1}>1\) and observe that
$$
\beta \pi_{\mathrm{ref}}(S)=1 / N<1 .
$$
Thus if we set \(\pi(y)=\beta \pi_{\mathrm{ref}}(y)\) for $y \in S$ we achieve 0 loss and have \(\sum_{y} \pi(y)<1\); we can allocate the remaining mass arbitrarily to preserve 0 offset loss and obtain a distribution.

Since zero-loss is achievable, the KL projection ensures that \(\pi_{\mathrm{OFF}, \beta}(y) \geq \beta \pi_{\mathrm{ref}}(y)\) for all $y \in \hat{S}$ as this is required to achieve zero loss. By the choice of $\beta$ this actions are covered. On the other hand, all other actions have \(\pi_{\mathrm{OFF}, \beta}(y)=\lambda \pi_{\mathrm{ref}}(y) \leq \pi_{\mathrm{ref}}(y)\) using the argument above that $\lambda \leq 1$. These actions are all not covered under the condition that \(1 / \pi_{\text {ref }}(S)>N\). Thus this choice of $\beta$ achieves coverage exactly \(\pi_{\text {ref }}(S \backslash \hat{S}) / \pi_{\text {ref }}(S)\). $\square$

Proof of Claim 1 of Theorem B.1. We first argue about $\pi_{\mathrm{ERM}}$. Note that the ERM is unique and is exactly the empirical distribution over the sample. In particular, we have \(\pi_{\mathrm{ERM}}(S \backslash \hat{S})=0\), and therefore these actions are always uncovered, so \(\operatorname{Cov}_{N}\left(\pi_{\text {ERM }}\right) \geq \pi_{\text {ref }}(S \backslash \hat{S}) / \pi_{\text {ref }}(S)\).

Next we turn to $\pi_{\mathrm{ABS}, \alpha}$. Here we consider two cases. First consider the case that $\alpha$ is large enough such that zero loss is not achievable. In this case, if we have a policy $\pi$ that allocates mass to unobserved actions, we can always strictly improve the loss by moving that mass onto some observed action that is not already saturated, and such an action must exist if zero loss is not achievable. Thus the loss minimizer must allocate no mass on unobserved actions. In this case the same reasoning as we applied to analyze the ERM holds.

Next, consider that $\alpha$ is small enough such that zero loss is achievable. As with the offset loss, we can translate the optimization problem to the form in Lemma B. 1 and take $f_{y}=\alpha \mathbf{1}\{y \in \hat{S}\}$. However, as in the proof of Lemma B.3, we still require that $\lambda \leq 1$. As before when \(1<N<1 / \pi_{\text {ref }}(S)\), this implies that all unobserved actions in $S$ are uncovered, establishing a lower bound of \(\pi_{\text {ref }}(S \backslash \hat{S}) / \pi_{\text {ref }}(S)\) in this regime. $\square$

Proof of Claim 2 in Theorem B.1. We construct an instance with three types of arms with $|\mathcal{Y}|=1+n / 4+3 n$ for some $n \in \mathbb{N}$ such that $\mathcal{Y}=\{o\} \cup H \cup L$ where $|H|=n / 4$ and $L=3 n$. Define
$$
\pi_{\mathrm{ref}}(o)=\frac{1}{2}, \quad \pi_{\mathrm{ref}}(h)=\frac{1}{n}, h \in H, \quad \pi_{\mathrm{ref}}(\ell)=\frac{1}{12 n}, \ell \in L .
$$
Set $S=H \cup L$ such that
$$
\pi^{\star}(h)=\frac{2}{n}, h \in H, \quad \pi^{\star}(\ell)=\frac{1}{6 n}, \ell \in L
$$
Recall that $\hat{S}$ is the set of actions observed in the sample. The high probability event is the intersection of two events: (1) $|\hat{S}|$ is sufficiently large and (2) \(\left|\hat{H}^{(1)}\right|\) is sufficiently large, where \(\hat{H}^{(1)}\) is the set of actions in $H$ that are observed exactly once.
To control $|\hat{S}|$, we first calculate:
$$
\begin{aligned}
& \mathbb{E}|\hat{S} \cap H|=\sum_{h \in H} \operatorname{Pr}[h \in \hat{S}]=\frac{n}{4}\left(1-(1-2 / n)^{n}\right) \geq \frac{n}{4}\left(1-e^{-2}\right) \\
& \mathbb{E}|\hat{S} \cap L|=\sum_{\ell \in L} \operatorname{Pr}[\ell \in \hat{S}]=3 n\left(1-(1-1 /(6 n))^{n}\right) \geq 3 n\left(1-e^{-1 / 6}\right)
\end{aligned}
$$

Observe that $|\hat{S}|=|\hat{S} \cap H|+|\hat{S} \cap L|$ and that the random variable $|\hat{S}|$ satisfies the conditions of McDiarmid's inequality with constant 1. Thus with probability at least $1-\delta / 2$ we have
$$
|\hat{S}| \geq \mathbb{E}|\hat{S}|-\sqrt{\frac{n \log (2 / \delta)}{2}} \geq \frac{n}{4}\left(1-e^{-2}\right)+3 n\left(1-e^{-1 / 6}\right)-\sqrt{\frac{n \log (2 / \delta)}{2}} .
$$
The constant on the \(\Theta(n)\) term is at least 0.676, and for \(\delta=\operatorname{poly}(1 / n)\) the \(\Theta(n)\) term dominates, and so we have that for $n$ sufficiently large, $|\hat{S}| \geq 2 n / 3$.
We control \(\left|\hat{H}^{(1)}\right|\) similarly:
$$
\mathbb{E}\left|\hat{H}^{(1)}\right|=\sum_{h \in H} \operatorname{Pr}\left[B_{h}=1\right]=\frac{n}{4}\left(n \cdot \frac{2}{n} \cdot(1-2 / n)^{n-1}\right) \geq \frac{n}{2} \cdot(1-2 / n)^{n-1}
$$
Here $B_{h}$ is a Binomially distributed random variable with parameters \((n, 2 / n)\). Again we apply McDiarmid's inequality, here \(\left|\hat{H}^{(1)}\right|\) satisfies the conditions with a constant of 2, so that with probability at least $1-\delta / 2$ we have
$$
\left|\hat{H}^{(1)}\right| \geq \mathbb{E}\left|\hat{H}^{(1)}\right|-\sqrt{2 n \log (2 / \delta)} \geq \frac{n}{2} \cdot(1-2 / n)^{n-1}-\sqrt{2 n \log (2 / \delta)}
$$
Observe that the first term asymptotically approaches $n \cdot e^{-2} / 2$ while the second term is lower order in $n$. Therefore for $n$ sufficiently large, we can see that \(\left|\hat{H}^{(1)}\right| \geq e^{-2} / 3\) with high probability.

Next we turn to the coverage calculations. Since $N=4 / 3$ satisfies \(1<N<1 / \pi_{\text {ref }}(S)=2\) we know that \(\inf _{\beta \geq 0} \operatorname{Cov}\left(\pi_{\mathrm{OFF}, \beta}\right)=\pi_{\text {ref }}(S \backslash \hat{S}) / \pi_{\text {ref }}(S)=\pi^{\star}(S \backslash \hat{S})\). This is exactly the expert mass of the unobserved set. On the other hand $\pi_{\text {ERM }}$ matches exactly the empirical frequencies in the data. In particular, we have \(\pi_{\text {ERM }}(h)=1 / n\) for \(h \in \hat{H}^{(1)}\). This implies
$$
\operatorname{Cov}\left(\pi_{\mathrm{ERM}}\right) \geq \sum_{y \in S \backslash \hat{S}} \pi^{\star}(y)+\sum_{y \in \hat{H}^{(1)}} \frac{2}{n} 1\left\{\frac{2 / n}{1 / n} \geq N\right\}=\pi^{\star}(S \backslash \hat{S})+\pi^{\star}\left(\hat{H}^{(1)}\right)
$$
Finally for $\pi_{\mathrm{ABS}, \alpha}$ we need to show that no absolute clipping threshold improves on this value. To cover any \(h \in \hat{H}^{(1)}\) we need \(\pi_{\mathrm{ABS}, \alpha}(h) \geq 3 /(2 n)\). Recall that we only see $|\hat{S}|$ actions in the dataset. If we set $\alpha \leq 1 /|\hat{S}|$ then we can achieve zero loss but from Lemma B. 1 we have
$$
\pi_{\mathrm{ABS}, \alpha}(h)=\max \left(\alpha, \lambda \pi_{\mathrm{ref}}(h)\right) \leq \max (1 /|\hat{S}|, 1 / n)<3 /(2 n) .
$$
Here recall that $\lambda \leq 1$ (otherwise \(\pi_{\mathrm{ABS}, \alpha}(h)\) will not be a distribution) and that we have $|\hat{S}|>\frac{2 n}{3}$ with high probability. In this regime, unobserved actions get mass \(\lambda \pi_{\text {ref }}(h)\) but this is insufficient to cover $\pi^{\star}$ (since \(\left.2=\pi^{\star}(y) / \pi_{\text {ref }}(y)=2 \geq N=4 / 3\right)\).
On the other hand, if $\alpha>1 /|\hat{S}|$ then, by Lemma B. 2 the unique minimizer of the absolute clipped loss is
$$
\pi_{\mathrm{ABS}, \alpha}(y)=\min (\alpha, \lambda \hat{\mu}(y))
$$
where \(\hat{\mu}(y)\) is the empirical distribution. The normalizing condition gives,
$$
1 \geq \sum_{y \in \hat{S}} \min (\alpha, \lambda \hat{\mu}(y)) \geq|\hat{S}| \min (\alpha, \lambda / n),
$$
which follows because every observed action is observed at least once, so it has empirical mass at least $1 / n$. Since $\alpha>1 /|\hat{S}|$ we must have $\lambda / n \leq 1 /|\hat{S}|$ and therefore actions \(h \in \hat{H}^{(1)}\) have \(\pi_{\mathrm{ABS}, \alpha}(h) \leq 1 /|\hat{S}| \leq \frac{3}{2 n}\) (while unobserved actions have \(\pi_{\mathrm{ABS}, \alpha}(y)=0\) ).

Thus in both cases we have
$$
\operatorname{Cov}\left(\pi_{\mathrm{ABS}, \alpha}\right) \geq \pi^{\star}(S \backslash \hat{S})+\pi^{\star}\left(\hat{H}^{(1)}\right)
$$

The high probability event implies that \(\left|\hat{H}^{(1)}\right|>0\) and so \(\pi^{\star}\left(\hat{H}^{(1)}\right)>0\), establishing strict separation. $\square$

Proof of Claim 3 in Theorem B.1. We use the same construction as in the proof of Claim 2. Recall that in that construction, zero absolute clipped loss is achievable only if $\alpha \leq 1 /|\hat{S}|$. Let \(\hat{H}^{(>1)}\) denote the set of actions in $H$ that are observed at least twice in the sample $\hat{S}$. Since \(\alpha \leq 1 /|\hat{S}| \leq 3 /(2 n)\) (since with high probability we have $|\hat{S}| \geq 2 n / 3$ ), via Lemma B.1, the absolute clipped loss solution satisfies
$$
\pi_{\mathrm{ABS}, \alpha}(h)=\max \left(\alpha, \lambda \pi_{\mathrm{ref}}(h)\right) \leq \max (1 /|\hat{S}|, 1 / n)<3 /(2 n)=\pi^{\star}(h) / N,
$$
if we take $N=4 / 3$ as above. Thus all actions in \(\hat{H}^{(<1)}\) violate coverage. We already saw that all actions in \(\hat{H}^{(1)}\) violate coverage as well as all unobserved actions in $L$. Thus the coverage is
$$
\inf _{\alpha: \min _{\pi} L_{\mathrm{ABS}, \alpha}(\pi)=0} \operatorname{Cov}\left(\pi_{\mathrm{ABS}, \alpha}\right)=\frac{1}{2}+\pi^{\star}(L \backslash \hat{S}) .
$$
On the other hand, ERM covers actions in \(\hat{H}^{(>1)}\), because for any \(h \in \hat{H}^{(>1)}\), we have \(\pi_{\text {ERM }}(h) \geq 2 / n\) while \(\pi^{\star}(h)=2 / n\). Thus ERM strictly dominates absolute clipping whenever \(\left|\hat{H}^{(>1)}\right|>0\). Since
$$
\mathbb{E}\left|\hat{H}^{(>1)}\right|=\sum_{h \in H} \operatorname{Pr}\left[B_{h}>1\right]=\frac{n}{4} \cdot\left(1-(1-2 / n)^{n}-2(1-2 / n)^{n-1}\right)
$$
where \(B_{h} \sim \operatorname{Binomial}(n, 2 / n)\), and since \(\left|\hat{H}^{(>1)}\right|\) satisfies the conditions of McDiarmid's inequality with constant 1, we have that with probability at least $1-\delta$
$$
\left|\hat{H}^{(>1)}\right| \geq \frac{n}{4} \cdot\left(1-(1-2 / n)^{n}-2(1-2 / n)^{n-1}\right)-\sqrt{\frac{n}{2} \log (1 / \delta)} .
$$
For sufficiently large $n$ the first term is larger than \(\left(1-3 e^{-2}\right) n / 5\) and the second term is lower order, so for sufficiently large $n$ we get that \(\left|H^{(>1)}\right|>1\) with high probability. $\square$

