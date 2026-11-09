# A Mathematical Derivations

# A. 1 Deriving the Optimum of the KL-Constrained Reward Maximization Objective

In this appendix, we will derive Eq. 4 Analogously to Eq. 3, we optimize the following objective:
$$
\max _{\pi} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi}[r(x, y)]-\beta \mathbb{D}_{\mathrm{KL}}\left[\pi(y \mid x) \| \pi_{\mathrm{ref}}(y \mid x)\right]
$$
under any reward function \(r(x, y)\), reference model $\pi_{\text {ref }}$ and a general non-parametric policy class. We now have:
$$
\begin{aligned}
\max _{\pi} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi} & {[r(x, y)]-\beta \mathbb{D}_{\mathrm{KL}}\left[\pi(y \mid x) \| \pi_{\mathrm{ref}}(y \mid x)\right] } \\
& =\max _{\pi} \mathbb{E}_{x \sim \mathcal{D}} \mathbb{E}_{y \sim \pi(y \mid x)}\left[r(x, y)-\beta \log \frac{\pi(y \mid x)}{\pi_{\mathrm{ref}}(y \mid x)}\right] \\
& =\min _{\pi} \mathbb{E}_{x \sim \mathcal{D}} \mathbb{E}_{y \sim \pi(y \mid x)}\left[\log \frac{\pi(y \mid x)}{\pi_{\mathrm{ref}}(y \mid x)}-\frac{1}{\beta} r(x, y)\right] \\
& =\min _{\pi} \mathbb{E}_{x \sim \mathcal{D}} \mathbb{E}_{y \sim \pi(y \mid x)}\left[\log \frac{\pi(y \mid x)}{\frac{1}{Z(x)} \pi_{\mathrm{ref}}(y \mid x) \exp \left(\frac{1}{\beta} r(x, y)\right)}-\log Z(x)\right]
\end{aligned}
$$
where we have partition function:
$$
Z(x)=\sum_{y} \pi_{\mathrm{ref}}(y \mid x) \exp \left(\frac{1}{\beta} r(x, y)\right) .
$$
Note that the partition function is a function of only $x$ and the reference policy $\pi_{\text {ref }}$, but does not depend on the policy $\pi$. We can now define
$$
\pi^{*}(y \mid x)=\frac{1}{Z(x)} \pi_{\mathrm{ref}}(y \mid x) \exp \left(\frac{1}{\beta} r(x, y)\right),
$$
which is a valid probability distribution as \(\pi^{*}(y \mid x) \geq 0\) for all $y$ and \(\sum_{y} \pi^{*}(y \mid x)=1\). Since \(Z(x)\) is not a function of $y$, we can then re-organize the final objective in Eq 12 as:
$$
\begin{gathered}
\min _{\pi} \mathbb{E}_{x \sim \mathcal{D}}\left[\mathbb{E}_{y \sim \pi(y \mid x)}\left[\log \frac{\pi(y \mid x)}{\pi^{*}(y \mid x)}\right]-\log Z(x)\right]= \\
\min _{\pi} \mathbb{E}_{x \sim \mathcal{D}}\left[\mathbb{D}_{\mathrm{KL}}\left(\pi(y \mid x) \| \pi^{*}(y \mid x)\right)-\log Z(x)\right]
\end{gathered}
$$
Now, since \(Z(x)\) does not depend on $\pi$, the minimum is achieved by the policy that minimizes the first KL term. Gibbs' inequality tells us that the KL-divergence is minimized at 0 if and only if the two distributions are identical. Hence we have the optimal solution:
$$
\pi(y \mid x)=\pi^{*}(y \mid x)=\frac{1}{Z(x)} \pi_{\mathrm{ref}}(y \mid x) \exp \left(\frac{1}{\beta} r(x, y)\right)
$$
for all $x \in \mathcal{D}$. This completes the derivation.

# A. 2 Deriving the DPO Objective Under the Bradley-Terry Model

It is straightforward to derive the DPO objective under the Bradley-Terry preference model as we have
$$
p^{*}\left(y_{1} \succ y_{2} \mid x\right)=\frac{\exp \left(r^{*}\left(x, y_{1}\right)\right)}{\exp \left(r^{*}\left(x, y_{1}\right)\right)+\exp \left(r^{*}\left(x, y_{2}\right)\right)}
$$
In Section 4 we showed that we can express the (unavailable) ground-truth reward through its corresponding optimal policy:
$$
r^{*}(x, y)=\beta \log \frac{\pi^{*}(y \mid x)}{\pi_{\mathrm{ref}}(y \mid x)}+\beta \log Z(x)
$$
Substituting Eq. 17 into Eq. 16 we obtain:
$$
\begin{aligned}
p^{*}\left(y_{1} \succ y_{2} \mid x\right) & =\frac{\exp \left(\beta \log \frac{\pi^{*}\left(y_{1} \mid x\right)}{\pi_{\text {ref }}\left(y_{1} \mid x\right)}+\beta \log Z(x)\right)}{\exp \left(\beta \log \frac{\pi^{*}\left(y_{1} \mid x\right)}{\pi_{\text {ref }}\left(y_{1} \mid x\right)}+\beta \log Z(x)\right)+\exp \left(\beta \log \frac{\pi^{*}\left(y_{2} \mid x\right)}{\pi_{\text {ref }}\left(y_{2} \mid x\right)}+\beta \log Z(x)\right)} \\
& =\frac{1}{1+\exp \left(\beta \log \frac{\pi^{*}\left(y_{2} \mid x\right)}{\pi_{\text {ref }}\left(y_{2} \mid x\right)}-\beta \log \frac{\pi^{*}\left(y_{1} \mid x\right)}{\pi_{\text {ref }}\left(y_{1} \mid x\right)}\right)} \\
& =\sigma\left(\beta \log \frac{\pi^{*}\left(y_{1} \mid x\right)}{\pi_{\text {ref }}\left(y_{1} \mid x\right)}-\beta \log \frac{\pi^{*}\left(y_{2} \mid x\right)}{\pi_{\text {ref }}\left(y_{2} \mid x\right)}\right) .
\end{aligned}
$$
The last line is the per-instance loss in Equation 7.

# A. 3 Deriving the DPO Objective Under the Plackett-Luce Model

The Plackett-Luce model [32, 23] is a generalization of the Bradley-Terry model over rankings (rather than just pair-wise comparisons). Similar to to the Bradley-Terry model, it stipulates that when presented with a set of possible choices, people prefer a choice with probability proportional to the value of some latent reward function for that choice. In our context, when presented with a prompt $x$ and a set of $K$ answers $y_{1}, \ldots, y_{K}$ a user would output a permutation $\tau:[K] \rightarrow[K]$, giving their ranking of the answers. The Plackett-Luce model stipulates that
$$
p^{*}\left(\tau \mid y_{1}, \ldots, y_{K}, x\right)=\prod_{k=1}^{K} \frac{\exp \left(r^{*}\left(x, y_{\tau(k)}\right)\right)}{\sum_{j=k}^{K} \exp \left(r^{*}\left(x, y_{\tau(j)}\right)\right)}
$$
Notice that when $K=2$, Equation 18 reduces to the Bradley-Terry model. However, for the general Plackett-Luce model, we can still utilize the results of Eq. 5 and substitute the reward function parameterized by its optimal policy. Similarly to Appendix A.2. the normalization constant \(Z(x)\) cancels out and we're left with:
$$
p^{*}\left(\tau \mid y_{1}, \ldots, y_{K}, x\right)=\prod_{k=1}^{K} \frac{\exp \left(\beta \log \frac{\pi^{*}\left(y_{\tau(k)} \mid x\right)}{\pi_{\mathrm{ref}}\left(y_{\tau(k)} \mid x\right)}\right)}{\sum_{j=k}^{K} \exp \left(\beta \log \frac{\pi^{*}\left(y_{\tau(j)} \mid x\right)}{\pi_{\mathrm{ref}}\left(y_{\tau(j)} \mid x\right)}\right)}
$$

Similarly to the approach of Section 4, if we have access to a dataset $\mathcal{D}=$ \(\left\{\tau^{(i)}, y_{1}^{(i)}, \ldots, y_{K}^{(i)}, x^{(i)}\right\}_{i=1}^{N}\) of prompts and user-specified rankings, we can use a parameterized model and optimize this objective with maximum-likelihood.:
$$
\mathcal{L}_{\mathrm{DPO}}\left(\pi_{\theta}, \pi_{\mathrm{ref}}\right)=-\mathbb{E}_{\tau, y_{1}, \ldots, y_{K}, x \sim \mathcal{D}}\left[\log \prod_{k=1}^{K} \frac{\exp \left(\beta \log \frac{\pi_{\theta}\left(y_{\tau(k)} \mid x\right)}{\pi_{\mathrm{ref}}\left(y_{\tau(k)} \mid x\right)}\right)}{\sum_{j=k}^{K} \exp \left(\beta \log \frac{\pi_{\theta}\left(y_{\tau(j)} \mid x\right)}{\pi_{\mathrm{ref}}\left(y_{\tau(j)} \mid x\right)}\right)}\right]
$$

# A. 4 Deriving the Gradient of the DPO Objective

In this section we derive the gradient of the DPO objective:
$$
\nabla_{\theta} \mathcal{L}_{\mathrm{DPO}}\left(\pi_{\theta} ; \pi_{\mathrm{ref}}\right)=-\nabla_{\theta} \mathbb{E}_{\left(x, y_{w}, y_{l}\right) \sim \mathcal{D}}\left[\log \sigma\left(\beta \log \frac{\pi_{\theta}\left(y_{l} \mid x\right)}{\pi_{\mathrm{ref}}\left(y_{l} \mid x\right)}-\beta \log \frac{\pi_{\theta}\left(y_{w} \mid x\right)}{\pi_{\mathrm{ref}}\left(y_{w} \mid x\right)}\right)\right]
$$
We can rewrite the RHS of Equation 21 as
$$
\nabla_{\theta} \mathcal{L}_{\mathrm{DPO}}\left(\pi_{\theta} ; \pi_{\mathrm{ref}}\right)=-\mathbb{E}_{\left(x, y_{w}, y_{l}\right) \sim \mathcal{D}}\left[\frac{\sigma^{\prime}(u)}{\sigma(u)} \nabla_{\theta}(u)\right],
$$
where \(u=\beta \log \frac{\pi_{\theta}\left(y_{l} \mid x\right)}{\pi_{\text {ref }}\left(y_{l} \mid x\right)}-\beta \log \frac{\pi_{\theta}\left(y_{w} \mid x\right)}{\pi_{\text {ref }}\left(y_{w} \mid x\right)}\).
Using the properties of sigmoid function \(\sigma^{\prime}(x)=\sigma(x)(1-\sigma(x))\) and \(\sigma(-x)=1-\sigma(x)\), we obtain the final gradient
$$
\begin{aligned}
& \quad \nabla_{\theta} \mathcal{L}_{\mathrm{DPO}}\left(\pi_{\theta} ; \pi_{\mathrm{ref}}\right)= \\
& -\mathbb{E}_{\left(x, y_{w}, y_{l}\right) \sim \mathcal{D}}\left[\beta \sigma\left(\beta \log \frac{\pi_{\theta}\left(y_{w} \mid x\right)}{\pi_{\mathrm{ref}}\left(y_{w} \mid x\right)}-\beta \log \frac{\pi_{\theta}\left(y_{l} \mid x\right)}{\pi_{\mathrm{ref}}\left(y_{l} \mid x\right)}\right)\left[\nabla_{\theta} \log \pi\left(y_{w} \mid x\right)-\nabla_{\theta} \log \pi\left(y_{l} \mid x\right)\right]\right],
\end{aligned}
$$
After using the reward substitution of \(\hat{r}_{\theta}(x, y)=\beta \log \frac{\pi_{\theta}(y \mid x)}{\pi_{\text {ref }}(y \mid x)}\) we obtain the final form of the gradient from Section 4.

# A. 5 Proof of Lemma 1 and 2

In this section, we will prove the two lemmas from Section 5.
Lemma 1 Restated. Under the Plackett-Luce preference framework, and in particular the Bradley-Terry framework, two reward functions from the same equivalence class induce the same preference distribution.

Proof. We say that two reward functions \(r(x, y)\) and \(r^{\prime}(x, y)\) are from the same equivalence class if \(r^{\prime}(x, y)=r(x, y)+f(x)\) for some function $f$. We consider the general Plackett-Luce (with the Bradley-Terry model a special case for $K=2$ ) and denote the probability distribution over rankings induced by a particular reward function \(r(x, y)\) as $p_{r}$. For any prompt $x$, answers $y_{1}, \ldots, y_{K}$ and ranking $\tau$ we have:
$$
\begin{aligned}
p_{r^{\prime}}\left(\tau \mid y_{1}, \ldots, y_{K}, x\right) & =\prod_{k=1}^{K} \frac{\exp \left(r^{\prime}\left(x, y_{\tau(k)}\right)\right)}{\sum_{j=k}^{K} \exp \left(r^{\prime}\left(x, y_{\tau(j)}\right)\right)} \\
& =\prod_{k=1}^{K} \frac{\exp \left(r\left(x, y_{\tau(k)}\right)+f(x)\right)}{\sum_{j=k}^{K} \exp \left(r\left(x, y_{\tau(j)}\right)+f(x)\right)} \\
& =\prod_{k=1}^{K} \frac{\exp (f(x)) \exp \left(r\left(x, y_{\tau(k)}\right)\right)}{\exp (f(x)) \sum_{j=k}^{K} \exp \left(r\left(x, y_{\tau(j)}\right)\right)} \\
& =\prod_{k=1}^{K} \frac{\exp \left(r\left(x, y_{\tau(k)}\right)\right)}{\sum_{j=k}^{K} \exp \left(r\left(x, y_{\tau(j)}\right)\right)} \\
& =p_{r}\left(\tau \mid y_{1}, \ldots, y_{K}, x\right),
\end{aligned}
$$
which completes the proof. $\square$

Lemma 2 Restated. Two reward functions from the same equivalence class induce the same optimal policy under the constrained RL problem.

Proof. Let us consider two reward functions from the same class, such that \(r^{\prime}(x, y)=r(x, y)+f(x)\) and, let us denote as $\pi_{r}$ and $\pi_{r^{\prime}}$ the corresponding optimal policies. By Eq. 4. for all $x, y$ we have
$$
\begin{aligned}
\pi_{r^{\prime}}(y \mid x) & =\frac{1}{\sum_{y} \pi_{\text {ref }}(y \mid x) \exp \left(\frac{1}{\beta} r^{\prime}(x, y)\right)} \pi_{\text {ref }}(y \mid x) \exp \left(\frac{1}{\beta} r^{\prime}(x, y)\right) \\
& =\frac{1}{\sum_{y} \pi_{\text {ref }}(y \mid x) \exp \left(\frac{1}{\beta}(r(x, y)+f(x))\right)} \pi_{\text {ref }}(y \mid x) \exp \left(\frac{1}{\beta}(r(x, y)+f(x))\right) \\
& =\frac{1}{\exp \left(\frac{1}{\beta} f(x)\right) \sum_{y} \pi_{\text {ref }}(y \mid x) \exp \left(\frac{1}{\beta} r(x, y)\right)} \pi_{\text {ref }}(y \mid x) \exp \left(\frac{1}{\beta} r(x, y)\right) \exp \left(\frac{1}{\beta} f(x)\right) \\
& =\frac{1}{\sum_{y} \pi_{\text {ref }}(y \mid x) \exp \left(\frac{1}{\beta} r(x, y)\right)} \pi_{\text {ref }}(y \mid x) \exp \left(\frac{1}{\beta} r(x, y)\right) \\
& =\pi_{r}(y \mid x)
\end{aligned}
$$
which completes the proof. $\square$

# A. 6 Proof of Theorem 1

In this section, we will expand on the results of Theorem 1.
Theorem 1 Restated. Assume, we have a reference model, such that \(\pi_{\text {ref }}(y \mid x)>0\) for all pairs of prompts $x$ and answers $y$ and a parameter $\beta>0$. All reward equivalence classes, as defined in Section 5 can be represented with the reparameterization \(r(x, y)=\beta \log \frac{\pi(y \mid x)}{\pi_{\text {ref }}(y \mid x)}\) for some model \(\pi(y \mid x)\).

Proof. Consider any reward function \(r(x, y)\), which induces an optimal model \(\pi_{r}(y \mid x)\) under the KL-constrained RL problem, with solution given by 4 Following Eq. 5, when we log-linearize both sides we obtain:
$$
r(x, y)=\beta \log \frac{\pi_{r}(y \mid x)}{\pi_{\text {ref }}(y \mid x)}+\beta \log Z(x)
$$
where \(Z(x)=\sum_{y} \pi_{\text {ref }}(y \mid x) \exp \left(\frac{1}{\beta} r(x, y)\right)\) (notice that \(Z(x)\) also depends on the reward function \(r)\). Using the operator \(r^{\prime}(x, y)=f\left(r, \pi_{\text {ref }}, \beta\right)(x, y)=r(x, y)-\beta \log Z(x)\), we see that this new reward function is within the equivalence class of $r$ and, we have:
$$
r^{\prime}(x, y)=\beta \log \frac{\pi_{r}(y \mid x)}{\pi_{\text {ref }}(y \mid x)}
$$
which completes the proof. $\square$

We can further expand on these results. We can see that if $r$ and $r^{\prime}$ are two reward functions in the same class, then
$$
f\left(r, \pi_{\mathrm{ref}}, \beta\right)(x, y)=\beta \log \frac{\pi_{r}(y \mid x)}{\pi_{\mathrm{ref}}(y \mid x)}=\beta \log \frac{\pi_{r}^{\prime}(y \mid x)}{\pi_{\mathrm{ref}}(y \mid x)}=f\left(r^{\prime}, \pi_{\mathrm{ref}}, \beta\right)(x, y)
$$
where the second equality follows from Lemma 2. We have proven that the operator $f$ maps all reward functions from a particular equivalence class to the same reward function. Next, we show that for every equivalence class of reward functions, the reward function that has the reparameterization outlined in Theorem 1 is unique.
Proposition 1. Assume, we have a reference model, such that \(\pi_{\text {ref }}(y \mid x)>0\) for all pairs of prompts $x$ and answers $y$ and a parameter $\beta>0$. Then every equivalence class of reward functions, as defined in Section 5; has a unique reward function \(r(x, y)\), which can be reparameterized as \(r(x, y)=\beta \log \frac{\pi(y \mid x)}{\pi_{\text {ref }}(y \mid x)}\) for some model \(\pi(y \mid x)\).

Proof. We will proceed using proof by contradiction. Assume we have two reward functions from the same class, such that \(r^{\prime}(x, y)=r(x, y)+f(x)\). Moreover, assume that \(r^{\prime}(x, y)=\beta \log \frac{\pi^{\prime}(y \mid x)}{\pi_{\text {ref }}(y \mid x)}\) for some model \(\pi^{\prime}(y \mid x)\) and \(r(x, y)=\beta \log \frac{\pi(y \mid x)}{\pi_{\text {ref }}(y \mid x)}\) for some model \(\pi(y \mid x)\), such that $\pi \neq \pi^{\prime}$. We then have
$$
r^{\prime}(x, y)=r(x, y)+f(x)=\beta \log \frac{\pi(y \mid x)}{\pi_{\text {ref }}(y \mid x)}+f(x)=\beta \log \frac{\pi(y \mid x) \exp \left(\frac{1}{\beta} f(x)\right)}{\pi_{\text {ref }}(y \mid x)}=\beta \log \frac{\pi^{\prime}(y \mid x)}{\pi_{\text {ref }}(y \mid x)}
$$
for all prompts $x$ and completions $y$. Then we must have \(\pi(y \mid x) \exp \left(\frac{1}{\beta} f(x)\right)=\pi^{\prime}(y \mid x)\). Since these are distributions, summing over $y$ on both sides, we obtain that \(\exp \left(\frac{1}{\beta} f(x)\right)=1\) and since $\beta>0$, we must have \(f(x)=0\) for all $x$. Therefore \(r(x, y)=r^{\prime}(x, y)\). This completes the proof. $\square$

We have now shown that every reward class has a unique reward function that can be represented as outlined in Theorem 1, which is given by \(f\left(r, \pi_{\text {ref }}, \beta\right)\) for any reward function in that class.

