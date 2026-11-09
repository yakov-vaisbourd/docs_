# 3 From PPO to REINFORCE

We scrutinize individual components of PPO that we consider not an ideal fit for RLHF. We explain the theoretical origin, motivate with the practical conditions of LLM RLHF, and provide empirical support from preliminary experiments.

\footnotetext{
${ }^{1}$ In collaboration with the Huggingface team, we provide an open-source implementation of RLOO in the TRL (von Werra et al., 2020) library: https://github.com/huggingface/trl
}

## 3.1 Revisiting the need for low-variance estimators

Actor-critic algorithms, such as PPO, were motivated in formulation by the high variance observed in traditional RL settings. PPO leverages lower-variance estimators of the total trajectory return to improve learning. These estimators are constructed by bootstrapping off a state-value function (Sutton et al., 1999; Schulman et al., 2018; Sutton and Barto, 2020). While bootstrapping reduces variance, the trade-off is the introduction of bias which risks optimizing for biased rewards.

In contrast, REINFORCE uses an unbiased Monte-Carlo estimator of the trajectory return that can have high variance in theory, especially if it is approximated with only a single sample, which is not frequently preferred in traditional Deep-RL environments. Recent work has offered a plethora of evidence that REINFORCE suffers from high variance and fails in the presence of large action spaces like NLP (Ranzato et al., 2016; Bahdanau et al., 2017; Ding and Soricut, 2017; Ammanabrolu and Hausknecht, 2020; Ammanabrolu et al., 2022; Martin et al., 2022; Korbak et al., 2022). However, we note that these findings were based on scenarios with poor conditioning when training from random or weak initialization as opposed to warm-starting it from a strong pre-trained model.

Here, we question whether this empirical evidence holds for RLHF. We posit that this is not a practical concern in fine-tuning LLMs due to the extremely strong initialization of the policy (a pre-trained LLM). In this setting, strong initialization coupled with prompt conditioning leads to the concentration of probability mass on a few tokens at each generation step, even though the number of possible actions is in theory enormous (refer to Appendix B for further discussion on the effect of conditioning). The optimization landscape is far less likely to present problems like destructively large and high-variance gradient updates. Thus, attempting to reduce variance further at the cost of introducing bias is not worth it.

Empirical support To validate this hypothesis, we vary the weight placed upon variance minimization and bias introduction. In the formulation of PPO in Sect. 2.1, the advantage estimator GAE (Schulman et al., 2018) is relied upon to trade-off bias and variance when estimating the true advantage function in PPO.

GAE introduces a hyper-parameter $\lambda \in[0,1]$ in the true advantage function, which balances bias and variance of the constructed estimator. The closer $\lambda$ is to 1 , the higher the observed variance. The optimal choice of where to set $\lambda=0$ depends on the environment. In a highly stochastic environment, minimizing variance at the cost of bias is a worthy trade-off. However, given a stable environment where variance is already low, the introduction of bias is needless.

At the extreme of 1 which imposes minimal bias at the trade-off of variance, the advantage term reduces to return the estimator used in Vanilla Policy Gradient (PG) REINFORCE, which directly builds on the REINFORCE estimator, by optimizing the trajectory returns starting from each token in the generation
$$
\sum_{i=t}^{T} \gamma^{T-i-1} R_{t}\left(x, y_{t}\right)-b_{\psi}\left(s_{t}\right)
$$
where \(b_{\psi}\left(s_{t}\right)\) is a learned baseline state $s_{t}$, akin to how a value network is learned in the traditional RL setting, using a standard MLE loss \(\frac{1}{2}\left(\sum_{i=t}^{T} \gamma^{T-i-1} R_{i}\left(x, y_{i}\right)-b_{\psi}\left(s_{t}\right)\right)^{2}\). Note that the key distinguishing factor between Vanilla PG and REINFORCE as referred to in this work, is that Vanilla PG uses the REINFORCE estimator on the trajectory return starting from the context formed by the prompt and a partial completion, whereas the REINFORCE estimator described in Section 2.2 is applied to the the full trajectory return. We will return to this distinction in the results Section 5.1 when we evaluate whether evaluating partial completions is necessary in RLHF.

In Figure 1, we present the results of evaluating the reward for PPO given GAE with different value of $\lambda$. Two variants impose minimal bias but invite high variance ( $\left(\hat{A}_{\lambda=1.00}\right.$ (Vanilla PG introduced above), and $\hat{A}_{\lambda=0.95}$ ) and two variants which over-index on minimizing variance at the cost of bias ( $\hat{A}_{\lambda=0.0}$, and $\hat{A}_{\lambda=0.5}$ ). Figure 1 plots the reward and observe that the most extreme variant Vanilla PG (unbiased $A_{\lambda=1.0}$ ) performs the best given it presents no bias at the risk of high variance. We observe a monotonically decreasing reward with decreasing $\lambda$. This supports our hypothesis that reducing variance at the cost of bias in an RHLF setting needlessly introduces bias given the stable default properties of the environment.

## 3.2 Clipping is Rarely Necessary in RLHF

Next, we turn to the clipping-ratio $\epsilon$ (see Eq. 6), which is used to prevent large policy updates when
$\frac{\pi_{\theta}}{\pi_{\text {old }}}$ deviates far from 1, i.e., to prevent updates that are too far off from the current policy (Schulman et al., 2017).

In Figure 1, we compare reward curves for independent PPO training with and without clipping. Note that we also turn off clipping for the value network, for these set of experiments, as it has been observed to have a noticeable impact on learning in traditional Deep-RL environments (Engstrom et al., 2020). The removal of these components does not impact learning meaningfully. We empirically found in our RLHF setting that the loss is actually clipped on average < 5\% of the time per batch, throughout training across all dataset and base-model pairings, which indicates that the learning regime is close to being "on-policy", with policies varying slowly from one iteration to the other.

To further validate this, we completely turn off clipping followed by removing the ratio $\frac{\pi_{\theta}}{\pi_{o l d}}$, while $\lambda=1$, which reduces the PPO loss to that of Vanilla PG. If anything the removal of clipping gives a slight boost in performance, validating our hypothesis that large off-policy updates in our optimization regime are rare and do not have catastrophic effects on learning as they do in traditional Deep-RL.

## 3.3 Modeling Partial Completions is Not Necessary

As described in Sect. 2, PPO models each token as an action whereas REINFORCE models the entire generation as a single action, as opposed to each token. In practice, in LLM RLHF a \(r(x, y)\) is only attributed to the <EOS> token, where for all other tokens, only \(\log \frac{\pi\left(y_{t} \mid s_{t}\right)}{\pi_{\text {ref }}\left(y_{t} \mid s_{t}\right)}\) composes \(R_{t}(x, y)\), which is not meaningful.

From a pure RL point of view, the environment dynamics are fully deterministic ( \(P_{D}\left(\left\{y_{<t+1, x}\right\} \mid s_{t}, y_{t}\right)=1\) ), meaning that our environment (context) changes deterministically based on the new token/action predicted. Hence, the problem can be reduced to a bandit problem, where the Markov Decision Process (MDP) consists of only the initial state as determined by the prompt, and the terminal state, which is always reached after the generation (Kreutzer et al., 2017; Nguyen et al., 2017b). Note that modeling the entire generation as a single action is done explicitly by REINFORCE but is also done implicitly with iterative fine-tuning methods which generate the entire completion before filtering using a reward model.

In the results section 5.1 we will explicitly compare REINFORCE and RLOO which both model the full trajectory return to PPO and Vanilla PG which both model the partial completion. We ask is modeling the entire generation as a single action sufficient to achieve similar or better performance in RLHF?

