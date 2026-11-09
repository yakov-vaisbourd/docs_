# 5 Results and Discussion

## 5.1 Reward Optimization

The objective of RLOO, REINFORCE with baseline, RAFT, PPO, and vanilla PG is to maximize the reward score, hence we compare the success of optimization for each method. Fig. 2 shows the test reward scores plotted during the training for all methods. On each dataset, we use the same reward model for all the methods, thus their test reward scores are directly comparable.

Modeling Partial Completions vs Full Generations In Fig. 2, we observe that methods that do not model partial completions such as REINFORCE with baseline, RLOO, and RAFT consistently outperform Vanilla PG and PPO where each token is modeled as an action (i.e. partial completions). In addition to their higher performance in reward optimization, these methods are more efficient than Vanilla PG and PPO since they do not require training a learned baseline and a value network as in Vanilla PG or PPO respectively. This suggests that modeling partial sequences is unnecessary.

Sampling Efficiency Given the same sampling budget, $k$, RLOO consistently outperforms RAFT throughout training in terms of reward scores. Noticeably, even with a smaller sampling budget $\mathrm{RLOO}_{k=2}$, RLOO either closely matches or outperforms $\mathrm{RAFT}_{k=4}$, across all datasets and models. In this setting, RLOO uses only half the onlinesample budget given the same step count.

This confirms that RLOO leads to better optimization by using all the samples generated, unlike RAFT where only the top-ranked sample is used for fine-tuning. Showcasing the same finding, Fig. 5 in Appendix A plots the rewards with respect to the number of samples generated during training regardless of the $k$ value.

![Figure](./REINFORCE_leave_1_out_images/7b6ccb24-aeaf-4405-a575-200815561112-08_522_1253_248_406.jpg)

**Figure 3: Sensitivity to KL weight Training curve for \(r(x, y)\) (left) $D_{K L**$ (right) and with $k=2$. Under higher KL, RAFT is not only worse than RLOO at optimizing reward, but also deviates further from the reference policy. Curves are exponentially smoothed for better readability.}

\begin{table}
\begin{tabular}[t]{|l|l|l|l|l|l|}
\hline Method & Len. & PPL & D-1 & D-2 & R-Var. \\
\hline RLOO ( $\mathrm{k}=4$ ) & 60.6 & 27.6 & 0.10 & 0.43 & 3.1 \\
\hline RAFT ( $\mathrm{k}=4$ ) & 62.4 & 30.1 & 0.10 & 0.43 & 3.2 \\
\hline RLOO ( $\mathrm{k}=2$ ) & 58.6 & 29.2 & 0.11 & 0.44 & 3.0 \\
\hline RAFT ( $\mathrm{k}=2$ ) & 52.8 & 28.9 & 0.12 & 0.47 & 3.1 \\
\hline RF. w/ B. & 47.2 & 27.2 & 0.13 & 0.50 & 2.7 \\
\hline V. PG & 39.1 & 39.0 & 0.15 & 0.54 & 3.7 \\
\hline PPO & 16.5 & 40.4 & 0.34 & 0.60 & 2.3 \\
\hline DPO & 104.4 & 33.8 & 0.08 & 0.39 & N/A \\
\hline
\end{tabular}

**Table 2: Language Fluency and Diversity Metrics on the Anthropic-HH dataset.**
\end{table}

## 5.2 Simulated Win Rates

Table 1 presents the win-rates against the original completions in TL;DR Summarize and AntrophicHH for each method. Here, we also include DPO.

Modeling partial completions is not necessary Recall that the key distinguishing factor between Vanilla PG and REINFORCE as referred to in this work, is that while Vanilla PG treats each token as an action, REINFORCE operates on the entire generation. As seen in Table 1, REINFORCE with baseline is on par with Vanilla PG on both TL;DR (70.7 vs 70.4) and HH (37.9 vs 36.4) datasets when using Pythia-based models. Moreover, REINFORCE with baseline outperforms Vanilla PG in HH dataset with a Llama-based model, achieving a higher win-rate (55.3 vs 52.3).

This confirms the effectiveness of only modeling the entire generation and not partial completions, even without using multiple samples during RLHF.

Win-rates are inline with test reward scores RLOO with $k=4$ achieves the highest win-rates, outperforming PPO by 10.3, 14.5, and 32.1 for TL;DR, HH (Pythia) and HH (Llama), respectively. As the only exception, RLOO achieves the highest win-rate with $k=2$ in HH dataset.

RLOO is more sample efficient than RAFT Comparing RLOO with RAFT, under the same sampling budget $k$, RLOO consistently outperforms RAFT in all datasets and models. When averaged across the three dataset and model pairings, RLOO achieves win-rates of 61.3 and 61.9 for $k=2$ and $k=4$, respectively, while RAFT scores 56.1 and 59.5, respectively. Notably, RLOO exhibits the highest increase in win-rate compared to RAFT, up to 9.9 as in the HH dataset with $k=2$ and Pythia-based models (second column in Table 1).

## 5.2.1 Alignment Tax

Table 2 shows various intrinsic evaluation metrics including perplexity and diversity scores of Llamabased models in Antrophic-HH datasets.

Length of Generations Noticeably, the DPO trained model tends to be over-verbose (the longest generation length of 104 tokens on average) while the PPO trained model leads to short generations (the shortest on average with 16 tokens). We provide example responses in Appendix F.

Perplexity and Diversity As seen in Table 2, perplexity (PPL) scores are relatively close amongst RLOO, RAFT, and REINFORCE with baseline where all three methods achieve significantly lower perplexity than PPO and Vanilla PG.

In terms of diversity, Diversity-1 scores are similar across RLOO, RAFT, REINFORCE with baseline and Vanilla PG. Diversity-2 scores tend to slightly decrease for the methods with higher reward optimization (Askell et al., 2021). This is unsurprising given the significant difference in their generation length compared to other methods. Overall, RLOO and REINFORCE with a baseline maintain fluency and diversity in generations in
comparison with other methods while achieving higher reward scores and win-rates.

Reward variance Lower reward variance is desirable for applications such as safety and harmlessness where there is a high risk associated with generating a low-reward sample. Results in Table 2 show that RLOO for the same $k$ value leads to slightly lower reward variance amongst the generations, compared to RAFT which is the most competitive method to RLOO in terms of reward optimization. Finally, Vanilla PG leads to the highest reward variance. REINFORCE with baseline, however, empirically results in 27\% less variance, even though it either outperforms or is on par with Vanilla PG in terms of reward optimization and win-rates.

## 5.2.2 Robustness

As previously noted, a major drawback of RAFT is that it only optimizes on the highest-ranked sample and discards the rest of the online samples. Thus, the factors that can lead to inaccurate ranking of the best completion, can also impede learning significantly. We demonstrate this fragility by showing the effects of 1) high $\beta$ for the KL-term and 2) inserted reward noise on RAFT in comparison to RLOO.

Mismatch from KL-penalty In Figure 3, we show the evolution of the KL distance and test reward curve \(r(x, y)\) throughout training for RLOO and RAFT, using $k=2$, on the HH dataset using Pythia-based models. We vary the KL regularization, using $\beta=\{0.25,0.5,1\}$. Here, a larger KL penalty in \(R(x, y)\) (higher $\beta$ ) potentially increases mismatches between rankings of the $k$ online samples. However, the choice of $\beta$ often depends on multiple factors such as the distribution of the data and output logits of the base model, which may not allow for a low $\beta$ value even with early-stopping.

We find that RAFT is more sensitive to higher KL regularization. In a low-regularized regime \((\beta=\{0.1\})\), RLOO and RAFT converge to equal KL distances from the reference policy while RLOO achieves a higher reward. However increased regularization with $\beta=\{0.25,0.5,1.0\}$, not only RAFT is worse at optimizing the reward, but also deviates more from the reference policy.

Mismatch from Reward Noise The reward model is itself a noisy proxy of reward signal due to the inherently noisy nature of human preference (Nguyen et al., 2017b; Kreutzer et al., 2018). Inspired by literature in Bayesian deep learning on

![Figure](./REINFORCE_leave_1_out_images/7b6ccb24-aeaf-4405-a575-200815561112-09_518_748_252_1066.jpg)

**Figure 4: Sensitivity to noise Drop in training rewards upon the addition of noise with $\sigma=\{1.0,3.0,5.0\**$ .RAFT shows a more significant drop in reward compared to RLOO. Curves are exponentially smoothed for better readability.}

modeling Aleatoric uncertainty (Kendall and Gal, 2017; Collier et al., 2021), to simulate the effect of such noise in different degrees, for each prompt, we add noise to the rewards. Concretely, we add noise $\epsilon$ to the output logits of the binary classifier: \(r_{\sigma}(x, y)=r(x, y)+\epsilon\) where \(\epsilon \sim \mathcal{N}\left(0, \sigma^{2}\right)\).

Figure 4 shows the drop in reward at different levels for noise $\sigma=\{1.0,3.0,5.0\}$. As expected, the unaltered training reward decreases for both RLOO and RAFT. However, the drop is far more pronounced for RAFT with $\sigma=\{3.0,5.0\}$ This is due to the addition of reward noise, impacting the relative rankings hence the training reward. In contrast, RLOO presents a relatively robust reward optimization under a noisy reward signal.

