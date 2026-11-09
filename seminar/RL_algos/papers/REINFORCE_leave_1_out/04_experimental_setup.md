# 4 Experimental Setup

## 4.1 Training Details

Datasets We report results on the TL;DR Summarize (Stiennon et al., 2020) and Anthropic Helpful and Harmless Dialogue (Bai et al., 2022a) datasets. The trainig split of TL;DR Summarize ${ }^{2}$ dataset contains 116k human-written instructions and 93k human-annotated preference pairs. The preprocessed Anthropic- $\mathrm{HH}^{3}$ dataset contains 112k many training preference pairs.

Models For both datasets, we use Pythia-6.9B (Biderman et al., 2023) as the pretrained basemodel. To ablate the effect of the pre-trained model quality on the human preference optimization, we also experiment with Llama-7B coupled with the Anthropic-HH dataset. To ensure a fair comparison across all methods, we use a context length of 512 tokens during both supervised fine-tuning and the reward model training. We initialize both the reward model and policy with the corresponding SFT checkpoint, unless noted otherwise.

Experimental Details For the TL;DR Summarize dataset, we use the dedicated SFT split. Since the original Antrophic-HH dataset does not include a separate SFT split, we use prompts and the preferred responses from the binary comparisons during the SFT stage similar to prior work (Yuan et al., 2023; Dong et al., 2023; Rafailov et al., 2023). In the preference training stage, we use the same prompts as in the SFT stage to generate completions. Further details on the experimental setup and hyper-parameters are given in Appendix D

## 4.2 Evaluation

Optimization Quality For all online methods (all methods except DPO), to measure how well they optimize the intrinsic objective , we report average rewards (using the training RM) on 1000 samples from the test set. To measure how well

\footnotetext{
${ }^{2}$ https://github.com/openai/summarize-from-feedback
${ }^{3}$ https://huggingface.co/datasets/Dahoas/full-hh-rlhf
}

![Figure](./REINFORCE_leave_1_out_images/7b6ccb24-aeaf-4405-a575-200815561112-07_550_1249_249_407.jpg)

**Figure 2: Comparison of test rewards for various optimization methods, throughout training. RLOO and REIN-FORCE with a moving average baseline, consistently achieve higher test rewards than RAFT and PPO throughout training. REINFORCE w/ B. refers to REINFORCE with a moving average reward baseline as in Equation 8.**

\begin{table}
\begin{tabular}[t]{|l|l|l|l|}
\hline Method & TL;DR & HH (Pythia) & HH (Llama) \\
\hline RLOO ( $\mathrm{k}=4$ ) & 77.9 & 43.7 & 64.1 \\
\hline RAFT ( $\mathrm{k}=4$ ) & 73.2 & 42.1 & 63.3 \\
\hline RLOO ( $\mathrm{k}=2$ ) & 74.2 & 47.6 & 62.2 \\
\hline RAFT ( $\mathrm{k}=2$ ) & 72.1 & 37.7 & 58.4 \\
\hline RF. w/ B. & 70.7 & 37.9 & 55.3 \\
\hline Vanilla PG & 70.4 & 36.4 & 52.3 \\
\hline PPO & 67.6 & 29.2 & 32.0 \\
\hline DPO & 66.6 & 39.0 & 61.9 \\
\hline
\end{tabular}

**Table 1: Final win rates on generations for held-out test prompts in the Anthropic-HH and TL;DR Summarize datasets. Metrics are reported for the checkpoint with the highest test reward.**
\end{table}
each method optimizes the extrinsic objective of aligning the models to human preference, we report simulated win rates in accordance with Alpacafarm framework (Dubois et al., 2024) where we use GPT-4 as a proxy for human evaluation.

Alignment Tax RLHF fine-tuning is often associated with a drop in diversity and language fluency which is referred to as alignment tax (Askell et al., 2021; Kirk et al., 2024). Hence, we also report metrics which serve as proxies for fluency and diversity similar to (Dong et al., 2023). To measure fluency, we report perplexity measured using the preferred completions from the test set, similar to (Dong et al., 2023). Finally, we measure the diversity coupled with length by using the average n-gram diversity Li et al. (2016).

