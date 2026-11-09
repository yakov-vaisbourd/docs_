# 2 Related Work

Large language models have been key to strong performance in a wide range of downstream tasks (Devlin et al., 2019; Radford et al., 2019; Liu et al., 2019; Raffel et al., 2020; Lewis et al., 2020). While finetuning has been a popular approach to transfer to new tasks (Devlin et al., 2019), it is often impractical to finetune a very large model (e.g. $\geq 10 \mathrm{~B}$ parameters). Brown et al. (2020) propose in-context learning as an alternative way to learn a new task. As depicted in Figure 2, the LM learns a new task via inference alone by conditioning on a concatenation of the training data as demonstrations, without any gradient updates.

In-context learning has been the focus of significant study since its introduction. Prior work proposes better ways of formulating the problem (Zhao et al., 2021; Holtzman et al., 2021; Min et al., 2021a), better ways of choosing labeled examples for the demonstrations (Liu et al., 2021; Lu et al., 2021; Rubin et al., 2021), meta-training with an explicit in-context learning objective (Chen et al., 2021; Min et al., 2021b), and learning to follow instructions as a variant of in-context learning (Mishra et al., 2021b; Efrat and Levy, 2020; Wei et al., 2022a; Sanh et al., 2022). At the same time, some work reports brittleness and oversensitivity for in-context learning (Lu et al., 2021; Zhao et al., 2021; Mishra et al., 2021a).

Relatively less work has been done to understand why in-context learning works. Xie et al. (2022) provide theoretical analysis that in-context learning can be formalized as Bayesian inference that

![Figure](./minRethinkingRoleDemonstrations2022_images/7452035e-c407-4143-88d7-35adf7d81a93-02_292_696_232_1091.jpg)

**Figure 2: An overview of in-context learning. The demonstrations consist of $k$ input-label pairs from the training data ( $k=3$ in the figure).**

\begin{table}
\begin{tabular}[t]{|l|l|l|l|}
\hline Model & \# Params & Public & Meta-trained \\
\hline GPT-2 Large & 774M & ✓ & × \\
\hline MetaICL & 774M & ✓ & ✓ \\
\hline GPT-J & 6B & ✓ & × \\
\hline fairseq 6.7B ${ }^{\dagger}$ & 6.7 B & ✓ & × \\
\hline fairseq 13B ${ }^{\dagger}$ & 13B & ✓ & × \\
\hline GPT-3 & $175 \mathrm{~B}^{\ddagger}$ & × & × \\
\hline
\end{tabular}

**Table 1: A list of LMs used in the experiments: GPT-2 (Radford et al., 2019), MetaICL (Min et al., 2021b), GPT-J (Wang and Komatsuzaki, 2021), fairseq LMs (Artetxe et al., 2021) and GPT-3 (Brown et al., 2020). 'Public' indicates whether the model weights are public; 'Meta-trained' indicates whether the model is meta-trained with an in-context learning objective. ${ **^{\dagger}$ We use dense models in Artetxe et al. (2021) and refer them as fairseq LMs for convenience. ${ }^{\ddagger}$ We use the Davinci API (the base version, not the instruct version) and assume it to be 175B, following Gao et al. (2021) and Artetxe et al. (2021).}
\end{table}
uses the demonstrations to recover latent concepts. Razeghi et al. (2022) show that in-context learning performance is highly correlated with term frequencies in the pretraining data. To the best of our knowledge, this paper is the first that provides an empirical analysis that investigates why in-context learning achieves performance gains over zero-shot inference. We find that the ground truth input-label mapping in the demonstrations has only a marginal effect, and measure the impact of finer-grained aspects of the demonstrations.

