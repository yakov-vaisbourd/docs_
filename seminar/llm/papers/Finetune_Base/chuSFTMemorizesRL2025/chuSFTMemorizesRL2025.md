\title{
SFT Memorizes, RL Generalizes: A Comparative Study of Foundation Model Post-training
}

\author{
Tianzhe Chu ${ }^{\text {★ }}$ Yuexiang Zhai ${ }^{\text {૪ }}$ * Jihan Yang ${ }^{\text {* }}$ Shengbang Tong ${ }^{\text {◇ }}$ Saining Xie ${ }^{\text {★★ }}$ Dale Schuurmans ${ }^{\text {★ }}$ Quoc V. Le ${ }^{\text {★ }}$ Sergey Levine ${ }^{\text {}}$ Yi Ma ${ }^{\text {◇ }}$
}

\begin{abstract}
Supervised fine-tuning (SFT) and reinforcement learning (RL) are widely used post-training techniques for foundation models. However, their respective role in enhancing model generalization in rule-based reasoning tasks remains unclear. This paper studies the comparative effect of SFT and RL on generalization and memorization, focusing on text-based and visual reasoning tasks. We introduce GeneralPoints, an arithmetic reasoning card game, and also consider V-IRL, a real-world navigation environment, to assess how models trained with SFT and RL generalize to unseen variants in both novel textual rules and visual domains. We show that RL, especially when trained with an outcomebased reward, generalizes in both the rule-based textual and visual environments. SFT, in contrast, tends to memorize the training data and struggles to generalize out-of-distribution in either scenario. Further analysis reveals that RL improves the model's underlying visual recognition capabilities, contributing to its enhanced generalization in visual domains. Despite RL's superior generalization, we show that SFT is still helpful for effective RL training: SFT stabilizes the model's output format, enabling subsequent RL to achieve its performance gains. These findings demonstrate the advantage of RL for acquiring generalizable knowledge in complex, multimodal tasks.
\end{abstract}

\footnotetext{
*Equal contribution . ${ }^{\text {৯ }}$ HKU, ${ }^{\text {V }}$ UC Berkeley, ${ }^{\text {* }}$ Google DeepMind, *NYU, *University of Alberta. All experiments are conducted outside of Google. Project page: https://tianzhechu.com/SFTvsRL. Correspondence to: Tianzhe Chu <tianzhechu@gmail.com>, Yuexiang Zhai <simonzhai@berkeley.edu>.

Proceedings of the $42{ }^{\text {nd }}$ International Conference on Machine Learning, Vancouver, Canada. PMLR 267, 2025. Copyright 2025 by the author(s).
}

