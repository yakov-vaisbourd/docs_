\title{
Back to Basics: Revisiting REINFORCE-Style Optimization for Learning from Human Feedback in LLMs
}

\author{
Arash Ahmadian ${ }^{\mathbf{1}, \mathbf{2}}$, Chris Cremer ${ }^{\mathbf{2}}$ Matthias Gallé ${ }^{\mathbf{2}}$, Marzieh Fadee ${ }^{\mathbf{1}}$, Julia Kreutzer ${ }^{\mathbf{1}}$, Olivier Pietquin ${ }^{\mathbf{2}}$, Ahmet Üstün ${ }^{\mathbf{1}}$, Sara Hooker ${ }^{\mathbf{1}}$ \\ ${ }^{1}$ Cohere For AI, ${ }^{2}$ Cohere \\ \{arash,olivier,ahmet,sarahooker\}@cohere.com
}

\begin{abstract}
AI alignment in the shape of Reinforcement Learning from Human Feedback (RLHF) is increasingly treated as a crucial ingredient for high performance large language models. Proximal Policy Optimization (PPO) has been positioned by recent literature as the canonical method for the RL part of RLHF. However, it involves both high computational cost and sensitive hyperparameter tuning. We posit that most of the motivational principles that led to the development of PPO are less of a practical concern in RLHF and advocate for a less computationally expensive method that preserves and even increases performance. We revisit the formulation of alignment from human preferences in the context of RL. Keeping simplicity as a guiding principle, we show that many components of PPO are unnecessary in an RLHF context and that far simpler REINFORCE-style optimization variants outperform both PPO and newly proposed "RLfree" methods such as DPO and RAFT. Our work suggests that careful adaptation to LLMs alignment characteristics enables benefiting from online RL optimization at low cost.
\end{abstract}

