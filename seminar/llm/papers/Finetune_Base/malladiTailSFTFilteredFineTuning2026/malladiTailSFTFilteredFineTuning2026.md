\title{
TailSFT: Filtered Fine-Tuning Improves Post-Training Performance
}

\author{
Sadhika Malladi ${ }^{\dagger}$ Samy Jelassi ${ }^{\ddagger}$ Dylan J. Foster ${ }^{\ddagger}$ Jordan T. Ash ${ }^{\ddagger}$ Akshay Krishnamurthy ${ }^{\ddagger}$
}

\begin{abstract}
Reinforcement learning post-training drives reasoning and agentic capabilities in modern AI systems, yet a growing body of work shows that it is most effective when used to fine-tune an already capable base model. We question whether existing pipelines yield models that are most suitable for reinforcement learning. Building on prior work highlighting the role of coverage and pass@ $K$ as predictors of post-RL performance, we design a simple modification to supervised fine-tuning, TailSFT, which filters out already fit sequences during training, thereby focusing learning on under-modeled regions, or the tail, of the data distribution. We justify and validate the design choices in TailSFT, particularly the specific filtering criteria, through a combination of controlled experiments and theoretical analysis. On OLMo-3 7B, TailSFT often improves pass@16 performance on math and coding evaluations, with gains up to 17\% absolute, while incurring minimal computational overhead. These higher-coverage checkpoints consistently translate to up to 4\% absolute pass@1 gains in subsequent GRPO runs, demonstrating that TailSFT checkpoints serve as better initializations for RL. We further introduce a lightweight diagnostic for identifying settings where TailSFT is most likely to help. More broadly, our results motivate a principled, stage-aware approach to model development, in which intermediate checkpoints are judged by how effectively they support subsequent training.
\end{abstract}

![Figure](./malladiTailSFTFilteredFineTuning2026_images/b0c2e9de-14c4-464c-9bb3-a4879161b9bf-01_645_1326_1394_391.jpg)

**Figure 1: TailSFT yields better coverage, possibly at the cost of worse cross-entropy, which translates to stronger post-RL performance. Standard SFT achieves lower cross-entropy, while TailSFT preserves higher pass@ $K$ at large $K$. Starting from this higher-coverage checkpoint, the same RL procedure yields higher pass@1. Curves are illustrative.**

\footnotetext{
${ }^{\dagger}$ University of California San Diego, sadhika.malladi98@gmail.com. Work partially completed while at Microsoft Research NYC.
${ }^{\ddagger}$ Microsoft Research NYC, \{samyjelassi,dylanfoster,ash.jordan,akshaykr\}@microsoft.com.
}

