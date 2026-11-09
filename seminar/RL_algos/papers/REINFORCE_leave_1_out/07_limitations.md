# 7 Limitations

As one of the limitations of our work, we do not study the reward model (RM) over-optimization, which refers to the problem when optimizing the
trajectory of the proxy reward, the model diverges from the "gold" reward objective (Gao et al., 2022). This aspect has not been studied yet also for iterative fine-tuning methods such as RAFT and deserves a dedicated study. We leave this to future work.

Another limitation is the exploration of RLOO baseline in a single token action framework, where partial sequences are modeled. In this work, we show that modeling partial sequences are unnecessary undertaking in the RLHF context. However, understanding how RLOO performs in such modeling formulation can potentially be useful in other contexts.

Finally, we limit our work to rewards from human preferences and we did not explore the other rewards such as ROUGE, BLEU, or other metrics used in the NLP.

