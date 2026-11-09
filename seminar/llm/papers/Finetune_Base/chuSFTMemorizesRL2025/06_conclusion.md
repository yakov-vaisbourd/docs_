# 6. Conclusion, Discussion, and Limitations

In this paper, we present a comprehensive analysis of the generalization effects of foundation model post-training techniques, specifically RL and SFT. Through extensive

![Figure](./chuSFTMemorizesRL2025_images/c29435c9-0689-4d6d-8ce6-4fa35255cad0-08_516_674_1009_1140.jpg)

**Figure 10: In-distribution vs. OOD performance growth on GP-L. We record RL experiments with different number of verification iterations (VIter) as scaling up training compute (color transparency).**

experiments on the GeneralPoints and V-IRL tasks, we demonstrated that RL exhibits superior performance in learning generalizable knowledge, while SFT tends to merely memorize the training data, across both the rule and visual variations. This phenomenon consistently occurs across multimodal arithmetic and spatial reasoning capabilities. In addition, we studied the effect of RL on visual recognition, the role of SFT, and the role of verification steps. During our study, two challenges were not resolved.

Failure of SFT on GP-VL. In Figure 5 for GP-VL, we observe that SFT fails to achieve a comparable indistribution performance with RL. To mitigate the variance introduced by hyperparameter choices, we additionally conduct 10 more experiments with different learning rates and tunable components (Figure 16), none of which exhibits a strong increasing trend like RL (Figure 17).

Given our observation that scaling up SFT degrades visual recognition capabilities (Figure 8), we hypothesize that SFT locally overfits to reasoning tokens while neglecting recognition tokens, possibly due to the higher frequency of reasoning tokens (see Figure 11 as example). We leave further investigation to future work.

Limits of RL in corner cases. As discussed in Section 5.4, SFT is necessary for effective RL training on Llama-3.2. We investigate applying RL to an overly-tuned SFT checkpoint. As demonstrated in Figure 19, RL is unable to recover out-of-distribution performance when starting from such a checkpoint. Example failure cases are illustrated in Figure 21, where the model collapses to the training rule. These results, together with findings in Section 5.4, indicate that RL has limited effectiveness when applied to extremely underfit or overfit initial checkpoints. Further research is needed to delineate the conditions under which SFT facilitates effective RL.

