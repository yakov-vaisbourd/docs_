# E Analysis of Cross-Domain Masked LM Loss

In Section §3.2, we provide RoBERTA's masked LM loss before and after dapt. We display crossdomain masked-LM loss in Table 12, where we evaluate masked LM loss on text samples in other domains after performing DAPT.

We observe that the cross-domain masked-LM loss mostly follows our intuition and insights from the paper, i.e. RoBERTA's pretraining corpus and News are closer, and BioMed to CS (relative to other domains). However, our analysis in §3.1 illustrates that Reviews and News also have some similarities. This is supported with the loss of RoBERTA that is adapted to News, calculated on a sample of Reviews. However, RoBERTA that is adapted to Reviews results in the highest loss for a News sample. This is the case for all domains. One of the properties that distinguishes Reviews from all other domains is that its documents are significantly shorter. In general, we find that cross-DAPT masked-LM loss can in some cases be a noisy predictor of domain similarity.

