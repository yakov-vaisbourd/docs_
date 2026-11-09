# 3. Accuracy Varies Highly Across Prompts

This section studies how GPT-3's accuracy changes as we vary each aspect of the prompt (training examples, permutation, format). We focus on a subset of the datasets to simplify our analysis; in Section 5 we show that our findings hold across all of the datasets we study.

GPT-3's accuracy depends highly on both selection and permutation of training examples. Concretely, we use a fixed prompt format and choose different random sets of training examples. For each set of training examples, we evaluate the accuracy for all possible permutations.

Figure 2 shows the results for SST-2 (4-shot, GPT-3 2.7B). Surprisingly, varying the permutation can be as important, or even more important, than which training examples are chosen. For example, varying the permutation of the training examples can cause accuracy to go from near chance (54.3\%) to near state-of-the-art (93.4\%). For a qualitative example of the sensitivity to permutations, see Table 2 in Appendix A. This high importance on example order is in contrast to standard machine learning, where the ordering of examples during training is typically an afterthought.

The variance persists with more data and larger models. Adding more training examples into the prompt does not necessarily reduce the variance in accuracy. We sweep over the number of training examples for three different datasets in Figure 1 (red curves). The variance remains high even when we use 16 training examples. Moreover, adding more training examples can sometimes hurt accuracy (e.g., mean accuracy drops from 36.0\% to 25.9\% for DBPedia 0-shot to 1-shot). The variance in accuracy can also remain high when using larger models, e.g., the left of Figure 1.

![Figure](./zhaoCalibrateUseImproving2021_images/bfa78706-72f5-4d26-85c6-dd246cd5745f-04_419_1644_256_213.jpg)

**Figure 4. Majority label and recency biases cause GPT-3 to become biased towards certain answers and help to explain the high variance across different examples and orderings. Above, we use 4-shot SST-2 with prompts that have different class balances and permutations, e.g., [P P N N] indicates two positive training examples and then two negative. We plot how often GPT-3 2.7B predicts Positive on the balanced validation set. When the prompt is unbalanced, the predictions are unbalanced (majority label bias). In addition, balanced prompts that have one class repeated near the end, e.g., end with two Negative examples, will have a bias towards that class (recency bias).**

GPT-3's accuracy depends highly on prompt format. We next keep the set of training examples and permutations fixed but vary the prompt format. We focus on SST-2, and we manually design an additional 14 prompt formats. The formats include question-answer templates, conversationstyle templates, prompts that resemble Web pages, and variations on the label names (all formats available in Table 7 in Appendix B). The accuracy for ten of the formats is shown in Figure 3. We find that some of the formats are better than others on average. However, all of the formats still suffer from high variance across different training sets.

