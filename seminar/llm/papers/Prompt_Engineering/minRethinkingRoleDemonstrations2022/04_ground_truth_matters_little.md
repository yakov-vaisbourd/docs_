# 4 Ground Truth Matters Little

## 4.1 Gold labels vs. random labels

To see the impact of correctly-paired inputs and labels in the demonstrations-which we call the ground truth input-label mapping-we compare the following three methods. ${ }^{4}$

No demonstrations is a typical zero-shot method that does not use any labeled data. A prediction is made via \(\operatorname{argmax}_{y \in \mathcal{C}} P(y \mid x)\), where $x$ is the test input and $\mathcal{C}$ is a small discrete set of possible labels.
Demonstrations w/ gold labels are used in a typical in-context learning method with $k$ labeled examples \(\left(x_{1}, y_{1}\right) \ldots\left(x_{k}, y_{k}\right)\). A concatenation of $k$ input-label pairs is used to make a prediction via \(\operatorname{argmax}_{y \in \mathcal{C}} P\left(y \mid x_{1}, y_{1} \ldots x_{k}, y_{k}, x\right)\).

\footnotetext{
${ }^{2}$ Three classification and three multi-choice: MRPC, RTE, Tweet_eval-hate, OpenbookQA, CommonsenseQA, COPA.
${ }^{3}$ Known to be better for imbalanced classes.
${ }^{4}$ Without loss of generality, all methods in Section 4 and 5 are described based on the direct method, but can be trivially converted to the channel method by flipping $x$ and $y$.
}

![Figure](./minRethinkingRoleDemonstrations2022_images/7452035e-c407-4143-88d7-35adf7d81a93-04_245_1478_230_329.jpg)

**Figure 4: Results with varying number of correct labels in the demonstrations. Channel and Direct used for classification and multi-choice, respectively. Performance with no demonstrations (blue) is reported as a reference.**

Demonstrations w/ random labels are formed with random labels, instead of gold labels from the labeled data. Each $x_{i}(1 \leq i \leq$ \(k)\) is paired with $\tilde{y}_{i}$ that is randomly sampled at uniform from $\mathcal{C}$. A concatenation of \(\left(x_{1}, \tilde{y}_{1}\right) \ldots\left(x_{k}, \tilde{y}_{k}\right)\) is then used to make a prediction via \(\operatorname{argmax}_{y \in \mathcal{C}} P\left(y \mid x_{1}, \tilde{y}_{1} \ldots x_{k}, \tilde{y}_{k}, x\right)\).

Results are reported in Figure 3. First, using the demonstrations with gold labels significantly improves the performance over no demonstrations, ${ }^{5}$ as it has been consistently found in much of prior work (Brown et al., 2020; Zhao et al., 2021; Liu et al., 2021). We then find that replacing gold labels with random labels only marginally hurts performance. The trend is consistent over nearly all models: models see performance drop in the range of 0-5\% absolute. There is less impact in replacing labels in multi-choice tasks (1.7\% on average) than in classification tasks (2.6\% absolute).

This result indicates that the ground truth inputlabel pairs are not necessary to achieve performance gains. This is counter-intuitive, given that correctly paired training data is critical in typical supervised training-it informs the model of the expected input-label correspondence required to perform the downstream task. Nonetheless, the models do achieve non-trivial performance on the downstream tasks. This strongly suggests that the models are capable of recovering the expected inputlabel correspondence for the task; however, it is not directly from the pairings in the demonstrations.

It is also worth noting that there is particularly little performance drop in MetaICL: 0.1-0.9\% absolute. This suggests that meta-training with an explicit in-context learning objective actually encourages the model to essentially ignore the input-

\footnotetext{
${ }^{5}$ There are some exceptions, e.g., in the classification tasks, Direct GPT-2, Direct GPT-J and Direct fairseq 6.7B models are not significantly better than random guessing on many datasets; Channel fairseq 13B has significantly better nodemonstrations performance compared to demonstrations with gold labels. We thus discuss the results from these models less significantly for the rest of analysis.
}

![Figure](./minRethinkingRoleDemonstrations2022_images/7452035e-c407-4143-88d7-35adf7d81a93-04_312_711_645_1094.jpg)

**Figure 5: Ablations on varying numbers of examples in the demonstrations \((k)\). Models that are the best under 13B in each task category (Channel MetaICL and Direct GPT-J, respectively) are used.**

label mapping and exploit other components of the demonstrations (more discussion in Section 5.4).

In Appendix C.2, we provide additional results showing that (1) selecting random labels from a true distribution of labels (instead of a uniform distribution) reduces the gap even further, and (2) the trends may depend on the dataset, although the overall trend is consistent over most datasets.

## 4.2 Ablations

For additional ablations, we experiment with 5 classification and 4 multi-choice datasets. ${ }^{6}$

Does the number of correct labels matter? To further examine the impact of correctness of labels in the demonstrations, we conduct an ablation study by varying the number of correct labels in the demonstrations. We evaluate "Demonstrations w/ $a \%$ correct labels" \((0 \leq a \leq 100)\) which consist of $k \times a / 100$ correct pairs and \(k \times(1-a / 100)\) incorrect pairs (see Algorithm 1 in Appendix B). Here, $a=100$ is the same as typical in-context learning, i.e., demonstrations w/ gold labels.

Results are reported in Figure 4. Model performance is fairly insensitive to the number of correct labels in the demonstrations. In fact, always using incorrect labels significantly outperforms no-

\footnotetext{
${ }^{6}$ Classification includes: MRPC, RTE, Tweet_eval-hate, SICK, poem-sentiment; Multi-choice includes OpenbookQA, CommonsenseQA, COPA and ARC.
}

![Figure](./minRethinkingRoleDemonstrations2022_images/7452035e-c407-4143-88d7-35adf7d81a93-05_245_1478_230_329.jpg)

**Figure 6: Results with minimal templates and manual templates. '+T' indicates that manual templates are used. Channel and Direct used for classification and multi-choice, respectively.**

demonstrations, e.g., preserving 92\%, 100\% and 97\% of improvements from using the demonstrations with MetaICL in classification, MetaICL in multi-choice, and GPT-J in multi-choice, respectively. In contrast, GPT-J in classification sees relatively significant performance drop with more incorrect labels, e.g., nearly 10\% drop in performance when always using incorrect labels. Still, always using incorrect labels is significantly better than no demonstrations.

Is the result consistent with varying $\boldsymbol{k}$ ? We study the impact of the number of input-label pairs \((k)\) in the demonstrations. Results are reported in Figure 5. First, using the demonstrations significantly outperforms the no demonstrations method even with small \(k(k=4)\), and performance drop from using gold labels to using random labels is consistently small across varying $k$, in the range of $0.8-1.6 \% .^{7}$ Interestingly, model performance does not increase much as $k$ increases when $k \geq 8$, both with gold labels and with random labels. This is in contrast with typical supervised training where model performance rapidly increases as $k$ increases, especially when $k$ is small. We hypothesize that larger labeled data is beneficial mainly for supervising the input-label correspondence, and other components of the data like the example inputs, example labels and the data format are easier to recover from the small data, which is potentially a reason for minimal performance gains from larger $k$ (more discussion in Section 5).

Is the result consistent with better templates? While we use minimal templates by default, we also explore manual templates, i.e., templates that are manually written in a dataset-specific manner, taken from prior work (details in Appendix B). Figure 6 shows that the trend-replacing gold labels with random labels barely hurting performance- holds with manual templates. It is worth noting

\footnotetext{
${ }^{7}$ With an exception of 4.4\% in classification with $k=4$, likely due to a high variance with a very small value of $k$.
}

![Figure](./minRethinkingRoleDemonstrations2022_images/7452035e-c407-4143-88d7-35adf7d81a93-05_249_718_662_1087.jpg)

**Figure 7: Four different aspects in the demonstrations: the input-label mapping, the distribution of the input text, the label space, and the use of input-label pairing as the format of the demonstrations.**

that using manual templates does not always outperform using minimal templates.

