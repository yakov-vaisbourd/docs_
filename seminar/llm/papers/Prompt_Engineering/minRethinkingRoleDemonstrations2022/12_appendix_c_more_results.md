# C More Experimental Results

# C. 1 Gold labels vs. random labels

Figure 11 shares the same interface as Figure 3, but all models are evaluated on 3 classification and 3 multi-choice datasets and are thus comparable to each other.

# C. 2 Random labels from true distribution of labels \& Task breakdown

In Section 4, random labels are sampled from the label space from a uniform distribution. We experiment with another variant of demonstrations in the classification tasks, where labels are randomly sampled from the true distribution of labels on the training data. This may have large impact if labels are far from uniform on the training data. Results indicate that performance drop from using gold labels is further reduced compared to using uniformly random labels: with Channel MetaICL, the gap is reduced from $1.9 \%$ to $1.3 \%$ absolute, and with Channel GPT-J, the gap is reduced from 5.0\% to 3.5\% absolute.

Figure 12 shows performance gap between using gold labels and using random labels per dataset. We find that the trend that the gap is smaller than previously thought is consistant across most datasets. Nonetheless, there are a few outlier datasets where performance gap is non-negligible, such as financial_phrasebank and a few hate speech detection datasets. Future work may investigate on which tasks the model makes more use of the correctly paired training data.

# C. 3 More variants of the demonstrations

We explored demonstrations with a constant label where all labels in the demonstrations are replaced with a constant text, "answer". Specifically, a prediction is made via $\operatorname{argmax}_{y \in \mathcal{C}} P\left(y \mid x_{1}\right.$, answer $\ldots x_{k}$, answer, $x$ ). This can be viewed as another way to remove the impact of the label space while keeping the impact of the distribution of the input text. However, results are consistently worse than the results of demonstrations with random English labels. We think this is because constant labels actually change the format of the demonstrations, since they can be viewed as part of a separator between different demonstration examples.

We also explored demonstrations with the test input where all inputs in the demonstrations are replaced with the test input, each paired with a ran-
dom label. Specifically, a prediction is made via \(\operatorname{argmax}_{y \in \mathcal{C}} P\left(y \mid x, \tilde{y}_{1} \ldots x, \tilde{y}_{k}, x\right)\), where $\tilde{y}_{i}(1 \leq$ $i \leq k$ ) is randomly sampled at uniform from $\mathcal{C}$. This variant is seemingly a reasonable choice given that it satisfies the condition that the inputs in the demonstrations come from the same distribution as the test input (since they are identical), and using random labels is as good as using gold labels. Nonetheless, we find that this variant is significantly worse than most other methods with demonstrations. We think this is because using the constant input for all demonstration example significantly changes the format of the sequence, since the input can be viewed as part of a separator between different demonstration examples.

![Figure](./minRethinkingRoleDemonstrations2022_images/7452035e-c407-4143-88d7-35adf7d81a93-16_1808_1482_355_330.jpg)

**Figure 12: Performance gap from using the demonstrations with gold labels to using the demonstrations with random labels. Datasets are sorted in descending order. The top two figures use random labels that are sampled at uniform, with Channel MetaICL and Channel GPT-J, respectively. The bottom two figures use random labels that are sampled from a true distribution of labels on the training data, with Channel MetaICL and Channel GPT-J, respectively.**

\begin{table}
\begin{tabular}[t]{|l|l|l|}
\hline Dataset & Type & Example \\
\hline \multirow{2}{*}{MRPC} & Minimal & sentence 1: Cisco pared spending to compensate for sluggish sales . [SEP] sentence 2: In response to sluggish sales , Cisco pared spending. In \{equivalent|not_equivalent\} \\
\hline & Manual & Cisco pared spending to compensate for sluggish sales . In The question is: In response to sluggish sales , Cisco pared spending . True or False? \n The answer is:\{True|False\} \\
\hline \multirow{2}{*}{RTE} & Minimal & sentence 1: The girl was found in Drummondville. [SEP] sentence 2: Drummondville contains the girl. \n \{entailment|not_entailment\} \\
\hline & Manual & The girl was found in Drummondville. In The question is: Drummondville contains the girl. True or False? \n The answer is:\{True|False\} \\
\hline \multirow{2}{*}{Tweet_eval-hate} & Minimal & The Truth about \#Immigration \n \{hate|non-hate\} \\
\hline & Manual & Tweet: The Truth about \#Immigration \n Sentiment: \{against|favor\} \\
\hline \multirow{2}{*}{SICK} & Minimal & sentence 1: A man is screaming. [SEP] sentence 2: A man is scared. In \{contradiction|entailment|neutral\} \\
\hline & Manual & A man is screaming. In The question is: A man is scared. True or False? \n The answer is: \{False|True|Not sure\} \\
\hline \multirow{2}{*}{poem-sentiment} & Minimal & willis sneered: \n \{negative|no_impact|positive\} \\
\hline & Manual & willis sneered: \n The sentiment is: \{negative|no_impact|positive\} \\
\hline \multirow[t]{2}{*}{OpenbookQA} & Minimal & What creates a valley? \n \{feet|rock|water|sand\} \\
\hline & Manual & The question is: What creates a valley? \n The answer is: \{feet|rock|water|sand\} \\
\hline \multirow[t]{2}{*}{CommonsenseQA} & Minimal & What blocks sunshine? \n \{summer|park|desktop|sea|moon\} \\
\hline & Manual & The question is: What blocks sunshine? \n The answer is: \{summer|park|desktop|sea|moon\} \\
\hline \multirow[t]{2}{*}{COPA} & Minimal & Effect: I coughed. \n \{Cause: I inhaled smoke.|Cause: I lowered my voice.\} \\
\hline & Manual & I coughed because \{I inhaled smoke.|I lowered my voice.\} \\
\hline \multirow{2}{*}{ARC} & Minimal & Which biome has the most vegetation? \n \{desert|forest|grassland|tundra\} \\
\hline & Manual & The question is: Which biome has the most vegetation? \n The answer is: \{desert|forest| grassland|tundra\} \\
\hline
\end{tabular}

**Table 3: A list of minimal templates taken from Ye et al. (2021); Min et al. (2021b) and manual templates taken from Holtzman et al. (2021); Zhao et al. (2021). Details provided in Appendix B. See Figure 6 for discussion in empirical results. The input and the label are in the red text and in the blue text, respectively. Note that | is used to separate different options for the labels.**
\end{table}

\begin{table}
\begin{tabular}[t]{|l|l|}
\hline Demos w/ gold labels & (Format ✓ Input distribution ✓ Label space ⇃ Input-label mapping ⇃) Circulation revenue has increased by 5\% in Finland and 4\% in Sweden in 2008. In positive Panostaja did not disclose the purchase price. In neutral \\
\hline Demos w/ random labels & (Format ✓ Input distribution ✓ Label space ⇃ Input-label mapping X) Circulation revenue has increased by 5\% in Finland and 4\% in Sweden in 2008. In neutral Panostaja did not disclose the purchase price. In negative \\
\hline OOD Demos w/ random labels & (Format ✓ Input distribution X Label space ✓ Input-label mapping X) Colour-printed lithograph. Very good condition. Image size: $15 \times 23$ 1/2 inches. \n neutral Many accompanying marketing claims of cannabis products are often well-meaning. In negative \\
\hline Demos w/ random English words & (Format ✓ Input distribution ✓ Label space X Input-label mapping X) Circulation revenue has increased by 5\% in Finland and 4\% in Sweden in 2008. In unanimity Panostaja did not disclose the purchase price. In wave \\
\hline Demos w/o labels & (Format X Input distribution ✓ Label space X Input-label mapping X) Circulation revenue has increased by 5\% in Finland and 4\% in Sweden in 2008. Panostaja did not disclose the purchase price. \\
\hline Demos labels only & (Format $\boldsymbol{X}$ Input distribution $\boldsymbol{X}$ Label space ✓ Input-label mapping $\boldsymbol{X}$ ) positive neutral \\
\hline
\end{tabular}

**Table 4: Example demonstrations when using methods in Section 5. The financial_phrasebank dataset with $\mathcal{C**=$ \{"positive", "neutral", "negative"\} is used. Red text indicates the text is sampled from an external corpus; blue text indicates the labels are randomly sampled from the label set; purple text indicates a random English word.}
\end{table}

\title{
Rethinking the Role of Demonstrations: What Makes In-Context Learning Work?
}

\author{
Anonymous ACL submission
}

We are glad that all reviewers find that the paper is novel (8jk5, LQ6N, 92YB, 7E5P), of interest to the broader NLP community (LQ6N, 92YB, 7E5P), supported by solid experiments (8jk5, LQ6N, 92YB, 7E5P), and well-written (8jk5, LQ6N, 92YB). Reviewers gave comments on more discussion, limitations and avenues for future work. We will incorporate them in the next version of the paper. ${ }^{1}$
Adding discussion on robustness of LMs (8jk5, 7E5P): We think the fact that LMs do not use the input-label correspondence does not necessarily mean that LMs are robust to other aspects of the demonstration. Nonetheless, it will be an interesting avenue for future work, given that LMs are highly sensitive to small changes in the demonstrations (??).
Adding discussion with respect to the model size (8jk5): Absolute differences are similar across different model sizes (Figure 3), but since the large models have higher absolute performance, the relative differences are larger with larger models.
When our findings hold or do not hold (8jk5): We believe that the findings will hold only when some notion of input-label correspondences are already captured during pre-training-for tasks whose input-label correspondences during pretraining is sparse, the demonstrations with random labels are highly unlikely to work. This has been shown by ? in a synthetic setup, and future work can revisit this with more natural data.
Risk in applying in-context learning, Recommendation to practitioners (8jk5, LQ6N): Since the models do not capture the correspondence from the demonstrations, it is possible that models are simply relying on some notation of input-label correspondence during pre-training. Thus, applying in-context learning for problems that were not

\footnotetext{
${ }^{1}$ We answered reviewers' questions on the OpenReview page, and address higher-level comments here.
}
in the pretraining data would be potentially risky, and practitioners may want to collect the labeled data and fine-tune the model.

Where do the gains from demonstrations come from? (92YB): We think gains from demonstrations are mainly due to the specification of the input distribution and the label space rather than the input-label correspondence, as Section 5 indicates. We will clarify this in the next version of the paper.

Concrete answer to "why" using random labels keeps reasonable performance (LQ6N): We think it is highly likely to be because the model has been exposed to some notion of input-label correspondence during pre-training, so that the demonstrations play a role of triggering them at inference.
Consideration when training large LMs (LQ6N): Due to compute limitations, we were not be able to provide recommendations that are empirically supported. Future work may investigate aspects of language model pretraining that affect the findings, including the pretraining data and the objective.
Word ordering matters? (7E5P): We think it is an important avenue for future work. For this paper, we did not include it in our scope due to requiring multiple times more experiments.
More task types, e.g., generation (7E5P): Extending this work to generation tasks is not very trivial because designing the demonstrations that do not keep the input-output correspondence but keep the output distribution is difficult. For instance, if we simply replace the output with a random output as we did in the classification tasks, it will destroy both the input-output correspondence and the output distribution. We hope future work can investigate more in this direction.
Stronger link with instruction-following models (7E5P): We will add discussion on ? who find that instructions that are irrelevant or even misleading lead to performance gains as much as "good"
instructions do.

# Limitation

This paper focuses on the tasks from established NLP benchmarks that have real natural language inputs. Synthetic tasks with more limited inputs may actually use the ground truth labels more, as observed by ?. Our paper mainly includes macrolevel analysis by examining the average performance over multiple NLP datasets, but different datasets may behave differently. Appendix discusses this aspect, including findings that there are larger gaps between using the ground truth labels and using the random labels in some dataset-model pairs (e.g., in the most extreme case, nearly 14\% absolute on the financial_phrasebank dataset with GPT-J). We believe it is important to understand in which cases the ground truth labels matter or not, which we leave for future work. Furthermore, our work is limited to classification and multi-choice tasks. Extension to the open-set tasks such as generation is not trivial, since it is unclear how to remove the input-output correspondence while keeping the correct output distribution. We leave the extension for future work.
