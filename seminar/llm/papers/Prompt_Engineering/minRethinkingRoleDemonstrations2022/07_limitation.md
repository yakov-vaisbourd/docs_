# Limitation

Effect of types of tasks and datasets. This paper focuses on the tasks from established NLP benchmarks that have real natural language inputs. Synthetic tasks with more limited inputs may actually use the ground truth labels more, as observed by Rong (2021).

We report macro-level analysis by examining the average performance over multiple NLP datasets, but different datasets may behave differently. Appendix C. 2 discusses this aspect, including findings that there are larger gaps between using the ground truth labels and using the random labels in some dataset-model pairs (e.g., in the most extreme case, nearly 14\% absolute on the financial_phrasebank dataset with GPT-J). Since the first version of our paper, Kim et al. (2022) showed that using negated labels substantially lowers the performance in classification. ${ }^{11}$ We believe it is important to understand to what extend the model needs the ground truth labels to successfully perform in-context learning.

Extensions to generation. Our experiments are limited to classification and multi-choice tasks. We hypothesize that ground truth output may not be

\footnotetext{
${ }^{10}$ We take the perspective that using the unlabeled training data is permitted (Kodirov et al., 2015; Wang et al., 2019b; Schick and Schütze, 2021).
${ }^{11}$ Note that Kim et al. (2022) estimate the random label performance by interpolating with the performance using negated labels, while our paper samples the random labels at uniform.
}
necessary for in-context learning in the open-set tasks such as generation, but leave this to future work. Extending of our experiments to such tasks is not trivial, because it requires a variation of the output which has incorrect input-output correspondence while keeping the correct output distribution (which is important based on our analysis in Section 5).

Since the first version of our paper, Madaan and Yazdanbakhsh (2022) conducted a similar analysis with the chain of thought prompting (Wei et al., 2022b) which generates a rationale to perform complex tasks such as math problems. Madaan and Yazdanbakhsh (2022) show that, while simply using a random rationale in the demonstrations (e.g., pairing with a rationale from a different example) significantly degrades the performance, other types of counterfactual rationales (e.g., wrong equations) do not degrade the performance as much as we thought. We refer to Madaan and Yazdanbakhsh (2022) for more discussions on what aspects of the rationale matter or do not matter.

