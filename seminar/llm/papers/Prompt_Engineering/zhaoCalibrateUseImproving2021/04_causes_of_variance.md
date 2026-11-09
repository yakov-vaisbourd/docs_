# 4. What Causes the High Variance?

We next analyze why GPT-3's accuracy varies across different training examples, permutations, and prompt formats. Concretely, we show that the variance arises because LMs are biased towards outputting answers that are (1) frequent in the prompt (majority label bias), (2) towards the end of the prompt (recency bias), and (3) common in the pre-training data (common token bias).

Majority Label Bias We find that GPT-3 is biased towards answers that are frequent in the prompt. A trivial case is when a text classification prompt has a class imbalance, e.g., more Positive than Negative sentiment examples. This is demonstrated in the "unbalanced" region of Figure 4: when one class is more common, GPT-3 2.7B is heavily biased towards predicting that class. Since the SST-2 sentiment analysis dataset is balanced, this bias causes large accuracy degradations. The majority label bias also explains why we frequently observe a drop in accuracy when moving from 0-shot to 1-shot-we found that the drop is due to the model frequently repeating the class of the one training example.

The majority label bias also occurs for generation tasks. On the validation set for 4-shot LAMA with GPT-3 2.7B, 50.2\% of the model predictions are a repeat of one of the four training answers (the correct repeat rate is 24.7\%). Overall, the majority label bias helps to explain why different choices for the training examples heavily influence GPT-3's accuracy- it shifts the distribution of model predictions.

Recency Bias The model's majority label bias is aggravated by its recency bias: the tendency to repeat answers that appear towards the end of the prompt. The "balanced" region of Figure 4 demonstrates this. For instance, when two Negative examples appear at the end (P P N N), the model will heavily prefer the Negative class. Moreover, the recency bias can outweigh the majority label bias, e.g., the "P P P N" training set leads to nearly 90\% of predictions being Negative, despite $\frac{3}{4}$ of the training examples being Positive.

Recency bias also affects generation tasks. For 4-shot LAMA, the training answers that are closer to the end of the prompt are more likely to be repeated by the model. Concretely, the model "overpredicts" the answer from the 1st, 2nd, 3rd, and 4th training example by 8.5\%, 8.3\%, 14.3\%, and $16.1 \%$, respectively. ${ }^{2}$ Overall, recency bias helps to explain why the permutation of the training examples is important-the ordering of the examples heavily influences the distribution of the model predictions.

Common Token Bias Finally, we find that GPT-3 is biased towards outputting tokens that are common in its pretraining distribution, which is likely suboptimal for the distribution of answers on the downstream task. A simple case of this occurs for the LAMA fact retrieval dataset, where the model often predicts common entities such as "America" when the ground-truth answer is instead a rare entity.

A more nuanced case of the common token bias occurs for

\footnotetext{
${ }^{2}$ Over all relations, as well as three different sets of training examples, the model repeats the training example at a rate of $20.7 \%, 19.8 \%, 29.9 \%$, and $26.8 \%$. The ground-truth repeat rate is $12.2 \%, 11.5 \%, 15.6 \%$, and $10.7 \%$. We define "overpredicts" as the model's repeat rate minus the ground-truth repeat rate.
}
text classification. Recall that the model makes predictions by generating the label name associated with each class. Because certain label names appear more frequently in the pre-training data, the model will be inherently biased towards predicting certain classes. For example, on DBPedia (a balanced 14-way topic classification dataset), GPT-3 predicts the "book" class $11 \times$ more often than the "artist" class. In fact, there is a moderate correlation \((r=0.67)\) between the frequency of a DBPedia label name and the rate at which GPT-3 predicts its class. ${ }^{3}$ Overall, the common token bias helps to explain why the choice of label names is important, and why the model struggles on rare answers.

The Impact of Biases on Model Predictions We find that the end result of the above three biases is typically a simple shift in the model's output distribution. For example, Figure 5 visualizes this shift for a SST-2 sentiment prompt.

![Figure](./zhaoCalibrateUseImproving2021_images/bfa78706-72f5-4d26-85c6-dd246cd5745f-05_174_767_960_217.jpg)

**Figure 5. The Positive class probability for 25 random test inputs for a particular sentiment analysis prompt. Negative ground-truth examples are marked with ● and Positive are marked with ●.**

The prompt used in Figure 5 and the model's intrinsic biases cause it to frequently predict high confidence for the Positive class. Since the default 50\% threshold is used to make predictions, this results in frequent false positives. Importantly, note that if we could optimally set the classification threshold (p(Positive) = 0.68 in this case), the classifier would be highly accurate (94\% on the validation set).

