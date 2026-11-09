# 1. Introduction

Few-shot learning-the ability to learn tasks with limited examples-is an important aspect of intelligence (Lake et al., 2015; Yogatama et al., 2019). Recent work shows that large neural language models can perform few-shot learning without finetuning (Radford et al., 2019; Brown et al., 2020). Specifically, GPT-3 (Brown et al., 2020) can perform numerous tasks when provided a few examples in a natural language prompt. For example, to perform sentiment analysis one can condition GPT-3 on a prompt such as:

Input: Subpar acting. Sentiment: Negative
Input: Beautiful film. Sentiment: Positive
Input: Amazing. Sentiment:

\footnotetext{
*Equal contribution ${ }^{1}$ UC Berkeley ${ }^{2}$ University of Maryland ${ }^{3}$ UC Irvine. Correspondence to: Eric Wallace <ericwallace@berkeley.edu>.
}
where the first two lines correspond to two training examples and the last line is a test example. To make predictions, the model predicts whether the subsequent token is more likely to be the word "Positive" or "Negative".

This style of few-shot "in-context" learning is interesting because it shows that the model can learn without parameter updates. And, more importantly, it has numerous practical advantages over the now-standard approach of finetuning (Radford et al., 2018; Devlin et al., 2019). First, it allows practitioners to "rapidly prototype" NLP models: changing the prompt immediately leads to a new model. Second, it provides a fully natural language interface to a machine learning model, which allows users-even those without technical expertise-to create NLP systems. Finally, since in-context learning reuses the same model for each task, it reduces memory requirements and system complexity when serving many different tasks.

However, despite these promises, we show that GPT-3's accuracy can be highly unstable across different prompts (Section 3). A prompt contains three components: a format, a set of training examples, and a permutation (ordering) for those examples. We show that different choices for these factors can lead to highly different accuracies, e.g., changing the permutation of the training examples in a sentiment analysis prompt can change accuracy from near chance (54\%) to near state-of-the-art (93\%). This instability implies that GPT-3 users, who typically design prompts manually, cannot expect to consistently obtain good accuracy.

We next analyze what causes this instability. We identify three pitfalls of language models that lead them to be biased toward certain answers during few-shot learning. In particular, they suffer from majority label bias, recency bias, and common token bias (Section 4). The majority label and recency biases lead the model to predict training answers that appear frequently or near the end of the prompt. For example, a prompt that ends with a Negative training example may cause a bias towards the Negative class. On the other hand, the common token bias leads the model to prefer answers that are frequent in its pre-training data, e.g., it prefers "United States" over "Saint Lucia", which is likely suboptimal for the task of interest.

We identify that these biases typically result in a shift in the output distribution of the model. We can thus coun-

![Figure](./zhaoCalibrateUseImproving2021_images/bfa78706-72f5-4d26-85c6-dd246cd5745f-02_548_1676_238_198.jpg)

**Figure 1. Few-shot learning can be highly unstable across different choices of the prompt. Above, we plot the mean accuracy (± one standard deviation) across different choices of the training examples for three different datasets and model sizes. We show that our method, contextual calibration, improves accuracy, reduces variance, and overall makes tools like GPT-3 more effective for end users.**

teract these biases by "calibrating" the output distribution. Concretely, we estimate the model's bias towards certain answers by feeding in a dummy test input that is content-free. In the prompt above for example, if we replace "Amazing." with the string "N/A", the model predicts 62\% Positive. We then fit the calibration parameters so that the content-free input has uniform scores for each answer. This contextual calibration procedure provides a good setting of the calibration parameters without additional training data.

We test the effectiveness of contextual calibration on a range of tasks (Section 5). Contextual calibration consistently improves GPT-3 and GPT-2's accuracy (up to 30.0\% absolute) across different choices of the prompt format and examples (e.g., Figure 1). It also makes the accuracy more stable across different prompts, thus mitigating the need for prompt engineering. Overall, contextual calibration is a simple method that makes language models better few-shot learners: it enables end users to obtain higher accuracy with considerably less effort.

