# 5. Contextual Calibration

Thus far, we have shown that GPT-3 is biased towards certain answers due to the prompt and the model's intrinsic biases. Here, we look to correct this by "calibrating" the model's output probabilities. ${ }^{4}$ A common technique for adjusting output probabilities is to apply an affine transformation (Platt, 1999; Guo et al., 2017):
$$
\hat{\mathbf{q}}=\operatorname{softmax}(\mathbf{W} \hat{\mathbf{p}}+\mathbf{b}),
$$
where a weight matrix W and a bias vector b are applied to the original probabilities $\hat{\mathbf{p}}$ to get the new probabilities

\footnotetext{
${ }^{3}$ The frequency of a token on the web is calculated using Google Ngrams https://books.google.com/ngrams. The predictions are from the 0-shot setting on the validation set.
${ }^{4}$ The output of GPT-3 is biased (its outputs are shifted), similar to how measurement devices such as voltage meters or weighing scales are biased. Just like how these devices require "calibration before use", where the devices' outputs are scaled/zeroed-out, we hope to apply a similar calibration procedure to LMs. This goal is distinct from statistical calibration (Brier, 1950; Guo et al., 2017), i.e., aligning a model's confidence estimate with its true accuracy.
}
$\hat{\mathbf{q}} .^{5}$ For classification tasks, $\hat{\mathbf{p}}$ is the set of probabilities that are associated with each label name, renormalized to one. For generation tasks, $\hat{\mathbf{p}}$ is the entire set of probabilities for the first token. ${ }^{6}$ In this paper, we restrict the matrix W to be diagonal, known as vector scaling (Guo et al., 2017), to prevent the parameters from growing quadratically in the size of $\hat{\mathbf{p}}$ (which is $\approx 50,000$ for generation tasks).

The main challenge in the zero- or few-shot setting is that we do not have data to learn $\mathbf{W}$ and $\mathbf{b}$. We thus propose a novel data-free procedure to infer a good setting of these parameters. The key idea is that the model's bias towards certain answers can be estimated by feeding in a contentfree input such as the string "N/A". For example, consider the two-shot prompt:

Input: Subpar acting. Sentiment: Negative
Input: Beautiful film. Sentiment: Positive
Input: N/A Sentiment:
where "N/A" serves as the test input. Ideally, GPT-3 would score this test input as 50\% Positive and 50\% Negative. However, the model's biases cause it to score this input as $61.8 \%$ Positive. Note that this error is contextual: a different choice of the training examples, permutation, and format will lead to different predictions for the content-free input.

We can correct this error by setting W and b so that the class scores for the content-free input are uniform. We first obtain $\hat{\mathbf{p}}$ for the content-free input, denoted $\hat{\mathbf{p}}_{\mathrm{cf}}$. We then set \(\mathbf{W}=\operatorname{diag}\left(\hat{\mathbf{p}}_{\mathrm{cf}}\right)^{-1}\) and $\mathbf{b}$ to the all-zero vector. ${ }^{7}$ To make test predictions, we compute $\mathbf{W} \hat{\mathbf{p}}+\mathbf{b}$ and take the argmax.

Implementation Details This contextual calibration procedure adds trivial amounts of computational overhead and is implemented in a few lines of code (compute and save $\hat{\mathbf{p}}_{\mathrm{cf}}$, adjust output probabilities). For the content-free input, many good choices exist, including "N/A", the empty string, and gibberish tokens. In all our experiments, we average the probabilities from three content-free inputs: "N/A", "[MASK]", and the empty string. ${ }^{8}$ One could also craft the content-free input in a task-specific manner. We explore this for LAMA, where we replace the subject with the contentfree input, e.g., we use "N/A was born in" as the input.

\footnotetext{
${ }^{5}$ This affine transformation is usually applied to the logits, i.e., prior to the softmax. However, we only have access to GPT-3's output probabilities in the OpenAI API.
${ }^{6}$ We only calibrate the prediction of the first output token for generation tasks. This is reasonable because, for the tasks we consider, we found that the model's predictions are highly deterministic after generating the first token.
${ }^{7}$ An alternate solution is to set $\mathbf{b}$ to $-\hat{\mathbf{p}}_{\mathrm{cf}}$ and $\mathbf{W}$ to the identity. Empirically, this alternate solution yields higher accuracy for generation tasks (where the dimensionality of $\hat{\mathbf{p}}$ is large). The solution in the main text performs better for classification.
${ }^{8}$ We found this simple ensemble to achieve the best results for AGNews, and we reuse it for all other datasets. See Section 5.2 for an ablation on the choice of content-free input.
}

\begin{table}
\begin{tabular}[t]{|l|l|l|l|l|l|l|l|l|l|}
\hline \multirow[t]{2}{*}{Dataset} & \multirow[t]{2}{*}{LM} & \multicolumn{2}{|c|}{0-shot} & \multicolumn{2}{|c|}{1-shot} & \multicolumn{2}{|c|}{4-shot} & \multicolumn{2}{|c|}{8-shot} \\
\hline & & Baseline & Ours & Baseline & Ours & Baseline & Ours & Baseline & Ours \\
\hline \multicolumn{10}{|l|}{Text Classification} \\
\hline \multirow[t]{2}{*}{AGNews} & 2.7B & $44.7_{0.0}$ & $\mathbf{6 3 . 2}_{0.0}$ & $33.0_{5.1}$ & $\mathbf{5 9 . 6}_{6.4}$ & $43.3_{8.3}$ & $\mathbf{7 1 . 1}_{8.5}$ & $50.8_{7.8}$ & $\mathbf{7 2 . 7}_{5.8}$ \\
\hline & 175B & $43.9_{0.0}$ & $\mathbf{7 3 . 9}_{0.0}$ & $62.1_{6.3}$ & $\mathbf{7 7 . 1}_{3.8}$ & $61.0_{10.9}$ & $\mathbf{8 5 . 9}_{1.3}$ & $79.1_{2.6}$ & $\mathbf{8 4 . 3}_{2.5}$ \\
\hline \multirow{2}{*}{TREC} & 2.7 B & $31.0_{0.0}$ & $\mathbf{3 8 . 8}_{0.0}$ & $24.3_{6.4}$ & $\mathbf{3 6 . 8}_{7.7}$ & $25.8_{11.5}$ & $\mathbf{3 8 . 6}_{13.2}$ & $29.3_{8.0}$ & $\mathbf{4 4 . 3}_{11.4}$ \\
\hline & 175B & $47.4_{0.0}$ & $\mathbf{5 7 . 4}_{0.0}$ & $57.7_{6.0}$ & $\mathbf{7 5 . 7}_{1.4}$ & $60.2_{7.6}$ & $\mathbf{6 9 . 7}_{1.4}$ & $45.6_{4.0}$ & $\mathbf{6 6 . 9}_{6.5}$ \\
\hline \multirow[t]{2}{*}{CB} & 2.7 B & $44.6_{0.0}$ & $\mathbf{5 0 . 0}_{0.0}$ & $\mathbf{3 3 . 8}_{16.6}$ & $33.0_{7.3}$ & $43.5_{11.9}$ & $\mathbf{5 4 . 2}_{4.7}$ & $43.9_{8.4}$ & $\mathbf{5 3 . 0}_{7.7}$ \\
\hline & 175B & $30.4_{0.0}$ & $\mathbf{4 8 . 2}_{0.0}$ & $50.9_{6.7}$ & $\mathbf{5 1 . 8}_{7.2}$ & $45.2_{19.4}$ & $\mathbf{6 0 . 7}_{6.7}$ & $59.6_{11.3}$ & $\mathbf{6 5 . 0}_{7.9}$ \\
\hline \multirow{2}{*}{RTE} & 2.7B & $44.8_{0.0}$ & $\mathbf{4 9 . 5}_{0.0}$ & $49.6_{2.9}$ & $\mathbf{5 0 . 4}_{2.7}$ & $44.0_{1.4}$ & $\mathbf{5 4 . 5}_{4.7}$ & $49.2_{1.9}$ & $\mathbf{5 4 . 8}_{2.8}$ \\
\hline & 175B & $\mathbf{5 7 . 8}_{0.0}$ & $\mathbf{5 7 . 8}_{0.0}$ & $\mathbf{6 2 . 9}_{2.7}$ & $62.8_{2.3}$ & $58.7_{11.9}$ & $\mathbf{6 0 . 4}_{8.1}$ & $\mathbf{6 6 . 2}_{5.8}$ & $65.5_{2.5}$ \\
\hline \multirow[t]{2}{*}{SST-2} & 2.7B & $57.2_{0.0}$ & $\mathbf{7 1 . 4}_{0.0}$ & $67.3_{7.9}$ & $\mathbf{7 9 . 1}_{8.3}$ & $59.1_{10.2}$ & $\mathbf{7 9 . 9}_{7.8}$ & $54.0_{4.3}$ & $\mathbf{8 2 . 0}_{5.5}$ \\
\hline & 175B & $71.6_{0.0}$ & $\mathbf{7 5 . 8}_{0.0}$ & $93.3_{2.8}$ & $\mathbf{9 4 . 7}_{1.4}$ & $93.6_{3.3}$ & $\mathbf{9 4 . 3}_{1.0}$ & $\mathbf{9 5 . 6}_{1.0}$ & $95.3_{0.7}$ \\
\hline \multirow{2}{*}{DBPedia} & 2.7 B & $36.0_{0.0}$ & $\mathbf{3 8 . 7}_{0.0}$ & $25.9_{4.4}$ & $\mathbf{6 1 . 6}_{2.9}$ & $61.0_{12.8}$ & $\mathbf{6 6 . 0}_{7.5}$ & $72.6_{4.5}$ & $\mathbf{7 4 . 8}_{5.0}$ \\
\hline & 175B & $22.0_{0.0}$ & $\mathbf{5 9 . 7}_{0.0}$ & $79.3_{3.0}$ & $\mathbf{8 5 . 3}_{2.2}$ & $84.6_{5.8}$ & $\mathbf{8 6 . 9}_{4.0}$ & $82.3_{7.8}$ & $\mathbf{8 6 . 9}_{1.9}$ \\
\hline \multicolumn{10}{|l|}{Fact Retrieval} \\
\hline \multirow[t]{2}{*}{LAMA} & 2.7 B & $14.0_{0.0}$ & $\mathbf{2 2 . 7}_{0.0}$ & $29.7_{1.8}$ & $\mathbf{3 1 . 6}_{1.3}$ & $35.8_{3.8}$ & $\mathbf{3 7 . 4}_{3.4}$ & $\mathbf{4 2 . 5}_{1.3}$ & $\mathbf{4 2 . 5}_{1.4}$ \\
\hline & 175B & $23.5_{0.0}$ & $\mathbf{3 0 . 1}_{0.0}$ & $48.9_{2.3}$ & $\mathbf{4 9 . 0}_{1.4}$ & $\mathbf{6 2 . 0}_{2.4}$ & $61.8_{2.9}$ & $\mathbf{6 3 . 8}_{1.0}$ & $63.6_{1.3}$ \\
\hline \multicolumn{10}{|l|}{Information Extraction} \\
\hline \multirow[t]{2}{*}{MIT-G} & 2.7B & $5.0_{0.0}$ & $\mathbf{5 . 7}_{0.0}$ & $26.7_{11.4}$ & $\mathbf{3 7 . 9}_{5.7}$ & $53.1_{7.8}$ & $\mathbf{5 4 . 7}_{6.0}$ & $59.0_{4.7}$ & $\mathbf{5 9 . 1}_{4.8}$ \\
\hline & 13B & $15.0_{0.0}$ & $\mathbf{1 8 . 7}_{0.0}$ & $47.3_{3.9}$ & $\mathbf{5 2 . 0}_{7.9}$ & $57.9_{4.8}$ & $\mathbf{5 8 . 9}_{4.0}$ & $59.0_{4.7}$ & $\mathbf{5 9 . 1}_{4.8}$ \\
\hline \multirow{2}{*}{MIT-D} & 2.7B & $46.3_{0.0}$ & $\mathbf{4 7 . 0}_{0.0}$ & $42.0_{13.0}$ & $\mathbf{5 3 . 5}_{13.5}$ & $73.5_{4.9}$ & $\mathbf{7 4 . 1}_{5.0}$ & $\mathbf{7 5 . 3}_{1.0}$ & $75.1_{1.3}$ \\
\hline & 13B & $36.3_{0.0}$ & $\mathbf{3 8 . 7}_{0.0}$ & $58.6_{21.4}$ & $\mathbf{7 2 . 8}_{4.0}$ & $75.4_{1.9}$ & $\mathbf{7 5 . 9}_{2.1}$ & $\mathbf{7 7 . 8}_{0.5}$ & $\mathbf{7 7 . 8}_{0.5}$ \\
\hline \multirow[t]{2}{*}{ATIS-A} & 2.7B & $10.8_{0.0}$ & $\mathbf{1 4 . 0}_{0.0}$ & $29.8_{12.8}$ & $\mathbf{3 3 . 1}_{9.4}$ & $43.0_{26.2}$ & $\mathbf{4 7 . 3}_{21.3}$ & $55.6_{5.0}$ & $\mathbf{5 8 . 8}_{4.0}$ \\
\hline & 13B & $49.5_{0.0}$ & $\mathbf{5 2 . 7}_{0.0}$ & $69.6_{17.4}$ & $\mathbf{7 1 . 8}_{17.1}$ & $67.5_{10.4}$ & $\mathbf{6 9 . 6}_{13.4}$ & $63.4_{4.6}$ & $\mathbf{6 4 . 5}_{4.0}$ \\
\hline \multirow{2}{*}{ATIS-D} & 2.7B & $6.4_{0.0}$ & $\mathbf{1 2 . 9}_{0.0}$ & $42.3_{28.8}$ & $\mathbf{6 5 . 6}_{20.8}$ & $75.0_{6.7}$ & $\mathbf{8 3 . 4}_{4.2}$ & $81.0_{8.8}$ & $\mathbf{8 8 . 3}_{3.7}$ \\
\hline & 13B & $4.0_{0.0}$ & $\mathbf{5 . 0}_{0.0}$ & $\mathbf{9 7 . 9}_{0.6}$ & $95.5_{4.6}$ & $\mathbf{9 8 . 0}_{0.6}$ & $97.8_{0.7}$ & $\mathbf{9 8 . 8}_{0.3}$ & $\mathbf{9 8 . 8}_{0.3}$ \\
\hline
\end{tabular}

**Table 1. Contextual calibration improves accuracy across a range of tasks. We show the mean and standard deviation across different choices of the training examples (the prompt format is fixed). The LM column indicates the GPT-3 size (see Appendix A for GPT-2 results). The Baseline column shows the standard approach of greedy decoding (Brown et al., 2020) and Ours corresponds to greedy decoding after modifying the output probabilities using contextual calibration. We bold the better result of the baseline and ours. MIT-G, MIT-D, ATIS-A, and ATIS-D indicate the MIT Genre, MIT Director, ATIS Airline, and ATIS Departure Date datasets.**
\end{table}

## 5.1. Results for Contextual Calibration

Here, we evaluate the effectiveness of contextual calibration across all of our datasets and LMs. We first use a fixed prompt format and select five different random sets of training examples, placing them in an arbitrary order in the prompt. We do not artificially balance the labels of the training examples for the classification tasks. We use the same sets of training examples for the baseline (standard decoding without calibration) and contextual calibration. We use labeling budgets of 0-8 examples; using more than 8shots causes the cost of querying the OpenAI API to become prohibitively expensive.

Table 1 shows the results and Figure 1 in Section 1 plots the same data for a subset of the tasks.

Improves Mean And Worst-Case Accuracy Contextual calibration dramatically improves GPT-3's average and worst-case accuracy, by up to 30.0\% absolute. These gains hold for both classification and generation tasks. Contextual calibration also sometimes allows GPT-3 2.7B to outperform the GPT-3 175B baseline-by up to 19.3\%-despite being over 50x smaller.

Can Reduce Variance Across Training Sets Figure 6 plots the difference in the standard deviation between the baseline and contextual calibration for all tasks from Table 1. Contextual calibration reduces the variance considerably in a majority of cases, and it does not increase variance by much in the remaining cases.

Reduces Drop from 0-shot to 1-shot For the baseline, there are four cases where there is a drop in accuracy when

![Figure](./zhaoCalibrateUseImproving2021_images/bfa78706-72f5-4d26-85c6-dd246cd5745f-07_170_758_249_221.jpg)

**Figure 6. Aside from improving mean accuracy, contextual calibration also reduces the standard deviation of accuracy across different choices of the training examples. We plot the difference in standard deviation between contextual calibration and the baseline from Table 1.**

moving from 0-shot to 1-shot (TREC, AGNews, DBpedia, SST-2). We attribute this drop to the majority label bias (see discussion in Section 4). Calibration removes this drop in three out of four cases.

Improves GPT-2 We also test GPT-2 1.5B (see Table 4 in Appendix A). We find that like GPT-3, GPT-2's accuracy also highly varies across different prompts. This suggests that the variance that we observe for few-shot in-context learning is a general problem for LMs. Second, contextual calibration works out-of-the-box for GPT-2-it improves the mean accuracy and reduces variance for most tasks.

Improves Accuracy Across Formats In our next set of experiments, we use a fixed set of training examples and vary the prompt format. We use the 15 prompt formats for SST-2 discussed in Section 3. We also create 15 prompt formats for each of three random relations in LAMA (P20, P159, P19) by using the paraphrases of the original LAMA templates generated by Jiang et al. (2020b). Figure 7 shows the results before and after calibration for SST-2, and Figure 9 in Appendix A show the results for LAMA. Contextual calibration improves the average and worst-case accuracy for both datasets, and reduces the variance for SST-2.

## 5.2. Ablations on Contextual Calibration

We finally conduct two analyses/ablations on contextual calibration. We first analyze how effective contextual calibration is at inferring a good setting of W. To do so, we compare its accuracy to an "oracle calibration" method that uses the validation set to find the best possible diagonal $\mathbf{W}$. We evaluate this oracle on AGNews, and find that contextual calibration is surprisingly close to it (Figure 8).

We also study how the choice of content-free input affects accuracy. In Table 3 in Appendix A, we show the accuracy for SST-2 and AGNews for different choices of the content-free input. The choice of content-free input matters, however, many good choices exist.

![Figure](./zhaoCalibrateUseImproving2021_images/bfa78706-72f5-4d26-85c6-dd246cd5745f-07_606_655_230_1141.jpg)

**Figure 7. GPT-3 has high variance across different prompt formats; contextual calibration reduces this variance and improves mean accuracy. We show the mean accuracy (± standard deviation) over 15 different prompt formats for SST-2.**

