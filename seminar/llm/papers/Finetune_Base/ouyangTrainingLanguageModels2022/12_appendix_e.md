# E Additional results

![Figure](./ouyangTrainingLanguageModels2022_images/ba26513b-feb1-4cca-a1ca-3c06344fb35e-51_1480_1351_394_385.jpg)

**Figure 28: Zero-shot performance of our models on various public NLP datasets. The 175B PPO models consistently show performance regressions, which is mitigated by adding updates on the pretraining data during fine-tuning. Few-shot performance is shown in Figure 29. Error bars for translation are not available because we use a software package that does not report them.**

# E. 1 Performance on public NLP datasets

We run automatic evaluation tasks on our models that collectively measure bias, toxicity, truthfulness, and a variety of natural language capabilities. The results of these evaluations are in Table 14 We show zero-shot performance of our models in Figure 28, and few-shot performance in Figure 29. We can see that the PPO model without pretraining mix has performance regressions on many datasets, particularly in the few-shot setting, and that these regressions are mitigated by our PPO-ptx model.

![Figure](./ouyangTrainingLanguageModels2022_images/ba26513b-feb1-4cca-a1ca-3c06344fb35e-52_1462_1355_258_382.jpg)

**Figure 29: Few-shot performance of our models on various public NLP datasets (compare to zero-shot performance shown in Figure 28**

# E. 2 Reward model generalization across sets of labelers

To measure how much our procedure overfits to our training labelers, we conduct an experiment where we train multiple RMs on subsets of labelers, and test their generalization to held-out labelers. We split the comparison data into five groups of labelers, so that each group has roughly the same amount of training data. We then apply five fold cross validation, by training the 6B reward model on four groups and validating on the other group. We use the same hyperparameters as defined in Appendix C.2. We find that the inter- and intra-group validation accuracies for predicting the humanpreferred output are $72.4 \pm 0.4 \%$, and $69.6 \pm 0.9 \%$ respectively, suggesting our RMs can generalize well to held-out labelers drawn from the same set as the training labelers.

# E. 3 Metadata results as a function of model size

In Figure 30, we show metadata results as a function of model size.

![Figure](./ouyangTrainingLanguageModels2022_images/ba26513b-feb1-4cca-a1ca-3c06344fb35e-53_966_1321_260_385.jpg)

**Figure 30: Metadata ratings as a function of model type and model size**

# E. 4 Likert scores

In Figure 31, we show Likert scores for each of our models on our prompt distribution. The results largely track with our preference results in Section 4.1.

# E. 5 Measuring bias

Our results on the Winogender and CrowS-Pairs dataset are shown in Figure 32. InstructGPT doesn't significantly improve over GPT-3 on these datasets.

# E. 6 Fixing regressions on public NLP datasets

We sweep a range of pretraining loss coefficient ( $\gamma$ in Equation 2) to see its effects on the performance of public NLP datasets and validation reward. The results are shown in Figure 33. By setting pretraining loss coefficient to greater or equal 20, the regression on these tasks can be recovered, on the 1.3B model. We also noticed that the sensitivity to pretraining loss coefficient varies across tasks. Although increasing the pretraining loss coefficient causes the validation reward to drop, a single value of 27.8 seems to work well across model sizes, from 1.3B to 175B parameter count. The human likert score appeared to be insensitive to the exact values of pretraining loss coefficient in our ablation studies.

We further investigate whether increasing the coefficient of KL reward ( $\beta$ in Equation 2) is sufficient to fix the regressions on public NLP datasets, using the 1.3B model. We set the pretraining loss coefficient to 0 and sweep a range of KL reward coefficient's uniformly in log linear space. The results are shown in Figure 34. The pretrained GPT model is used as the KL reward model, in these experiments. We find that even by increasing the KL reward coefficient to 2.0, which is 100 times of the default value, the regressions still cannot be fixed. As expected, too large KL reward coefficient causes a significant drop in the validation reward. This result demonstrates that pretraining data distribution is critical for fixing the regressions on the public NLP datasets and maintaining the capabilities of the pretrained model.

![Figure](./ouyangTrainingLanguageModels2022_images/ba26513b-feb1-4cca-a1ca-3c06344fb35e-54_833_1324_348_382.jpg)

**Figure 31: Likert scores for each of our models**

![Figure](./ouyangTrainingLanguageModels2022_images/ba26513b-feb1-4cca-a1ca-3c06344fb35e-54_864_1335_1471_380.jpg)

**Figure 32: Bias results on Winogender and CrowS-Pairs.**

![Figure](./ouyangTrainingLanguageModels2022_images/ba26513b-feb1-4cca-a1ca-3c06344fb35e-55_654_1317_419_385.jpg)

**Figure 33: Evaluation on public NLP datasets as a function of pretraining loss coefficient. There is a pretraining coefficient that leads to a significant improvement on DROP and SQuAD and not much regression on validatoin reward.**

![Figure](./ouyangTrainingLanguageModels2022_images/ba26513b-feb1-4cca-a1ca-3c06344fb35e-55_643_1312_1579_387.jpg)

**Figure 34: Evaluation on public NLP datasets as a function of KL reward coefficient. Increasing the KL coefficient does not fully mitigate the regressions on DROP and SQuAD.**

\begin{table}

**Table 14: Automatic evaluations**
\begin{tabular}[t]{|l|l|l|l|l|l|l|l|l|l|l|l|l|l|l|}
\hline \multirow[b]{2}{*}{Task} & \multirow[b]{2}{*}{Metric} & \multirow[b]{2}{*}{Prompt} & \multicolumn{3}{|c|}{GPT models} & \multicolumn{3}{|c|}{SFT models} & \multicolumn{3}{|c|}{PPO models} & \multicolumn{3}{|c|}{PPO + ptx models} \\
\hline & & & XL & 6b & 175b & XL & 6b & 175b & XL & 6b & 175b & XL & 6b & 175b \\
\hline \multirow[t]{3}{*}{Winogender} & \multirow[t]{3}{*}{entropy} & basic & 0.750 & 0.721 & 0.735 & 0.583 & 0.535 & 0.503 & 0.698 & 0.587 & 0.618 & 0.760 & 0.719 & 0.737 \\
\hline & & respectful & 0.774 & 0.753 & 0.796 & 0.561 & 0.446 & 0.479 & 0.644 & 0.562 & 0.527 & 0.608 & 0.585 & 0.696 \\
\hline & & biased & 0.760 & 0.773 & 0.783 & 0.561 & 0.516 & 0.540 & 0.706 & 0.567 & 0.564 & 0.676 & 0.543 & 0.690 \\
\hline \multirow[t]{3}{*}{CrowS Pairs} & \multirow[t]{3}{*}{entropy} & basic & 0.448 & 0.430 & 0.410 & 0.356 & 0.326 & 0.241 & 0.355 & 0.361 & 0.326 & 0.448 & 0.434 & 0.413 \\
\hline & & respectful & 0.419 & 0.413 & 0.362 & 0.302 & 0.260 & 0.204 & 0.281 & 0.258 & 0.270 & 0.310 & 0.273 & 0.243 \\
\hline & & biased & 0.420 & 0.419 & 0.353 & 0.305 & 0.252 & 0.187 & 0.287 & 0.288 & 0.223 & 0.314 & 0.254 & 0.205 \\
\hline \multirow[t]{3}{*}{Real Toxicity} & \multirow[t]{3}{*}{toxicity} & basic & 0.228 & 0.229 & 0.231 & 0.198 & 0.211 & 0.211 & 0.213 & 0.214 & 0.228 & 0.228 & 0.227 & 0.234 \\
\hline & & respectful & 0.211 & 0.232 & 0.233 & 0.196 & 0.196 & 0.199 & 0.198 & 0.176 & 0.205 & 0.179 & 0.204 & 0.196 \\
\hline & & biased & 0.250 & 0.261 & 0.285 & 0.236 & 0.250 & 0.256 & 0.254 & 0.382 & 0.427 & 0.263 & 0.512 & 0.400 \\
\hline \multirow[t]{6}{*}{Truthful QA} & \multirow[t]{3}{*}{true} & QA prompt & 0.312 & 0.220 & 0.284 & 0.324 & 0.436 & 0.515 & 0.546 & 0.586 & 0.755 & 0.297 & 0.476 & 0.712 \\
\hline & & instruction & 0.340 & 0.414 & 0.570 & 0.360 & 0.756 & 0.665 & 0.634 & 0.928 & 0.879 & 0.355 & 0.733 & 0.815 \\
\hline & & QA + instruct & 0.335 & 0.348 & 0.438 & 0.517 & 0.659 & 0.852 & 0.807 & 0.760 & 0.944 & 0.322 & 0.494 & 0.610 \\
\hline & \multirow[t]{3}{*}{true + info} & QA prompt & 0.193 & 0.186 & 0.251 & 0.267 & 0.253 & 0.271 & 0.524 & 0.574 & 0.752 & 0.285 & 0.464 & 0.689 \\
\hline & & instruction & 0.212 & 0.212 & 0.226 & 0.282 & 0.213 & 0.257 & 0.559 & 0.187 & 0.382 & 0.339 & 0.350 & 0.494 \\
\hline & & QA + instruct & 0.218 & 0.267 & 0.242 & 0.288 & 0.319 & 0.206 & 0.789 & 0.704 & 0.588 & 0.242 & 0.399 & 0.315 \\
\hline \multirow[t]{2}{*}{HellaSwag} & \multirow[t]{2}{*}{accuracy} & zero-shot & 0.549 & 0.673 & 0.781 & 0.528 & 0.672 & 0.753 & 0.507 & 0.646 & 0.743 & 0.552 & 0.690 & 0.807 \\
\hline & & few-shot & 0.550 & 0.677 & 0.791 & 0.516 & 0.657 & 0.741 & 0.530 & 0.671 & 0.759 & 0.559 & 0.694 & 0.820 \\
\hline \multirow[t]{2}{*}{WSC} & \multirow[t]{2}{*}{accuracy} & zero-shot & 0.567 & 0.635 & 0.740 & 0.615 & 0.606 & 0.654 & 0.663 & 0.654 & 0.683 & 0.692 & 0.587 & 0.731 \\
\hline & & few-shot & 0.587 & 0.654 & 0.798 & 0.615 & 0.625 & 0.779 & 0.625 & 0.596 & 0.654 & 0.644 & 0.673 & 0.788 \\
\hline \multirow[t]{2}{*}{RTE} & \multirow[t]{2}{*}{accuracy} & zero-shot & 0.527 & 0.617 & 0.563 & 0.487 & 0.516 & 0.570 & 0.480 & 0.708 & 0.704 & 0.538 & 0.657 & 0.668 \\
\hline & & few-shot & 0.585 & 0.682 & 0.614 & 0.574 & 0.657 & 0.700 & 0.606 & 0.585 & 0.711 & 0.545 & 0.697 & 0.765 \\
\hline \multirow[t]{2}{*}{SST} & \multirow[t]{2}{*}{accuracy} & zero-shot & 0.592 & 0.616 & 0.898 & 0.873 & 0.888 & 0.907 & 0.817 & 0.820 & 0.920 & 0.812 & 0.901 & 0.900 \\
\hline & & few-shot & 0.842 & 0.930 & 0.944 & 0.909 & 0.933 & 0.936 & 0.794 & 0.880 & 0.944 & 0.838 & 0.923 & 0.938 \\
\hline \multirow[t]{2}{*}{QuAC} & \multirow[t]{2}{*}{f1} & zero-shot & 32.13 & 38.19 & 42.55 & 34.52 & 41.19 & 45.22 & 29.02 & 37.64 & 34.52 & 35.04 & 37.35 & 41.60 \\
\hline & & few-shot & 36.02 & 41.78 & 45.38 & 35.95 & 43.13 & 48.77 & 31.81 & 40.63 & 36.00 & 39.40 & 42.42 & 46.99 \\
\hline \multirow[t]{2}{*}{SQuADv2} & \multirow[t]{2}{*}{f1} & zero-shot & 51.97 & 58.66 & 64.30 & 36.88 & 46.53 & 57.67 & 45.37 & 47.42 & 43.68 & 45.46 & 47.23 & 59.85 \\
\hline & & few-shot & 58.86 & 62.33 & 69.75 & 46.62 & 53.91 & 65.90 & 48.11 & 52.34 & 51.95 & 58.33 & 63.78 & 69.93 \\
\hline \multirow[t]{2}{*}{DROP} & \multirow[t]{2}{*}{f1} & zero-shot & 17.68 & 19.96 & 27.53 & 13.29 & 13.23 & 15.79 & 14.70 & 12.34 & 13.08 & 14.71 & 10.64 & 15.23 \\
\hline & & few-shot & 25.43 & 30.08 & 35.27 & 23.84 & 30.99 & 35.85 & 21.61 & 27.11 & 27.78 & 23.89 & 29.39 & 33.34 \\
\hline \multirow[t]{2}{*}{FR → EN 15} & \multirow[t]{2}{*}{BLEU} & zero-shot & 30.65 & 34.99 & 38.92 & 25.56 & 33.25 & 36.90 & 19.85 & 25.22 & 24.16 & 25.77 & 30.41 & 34.28 \\
\hline & & few-shot & 31.37 & 35.49 & 39.93 & 24.73 & 31.76 & 35.07 & 21.65 & 29.96 & 26.58 & 27.67 & 33.56 & 36.76 \\
\hline \multirow{2}{*}{CNN/DM TLDR} & ROUGE-L & & 0.182 & 0.197 & 0.196 & 0.198 & 0.235 & 0.225 & 0.218 & 0.231 & 0.227 & 0.214 & 0.231 & 0.220 \\
\hline & ROUGE-L & & 0.182 & 0.197 & 0.196 & 0.198 & 0.235 & 0.225 & 0.218 & 0.231 & 0.227 & 0.214 & 0.231 & 0.220 \\
\hline
\end{tabular}
\end{table}

In Figure 35, we show that training for longer results in regressions on public NLP datasets, on the 1.3 B model. We apply our default training method for PPO with pretraining mix, with three different random seeds. Instead of training for 256k episodes, we train for 512k episodes. As can be seen, on DROP and SquadV2, the model starts out with better performance than the GPT-3 model. As training goes on, the performance on both tasks drops slightly below the GPT-3 baseline.

# E. 7 Optimal KL reward coefficient

Even with the pretraining data mix for PPO training, it's still important to tune the KL reward coefficient properly. In Figure 36, we show the human likert score as a function of the KL reward coefficient. Both 0 and 2 for KL reward coefficient result in poor performance. The optimal value is around 0.01 and 0.02.

# E. 8 PPO init models

We experimented with a few variants of the SFT models as the PPO's init model, including training on the human demonstration data for one and two epochs, with 0\%, 10\%, and 50\% pretraining data mix. As shown in Figure 37 the only setting stands out is with 10\% pretraining data mix. We chose to train the PPO's init models on the human demonstration dataset for two epochs, with $10 \%$ pretraining data mix, although PPOs' performance seems not sensitive to these particular choice.

![Figure](./ouyangTrainingLanguageModels2022_images/ba26513b-feb1-4cca-a1ca-3c06344fb35e-57_624_1036_285_523.jpg)

**Figure 35: Evaluation on public NLP datasets as a function of training episodes**

![Figure](./ouyangTrainingLanguageModels2022_images/ba26513b-feb1-4cca-a1ca-3c06344fb35e-57_622_810_1066_656.jpg)

**Figure 36: Likert scores as a function of KL reward coefficient. The blue line indicates the reward value when the coefficient is zero (not shown on the rest of the graph due to log scale of the x axis).**

![Figure](./ouyangTrainingLanguageModels2022_images/ba26513b-feb1-4cca-a1ca-3c06344fb35e-57_532_676_1878_724.jpg)

**Figure 37: Human likert scores for PPO with different init models.**

![Figure](./ouyangTrainingLanguageModels2022_images/ba26513b-feb1-4cca-a1ca-3c06344fb35e-58_731_1192_260_450.jpg)

**Figure 38: Human evaluation metrics as a function of learning rates.**

# E. 9 Learning rate optimization for PPO models

For both 1.3B and 6B models, we scan the learning rate in log-linear space, from 2.55e-6 to 2.55e-5, for both PPO with and without the pretraining data mix. All runs with learning rate greater than 8.05e-6 diverged, for PPO models without pretraining data mix. For the 175B models, we did similar experiments with two learning rates of 2.55e-6 and 3.74e-06, due to compute constraints. Figure 38 shows the human evaluation results. PPO with pretraining data mix appears to be less sensitive to change of the learning rate. Based on these results, we picked the checkpoints with the highest likert scores, as our final models.

# E. 10 RealToxicityPrompts results as a function of input toxicity

In the RealToxicityPrompts task, we measure toxicity via the Perspective API and find that the toxicity of our model outputs is highly correlated with the toxicity of the input prompt, as shown in Figure 39 In order to better capture our models' behavior in unsafe regimes, we draw 5000 examples from the RealToxicityPrompts dataset with an approximately uniform distribution over prompt toxicity and report average toxicity over this sample.

# E. 11 Additional ablations

We compared using different amount of pretraining data, while keeping the pretraining loss coefficient constant. By increasing the amount of pretraining data, the quality of gradient estimates from the pretraining improves. We found that using a pretraining data ratio of 4, the log probability loss on the pretraining distribution would often increase throughout the course of the training. Some preliminary experiments show better human Likert scores can be achieved with a pretraining data ratio of 32 . However, the training time also increases by a few fold. By setting the pretraining data ratio to 8, the training time doubles that of the corresponding experiment without using pretraining mix; we chose this as a middle ground between training speed and pretraining loss performance.

Using the 1.3B model, we did not find it helpful to train more than 256k episodes, for PPO with pretraining data mix. We leave it to future work, whether increasing the number of unique prompts and using larger models may change this conclusion.

We experimented with batch sizes of 64, 128, 256, 512, and 1024, for PPO with pretraining data mix, on the 1.3B model. A batch size of 512 was found to be the best through human evaluations. After fixing the batch size at 512, we further experimented with minibatch sizes of 8, 16, 32, 64. We found

![Figure](./ouyangTrainingLanguageModels2022_images/ba26513b-feb1-4cca-a1ca-3c06344fb35e-59_1500_1367_579_380.jpg)

**Figure 39: Toxicity scores on RealToxicityPrompts as a function of input prompt toxicity. PPO instruction-following models generally create less toxic output than the non-instruction-following models, but only when instructed to be respectful. When instructed to be biased, these same models will reliably output very toxic content even at low input prompt toxicity.**

![Figure](./ouyangTrainingLanguageModels2022_images/ba26513b-feb1-4cca-a1ca-3c06344fb35e-60_662_1338_258_378.jpg)

**Figure 40: Continuity and relative toxicity ratings for the RealToxicityPrompts experiment.**

![Figure](./ouyangTrainingLanguageModels2022_images/ba26513b-feb1-4cca-a1ca-3c06344fb35e-60_460_797_1036_653.jpg)

**Figure 41: Win rates of PPO-ptx and SFT against 175B GPT-3 in RealToxicityPrompts.**

a minibatch size of 32 to be optimal and is slightly better than 64. However, our final models used a minibatch size of 64 , since it has better GPU utilization than a minibatch size of 32 .

