# C Data Contamination Analysis

One reasonable concern is that since the pretraining corpus of FLAN has more than 2 trillion tokens, it is possible that examples from a given evaluation dataset may have already been seen verbatim by the model during pre-training, hence inflating the performance of our purported zero-shot model. To this end, like GPT-3 (Brown et al., 2020), we perform post-hoc data contamination analysis to

\begin{table}
\begin{tabular}[t]{|l|l|l|l|l|l|l|}
\hline \multirow[b]{2}{*}{Finetuning prompt} & \multirow[b]{2}{*}{Inference prompt} & \multicolumn{5}{|r|}{Zero-shot performance on unseen task cluster} \\
\hline & & NLI & Read. Comp. & ClosedBook QA & Translation & Four-Task Average \\
\hline Natural instructions (= FLAN) & Natural instructions & 56.2 & 77.4 & 56.6 & 30.7 & 55.2 \\
\hline No template & Natural instructions & 50.5 & 58.2 & 25.5 & 15.0 & 37.3 \\
\hline Task/dataset name & Natural instructions & 52.8 & 63.0 & 44.8 & 25.9 & 46.6 \\
\hline Task/dataset name & Task/dataset name & 60.2 & 64.9 & 40.8 & 21.9 & 47.0 \\
\hline
\end{tabular}

**Table 3: Ablation study result using models where instructions are removed from the finetuning process. In "no template," only inputs and outputs are given, which does not distinguish among tasks during multi-task finetuning. In "task/dataset name", inputs during multi-task finetuning are prepended with the name of the task and dataset (e.g., "[Translation: WMT'14 to French] The dog runs") NLI datasets: ANLI R1-R3, CB, and RTE; reading comprehension datasets: BoolQ, MultiRC, and OpenbookQA; closed-book QA datasets: ARC-c, ARC-e, NQ, and TQA; translation datasets: WMT'14 Fr↔En, WMT'16 De↔En, and WMT'16 Ro↔En. Notably, training with task/dataset name achieved a high NLI score largely because it achieved a score of 83.9 on the CB dataset, for which the validation set only has 56 examples (FLAN also gets 83.9 with the best dev template, but the average template was only 64.1).**
\end{table}

\begin{table}
\begin{tabular}[t]{|l|l|l|l|l|l|l|l|l|l|}
\hline \multirow{2}{*}{} & \multirow[b]{2}{*}{Prompt tuning train. examples} & \multicolumn{8}{|c|}{Prompt Tuning Analysis} \\
\hline & & BoolQ acc. & CB acc. & CoPA acc. & MultiRC F1 & ReCoRD acc. & RTE acc. & WiC acc. & WSC acc. \\
\hline LaMDA-PT & \multirow{2}{*}{32} & 55.5 & 55.4 & 87.0 & 65.4 & 78.0 & 52.4 & 51.6 & 65.4 \\
\hline FLAN & & 77.5 & 87.5 & 91.0 & 76.8 & 80.8 & 83.0 & 57.8 & 70.2 \\
\hline LaMDA-PT & \multirow{2}{*}{full dataset} & 82.8 & 87.5 & 90.0 & 78.6 & 84.8 & 82.0 & 54.9 & 72.7 \\
\hline FLAN & & 86.3 & 98.2 & 94.0 & 83.4 & 85.1 & 91.7 & 74.0 & 86.5 \\
\hline
\end{tabular}

**Table 4: FLAN (instruction tuning) responds better to continuous inputs attained via prompt tuning than LaMDA-PT (no instruction tuning). When prompt tuning on a given dataset, no tasks from the same cluster as that dataset were seen during instruction tuning.**
\end{table}
investigate whether the performance of the model is in fact inflated by evaluating on examples that occurred in the pretraining dataset.

Our data contamination procedure follows the setup of Brown et al. (2020), which, for each benchmark, produces a "clean" version that removes all potentially leaked examples, defined as examples for which any $n$-gram ( $n$ varies per dataset but is roughly 13) overlapped with anything in the pretraining corpus. We use the same $n$ per dataset as Brown et al. (2020) and also split on spaces. We then evaluate our model on this clean subset, comparing against model performance on the original dataset (clean + dirty). Lower performance on the clean subset would suggest that data contamination leads to inflated results.

Figure 12 summarizes these results, with the exact numbers given in Table 5. We see several trends very similar to those in the GPT-3 paper: (1) many datasets had a substantial number of examples that overlapped with the pretraining data, (2) across all datasets, we do not see a correlation that evaluating on clean data does worse than evaluating on the total dataset, and (3) as datasets had fewer clean examples, there was higher variance in the percent change in performance (likely due to a smaller number of clean examples).

Like GPT-3, we also found that DROP and SQuADv2 had almost total overlap with the pretraining data. We follow their procedure of manually inspecting the data, and find that most overlapping $n$ grams were only in the contexts of examples (99.6\% for DROP and 97.2\% for SQuADv2). Overlaps never occurred in both the question and answer for DROP, and only occurred for both the question and answer for SQuADv2 in 5 of the 11,153 evaluation examples. Hence, for these two datasets, the

![Figure](./weiFinetunedLanguageModels2022_images/aa25be03-d0f6-48ca-994b-33ce39180820-26_480_1360_283_380.jpg)

**Figure 12: Like GPT-3, we also measured performance on cleaned versions of our datasets, which had high confidence to be unseen in the pretraining data of FLAN. We do not see a correlation that FLAN performed better on evaluation sets for which examples occurred more often in the pretraining data. When the percent of clean data is very small, there are fewer examples for computing the clean performance, which leads to high variance.**

model gains only background information and cannot memorize the answer to any specific questions (aside from the five examples in SQuADv2).

ANLI R1 and R2 (Nie et al., 2020) also had almost complete data contamination, to a much higher degree than GPT-3. Upon further inspection, we see that most overlaps occur in example contexts and not hypotheses (97.3\% for ANLI R1 and 98.2\% for ANLI R2). As ANLI R1 and R2 are based entirely from Wikipedia examples (R3 is not), we posit that this higher degree of contamination in our pretraining dataset compared with GPT-3's is potentially due to using a more-recent version of Wikipedia that includes the contexts used in ANLI R1 and R2 (which were collected in 2019). Because seeing a particular context in pretraining does not help with the NLI task given a new, unseen sentence, we think it is unlikely that these overlaps affected performance on the two datasets.

Of the remaining datasets, only ReCoRD and PIQA had a clean subset performance that was lower than the overall evaluation set performance by more than 1\%. These two datasets are language modeling (i.e., "what's the best continuation of this sentence?"), and so it is more likely compared with previous tasks that seeing a complete sentence in the pretraining data could help the model predict the right answer in downstream evaluations. For PIQA, both the goal and solution had overlaps in 93 of the 1,838 evaluation examples, and for ReCoRD, the query had overlaps in 2,320 of the 10,000 training examples. We hence mark these results with an asterisk * in Table 2. Brown et al. (2020) also reported substantial contamination rates for these two datasets (61\% dirty for ReCoRD and 29\% for PIQA), and also mark PIQA with an asterisk.

As this overlap analysis follows that performed in Brown et al. (2020), we reiterate the same caveats: the conservative nature of our $n$-gram matching procedure likely introduces additional false positives; there are no guarantees that the clean subset is drawn from the same distribution as the overall subset; and, accurately detecting test contamination is a relatively new research area without established best practices. Moreover, as our pretraining corpus is almost five times larger than that used for GPT-3 (which was 500B tokens), it is possible that there are more false positives in detecting dirty data.

\begin{table}
\begin{tabular}[t]{|l|l|l|l|l|l|l|l|}
\hline Dataset & Metric & Total count & Total acc/F1/BLEU & Clean count & Clean acc/F1/BLEU & \% clean & \% Diff (clean overall) \\
\hline DROP & F1 & 9,536 & 22.4 & 61 & 33.0 & 0.6 & 47.4 \\
\hline SQuADv2 & F1 & 11,873 & 41.3 & 106 & 38.7 & 0.9 & -6.2 \\
\hline ANLI R1 & acc & 1,000 & 48.1 & 14 & 57.1 & 1.4 & 18.8 \\
\hline ANLI R2 & acc & 1,000 & 42.9 & 21 & 38.1 & 2.1 & -11.2 \\
\hline ReCoRD & acc & 10,000 & 4.6 & 3,203 & 4.5 & 32.0 & -2.7 \\
\hline MultiRC & acc & 4,848 & 75.4 & 1,972 & 75.7 & 40.7 & 0.5 \\
\hline PIQA & acc & 1,838 & 23.7 & 896 & 23.3 & 48.7 & -1.7 \\
\hline ANLI R3 & acc & 1,200 & 44.2 & 718 & 45.3 & 59.8 & 2.5 \\
\hline HellaSwag & acc & 10,042 & 28.5 & 6,578 & 28.7 & 65.5 & 0.7 \\
\hline RTE & acc & 2,77 & 84.1 & 183 & 84.2 & 66.1 & 0.0 \\
\hline WMT'14 En→Fr & BLEU & 3,003 & 31.3 & 2,243 & 31.5 & 74.7 & 0.9 \\
\hline WMT'14 Fr→En & BLEU & 3,003 & 34.0 & 2,243 & 34.1 & 74.7 & 0.2 \\
\hline BoolQ & acc & 3,270 & 76.5 & 2,515 & 76.3 & 76.9 & -0.4 \\
\hline TQA (tfds-dev) & F1 & 11,313 & 62.2 & 8,731 & 62.0 & 77.2 & -0.2 \\
\hline ARC Easy & acc & 2,365 & 79.5 & 1,888 & 79.0 & 79.8 & -0.6 \\
\hline ARC Challenge & acc & 1,165 & 63.1 & 983 & 64.2 & 84.4 & 1.7 \\
\hline OpenbookQA & acc & 500 & 74.6 & 425 & 74.8 & 85.0 & 0.3 \\
\hline WMT'16 En→De & BLEU & 2,999 & 22.7 & 2,569 & 23.0 & 85.7 & 1.4 \\
\hline WMT'16 De→En & BLEU & 2,999 & 38.6 & 2,569 & 38.7 & 85.7 & 0.2 \\
\hline WMT'16 En→Ro & BLEU & 1,999 & 15.5 & 1,752 & 15.4 & 87.6 & -0.7 \\
\hline WMT' 16 Ro→En & BLEU & 1,999 & 36.7 & 1,752 & 36.8 & 87.6 & 0.1 \\
\hline COPA & acc & 100 & 88.0 & 91 & 87.9 & 91.0 & -0.1 \\
\hline CB & acc & 56 & 41.1 & 53 & 41.5 & 94.6 & 1.1 \\
\hline NQ & F1 & 3,610 & 24.5 & 3,495 & 24.3 & 96.8 & -0.5 \\
\hline StoryCloze & acc & 1,871 & 92.1 & 1,864 & 92.1 & 99.6 & 0.0 \\
\hline Winogrande & acc & 1,267 & 39.4 & 1,265 & 39.4 & 99.8 & 0.2 \\
\hline
\end{tabular}

**Table 5: Overlap statistics for the subset of datasets that are also used in GPT-3, sorted from dirtiest to cleanest. An evaluation example was dirty if it had any $n$-gram collision with the pretraining corpus. We computed these results for FLAN's performance using only a single template for each dataset, so they differ slightly compared with the results for average performance over all templates.**
\end{table}

