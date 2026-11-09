# 5 Resilience to Domain Shift

By freezing the core language model parameters, prompt tuning prevents the model from modifying its general understanding of language. Instead, prompt representations indirectly modulate the representation of the input. This reduces the model's ability to overfit to a dataset by memorizing spe-

\begin{table}
\begin{tabular}[t]{|l|l|l|l|l|}
\hline Train & Eval & Tuning & Accuracy & F1 \\
\hline \multirow[t]{2}{*}{QQP} & \multirow[t]{2}{*}{MRPC} & Model & $73.1 \pm 0.9$ & $81.2 \pm 2.1$ \\
\hline & & Prompt & 76.3 ±0.1 & $\mathbf{8 4 . 3} \pm 0.3$ \\
\hline \multirow[t]{2}{*}{MRPC} & \multirow[t]{2}{*}{QQP} & Model & 74.9 ±1.3 & 70.9 ±1.2 \\
\hline & & Prompt & $\mathbf{7 5 . 4} \pm 0.8$ & $69.7 \pm 0.3$ \\
\hline
\end{tabular}

**Table 2: Mean and stddev of zero-shot domain transfer between two paraphrase detection tasks.**
\end{table}
cific lexical cues and spurious correlations. This restriction suggests that prompt tuning may improve robustness to domain shifts, where the distribution of inputs differs between training and evaluation.

We investigate zero-shot domain transfer on two tasks: question answering (QA) and paraphrase detection. For question answering, we use the MRQA 2019 shared task on generalization (Fisch et al., 2019). This task collects extractive QA datasets in a unified format and tests how models trained on "in-domain" datasets perform when evaluated on "out-of-domain" datasets. For our experiments, we train on SQuAD (Rajpurkar et al., 2016) and evaluate on each of the out-of-domain datasets. ${ }^{11}$

Table 1 shows that prompt tuning outperforms model tuning on the majority of out-of-domain datasets, with a remarkable 12.5 point F1 gap between the two approaches on TextbookQA. We observe larger gains from prompt tuning in cases of larger domain shifts (e.g. to Biomedical in BioASQ or to Textbooks in TextbookQA). Of the datasets where model tuning is better, we see that DROP shares a domain (Wikipedia) with SQuAD and is thus one of the smallest domain transfers.

As a second test of robustness to domain shift, we explore transfer between two paraphrase detection tasks from GLUE (Wang et al., 2019b). The first task is QQP (Iyer et al., 2017), which asks if two questions from the community Q\&A site Quora are "duplicates". The second task is MRPC (Dolan and Brockett, 2005), which asks if two sentences drawn from news articles are paraphrases. We test transfer in both directions (QQP ⇔ MRPC). As before, we train on the "in-domain" task, select checkpoints using in-domain validation, and evaluate zero-shot on the "out-of-domain" task.

Table 2 shows that training a lightweight prompt on the QQP data and evaluating on MRPC gives much better performance than tuning the entire

\footnotetext{
${ }^{11}$ We select checkpoints based on SQuAD validation F1. The out-of-domain datasets are TextbookQA (Kembhavi et al., 2017), RACE (Lai et al., 2017), BioASQ (http://bioasq. org/), RE (Levy et al., 2017), DuoRC (Saha et al., 2018), and DROP (Dua et al., 2019).
}

\begin{table}
\begin{tabular}[t]{|l|l|l|l|l|}
\hline Dataset & Metric & Average & Best & Ensemble \\
\hline BoolQ & acc. & 91.1 & 91.3 & 91.7 \\
\hline CB & acc./F1 & 99.3 / 99.0 & 100.00 / 100.00 & 100.0 / 100.0 \\
\hline COPA & acc. & 98.8 & 100.0 & 100.0 \\
\hline MultiRC & $\mathrm{EM} / \mathrm{F} 1_{a}$ & 65.7 / 88.7 & 66.3 / 89.0 & 67.1 / 89.4 \\
\hline ReCoRD & EM/F1 & 92.7 / 93.4 & 92.9 / 93.5 & 93.2 / 93.9 \\
\hline RTE & acc. & 92.6 & 93.5 & 93.5 \\
\hline WiC & acc. & 76.2 & 76.6 & 77.4 \\
\hline WSC & acc. & 95.8 & 96.2 & 96.2 \\
\hline \multicolumn{2}{|l|}{SuperGLUE (dev)} & 90.5 & 91.0 & 91.3 \\
\hline
\end{tabular}

**Table 3: Performance of a five-prompt ensemble built from a single frozen T5-XXL model exceeds both the average and the best among the five prompts.**
\end{table}
model (+3.2 accuracy and +3.1 F1). The results are much closer in the other direction, with prompt tuning showing a small improvement in accuracy and a small drop in F1. These results support the view that model tuning may be over-parameterized and more prone to overfit the training task, to the detriment of similar tasks in different domains.

