# 1 Introduction

An important goal of artificial intelligence is to develop models that can generalize to unseen tasks. In natural language processing (NLP), pretrained language models have made significant progress towards this goal, as they can perform tasks given natural language descriptions (Brown et al., 2020, inter alia). Further progress has been made by finetuning language models on a collection of tasks phrased as instructions, which enables models to respond better to instructions and reduces the need for few-shot exemplars (Ouyang et al., 2022; Wei et al., 2021; Sanh et al., 2021, inter alia).

In this paper we advance instruction finetuning in several ways. First, we study the impact of scaling on instruction finetuning. Our experiments show that instruction finetuning does scale well with the number of tasks and the size of the model. Their respective scaling behaviors suggest that future research should scale up the number of tasks and the size of the model even further. Second, we study the effect of finetuning on the ability of the models to perform reasoning tasks. Our experiments show that whereas prior instruction finetuning methods that do not include chain-of-thought (CoT; Wei et al., 2022b) severely degrade performance on CoT evaluations, adding just nine CoT datasets into the finetuning mixture enables better performance on all evaluations.

Based on these findings, we train Flan-PaLM by using a 540B-parameter model, increasing the number of finetuning tasks to 1.8K, and including CoT data. Flan-PaLM outperforms PaLM, achieving new state-of-theart on several benchmarks. For instance, Flan-PaLM's improved reasoning abilities enable it to leverage CoT and self-consistency (Wang et al., 2022c) to achieve 75.2\% on Massive Multi-task Language Understanding (MMLU; Hendrycks et al., 2020). Flan-PaLM also has improved multilingual abilities compared to PaLM, such as $14.9 \%$ absolute improvement on one-shot TyDiQA (Clark et al., 2020) and $8.1 \%$ on arithmetic reasoning in under-represented languages (Shi et al., 2022). In human rater evaluations, Flan-PaLM substantially outperforms PaLM on a challenging set of open-ended generation questions, suggesting improved usability. Moreover, we found that instruction finetuning also improves performance across several responsible AI evaluation benchmarks.

In addition, we also instruction-finetune Flan-T5 models (80M to 11B). These checkpoints have strong zeroshot, few-shot, and CoT abilities, outperforming prior public checkpoints such as T5 (Raffel et al., 2020). For example, Flan-T5 11B outperforms T5 11B by double-digit improvements and even outperforms PaLM 62B on some challenging BIG-Bench tasks (Srivastava et al., 2022). Overall, our results underscore how instruction finetuning can improve performance across a range of models, prompting setups, and evaluation tasks.

\begin{table}
\begin{tabular}[t]{|l|l|l|}
\hline - & Random & 25.0 \\
\hline - & Average human rater & 34.5 \\
\hline May 2020 & GPT-3 5-shot & 43.9 \\
\hline Mar. 2022 & Chinchilla 5-shot & 67.6 \\
\hline Apr. 2022 & PaLM 5-shot & 69.3 \\
\hline Oct. 2022 & Flan-PaLM 5-shot & 72.2 \\
\hline & Flan-PaLM 5-shot: CoT + SC & 75.2 \\
\hline - & Average human expert & 89.8 \\
\hline & Jun. 2023 forecast (Hypermind) & 73.2 \\
\hline & Jun. 2024 forecast (Hypermind) & 75.0 \\
\hline & Jun. 2023 forecast (Metaculus) & 82.7 \\
\hline & Jun. 2024 forecast (Metaculus) & 87.6 \\
\hline
\end{tabular}

**Table 1: Average 5-shot MMLU scores (\%) for 57 tasks with model and human accuracy comparisons (Hendrycks et al., 2020). Forecasts were made in July 2022 by competitive human forecasters, regarding a single model (Steinhardt, 2021); see https://prod.hypermind.com/ngdp/en/showcase2/showcase.html?sc= JSAI and https://www.metaculus.com/questions/11676/mmlu-sota-in-2023-2025/. CoT + SC: chainof-thought prompting with self-consistency (Wang et al., 2022b).**
\end{table}

![Figure](./chungScalingInstructionFinetunedLanguage2022_images/5b970e92-029a-4684-b206-dafd7ac86e22-03_580_1594_266_258.jpg)

**Figure 2: Our finetuning data comprises 473 datasets, 146 task categories, and 1,836 total tasks. Details for the tasks used in this paper is given in Appendix F.**

\begin{itemize}
\item[-] A Dataset is an original data source (e.g. SQuAD).
\item[-] A Task Category is unique task setup (e.g. the SQuAD dataset is configurable for multiple task categories such as extractive question answering, query generation, and context generation).
\item[-] A Task is a unique <dataset, task category> pair, with any number of templates which preserve the task category (e.g. query generation on the SQuAD dataset.)
\end{itemize}

![Figure](./chungScalingInstructionFinetunedLanguage2022_images/5b970e92-029a-4684-b206-dafd7ac86e22-03_324_1611_1104_258.jpg)

**Figure 2: Our finetuning data comprises 473 datasets, 146 task categories, and 1,836 total tasks. Details for the tasks used in this paper is given in Appendix F.**

