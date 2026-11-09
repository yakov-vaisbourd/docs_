# F Finetuning Data Card

We present the data card, following the format proposed by Pushkarna et al. (2021).

Dataset Owners. flan-core@google.com

Dataset Overview. Our final set of finetuning tasks is sourced from a combination of tasks from FLAN, T0, Natural Instructions, along with some dialog, program synthesis, and chain-of-thought reasoning tasks, as described in Figure 2. We provide specific pointers and citations in Table 24. All data sources are publicly available. We also remove all MMLU tasks from Natural Instructions to preserve its role as a broad benchmark of 57 held-out tasks for evaluation. In total, there are 1,836 tasks.

\begin{table}
\begin{tabular}[t]{|l|l|l|}
\hline Source Mixture & Task Name & Reference \\
\hline \multicolumn{3}{|l|}{Collections} \\
\hline Muffin & FLAN & Wei et al. (2021) \\
\hline T0-SF & T0 & Sanh et al. (2021) \\
\hline Natural Instructions v2 & Natural Instructions & Wang et al. (2022c) \\
\hline \multicolumn{3}{|l|}{Individual Datasets} \\
\hline Muffin & QReCC & Anantha et al. (2021) \\
\hline Muffin & Task Master & Byrne et al. (2019) \\
\hline Muffin & Wiki Dialog & Dai et al. (2022) \\
\hline Muffin & Dr Repair - Error Comments & Yasunaga and Liang (2020) \\
\hline Muffin & Dr Repair - Line Numbers & Yasunaga and Liang (2020) \\
\hline Muffin & Dr Repair - No Errors & Yasunaga and Liang (2020) \\
\hline Muffin & Dr Repair - Plain Code & Yasunaga and Liang (2020) \\
\hline Muffin & DeepMind Coding Contents & Yasunaga and Liang (2020) \\
\hline Muffin & Lambada & Paperno et al. (2016) \\
\hline Muffin & UnifiedQA & Khashabi et al. (2020) \\
\hline Reasoning & GSM8k & Cobbe et al. (2021) \\
\hline Reasoning & StrategyQA & Geva et al. (2021) \\
\hline Reasoning & AQuA & Ling et al. (2017) \\
\hline Reasoning & Creak & Onoe et al. (2021) \\
\hline Reasoning & ECQA & Aggarwal et al. (2021) \\
\hline Reasoning & ESNLI & Camburu et al. (2018) \\
\hline Reasoning & QASC & Khot et al. (2020) \\
\hline Reasoning & QED & Lamm et al. (2021) \\
\hline Reasoning & SenseMaking & Wang et al. (2019b) \\
\hline
\end{tabular}

**Table 24: Here we list and cite collections and individual datasets used in instruction finetuning. For natural instructions, we removed all tasks related to MMLU, our held-out evaluation set.**
\end{table}

Risk and Mitigation. Flan data is built on top of publicly available datasets. Unwanted data distribution such bias might present in those publicly available datasets and therefore, Flan. Downstream users should be aware of the potential risks.

Maintenance status. Limited Maintenance. The data will not be updated, but any technical issues will be addressed.

Maintenance plan. Any technical issues will be addressed. No new versions are planned.

Versioning: No new versions will be available.
Updates: Updates are only limited to bug and error fixes.
Errors: Error handling will be considered case by case.
Feedback: Email lehou@google.com and cc flan-core@google.com.

Example: typical data point.

Inputs: Answer the following question: who played luke skywalker in the new star wars films?
Targets: Mark Richard Hamill

Sensitive human attributes. The upstream data sources may contain unknown sensitive human attributes.

Data distributions. Figure 17 shows the top 20 task category counts per mixture. Figure 18 counts the occurrence of languages present over 1,836 included tasks. Most tasks only include one language, while many translation/localization-related tasks include 2 or more. Notably, 60 languages are present in total, including many lower-resource languages, though their frequency is low.

![Figure](./chungScalingInstructionFinetunedLanguage2022_images/5b970e92-029a-4684-b206-dafd7ac86e22-49_1357_1649_661_238.jpg)

**Figure 17: The top 20 task category counts per mixture.**

![Figure](./chungScalingInstructionFinetunedLanguage2022_images/5b970e92-029a-4684-b206-dafd7ac86e22-50_1762_1579_455_260.jpg)

**Figure 18: The languages represented across tasks in our full multi-task finetuning data suite.**

