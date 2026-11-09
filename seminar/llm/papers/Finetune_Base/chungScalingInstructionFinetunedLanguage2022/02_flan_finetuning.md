# 2 Flan Finetuning

We instruction-finetune on a collection of data sources (Figure 2) with a variety of instruction template types (Figure 3). We call this finetuning procedure Flan (Finetuning language models; Wei et al., 2021) and prepend "Flan" to the resulting finetuned models (e.g., Flan-PaLM). ${ }^{2}$ We show that Flan works across several model sizes and architectures (Table 2).

## 2.1 Finetuning Data

Task mixtures. Prior literature has shown that increasing the number of tasks in finetuning with instructions improves generalization to unseen tasks (Wei et al., 2021; Sanh et al., 2021, inter alia). In this paper we scale to 1,836 finetuning tasks by combining four mixtures from prior work: Muffin, T0-SF, NIV2, and CoT, as summarized in Figure 2. Muffin ${ }^{3}$ (80 tasks) comprises 62 tasks from Wei et al. (2021) and 26 new tasks that we added in this work, including dialog data (Byrne et al., 2019; Anantha et al., 2021; Dai et al., 2022) and program synthesis data (Yasunaga and Liang, 2020; Li et al., 2022). T0-SF (193 tasks) comprises tasks from T0 (Sanh et al., 2021) that do not overlap with the data used in Muffin (SF stands for "sans Flan"). NIV2 (1554 tasks) comprises tasks from Wang et al. (2022c). ${ }^{4}$

\footnotetext{
${ }^{2}$ We use "Flan" to refer to our finetuning procedure. "FLAN" is a model in Wei et al. (2021).
${ }^{3}$ Multi-task finetuning with instructions.
${ }^{4}$ We removed 44 tasks related to MMLU (Hendrycks et al., 2020), since MMLU is used for evaluation.
}

![Figure](./chungScalingInstructionFinetunedLanguage2022_images/5b970e92-029a-4684-b206-dafd7ac86e22-04_652_1600_265_265.jpg)

**Figure 3: Combinations of finetuning data formats in this work. We finetune with and without exemplars, and also with and without chain-of-thought. In addition, we have some data formats without instructions but with few-shot exemplars only, like in Min et al. (2022) (not shown in the figure). Note that only nine chain-of-thought (CoT) datasets use the CoT formats.**

Chain-of-thought finetuning mixture. The fourth finetuning data mixture (reasoning) involves CoT annotations, which we use to explore whether finetuning on CoT annotations improves performance on unseen reasoning tasks. We create a new mixture of nine datasets from prior work for which human raters manually wrote CoT annotations for a training corpus. These nine datasets include tasks such as arithmetic reasoning (Cobbe et al., 2021), multi-hop reasoning (Geva et al., 2021), and natural language inference (Camburu et al., 2020). We manually compose ten instruction templates per task. A data card is in Appendix F.

Templates and formatting. For Muffin, T0-SF, and NIV2, we use instructional templates for each task as given by the creators of the mixtures. For CoT, we manually write around ten instruction templates for each of the nine datasets. To create few-shot templates, we write a variety of exemplar delimiters (e.g., "Q:"/"A:") and apply them randomly at the example level. An example of formatting for both with and without exemplars, as well as with and without CoT, is shown in Figure 3.

## 2.2 Finetuning procedure

In this paper, we apply instruction finetuning across a broad range of model families, including T5 (Raffel et al., 2020), PaLM (Chowdhery et al., 2022), and U-PaLM (Tay et al., 2022b). These model families span a range of sizes, from Flan-T5-small (80M parameters), to PaLM and U-PaLM (540B parameters). For each model, we apply the same training procedure, except for a few hyperparameters: learning rate, batch size, dropout, and finetuning steps. We use a constant learning rate schedule and finetune using the Adafactor optimizer (Shazeer and Stern, 2018). We use packing (Raffel et al., 2020) to combine multiple training examples into a single sequence, separating inputs from targets using an end-of-sequence token. Masking is applied to prevent the tokens from attending to others across the packed example boundary. The number of finetuning steps, learning rate, batch size, and dropout for each model are given in Appendix E. For each model, we use a single checkpoint for all evaluations; the optimal step was chosen based on periodic evaluations (every 2k to 10k steps depending the model size) of the held-out tasks, and we used the same number of checkpoint steps across all ablation runs for a given model. Notably, the amount of compute used for finetuning is only a small fraction relative to the training compute, as shown in Table 2. For example, we only use 0.2\% of the pre-training compute to instruction-finetune Flan-PaLM 540B (approximately 512 v4 TPU chips for 37 hours). We use the JAX-based T5X framework (Bradbury et al., 2018; Roberts et al., 2022).

\begin{table}
\begin{tabular}[t]{|l|l|l|l|l|l|l|}
\hline Params & Model & Architecture & Pre-training Objective & Pre-train FLOPs & Finetune FLOPs & \% Finetune Compute \\
\hline 80M & Flan-T5-Small & encoder-decoder & span corruption & $1.8 \mathrm{E}+20$ & $2.9 \mathrm{E}+18$ & 1.6\% \\
\hline 250M & Flan-T5-Base & encoder-decoder & span corruption & $6.6 \mathrm{E}+20$ & 9.1E+18 & 1.4\% \\
\hline 780M & Flan-T5-Large & encoder-decoder & span corruption & 2.3E+21 & $2.4 \mathrm{E}+19$ & 1.1\% \\
\hline 3B & Flan-T5-XL & encoder-decoder & span corruption & 9.0E+21 & 5.6E+19 & 0.6\% \\
\hline 11B & Flan-T5-XXL & encoder-decoder & span corruption & $3.3 \mathrm{E}+22$ & $7.6 \mathrm{E}+19$ & 0.2\% \\
\hline 8B & Flan-PaLM & decoder-only & causal LM & $3.7 \mathrm{E}+22$ & 1.6E+20 & 0.4\% \\
\hline 62B & Flan-PaLM & decoder-only & causal LM & $2.9 \mathrm{E}+23$ & 1.2E+21 & 0.4\% \\
\hline 540B & Flan-PaLM & decoder-only & causal LM & $2.5 \mathrm{E}+24$ & 5.6E+21 & 0.2\% \\
\hline 62B & Flan-cont-PaLM & decoder-only & causal LM & 4.8E+23 & 1.8E+21 & 0.4\% \\
\hline 540B & Flan-U-PaLM & decoder-only & prefix LM + span corruption & 2.5E+23 & 5.6E+21 & 0.2\% \\
\hline
\end{tabular}

**Table 2: Across several models, instruction finetuning only costs a small amount of compute relative to pre-training. T5: Raffel et al. (2020). PaLM and cont-PaLM (also known as PaLM 62B at 1.3T tokens): Chowdhery et al. (2022). U-PaLM: Tay et al. (2022b).**
\end{table}

## 2.3 Evaluation protocol

Evaluation benchmarks. We focus on performance on held-out tasks which were not included as part of the finetuning data. We are interested in Flan-PaLM's overall capabilities on world knowledge and reasoning tasks. Thus, we evaluate the model on a range of different benchmarks, including multilingual ones. We do not use the evaluation set from Brown et al. (2020) since almost all of those tasks have training sets that are included in our finetuning mixture. Instead, we use the following challenging benchmarks, for which current language models still perform well below expert human raters. (1) MMLU (Hendrycks et al., 2020) includes exam questions from 57 tasks such as mathematics, history, law, and medicine. (2) BBH includes 23 challenging tasks from BIG-Bench (Srivastava et al., 2022) for which PaLM performs below an average human rater (Suzgun et al., 2022). (3) TyDiQA (Clark et al., 2020) is a question-answering benchmark across 8 typologically diverse languages. (4) MGSM (Shi et al., 2022) is a multilingual benchmark of math word problems from Cobbe et al. (2021) manually translated into 10 languages. These benchmarks were also used in the PaLM paper (Chowdhery et al., 2022), which did not find any meaningful data contamination with pre-training data, consistent with data contamination analyses in previous work (Brown et al., 2020; Wei et al., 2021; Du et al., 2022). Responsible AI evaluations are discussed in Appendix C.

Evaluation methods and metrics. For MMLU and BBH, we evaluate both the ability to directly predict the answer via direct prompting, where the model directly gives the answer (Brown et al., 2020; Srivastava et al., 2022), as well as via chain-of-thought (CoT) prompting, where the model must provide a reasoning chain before giving the final answer (Wei et al., 2022b). For TyDiQA, we only measure direct prompting exact-match score, since highlighting the portion of a passage with the correct answer may not require sophisticated reasoning. For MGSM, we only measure CoT prompting accuracy since direct prompting has very low performance. For all benchmarks we use the given few-shot exemplars, with the number of exemplars following prior work: five-shot for MMLU, three-shot for BBH, one-shot for TyDiQA, and 8-shot for MGSM. For a given model we also report a single "normalized average" metric, following the "normalized preferred metric" in BIG-Bench (Srivastava et al., 2022). ${ }^{5}$ Our normalized average metric is the macro-average over six normalized scores: MMLU-Direct, MMLU-CoT, BBH-Direct, BBH-CoT, TyDiQA-Direct, and MGSM-CoT. Results for all tasks in each benchmark are reported in Appendix D. Some Responsible AI benchmarks use additional methods for generative tasks described in Appendix C.

\footnotetext{
${ }^{5}$ A normalized metric scales an evaluation number with respect to a task-specific lower bound such as random guessing baseline for a multiple choice question. For example, if random guessing produces $50 \%$ accuracy and the max accuracy of $100 \%$, then a raw accuracy of 55\% would be be normalized to 10\%, and a raw accuracy of 45\% would be normalized to -10\% since it is worse than random.
}

