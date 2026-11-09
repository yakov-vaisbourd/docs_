# 1 Introduction

Language models are pretrained to predict the next token at an incredible scale, allowing them to learn general-purpose representations that can be transferred to nearly any language understanding or generation task. To enable this transfer, various methods for aligning language models have thus been proposed, primarily focusing on instruction tuning [Mishra et al., 2021, Wei et al., 2022a, Sanh et al., 2022] over large multi-million-example datasets [Chung et al., 2022. Beeching et al., 2023. Köpf et al., 2023], and more recently reinforcement learning from human feedback (RLHF) [Bai] et al., 2022a| Ouyang et al., 2022], collected over millions of interactions with human annotators. Existing alignment methods require significant amounts of compute and specialized data to achieve ChatGPT-level performance. However, we demonstrate that, given a strong pretrained language model, remarkably strong performance can be achieved by simply fine-tuning on 1,000 carefully curated training examples.

We hypothesize that alignment can be a simple process where the model learns the style or format for interacting with users, to expose the knowledge and capabilities that were already acquired during

\begin{table}
\begin{tabular}[t]{|l|l|l|l|}
\hline Source & \#Examples & Avg Input Len. & Avg Output Len. \\
\hline \multicolumn{4}{|l|}{Training} \\
\hline Stack Exchange (STEM) & 200 & 117 & 523 \\
\hline Stack Exchange (Other) & 200 & 119 & 530 \\
\hline wikiHow & 200 & 12 & 1,811 \\
\hline Pushshift r/WritingPrompts & 150 & 34 & 274 \\
\hline Natural Instructions & 50 & 236 & 92 \\
\hline Paper Authors (Group A) & 200 & 40 & 334 \\
\hline \multicolumn{4}{|l|}{Dev} \\
\hline Paper Authors (Group A) & 50 & 36 & N/A \\
\hline \multicolumn{4}{|l|}{Test} \\
\hline Pushshift r/AskReddit & 70 & 30 & N/A \\
\hline Paper Authors (Group B) & 230 & 31 & N/A \\
\hline
\end{tabular}

**Table 1: Sources of training prompts (inputs) and responses (outputs), and test prompts. The total amount of training data is roughly 750,000 tokens, split over exactly 1,000 sequences.**
\end{table}
pretraining. To test this hypothesis, we curate 1,000 examples that approximate real user prompts and high-quality responses. We select 750 top questions and answers from community forums, such as Stack Exchange and wikiHow, sampling for quality and diversity. In addition, we manually write 250 examples of prompts and responses, while optimizing for task diversity and emphasizing a uniform response style in the spirit of an AI assistant. Finally, we train LIMA, a pretrained 65B-parameter LLaMa model [Touvron et al., 2023] fine-tuned on this set of 1,000 demonstrations.

We compare LIMA to state-of-the-art language models and products across 300 challenging test prompts. In a human preference study, we find that LIMA outperforms RLHF-trained DaVinci003 from OpenAI, which was trained with RLHF, as well as a 65B-parameter reproduction of Alpaca [Taori et al., 2023], which was trained on 52,000 examples. While humans typically prefer responses from GPT-4, Claude, and Bard over LIMA, this is not always the case; LIMA produces equal or preferrable responses in 43\%, 46\%, and 58\% of the cases, respectively. Repeating the human preference annotations with GPT-4 as the annotator corroborates our findings. Analyzing LIMA responses on an absolute scale reveals that 88\% meet the prompt requirements, and 50\% are considered excellent.

Ablation experiments reveal vastly diminishing returns when scaling up data quantity without also scaling up prompt diversity, alongside major gains when optimizing data quality. In addition, despite having zero dialogue examples, we find that LIMA can conduct coherent multi-turn dialogue, and that this ability can be dramatically improved by adding only 30 hand-crafted dialogue chains to the training set. Overall, these remarkable findings demonstrate the power of pretraining and its relative importance over large-scale instruction tuning and reinforcement learning approaches.

