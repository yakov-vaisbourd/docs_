\title{
Scaling Instruction-Finetuned Language Models
}

\author{
Hyung Won Chung* Le Hou* Shayne Longpre* Barret Zoph ${ }^{\dagger}$ Yi Tay ${ }^{\dagger}$ William Fedus ${ }^{\dagger}$ Yunxuan Li Xuezhi Wang Mostafa Dehghani Siddhartha Brahma Albert Webson Shixiang Shane Gu Zhuyun Dai Mirac Suzgun Xinyun Chen Aakanksha Chowdhery Alex Castro-Ros Marie Pellat Kevin Robinson Dasha Valter Sharan Narang Gaurav Mishra Adams Yu Vincent Zhao Yanping Huang Andrew Dai Hongkun Yu Slav Petrov Ed H. Chi Jeff Dean Jacob Devlin Adam Roberts Denny Zhou Quoc V. Le Jason Wei*
}

Google

\begin{abstract}
Finetuning language models on a collection of datasets phrased as instructions has been shown to improve model performance and generalization to unseen tasks. In this paper we explore instruction finetuning with a particular focus on (1) scaling the number of tasks, (2) scaling the model size, and (3) finetuning on chain-of-thought data. We find that instruction finetuning with the above aspects dramatically improves performance on a variety of model classes (PaLM, T5, U-PaLM), prompting setups (zero-shot, few-shot, CoT), and evaluation benchmarks (MMLU, BBH, TyDiQA, MGSM, open-ended generation, RealToxicityPrompts). For instance, Flan-PaLM 540B instruction-finetuned on 1.8K tasks outperforms PaLM 540B by a large margin (+9.4\% on average). Flan-PaLM 540B achieves state-of-the-art performance on several benchmarks, such as 75.2\% on five-shot MMLU. We also publicly release Flan-T5 checkpoints, ${ }^{1}$ which achieve strong few-shot performance even compared to much larger models, such as PaLM 62B. Overall, instruction finetuning is a general method for improving the performance and usability of pretrained language models.
\end{abstract}

![Figure](./chungScalingInstructionFinetunedLanguage2022_images/5b970e92-029a-4684-b206-dafd7ac86e22-01_670_1341_1532_393.jpg)

**Figure 1: We finetune various language models on 1.8 K tasks phrased as instructions, and evaluate them on unseen tasks. We finetune both with and without exemplars (i.e., zero-shot and few-shot) and with and without chain-of-thought, enabling generalization across a range of evaluation scenarios.**

\footnotetext{
*Equal contribution. Correspondence: lehou@google.com.
${ }^{\dagger}$ Core contributor.
${ }^{1}$ Public checkpoints: https://github.com/google-research/t5x/blob/main/docs/models.md\#flan-t5-checkpoints.
}

