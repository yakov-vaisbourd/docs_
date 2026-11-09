# 1 Introduction

Language models (LMs) at scale, such as GPT-3 (Brown et al., 2020), have been shown to perform few-shot learning remarkably well. They are less successful at zero-shot learning, however. For example, GPT-3's zero-shot performance is much worse than few-shot performance on tasks such as reading comprehension, question answering, and natural language inference. One potential reason is that, without few-shot exemplars, it is harder for models to perform well on prompts that are not similar to the format of the pretraining data.

In this paper, we explore a simple method to improve the zero-shot performance of large language models, which would expand their reach to a broader audience. We leverage the intuition that NLP tasks can be described via natural language instructions, such as "Is the sentiment of this movie review positive or negative?" or "Translate 'how are you' into Chinese." We take a pretrained language model of 137B parameters and perform instruction tuning-finetuning the model on a mixture of more than 60 NLP datasets expressed via natural language instructions. We refer to this resulting model as FLAN, for Finetuned Language Net.

To evaluate the zero-shot performance of FLAN on unseen tasks, we group NLP datasets into clusters based on their task types and hold out each cluster for evaluation while instruction tuning FLAN on all other clusters. For example, as shown in Figure 1, to evaluate FLAN's ability to perform natural language inference, we instruction tune the model on a range of other NLP tasks such as commonsense reasoning, translation, and sentiment analysis. As this setup ensures that FLAN has not seen any natural language inference tasks in instruction tuning, we then evaluate its ability to perform zero-shot natural language inference.

Our evaluations show that FLAN substantially improves the zero-shot performance of the base 137B-parameter model. FLAN's zero-shot also outperforms 175B-parameter GPT-3's zero-shot on 20 of 25 datasets that we evaluate, and even outperforms GPT-3's few-shot by a large margin on ANLI, RTE, BoolQ, AI2-ARC, OpenbookQA, and StoryCloze. In ablation studies, we find that increasing the number of task clusters in instruction tuning improves performance on unseen tasks and that the benefits of instruction tuning emerge only with sufficient model scale.

Instruction tuning is a simple method that, as depicted in Figure 2, combines appealing aspects of both the pretrain-finetune and prompting paradigms by using supervision via finetuning to improve language model's responses to inference-time text interactions. Our empirical results demonstrate promising abilities of language models to perform tasks described purely via instructions. Source code for loading the instruction tuning dataset used for FLAN is publicly available at https://github.com/google-research/flan.

![Figure](./weiFinetunedLanguageModels2022_images/aa25be03-d0f6-48ca-994b-33ce39180820-02_421_1290_1701_412.jpg)

**Figure 2: Comparing instruction tuning with pretrain-finetune and prompting.**

