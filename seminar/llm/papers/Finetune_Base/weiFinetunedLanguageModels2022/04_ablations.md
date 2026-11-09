# 4 Ablation Studies \& Further Analysis

## 4.1 Number of instruction tuning clusters

As the core question of our paper asks how instruction tuning improves a model's zero-shot performance on unseen tasks, in this first ablation we examine how performance is affected by the number of clusters and tasks used in instruction tuning. For this setup, we hold out NLI, closed-book QA, and commonsense reasoning as evaluation clusters, and use the seven remaining clusters for instruction tuning. ${ }^{3}$ We show results for one to seven instruction tuning clusters, where clusters are added in decreasing order of number of tasks per cluster.

Figure 6 shows these results. As expected, we observe that average performance across the three held-out clusters improves as we add additional clusters and tasks to instruction tuning (with the exception of the sentiment analysis cluster), confirming the benefits of our proposed instruction tuning approach on zero-shot performance on novel tasks. It is further interesting to see that, for the seven clusters we test, the performance does not appear to saturate, implying that performance may further improve with even more clusters added to instruction tuning. Of note, this ablation does not allow us to draw conclusions about which instruction tuning cluster contributes the most to each evaluation cluster, although we see minimal added value from the sentiment analysis cluster.

![Figure](./weiFinetunedLanguageModels2022_images/aa25be03-d0f6-48ca-994b-33ce39180820-06_522_853_1734_634.jpg)

**Figure 6: Adding additional task clusters to instruction tuning improves zero-shot performance on held-out task clusters. The evaluation tasks are the following. Commonsense: CoPA, HellaSwag, PiQA, and StoryCloze. NLI: ANLI R1-R3, QNLI, RTE, SNLI, and WNLI. Closed-book QA: ARC easy, ARC challenge, Natural Questions, and TriviaQA.**

\footnotetext{
${ }^{3}$ We do not use the paraphrase or reading comprehension with commonsense clusters for instruction tuning in this ablation because they are too similar to NLI and commmonsense reasoning, respectively.
}

## 4.2 Scaling laws

As Brown et al. (2020) shows that zero and few-shot capabilities of language models substantially improve for larger models, we next explore how the benefits of instruction tuning are affected by model scale. Using the same cluster split as in the previous ablation study, we evaluate the effect of instruction tuning on models of size 422M, 2B, 8B, 68B, and 137B parameters.

Figure 7 shows these results. We see that for the two models on the order of 100B parameters, instruction tuning substantially improves performance on held-out tasks, as is expected given the prior results in our paper. The behavior on held-out tasks for the 8B and smaller models, however, is thought- with the remaining capacity.

## 4.3 Role of instructions

In a final ablation study, we explore the role of instructions during finetuning, as one possibility is that performance gains come entirely from multi-task finetuning and the model could perform just as well without instructions. We hence consider two finetuning setups without instructions. In a no template setup, only inputs and outputs were given to the model (e.g., for translation the input would be "The dog runs." and the output would be "Le chien court."). In a dataset name setup, each input is prepended with the name of the task and dataset (e.g., for translation to French, the input would be "[Translation: WMT'14 to French] The dog runs.").

We compare these two ablations to FLAN's finetuning procedure, which used natural instructions (e.g., "Please translate this sentence to French: 'The dog

![Figure](./weiFinetunedLanguageModels2022_images/aa25be03-d0f6-48ca-994b-33ce39180820-07_481_726_288_1025.jpg)

**Figure 7: Whereas instruction tuning helps large models generalize to new tasks, for small models it actually hurts generalization to unseen tasks, potentially because all model capacity is used to learn the mixture of instruction tuning tasks.**

provoking-instruction tuning actually hurts performance on held-out tasks. One potential explanation for this result could be that for small-scale models, learning the ~40 tasks used during instruction tuning fills the entire model capacity, causing these models to perform worse on new tasks. Under this potential explanation, for the larger scale models, instruction tuning fills up some model capacity but also teaches these models how to follow instructions, allowing them to generalize to new tasks

Figure 8: Ablation study result using models with instructions removed from finetuning (FT). runs.""). We perform evaluations for four held-out clusters from Figure 5. For the no template setup, we used the FLAN instructions during zero-shot inference (because if we used no template, the model would not know what task to perform). For models finetuned on dataset name only, we report zero-shot performance for FLAN instructions as well as using the dataset name. Figure 8 shows the results-both ablation configurations performed substantially worse than FLAN, indicating that training with instructions is crucial for zero-shot performance on unseen tasks.

## 4.4 Instructions with Few-Shot Exemplars

So far, we have focused on instruction tuning in the zero-shot setting. Here, we study how instruction tuning can be used when few-shot exemplars are available at inference time. The format for the few-shot setting builds on the zero-shot format. For some input $x$ and output $y$, let instruct \((x)\) denote the zero-shot instructions. Then, given $k$ few-shot exemplars \(\left(x_{i}, y_{i}\right)_{i=1}^{k}\) and a new input $x$, the instruction format for the few-shot setting is " \(\operatorname{instruct}\left(x_{1}\right) \oplus y_{1} \oplus \operatorname{instruct}\left(x_{2}\right) \oplus y_{2} \oplus \ldots \oplus\)
instruct \(\left(x_{k}\right) \oplus y_{k} \oplus \operatorname{instruct}(x)\) ", where ⊕ denotes string concatenation with a delimiter token inserted in between. At both training and inference time, exemplars are randomly drawn from the training set, and the number of exemplars is capped at 16 and such that the total sequence length is less than 960 tokens. Our experiment uses the same task splits and evaluation procedure as §3, such that few-shot exemplars for an unseen task are only used at inference time.

As shown in Figure 9, few-shot exemplars improve the performance on all task clusters, compared with zero-shot FLAN. Exemplars are especially effective for tasks with large/complex output spaces, such as struct to text, translation, and closed-book QA, potentially because exemplars help the model better understand the output format. In addition, for all task clusters, standard deviation among templates is lower for few-shot FLAN, indicating reduced sensitivity to prompt engineering.

![Figure](./weiFinetunedLanguageModels2022_images/aa25be03-d0f6-48ca-994b-33ce39180820-08_331_1381_728_370.jpg)

**Figure 9: Adding few-shot exemplars to FLAN is a complementary method for improving the performance of instruction-tuned models. The orange bars indicate standard deviation among templates, averaged at the dataset level for each task cluster.**

## 4.5 Instruction Tuning Facilitates Prompt Tuning

As we've seen that instruction tuning improves the ability of a model to respond to instructions, it follows that, if FLAN is indeed more amenable to performing NLP tasks, then it should also achieve better performance when performing inference using soft prompts, represented by prepended continuous variables optimized via prompt tuning (Li \& Liang, 2021; Lester et al., 2021). As further analysis, we train continuous prompts for each of the SuperGLUE (Wang et al., 2019a) tasks in accordance with the cluster splits from §2.2 such that when prompt-tuning on task $\mathcal{T}$, no tasks in the same cluster as $\mathcal{T}$ were seen during instruction tuning. Our prompt tuning setup follows the procedure of Lester et al. (2021) except that we use a prompt length of 10, weight decay of 1e-4, and did not use dropout on the attention scores; we found in preliminary experiments that these changes improved the performance of LaMDA-PT.

Figure 10 shows the results of these prompt tuning experiments for both using a fully-supervised training set and in a low-resource setting with only 32 training examples. We see that in all scenarios, prompt tuning works better with FLAN than LaMDA-PT. In many cases, especially for the low-resource setting, prompt tuning on FLAN even achieves more than 10\% improvement over prompt tuning on the LaMDA-PT. This result exemplifies in another way how instruction tuning can result in a checkpoint that is more desirable for performing NLP tasks.

