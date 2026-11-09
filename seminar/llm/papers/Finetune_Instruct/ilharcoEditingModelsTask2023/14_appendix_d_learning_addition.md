# D Learning via addition

In all experiments, we add task vectors together and use a single scaling coefficient for the sum of the vectors, $\lambda \in\{0,0.05,0.1, \cdots, 1.0\}$. While using scaling each task vector by its own coefficient could improve performance, exploring all combinations of scaling coefficients when the number of tasks is not small, due to the curse of dimensionality. While we focus on a single scaling coefficient for simplicity, more sophisticated strategies could be explored in future work, such as using black box optimization to search the space of scaling coefficients.

![Figure](./ilharcoEditingModelsTask2023_images/a9c92ee0-c150-48d6-ad01-59723be068c6-24_507_1364_1951_380.jpg)

**Figure 13: Toxicity results using RealToxicityPrompts [29], for various GPT-2 models.**

![Figure](./ilharcoEditingModelsTask2023_images/a9c92ee0-c150-48d6-ad01-59723be068c6-25_468_674_292_724.jpg)

**Figure 14: The impact of random seeds when fine-tuning. Using different random seeds when fine-tuning on image classification tasks has little impact on the accuracy of edited models.**

Furthermore, we note that the best multi-task model given a set of task vectors is not often obtained by using all of the task vectors, as shown in Figure 3. Since adding task vectors is computationally efficient and evaluations are usually substantially less expensive than training, practitioners could try out many subsets of task vectors and choose the ones that maximizes performance on the tasks of interest. Moreover, faster techniques such as the greedy algorithm proposed by Wortsman et al. [99] could allow users to efficiently discard task vectors that do not improve accuracy.

# D. 1 The impact of random seeds

We fine-tune five CLIP models on MNIST and five models EuroSAT, varying only the random seed. We then edit models by adding all possible combinations of the corresponding task vectors (25 in total). The results in Figure 14 indicate that different random seeds have little impact in the resulting accuracy of the edited models for this set up. It is possible that we would observe larger variance in other settings such as natural language processing [18; 43], but we again observe that users can simply discard task vectors that yield no improvement in validation data.

# D. 2 Multi-task training

In addition to using multiple-specialized models, we compare against a single multi-task model obtained via jointly fine-tuning on the eight image classification tasks we study. We fine-tune with the same hyper-parameters described in Appendix B.1, also freezing the classification heads.

Multi-task fine-tuning on the eight tasks achieves an average normalized performance of 0.994, compared to the best result obtained with task vectors, 0.912 (recall that 1.0 is obtained with multiple specialized models). Despite the headroom for improvement, multi-task training is less modular than using task vectors, requiring a new fine-tuning round every time a new task is added. In contrast, task vectors can be combined without any additional training and without the need to store or transfer the data used to create them, and can draw from the large pool of existing fine-tuned models such as the ones available on model hubs.

# D. 3 Scaling coefficients

In Figure 15 (left), we show the optimal scaling coefficients for the experiments where task vectors are added together. Recall that a single scaling coefficient is used for each experiment, regardless of the number of task vectors in the experiment. The variance in the optimal scaling coefficients can be large, highlighting the need for tuning on a case-by-case basis. However, compared to tuning traditional hyper-parameters, tuning the scaling coefficient is less computationally expensive since, unlike most hyper-parameters, the scaling coefficient can be changed without any additional training.

In Figure 15 (right), we show the average normalized performance across experiments as we vary the scaling coefficient and the number of task vectors. Scaling coefficients in the range 0.3 to 0.5

![Figure](./ilharcoEditingModelsTask2023_images/a9c92ee0-c150-48d6-ad01-59723be068c6-26_423_1367_287_380.jpg)

**Figure 15: The effect of scaling coefficients when adding task vectors. Left: Optimal scaling coefficients when adding task vectors. Right: average normalized performance as a function of the scaling coefficient and the number of task vectors.**

![Figure](./ilharcoEditingModelsTask2023_images/a9c92ee0-c150-48d6-ad01-59723be068c6-26_376_747_943_688.jpg)

**Figure 16: Building multi-task models by adding task vectors. Unlike results shown in Figure 3, here performance is averaged only over the tasks used to build the task vectors in each experiment.**

produce close to optimal results in many cases, although we recommend tuning this parameter when possible for best results.

# D. 4 Accuracy on subsets of tasks

Complementing our results in the main paper, we show in Figure 16 the average performance for all subsets task vectors, averaged only over the tasks that originated the task vectors (recall that in Figure 3 we presented the normalized accuracy averaged over all tasks). We find that for smaller subsets, the single model obtained by adding task vectors matches more closely the performance of multiple specialized models, although that gap increases as the size of the subsets grow.

# D. 5 ImageNet experiments

In addition to results presented in Section 4.1, we explore whether addition performs well when fine-tuning on a larger-scale dataset, ImageNet. We fine-tune with the same hyper-parameters as described in Appendix B.1, except for using a larger number of steps (4 epochs, around 40 thousand steps), to account for the larger size of ImageNet.

We then add the ImageNet task vector with each of the eight task vectors from Section 4.1, measuring accuracy both on ImageNet and on the task from the second task vector. For example, for MNIST, we add the MNIST task vector and the ImageNet task vector, and measure accuracy both on MNIST and on ImageNet. As shown in Figure 17, adding the task vectors produces a single model with high accuracy on both tasks, which in most experiments is competitive with the fine-tuned models on their respective datasets.

![Figure](./ilharcoEditingModelsTask2023_images/a9c92ee0-c150-48d6-ad01-59723be068c6-27_649_1366_284_378.jpg)

**Figure 17: Adding pairs of task vectors containing a task vector from ImageNet. For all eight other target tasks from Section 4.1, adding their task vector with an ImageNet produces a model with high accuracy both on that task and on ImageNet.**

# D. 6 Adding pairs of task vectors from NLP tasks

In this section, we present results for building multi-task models using checkpoints that were not finetuned by the authors, and were instead downloaded directly from a hub that hosts model checkpoints publicly (the Hugging Face Hub). ${ }^{9}$

Our motivation is aligned that from with previous work on building multi-task models [79; 45; 103; 68; 96; 84; 67; 94].

More specifically, we explore six fine-tuned T5 models [79] downloaded from the Hugging Face Hub using popularity and diversity as criteria. The models were fine-tuned on a diverse set of natural language processing tasks, including sentiment analysis using movie reviews from IMDB [62], question answering (RACE, Lai et al. [49]; QASC, Khot et al. [46]), summarization (MultiNews, Fabbri et al. [24]), question generation (SQuAD, Rajpurkar et al. [80]); and constrained text generation (CommonGen, Lin et al. [56]). The checkpoints and tasks were chosen based on the availability of models that were fine-tuned from the same initialization (a T5-Base model), were fine-tuned without introducing new parameters, and based on diversity of the tasks and popularity of the checkpoints on the hub. The specific checkpoints we use are:
\begin{itemize}
\item[-] IMDB: mrm8488/t5-base-finetuned-imdb-sentiment
\item[-] RACE: mrm8488/t5-base-finetuned-race
\item[-] QASC: mrm8488/t5-base-finetuned-qasc
\item[-] MultiNews: mrm8488/t5-base-finetuned-summarize-news
\item[-] SQuAD: mrm8488/t5-base-finetuned-question-generation-ap
\item[-] CommonGen: mrm8488/t5-base-finetuned-common_gen
\end{itemize}

For evaluation, we use accuracy for the text classification task (IMDB), exact match for question answering tasks (RACE and QASC) and ROUGE-2 ${ }^{10}$ for text generation tasks (MultiNews, SQuAD question generation, and CommonGen). As in Section 4.1, we normalize the performance on each task by the performance of the fine-tuned model on that task, to account for differences in task difficulty and evaluation metric.

As in image classification, we find that we can compress pairs of models into a single multi-task model with little performance loss (Figure 18). These results are somewhat surprising, since the

\footnotetext{
${ }^{9}$ https://huggingface.co/models
${ }^{10}$ https://huggingface.co/spaces/evaluate-metric/rouge
}

![Figure](./ilharcoEditingModelsTask2023_images/a9c92ee0-c150-48d6-ad01-59723be068c6-28_493_1348_290_385.jpg)

**Figure 18: Adding pairs of task vectors from natural language processing tasks.**

gap between the pre-trained model and fine-tuned models is much larger, and tasks vary widely in terms of input domain, length, and output type. Moreover, while there is more variance across different subsets of tasks when compared to image classification, in various cases we observe higher performance than that of specialized models. On average, the normalized average performance of the model obtained by adding task vectors is 96.7\%.

# D. 7 GLUE EXPERIMENTS

In this section, we describe the experimental setup used for investigations presented in Section 4.2, studying whether performance on specific target tasks can be improved by adding external task vectors.

Our experiments use T5-base models, fine-tuned on four tasks from the GLUE benchmark:
\begin{itemize}
\item[-] Microsoft Research Paraphrase Corpus (MRPC; Dolan \& Brockett [19]) is a paraphrase task containing pairs of sentences labeled as either nearly semantically equivalent or not. The dataset is evaluated using the average of $F_{1}$ and accuracy.
\item[-] Recognizing Textual Entailment (RTE; Wang et al. [93]) is a dataset where models are tasked to predict whether a sentence entails or contradicts another sentence. The data is originally from a series of datasets [14; 5; 31; 6]. Accuracy is used as the evaluation metric.
\item[-] Corpus of Linguistic Acceptability (CoLA; Warstadt et al. [95]) is a dataset with sentences labeled as either grammatical or ungrammatical. Models are evaluated on Matthews correlation (MCC; [64]), which ranges between -1 and 1.
\item[-] Stanford Sentiment Treebank (SST-2; Socher et al. [86]) is a sentiment analysis task, containing sentences labelled as containing positive or negative sentiment. Accuracy is used as the evaluation metric.
\end{itemize}
For all tasks, we split the training set into two subsets, one used for fine-tuning and one used for determining the best external task vector, with the same size as the original validation sets. For fine-tuning, we use a batch size of 32 , learning rate $1 \mathrm{e}-5$ and fine-tune for 5 epochs using AdamW and a linear learning rate schedule. All results are averaged over 3 random seeds. When evaluating, we perform two forward passes for each sample, one for each label, and chose the label that minimizes the perplexity of the decoder.

