# 4 Learning via Addition

We now turn our attention to adding task vectors, either to build multi-task models that are proficient on multiple tasks simultaneously, or to improve single-task performance. This operation allows us to reuse and transfer knowledge either from in-house models, or from the multitude of publicly available fine-tuned models, without additional training or access to training data. We explore addition on various image classification and natural language processing tasks.

## 4.1 Image classification

We start with the same eight models used in Section 3, fine-tuned on a diverse set of image classification tasks (Cars, DTD, EuroSAT, GTSRB, MNIST, RESISC45, SUN397 and SVHN). In Figure 2, we

![Figure](./ilharcoEditingModelsTask2023_images/a9c92ee0-c150-48d6-ad01-59723be068c6-05_484_1224_290_446.jpg)

**Figure 2: Adding pairs of task vectors from image classification tasks. Adding task vectors from two tasks improves accuracy on both, resulting in a single model that is competitive with using two specialized fine-tuned models.**

![Figure](./ilharcoEditingModelsTask2023_images/a9c92ee0-c150-48d6-ad01-59723be068c6-05_459_719_1007_380.jpg)

**Figure 3: Adding task vectors builds multi-task models for image classification tasks. Accuracy is averaged over all downstream tasks. When more task vectors are available, better multi-task vectors can be built. Each point represents an experiment with a subset of the eight tasks we study, and the solid line connects the average performance for each subset size. Recall that the average normalized accuracy of using multiple fine-tuned models is always one. Additional details and experiments are in Appendix D.**

show the accuracy obtained by adding all pairs of task vectors from these tasks. To account for the difference in difficulty of the tasks, we normalize accuracy on each task by the accuracy of the model fine-tuned on that task. After normalizing, the performance of fine-tuned models on their respective tasks is one, and so the average performance of using multiple specialized models is also one. As shown in Figure 2, adding pairs of task vectors leads to a single model that outperforms the zero-shot model by a large margin, and is competitive with using two specialized models ( $98.9 \%$ normalized accuracy on average).

Beyond pairs of tasks, we explore adding task vectors for all possible subsets of the tasks ( $2^{8}$ in total). In Figure 3, we show how the normalized accuracy of the resulting models, averaged over all the eight tasks. As the number of available task vectors increases, better multi-task models can be produced. When all task vectors are available, the best model produced by adding task vectors reaches an average performance of 91.2\%, despite compressing several models into one. Additional experiments and details are presented in Appendix D.

## 4.2 Natural language processing

In addition to building multi-task models, we explore whether adding task vectors is a useful way of improving performance on a single target task. Towards this goal, we first fine-tune T5-base models on four tasks from the GLUE benchmark [93], as in Wortsman et al. [99]. Then, we search for compatible checkpoints on Hugging Face Hub, finding 427 candidates in total. We try adding each of the corresponding task vectors to our fine-tuned models, choosing the best checkpoint and scaling coefficient based on held-out validation data. As shown in Table 3, adding task vectors can improve performance on target tasks, compared to fine-tuning. Additional details and experiments-including building multi-task models from public checkpoints from Hugging Face Hub-are presented in Appendix D.

\begin{table}

**Table 3: Improving performance on target tasks with external task vectors. For four text classification tasks from the GLUE benchmark, adding task vectors downloaded from the Hugging Face Hub can improve accuracy of fine-tuned T5 models. Appendix D. 6 shows additional details.**
\begin{tabular}[t]{|l|l|l|l|l|l|}
\hline Method & MRPC & RTE & CoLA & SST-2 & Average \\
\hline Zero-shot & 74.8 & 52.7 & 8.29 & 92.7 & 57.1 \\
\hline Fine-tuned & 88.5 & 77.3 & 52.3 & 94.5 & 78.1 \\
\hline Fine-tuned + task vectors & 89.3 (+0.8) & 77.5 (+0.2) & 53.0 (+0.7) & 94.7 (+0.2) & 78.6 (+0.5) \\
\hline
\end{tabular}
\end{table}

\begin{table}

**Table 4: Improving domain generalization with task analogies. Using an auxiliary task for which labeled data is available and unlabeled data from both the auxiliary and the target datasets, task analogies improve the accuracy for multiple T5 models and two sentiment analysis target tasks [102; 65], without using any labeled data from the target tasks.**
\begin{tabular}[t]{|l|l|l|l|l|l|l|}
\hline \multirow[b]{2}{*}{Method} & \multicolumn{3}{|c|}{target = Yelp} & \multicolumn{3}{|c|}{target = Amazon} \\
\hline & T5-small & T5-base & T5-large & T5-small & T5-base & T5-large \\
\hline Fine-tuned on auxiliary & 88.6 & 92.3 & 95.0 & 87.9 & 90.8 & 94.8 \\
\hline Task analogies & 89.9 & 93.0 & 95.1 & 89.0 & 92.7 & 95.2 \\
\hline Fine-tuned on target & 91.1 & 93.4 & 95.5 & 90.2 & 93.2 & 95.5 \\
\hline
\end{tabular}
\end{table}

