# B Further Ablation Studies and Analysis

# B. 1 Datasets per Task Cluster \& Templates per Dataset

Our primary hypothesis is that instruction tuning on a diverse set of tasks improves performance on unseen tasks. §4.1 showed that adding more task clusters improves performance; here, we further explore whether adding additional datasets improves performance when the number of task clusters is held constant. We use the same split as in §4.1, where the NLI, commonsense reasoning, and closed-book QA clusters are held-out, and seven other task clusters remain for instruction tuning. For these seven task clusters, we instruction tune models using just one dataset per task cluster and using four datasets per task cluster (for task clusters that did not have four tasks, we just used all available tasks). In addition, we simultaneously explore the role of the number of instruction templates per dataset; as mentioned in §2.1, for each dataset we manually composed ten instructional templates for instruction tuning. Here, we instruction tune models using 1, 4, and 10 templates per dataset.

Figure 11 shows these results. Using more datasets per cluster improved performance by almost 10\% on average across the three held-out clusters. Using more templates per dataset, however, had a comparatively negligible effect on performance when there was one task per cluster, which disappeared when there were four tasks per cluster. The small effect of templates is striking given our original motivation that composing ten templates per task would mitigate overfitting to any particular template. This results serves to underscore, however, the unpredictability of finetuning large language models, as one hypothesis is that models at such scale do not easily overfit to a finetuning single task.

![Figure](./weiFinetunedLanguageModels2022_images/aa25be03-d0f6-48ca-994b-33ce39180820-24_477_741_1147_681.jpg)

**Figure 11: Effect of datasets per task cluster and templates per dataset on performance on three held-out clusters: NLI, commonsense reasoning, and closed-book QA. Adding more datasets per task cluster substantially improves performance. Using more templates per dataset, however, only had a very small effect on performance, which disappeared when there were sufficient dataset per task cluster.**

# B. 2 Role of instructions during finetuning

The per-cluster results for the ablation study from §4.3 are shown in Table 3

# B. 3 Further Analysis: Instruction Tuning Facilitates Prompt Tuning

The per-dataset results for the analysis in §4.5 are given in Table 4. As the above tasks are all classification, further work in this direction might include tasks such as summarization or question answering, or try to finetune the model using the supervised datasets.

