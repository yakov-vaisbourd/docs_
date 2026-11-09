# 3 Experimental Setup

We describe the experimental setup used in our analysis (Section 4 and 5).

Models. We experiment with 12 models in total. We include 6 language models (Table 1), all of which are decoder-only, dense LMs. We use each LM with two inference methods, direct and channel, following Min et al. (2021a). The sizes of LMs vary from 774M to 175B. We include the

![Figure](./minRethinkingRoleDemonstrations2022_images/7452035e-c407-4143-88d7-35adf7d81a93-03_580_1493_228_322.jpg)

**Figure 3: Results when using no-demonstrations, demonstrations with gold labels, and demonstrations with random labels in classification (top) and multi-choice tasks (bottom). The first eight models are evaluated on 16 classification and 10 multi-choice datasets, and the last four models are evaluated on 3 classification and 3 multichoice datasets. See Figure 11 for numbers comparable across all models. Model performance with random labels is very close to performance with gold labels (more discussion in Section 4.1).**

largest dense LM (GPT-3) and the largest publicly released dense LM (fairseq 13B) at the time of conducting experiments. We also include MetaICL, which is initialized from GPT-2 Large and then meta-trained on a collection of supervised datasets with an in-context learning objective, and ensure that our evaluation datasets do not overlap with those used at meta-training time.

Evaluation Data. We evaluate on 26 datasets, including sentiment analysis, paraphrase detection, natural language inference, hate speech detection, question answering, and sentence completion (full list and references provided in Appendix A). ${ }^{1}$ All datasets are classification and multi-choice tasks.

We use these datasets because they (1) are true low-resource datasets with less than 10K training examples, (2) include well-studied benchmarks from GLUE (Wang et al., 2018) and SuperGLUE (Wang et al., 2019a), and (3) cover diverse domains including science, social media, finance, and more.

Other Details. We use $k=16$ examples as demonstrations by default for all experiments in the paper, unless otherwise specified. Examples are sampled at uniform from the training data. We choose a set of $k$ training examples using 5 different random seeds and run experiments 5 times. For fairseq 13B and GPT-3, due to limited resources, we experiment with a subset of 6

\footnotetext{
${ }^{1}$ For convenience, we use 'labels' to refer to the output for the task, though our datasets include non-classification tasks.
}
datasets ${ }^{2}$ and 3 random seeds. We report Macro$\mathrm{F} 1^{3}$ for classification tasks and Accuracy for multichoice tasks. We compute per-dataset average over seeds, and then report macro-average over datasets. We use the minimal templates in forming an input sequence from an example. We refer to Appendix B for more details. All experiments are reproducible from github.com/Alrope123/ rethinking-demonstrations.

