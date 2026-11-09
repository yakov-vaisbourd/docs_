# 2 FLAN: Instruction Tuning Improves Zero-Shot Learning

The motivation of instruction tuning is to improve the ability of language models to respond to NLP instructions. The idea is that by using supervision to teach an LM to perform tasks described via instructions, the LM will learn to follow instructions and do so even for unseen tasks. To evaluate performance on unseen tasks, we group datasets into clusters by task type and hold out each task cluster for evaluation while instruction tuning on all remaining clusters.

## 2.1 Tasks \& Templates

As creating an instruction tuning dataset with many tasks from scratch would be resource-intensive, we transform existing datasets from the research community into an instructional format. We aggregate 62 text datasets that are publicly available on Tensorflow Datasets, including both language understanding and language generation tasks, into a single mixture. Figure 3 shows these datasets- each dataset is categorized into one of twelve task clusters, for which datasets in a given cluster are of the same task type. Descriptions, sizes, and examples of each dataset are shown in Appendix G

![Figure](./weiFinetunedLanguageModels2022_images/aa25be03-d0f6-48ca-994b-33ce39180820-03_432_1373_617_376.jpg)

**Figure 3: Datasets and task clusters used in this paper (NLU tasks in blue; NLG tasks in teal).**

For each dataset, we manually compose ten unique templates that use natural language instructions to describe the task for that dataset. While most of the ten templates describe the original task, to increase diversity, for each dataset we also include up to three templates that "turned the task around," (e.g., for sentiment classification we include templates asking to generate a movie review). We then instruction tune a pretrained language model on the mixture of all datasets, with examples in each dataset formatted via a randomly selected instruction template for that dataset. Figure 4 shows multiple instruction templates for a natural language inference dataset.

![Figure](./weiFinetunedLanguageModels2022_images/aa25be03-d0f6-48ca-994b-33ce39180820-03_425_1285_1439_417.jpg)

**Figure 4: Multiple instruction templates describing a natural language inference task.**

## 2.2 Evaluation Splits

We are interested in how FLAN performs on tasks not seen in instruction tuning, and so it is crucial to define what counts as an unseen task. Whereas some prior work defines unseen tasks by disallowing the same dataset to appear in training, we use a more conservative definition that leverages the task clusters from Figure 3. In this work, we only consider dataset $\mathcal{D}$ unseen at evaluation time if no datasets from any task clusters that $\mathcal{D}$ belongs to were seen during instruction tuning. For instance, if $\mathcal{D}$ is an entailment task, then no entailment datasets appeared in instruction tuning, and we instruction-tuned on all other clusters. ${ }^{1}$ Hence, to evaluate zero-shot FLAN on $c$ task clusters, we instruction tune $c$ models, where each model holds out a different task cluster for evaluation.

\footnotetext{
${ }^{1}$ When evaluating on the read. comp. with commonsense cluster, both read. comp. and commonsense reasoning were dropped from instruction tuning. Conversely, the read. comp. with commonsense cluster was not used for instruction tuning when evaluating on read. comp. or commonsense reasoning. We also drop the paraphrase cluster from instruction tuning when evaluating on NLI tasks and vice-versa.
}

## 2.3 Classification with Options

The output space for a given task is either one of several classes (classification) or free text (generation). As FLAN is an instruction-tuned version of a decoder-only language model, it naturally responds in free text, and so no further modifications are needed for generation tasks.

For classification tasks, prior work (Brown et al., 2020) used a rank classification approach where, for example, only two outputs ("yes" and "no") are considered and the higher probability one is taken as the model's prediction. Though this procedure is logically sound, it is imperfect in that the probability mass for answers may have an undesired distribution among ways of saying each answer (e.g., a large number of alternative ways of saying "yes" may lower the probability mass assigned to "yes"). Therefore, we include an options suffix, in which we append the token OPTIONS to the end of a classification task along with a list of the output classes for that task. This makes the model aware of which choices are desired when responding to classification tasks. Example use of options is shown in the NLI and commonsense examples in Figure 1

## 2.4 Training Details

Model architecture and pretraining. In our experiments, we use LaMDA-PT, a dense left-to-right, decoder-only transformer language model of 137B parameters (Thoppilan et al., 2022). This model is pretrained on a collection of web documents (including those with computer code), dialog data, and Wikipedia, tokenized into 2.49T BPE tokens with a 32 k vocabulary using the SentencePiece library (Kudo \& Richardson, 2018). Around 10\% of the pretraining data was non-English. Note that LaMDA-PT only has language model pretraining (c.f. LaMDA, which was finetuned for dialog).

Instruction tuning procedure. FLAN is the instruction-tuned version of LaMDA-PT. Our instruction tuning pipeline mixes all datasets and randomly samples from each dataset. To balance the different sizes of datasets, we limit the number of training examples per dataset to 30k and follow the examples-proportional mixing scheme (Raffel et al., 2020) with a mixing rate maximum of $3 \mathrm{k}^{2}$ We finetune all models for 30k gradient steps with a batch size of 8,192 tokens using the Adafactor Optimizer (Shazeer \& Stern, 2018) with a learning rate of 3e-5. The input and target sequence lengths used in finetuning are 1024 and 256, respectively. We use packing (Raffel et al., 2020) to combine multiple training examples into a single sequence, separating inputs from targets using a special EOS token. This instruction tuning takes around 60 hours on a TPUv3 with 128 cores. For all evaluations, we report results on the final checkpoint trained for 30k steps.

