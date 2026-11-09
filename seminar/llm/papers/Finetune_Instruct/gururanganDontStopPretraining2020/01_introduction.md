# 1 Introduction

Today's pretrained language models are trained on massive, heterogeneous corpora (Raffel et al., 2019; Yang et al., 2019). For instance, RoBERTA (Liu et al., 2019) was trained on over 160 GB of uncompressed text, with sources ranging from Englishlanguage encyclopedic and news articles, to literary works and web content. Representations learned by such models achieve strong performance across many tasks with datasets of varying sizes drawn from a variety of sources (e.g., Wang et al., 2018, 2019). This leads us to ask whether a tasks textual domain-a term typically used to denote a distribution over language characterizing a given topic or genre (such as "science" or "mystery novels")-is still relevant. Do the latest large pretrained models work universally or is it still helpful to build

![Figure](./gururanganDontStopPretraining2020_images/afe5ddec-74f8-4c27-a385-ddd9ceebaf9f-01_455_734_799_1075.jpg)

**Figure 1: An illustration of data distributions. Task data is comprised of an observable task distribution, usually non-randomly sampled from a wider distribution (light grey ellipsis) within an even larger target domain, which is not necessarily one of the domains included in the original LM pretraining domain - though overlap is possible. We explore the benefits of continued pretraining on data from the task distribution and the domain distribution.**

separate pretrained models for specific domains?
While some studies have shown the benefit of continued pretraining on domain-specific unlabeled data (e.g., Lee et al., 2019), these studies only consider a single domain at a time and use a language model that is pretrained on a smaller and less diverse corpus than the most recent language models. Moreover, it is not known how the benefit of continued pretraining may vary with factors like the amount of available labeled task data, or the proximity of the target domain to the original pretraining corpus (see Figure 1).

We address this question for one such highperforming model, RoBERTA (Liu et al., 2019) (§2). We consider four domains (biomedical and computer science publications, news, and reviews; §3) and eight classification tasks (two in each domain). For targets that are not already in-domain for ROBERTA, our experiments show that contin-
ued pretraining on the domain (which we refer to as domain-adaptive pretraining or DAPT) consistently improves performance on tasks from the target domain, in both high- and low-resource settings.

Above, we consider domains defined around genres and forums, but it is also possible to induce a domain from a given corpus used for a task, such as the one used in supervised training of a model. This raises the question of whether pretraining on a corpus more directly tied to the task can further improve performance. We study how domainadaptive pretraining compares to task-adaptive pretraining, or TAPT, on a smaller but directly taskrelevant corpus: the unlabeled task dataset (§4), drawn from the task distribution. Task-adaptive pretraining has been shown effective (Howard and Ruder, 2018), but is not typically used with the most recent models. We find that tapt provides a large performance boost for RoBERTA, with or without domain-adaptive pretraining.

Finally, we show that the benefits from taskadaptive pretraining increase when we have additional unlabeled data from the task distribution that has been manually curated by task designers or annotators. Inspired by this success, we propose ways to automatically select additional task-relevant unlabeled text, and show how this improves performance in certain low-resource cases (§5). On all tasks, our results using adaptive pretraining techniques are competitive with the state of the art.

In summary, our contributions include:
\begin{itemize}
\item[-] a thorough analysis of domain- and taskadaptive pretraining across four domains and eight tasks, spanning low- and high-resource settings;
\item[-] an investigation into the transferability of adapted LMs across domains and tasks; and
\item[-] a study highlighting the importance of pretraining on human-curated datasets, and a simple data selection strategy to automatically approach this performance.
\end{itemize}

Our code as well as pretrained models for multiple domains and tasks are publicly available. ${ }^{1}$

