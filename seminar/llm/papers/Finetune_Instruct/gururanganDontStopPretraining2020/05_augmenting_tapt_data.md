# 5 Augmenting Training Data for Task-Adaptive Pretraining

In §4, we continued pretraining the LM for task adaptation using only the training data for a supervised task. Inspired by the success of TAPT, we next investigate another setting where a larger pool of unlabeled data from the task distribution exists,

\begin{table}
\begin{tabular}[t]{|l|l|l|l|}
\hline Pretraining & BioMed RCT-500 & News HyP. & Reviews IMDB ${ }^{\dagger}$ \\
\hline TAPT & $79.8_{1.4}$ & $90.4_{5.2}$ & $95.5_{0.1}$ \\
\hline DAPT + TAPT & $83.0_{0.3}$ & $90.0_{6.6}$ & $95.6_{0.1}$ \\
\hline Curated-TAPT & $83.4_{0.3}$ & 89.99.5 & $95.7_{0.1}$ \\
\hline DAPT + Curated-TAPT & $\mathbf{8 3 . 8}_{0.5}$ & $\mathbf{9 2 . 1}_{3.6}$ & $\mathbf{9 5 . 8}_{0.1}$ \\
\hline
\end{tabular}

**Table 7: Mean test set macro- $F_{1**$ (for HyP. and IMDB) and micro- $F_{1}$ (for RCT-500), with CuratedTAPT across five random seeds, with standard deviations as subscripts. † indicates high-resource settings.}
\end{table}
typically curated by humans.
We explore two scenarios. First, for three tasks (RCT, HyperPartisan, and IMDB) we use this larger pool of unlabeled data from an available human-curated corpus (§5.1). Next, we explore retrieving related unlabeled data for TAPT, from a large unlabeled in-domain corpus, for tasks where extra human-curated data is unavailable (§5.2).

## 5.1 Human Curated-tapt

Dataset creation often involves collection of a large unlabeled corpus from known sources. This corpus is then downsampled to collect annotations, based on the annotation budget. The larger unlabeled corpus is thus expected to have a similar distribution to the task's training data. Moreover, it is usually available. We explore the role of such corpora in task-adaptive pretraining.

Data We simulate a low-resource setting RCT500, by downsampling the training data of the RCT dataset to 500 examples (out of 180 K available), and treat the rest of the training data as unlabeled. The HyperPartisan shared task (Kiesel et al., 2019) has two tracks: low- and high-resource. We use 5K documents from the high-resource setting as Curated-TAPT unlabeled data and the original lowresource training documents for task fine-tuning. For IMDB, we use the extra unlabeled data manually curated by task annotators, drawn from the same distribution as the labeled data (Maas et al., 2011).

Results We compare Curated-TAPT to TAPT and dapt + tapt in Table 7. Curated-tapt further improves our prior results from §4 across all three datasets. Applying Curated-TAPT after adapting to the domain results in the largest boost in performance on all tasks; in HyperPartisan, dapt + Curated-tapt is within standard deviation of Curated-tapt. Moreover, curated-tapt achieves

![Figure](./gururanganDontStopPretraining2020_images/afe5ddec-74f8-4c27-a385-ddd9ceebaf9f-07_567_754_219_1069.jpg)

**Figure 3: An illustration of automated data selection (§5.2). We map unlabeled ChemProt and 1M BioMed sentences to a shared vector space using the VAMPIRE model trained on these sentences. Then, for each ChemProt sentence, we identify $k$ nearest neighbors, from the BioMed domain.**

\begin{table}
\begin{tabular}[t]{|l|l|l|l|}
\hline \multirow[b]{2}{*}{Pretraining} & \multicolumn{2}{|c|}{BioMed} & CS \\
\hline & ChemProt & RCT-500 & ACL-ARC \\
\hline ROBERTA & $81.9_{1.0}$ & $79.3_{0.6}$ & $63.0_{5.8}$ \\
\hline TAPT & $82.6_{0.4}$ & $79.8_{1.4}$ & $67.4_{1.8}$ \\
\hline RAND-TAPT & $81.9_{0.6}$ & $80.6_{0.4}$ & $69.7_{3.4}$ \\
\hline 50nn-tapt & $83.3_{0.7}$ & $80.8_{0.6}$ & $70.7_{2.8}$ \\
\hline 150NN-TAPT & $83.2_{0.6}$ & $81.2_{0.8}$ & $73.3_{2.7}$ \\
\hline 500nn-tapt & $83.3_{0.7}$ & $81.7_{0.4}$ & $\mathbf{7 5 . 5}_{1.9}$ \\
\hline DAPT & $\mathbf{8 4 . 2}_{0.2}$ & $\mathbf{8 2 . 5}_{0.5}$ & $75.4_{2.5}$ \\
\hline
\end{tabular}

**Table 8: Mean test set micro- $F_{1**$ (for ChemProt and RCT) and macro- $F_{1}$ (for ACL-ARC), across five random seeds, with standard deviations as subscripts, comparing RAND-TAPT (with 50 candidates) and $k$ NNTAPT selection. Neighbors of the task data are selected from the domain data.}
\end{table}
95\% of the performance of DAPT + TAPT with the fully labeled RCT corpus (Table 5) with only 0.3\% of the labeled data. These results suggest that curating large amounts of data from the task distribution is extremely beneficial to end-task performance. We recommend that task designers release a large pool of unlabeled task data for their tasks to aid model adaptation through pretraining.

## 5.2 Automated Data Selection for TAPT

Consider a low-resource scenario without access to large amounts of unlabeled data to adequately benefit from TAPT, as well as absence of computational resources necessary for DAPT (see Table 9 for details of computational requirements for different pretraining phases). We propose simple unsuper-
vised methods to retrieve unlabeled text that aligns with the task distribution, from a large in-domain corpus. Our approach finds task-relevant data from the domain by embedding text from both the task and domain in a shared space, then selects candidates from the domain based on queries using the task data. Importantly, the embedding method must be lightweight enough to embed possibly millions of sentences in a reasonable time.

Given these constraints, we employ VAMPIRE (Gururangan et al., 2019; Figure 3), a lightweight bag-of-words language model. We pretrain VAM-PIRE on a large deduplicated ${ }^{3}$ sample of the domain (1M sentences) to obtain embeddings of the text from both the task and domain sample. We then select $k$ candidates of each task sentence from the domain sample, in embeddings space. Candidates are selected (i) via nearest neighbors selection ( $k$ NN-TAPT) ${ }^{4}$, or (ii) randomly (RAND-TAPT). We continue pretraining RoBERTA on this augmented corpus with both the task data (as in TAPT) as well as the selected candidate pool.

Results Results in Table 8 show that $k$ NN-tapt outperforms TAPT for all cases. RAND-TAPT is generally worse than $k$ NN-TAPT, but within a standard deviation arising from 5 seeds for RCT and ACL-ARC. As we increase $k, k$ NN-TAPT performance steadily increases, and approaches that of dapt. Appendix F shows examples of nearest neighbors of task data. Future work might consider a closer study of $k$ NN-TAPT, more sophisticated data selection methods, and the tradeoff between the diversity and task relevance of selected examples.

## 5.3 Computational Requirements

The computational requirements for all our adaptation techniques on RCT-500 in the BioMed domain in Table 9. TAPT is nearly 60 times faster to train than DAPT on a single v3-8 TPU and storage requirements for DAPT on this task are 5.8M times that of tapt. Our best setting of DAPT + TAPT amounts to three phases of pretraining, and at first glance appears to be very expensive. However, once the LM has been adapted to a broad domain, it can be reused for multiple tasks within that domain, with only a single additional TAPT phase per task. While Curated-tapt tends to achieve the best cost-

\footnotetext{
${ }^{3}$ We deduplicated this set to limit computation, since different sentences can share neighbors.
${ }^{4}$ We use a flat search index with cosine similarity between embeddings with the FAISS (Johnson et al., 2019) library.
}

\begin{table}
\begin{tabular}[t]{|l|l|l|l|l|}
\hline Pretraining & Steps & Docs. & Storage & $F_{1}$ \\
\hline RoBERTA & - & - & - & $79.3_{0.6}$ \\
\hline TAPT & 0.2K & 500 & 80KB & $79.8_{1.4}$ \\
\hline 50nn-tapt & 1.1K & 24 K & 3MB & $80.8_{0.6}$ \\
\hline 150nn-tapt & 3.2K & 66K & 8MB & $81.2_{0.8}$ \\
\hline 500nn-tapt & 9.0K & 185K & 24MB & $81.7_{0.4}$ \\
\hline Curated-tapt & 8.8K & 180K & 27MB & $\mathbf{8 3 . 4}_{0.3}$ \\
\hline DAPT & 12.5 K & 25M & 47GB & $82.5_{0.5}$ \\
\hline DAPT + TAPT & 12.6K & 25M & 47GB & $83.0_{0.3}$ \\
\hline
\end{tabular}

**Table 9: Computational requirements for adapting to the RCT-500 task, comparing dapt (§3) and the various tapt modifications described in §4 and §5.**
\end{table}
benefit ratio in this comparison, one must also take into account the cost of curating large in-domain data. Automatic methods such as $k$ NN-TAPT are much cheaper than DAPT.

