# 4 Task-Adaptive Pretraining

Datasets curated to capture specific tasks of interest tend to cover only a subset of the text available within the broader domain. For example, the ChemProt dataset for extracting relations between chemicals and proteins focuses on abstracts of recently-published, high-impact articles from hand-selected PubMed categories (Krallinger et al., 2017, 2015). We hypothesize that such cases where the task data is a narrowly-defined subset of the broader domain, pretraining on the task dataset itself or data relevant to the task may be helpful.

Task-adaptive pretraining (TAPT) refers to pretraining on the unlabeled training set for a given task; prior work has shown its effectiveness (e.g. Howard and Ruder, 2018). Compared to domainadaptive pretraining (DAPT; §3), the task-adaptive approach strikes a different trade-off: it uses a far smaller pretraining corpus, but one that is much more task-relevant (under the assumption that the training set represents aspects of the task well). This makes TAPT much less expensive to run than DAPT, and as we show in our experiments, the performance of TAPT is often competitive with that of DAPT.

## 4.1 Experiments

Similar to dapt, task-adaptive pretraining consists of a second phase of pretraining RoBERTA, but only on the available task-specific training data. In contrast to dapt, which we train for 12.5K steps, we perform tapt for 100 epochs. We artificially augment each dataset by randomly masking different words (using the masking probability of 0.15) across epochs. As in our DAPT experiments, we pass the final layer [CLS] token representation to a task-specific feedforward layer for classification (see Table 14 in Appendix for more hyperparameter details).

Our results are shown in the TAPT column of Table 5. TAPT consistently improves the ROBERTA baseline for all tasks across domains. Even on the news domain, which was part of ROBERTA pretraining corpus, TAPT improves over RoBERTA, showcasing the advantage of task adaptation. Particularly remarkable are the relative differences between TAPT and DAPT. DAPT is more resource intensive (see Table 9 in §5.3), but tapt manages to match its performance in some of the tasks, such as Scierc. In RCT, HyperPartisan, AGNews, Helpfulness, and IMDB, the results even exceed those of dapt, highlighting the efficacy of this cheaper adaptation technique.

\begin{table}
\begin{tabular}[t]{|l|l|l|l|l|l|}
\hline \multirow[b]{2}{*}{Domain} & \multirow[b]{2}{*}{Task} & \multirow[b]{2}{*}{ROBERTA} & \multicolumn{3}{|c|}{Additional Pretraining Phases} \\
\hline & & & DAPT & TAPT & DAPT + TAPT \\
\hline \multirow{2}{*}{BioMed} & ChemProt & $81.9_{1.0}$ & $84.2_{0.2}$ & $82.6_{0.4}$ & $\mathbf{8 4 . 4}_{0.4}$ \\
\hline & ${ }^{\dagger}$ RCT & $87.2_{0.1}$ & $87.6_{0.1}$ & $87.7_{0.1}$ & $\mathbf{8 7 . 8}_{0.1}$ \\
\hline \multirow{2}{*}{CS} & ACL-ARC & $63.0_{5.8}$ & $75.4_{2.5}$ & $67.4_{1.8}$ & $\mathbf{7 5 . 6}_{3.8}$ \\
\hline & Scierc & $77.3_{1.9}$ & $80.8_{1.5}$ & $79.3_{1.5}$ & $\mathbf{8 1 . 3}_{1.8}$ \\
\hline \multirow{2}{*}{News} & HyperPartisan & $86.6_{0.9}$ & $88.2_{5.9}$ & $\mathbf{9 0 . 4}_{5.2}$ & $90.0_{6.6}$ \\
\hline & ${ }^{\dagger}$ AGNews & $93.9_{0.2}$ & $93.9_{0.2}$ & $94.5_{0.1}$ & $\mathbf{9 4 . 6}_{0.1}$ \\
\hline \multirow{2}{*}{Reviews} & ${ }^{\dagger}$ Helpfulness & $65.1_{3.4}$ & $66.5_{1.4}$ & $68.5_{1.9}$ & $\mathbf{6 8 . 7}_{1.8}$ \\
\hline & ${ }^{\dagger}$ IMDB & $95.0_{0.2}$ & $95.4_{0.1}$ & $95.5_{0.1}$ & $\mathbf{9 5 . 6}_{0.1}$ \\
\hline
\end{tabular}

**Table 5: Results on different phases of adaptive pretraining compared to the baseline ROBERTA (col. 1). Our approaches are dapt (col. 2, §3), tapt (col. 3, §4), and a combination of both (col. 4). Reported results follow the same format as Table 3. State-of-the-art results we can compare to: ChemProt (84.6), RCT (92.9), ACL-ARC (71.0), Scierc (81.8), HyperPartisan (94.8), AGNews (95.5), IMDB (96.2); references in §A.2.**
\end{table}

\begin{table}
\begin{tabular}[t]{|l|l|l|l|l|l|}
\hline BioMed & RCT & ChemProt & CS & ACL-ARC & SciERC \\
\hline TAPT & $87.7_{0.1}$ & $82.6_{0.5}$ & TAPT & $67.4_{1.8}$ & $79.3_{1.5}$ \\
\hline Transfer-TAPT & \(87.1_{0.4}(\downarrow 0.6)\) & \(80.4_{0.6}(\downarrow 2.2)\) & Transfer-TAPT & \(64.1_{2.7}(\downarrow 3.3)\) & \(79.1_{2.5}(\downarrow 0.2)\) \\
\hline News & HyperPartisan & AGNews & Reviews & Helpfulness & IMDB \\
\hline TAPT & 89.99.5 & $94.5_{0.1}$ & TAPT & $68.5_{1.9}$ & $95.7_{0.1}$ \\
\hline Transfer-TAPT & \(82.2_{7.7}(\downarrow 7.7)\) & \(93.9_{0.2}(\downarrow 0.6)\) & Transfer-TAPT & \(65.0_{2.6}(\downarrow 3.5)\) & \(95.0_{0.1}(\downarrow 0.7)\) \\
\hline
\end{tabular}

**Table 6: Though TAPT is effective (Table 5), it is harmful when applied across tasks. These findings illustrate differences in task distributions within a domain.**
\end{table}

Combined Dapt and tapt We investigate the effect of using both adaptation techniques together. We begin with RoBERTA and apply DAPT then TAPT under this setting. The three phases of pretraining add up to make this the most computationally expensive of all our settings (see Table 9). As expected, combined domain- and task-adaptive pretraining achieves the best performance on all tasks (Table 5). ${ }^{2}$

Overall, our results show that DAPT followed by TAPT achieves the best of both worlds of domain and task awareness, yielding the best performance. While we speculate that TAPT followed by DAPT would be susceptible to catastrophic forgetting of the task-relevant corpus (Yogatama et al., 2019), alternate methods of combining the procedures may result in better downstream performance. Future work may explore pretraining with a more sophisticated curriculum of domain and task distributions.

\footnotetext{
${ }^{2}$ Results on HyperPartisan match those of tapt, within a standard deviation arising from the five seeds.
}

Cross-Task Transfer We complete the comparison between DAPT and TAPT by exploring whether adapting to one task transfers to other tasks in the same domain. For instance, we further pretrain the LM using the RCT unlabeled data, fine-tune it with the ChemProt labeled data, and observe the effect. We refer to this setting as Transfer-TAPT. Our results for tasks in all four domains are shown in Table 6. We see that tapt optimizes for single task performance, to the detriment of cross-task transfer. These results demonstrate that data distributions of tasks within a given domain might differ. Further, this could also explain why adapting only to a broad domain is not sufficient, and why TAPT after DAPT is effective.

