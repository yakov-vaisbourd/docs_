# 3 Domain-Adaptive Pretraining

Our approach to domain-adaptive pretraining (DAPT) is straightforward-we continue pretraining RoBERTA on a large corpus of unlabeled domain-specific text. The four domains we focus on are biomedical (BioMed) papers, computer science (CS) papers, newstext from RealNews, and Amazon reviews. We choose these domains because they have been popular in previous work, and datasets for text classification are available in each. Table 1 lists the specifics of the unlabeled datasets in all four domains, as well as ROBERTA's training corpus. ${ }^{1}$

## 3.1 Analyzing Domain Similarity

Before performing DAPT, we attempt to quantify the similarity of the target domain to ROBERTA's pretraining domain. We consider domain vocabularies containing the top 10K most frequent unigrams (excluding stopwords) in comparably sized

\footnotetext{
${ }^{1}$ For BioMed and CS, we used an internal version of S2ORC that contains papers that cannot be released due to copyright restrictions.
}

\begin{table}
\begin{tabular}[t]{|l|l|l|l|l|l|}
\hline Domain & Pretraining Corpus & \# Tokens & Size & $\mathcal{L}_{\text {Rob. }}$ & $\mathcal{L}_{\text {DAPT }}$ \\
\hline BioMed & 2.68M full-text papers from S2ORC (Lo et al., 2020) & 7.55B & 47GB & 1.32 & 0.99 \\
\hline CS & 2.22M full-text papers from S2ORC (Lo et al., 2020) & 8.10B & 48GB & 1.63 & 1.34 \\
\hline News & 11.90M articles from RealNews (Zellers et al., 2019) & 6.66B & 39GB & 1.08 & 1.16 \\
\hline Reviews & 24.75M Amazon reviews (He and McAuley, 2016) & 2.11B & 11GB & 2.10 & 1.93 \\
\hline ROBERTA (baseline) & see Appendix §A. 1 & N/A & 160 GB & ${ }^{\ddagger} 1.19$ & - \\
\hline
\end{tabular}

**Table 1: List of the domain-specific unlabeled datasets. In columns 5 and 6, we report RoBERTA's masked LM loss on 50K randomly sampled held-out documents from each domain before \(\left(\mathcal{L**_{\text {ROB. }}\right)\) and after \(\left(\mathcal{L}_{\text {DAPT }}\right)\) DAPT (lower implies a better fit on the sample). ‡ indicates that the masked LM loss is estimated on data sampled from sources similar to ROBERTA's pretraining corpus.}
\end{table}

\begin{table}
\begin{tabular}[t]{|l|l|l|l|l|l|}
\hline PT & 100.0 & 54.1 & 34.5 & 27.3 & 19.2 \\
\hline News & 54.1 & 100.0 & 40.0 & 24.9 & 17.3 \\
\hline Reviews & 34.5 & 40.0 & 100.0 & 18.3 & 12.7 \\
\hline BioMed & 27.3 & 24.9 & 18.3 & 100.0 & 21.4 \\
\hline CS & 19.2 & 17.3 & 12.7 & 21.4 & 100.0 \\
\hline & PT & News & Reviews & BioMed & CS \\
\hline
\end{tabular}

**Figure 2: Vocabulary overlap (\%) between domains. PT denotes a sample from sources similar to ROBERTA's pretraining corpus. Vocabularies for each domain are created by considering the top 10K most frequent words (excluding stopwords) in documents sampled from each domain.**
\end{table}
random samples of held-out documents in each domain's corpus. We use 50K held-out documents for each domain other than Reviews, and 150K held-out documents in Reviews, since they are much shorter. We also sample 50K documents from sources similar to ROBERTA's pretraining corpus (i.e., BookCorpus, Stories, Wikipedia, and RealNews) to construct the pretraining domain vocabulary, since the original pretraining corpus is not released. Figure 2 shows the vocabulary overlap across these samples. We observe that ROBERTA's pretraining domain has strong vocabulary overlap with News and Reviews, while CS and BioMed are far more dissimilar to the other domains. This simple analysis suggests the degree of benefit to be expected by adaptation of ROBERTA to different domains-the more dissimilar the domain, the higher the potential for dapt.

## 3.2 Experiments

Our LM adaptation follows the settings prescribed for training RoBERTA. We train RoBERTA on each domain for 12.5K steps, which amounts to single pass on each domain dataset, on a v3-8 TPU; see other details in Appendix B. This second phase of pretraining results in four domain-adapted LMs, one for each domain. We present the masked LM loss of RoBERTA on each domain before and after dapt in Table 1. We observe that masked LM loss decreases in all domains except News after dapt, where we observe a marginal increase. We discuss cross-domain masked LM loss in Appendix §E.

Under each domain, we consider two text classification tasks, as shown in Table 2. Our tasks represent both high- and low-resource ( $\leq 5$ K labeled training examples, and no additional unlabeled data) settings. For HyperPartisan, we use the data splits from Beltagy et al. (2020). For RCT, we represent all sentences in one long sequence for simultaneous prediction.

Baseline As our baseline, we use an off-the-shelf ROBERTA-base model and perform supervised fine-tuning of its parameters for each classification task. On average, ROBERTA is not drastically behind the state of the art (details in Appendix §A.2), and serves as a good baseline since it provides a single LM to adapt to different domains.

Classification Architecture Following standard practice (Devlin et al., 2019) we pass the final layer [CLS] token representation to a task-specific feedforward layer for prediction (see Table 14 in Appendix for more hyperparameter details).

Results Test results are shown under the dapt column of Table 3 (see Appendix §C for validation results). We observe that DAPT improves over RoBERTA in all domains. For BioMed, CS, and Reviews, we see consistent improve-

\begin{table}
\begin{tabular}[t]{|l|l|l|l|l|l|l|l|}
\hline Domain & Task & Label Type & Train (Lab.) & Train (Unl.) & Dev. & Test & Classes \\
\hline \multirow{2}{*}{BioMed} & ChemProt & relation classification & 4169 & - & 2427 & 3469 & 13 \\
\hline & ${ }^{\dagger}$ RCT & abstract sent. roles & 18040 & - & 30212 & 30135 & 5 \\
\hline \multirow{2}{*}{CS} & ACL-ARC & citation intent & 1688 & - & 114 & 139 & 6 \\
\hline & SciERC & relation classification & 3219 & - & 455 & 974 & 7 \\
\hline \multirow{2}{*}{News} & HyperPartisan & partisanship & 515 & 5000 & 65 & 65 & 2 \\
\hline & ${ }^{\dagger}$ AGNews & topic & 115000 & - & 5000 & 7600 & 4 \\
\hline \multirow{2}{*}{Reviews} & ${ }^{\dagger}$ Helpfulness & review helpfulness & 115251 & - & 5000 & 25000 & 2 \\
\hline & ${ }^{\dagger}$ IMDB & review sentiment & 20000 & 50000 & 5000 & 25000 & 2 \\
\hline
\end{tabular}

**Table 2: Specifications of the various target task datasets. † indicates high-resource settings. Sources: ChemProt (Kringelum et al., 2016), RCT (Dernoncourt and Lee, 2017), ACL-ARC (Jurgens et al., 2018), SciERC (Luan et al., 2018), HyperPartisan (Kiesel et al., 2019), AGNews (Zhang et al., 2015), Helpfulness (McAuley et al., 2015), IMDB (Maas et al., 2011).**
\end{table}

\begin{table}
\begin{tabular}[t]{|l|l|l|l|l|}
\hline Dom. & Task & RoBA. & DAPT & ¬DAPT \\
\hline \multirow{2}{*}{BM} & ChemProt & $81.9_{1.0}$ & $\mathbf{8 4 . 2}_{0.2}$ & $79.4_{1.3}$ \\
\hline & ${ }^{\dagger}$ RCT & $87.2_{0.1}$ & $\mathbf{8 7 . 6}_{0.1}$ & $86.9_{0.1}$ \\
\hline \multirow{2}{*}{CS} & ACL-ARC & $63.0_{5.8}$ & $\mathbf{7 5 . 4}_{2.5}$ & $66.4_{4.1}$ \\
\hline & SciERC & $77.3_{1.9}$ & $\mathbf{8 0 . 8}_{1.5}$ & $79.2_{0.9}$ \\
\hline \multirow{2}{*}{News} & HyP. & $86.6_{0.9}$ & $\mathbf{8 8 . 2}_{5.9}$ & $76.4_{4.9}$ \\
\hline & †AGNews & $\mathbf{9 3 . 9}_{0.2}$ & $\mathbf{9 3 . 9}_{0.2}$ & $93.5_{0.2}$ \\
\hline \multirow{2}{*}{Rev.} & ${ }^{\dagger}$ Helpful. & $65.1_{3.4}$ & $\mathbf{6 6 . 5}_{1.4}$ & $65.1_{2.8}$ \\
\hline & ${ }^{\dagger}$ IMDB & $95.0_{0.2}$ & $\mathbf{9 5 . 4}_{0.2}$ & $94.1_{0.4}$ \\
\hline
\end{tabular}

**Table 3: Comparison of RoBERTA (RoBa.) and DAPT to adaptation to an irrelevant domain ( $\neg$ DAPT). Reported results are test macro- $F_{1**$, except for ChemProt and RCT, for which we report micro- $F_{1}$, following Beltagy et al. (2019). We report averages across five random seeds, with standard deviations as subscripts. † indicates high-resource settings. Best task performance is boldfaced. See §3.3 for our choice of irrelevant domains.}
\end{table}
ments over RoBERTA, demonstrating the benefit of DAPT when the target domain is more distant from RoBERTA's source domain. The pattern is consistent across high- and low- resource settings. Although DAPT does not increase performance on AGNews, the benefit we observe in HyperPar-TISAN suggests that DAPT may be useful even for tasks that align more closely with RoBERTA's source domain.

## 3.3 Domain Relevance for DAPT

Additionally, we compare DAPT against a setting where for each task, we adapt the LM to a domain outside the domain of interest. This controls for the case in which the improvements over RoBERTA might be attributed simply to exposure to more data, regardless of the domain. In this setting, for News, we use a CS LM; for Reviews, a BioMed LM; for CS, a News LM; for BioMed, a Reviews LM. We use the vocabulary overlap statistics in Figure 2 to guide these choices.

Our results are shown in Table 3, where the last column (¬DAPT) corresponds to this setting. For each task, DAPT significantly outperforms adapting to an irrelevant domain, suggesting the importance of pretraining on domain-relevant data. Furthermore, we generally observe that ¬DAPT results in worse performance than even ROBERTA on end-tasks. Taken together, these results indicate that in most settings, exposure to more data without considering domain relevance is detrimental to end-task performance. However, there are two tasks (SciERC and ACL-ARC) in which ¬DAPT marginally improves performance over RoBERTA. This may suggest that in some cases, continued pretraining on any additional data is useful, as noted in Baevski et al. (2019).

## 3.4 Domain Overlap

Our analysis of DAPT is based on prior intuitions about how task data is assigned to specific domains. For instance, to perform dapt for Helpfulness, we only adapt to Amazon reviews, but not to any RealNews articles. However, the gradations in Figure 2 suggest that the boundaries between domains are in some sense fuzzy; for example, 40\% of unigrams are shared between Reviews and News. As further indication of this overlap, we also qualitatively identify documents that overlap cross-domain: in Table 4, we showcase reviews and RealNews articles that are similar to these reviews (other examples can be found in Appendix §D). In fact, we find that adapting RoBERTA to

\begin{table}
\begin{tabular}[t]{|l|l|}
\hline IMDB review & RealNews article \\
\hline \begin{tabular}[t]{l}
The Shop Around the Corner is one of the great films from director \\
Ernst Lubitsch . In addition to the talents of James Stewart and Margaret Sullavan , it's filled with a terrific cast of top character actors such as Frank Morgan and Felix Bressart. [...] The makers of You've Got Mail claim their film to be a remake , but that's just nothing but a lot of inflated self praise. Anyway, if you have an affection for romantic comedies of the 1940 's, you'll find The Shop Around the Corner to be nothing short of wonderful. Just as good with repeat viewings.
\end{tabular} & [...] Three great festive films... The Shop Around the Corner (1940) Delightful Comedy by Ernst Lubitsch stars James Stewart and Margaret Sullavan falling in love at Christmas. Remade as Youve Got Mail. [...] \\
\hline Helpfulness review & RealNews article \\
\hline Simply the Best! I've owned countless Droids and iPhones, but this one destroys them all. Samsung really nailed it with this one, extremely fast , very pocketable, gorgeous display , exceptional battery life , good audio quality, perfect GPS \& WiFi performance, transparent status bar, battery percentage, ability to turn off soft key lights, superb camera for a smartphone and more! [...] & Were living in a world with a new Samsung. [...] more on battery life later [...] Exposure is usually spot on and focusing is very fast. [...] The design, display, camera and performance are all best in class, and the phone feels smaller than it looks. [...] \\
\hline
\end{tabular}

**Table 4: Examples that illustrate how some domains might have overlaps with others, leading to unexpected positive transfer. We highlight expressions in the reviews that are also found in the RealNews articles.**
\end{table}

News not as harmful to its performance on ReVIEWS tasks (DAPT on News achieves $65.5_{2.3}$ on Helpfulness and $95.0_{0.1}$ on IMDB).

Although this analysis is by no means comprehensive, it indicates that the factors that give rise to observable domain differences are likely not mutually exclusive. It is possible that pretraining beyond conventional domain boundaries could result in more effective DAPT; we leave this investigation to future work. In general, the provenance of data, including the processes by which corpora are curated, must be kept in mind when designing pretraining procedures and creating new benchmarks that test out-of-domain generalization abilities.

