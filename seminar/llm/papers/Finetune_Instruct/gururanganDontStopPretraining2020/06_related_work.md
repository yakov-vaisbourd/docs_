# 6 Related Work

Transfer learning for domain adaptation Prior work has shown the benefit of continued pretraining in domain (Alsentzer et al., 2019; Chakrabarty et al., 2019; Lee et al., 2019). ${ }^{5}$ We have contributed further investigation of the effects of a shift between a large, diverse pretraining corpus and target domain on task performance. Other studies (e.g., Huang et al., 2019) have trained language models (LMs) in their domain of interest, from scratch. In contrast, our work explores multiple domains, and is arguably more cost effective, since we continue pretraining an already powerful LM.

Task-adaptive pretraining Continued pretraining of a LM on the unlabeled data of a given task (TAPT) has been show to be beneficial for endtask performance (e.g. in Howard and Ruder, 2018; Phang et al., 2018; Sun et al., 2019). In the presence of domain shift between train and test data distributions of the same task, domain-adaptive pretraining (DAPT) is sometimes used to describe what we term TAPT (Logeswaran et al., 2019; Han and Eisenstein, 2019). Related approaches include language modeling as an auxiliary objective to task classifier fine-tuning (Chronopoulou et al., 2019; Radford et al., 2018) or consider simple syntactic structure of the input while adapting to task-specific

\footnotetext{
${ }^{5}$ In contrast, Peters et al. (2019) find that the JensenShannon divergence on term distributions between BERT's pretraining corpora and each MultiNLI domain (Williams et al., 2018) does not predict its performance, though this might be an isolated finding specific to the MultiNLI dataset.
}

\begin{table}
\begin{tabular}[t]{|l|l|l|l|}
\hline \multirow{2}{*}{} & \multicolumn{3}{|c|}{Training Data} \\
\hline & Domain (Unlabeled) & Task (Unlabeled) & Task (Labeled) \\
\hline ROBERTA & & & ✓ \\
\hline DAPT & ✓ & & ✓ \\
\hline TAPT & & ✓ & ✓ \\
\hline DAPT + TAPT & ✓ & ✓ & ✓ \\
\hline $k \mathrm{NN}$-TAPT & (Subset) & ✓ & ✓ \\
\hline Curated-TAPT & & (Extra) & ✓ \\
\hline
\end{tabular}

**Table 10: Summary of strategies for multi-phase pretraining explored in this paper.**
\end{table}
data (Swayamdipta et al., 2019). We compare DAPT and TAPT as well as their interplay with respect to dataset size for continued pretraining (hence, expense of more rounds of pretraining), relevance to a data sample of a given task, and transferability to other tasks and datasets. See Table 11 in Appendix §A for a summary of multi-phase pretraining strategies from related work.

Data selection for transfer learning Selecting data for transfer learning has been explored in NLP (Moore and Lewis, 2010; Ruder and Plank, 2017; Zhang et al., 2019, among others). Dai et al. (2019) focus on identifying the most suitable corpus to pretrain a LM from scratch, for a single task: NER, whereas we select relevant examples for various tasks in §5.2. Concurrent to our work, Aharoni and Goldberg (2020) propose data selection methods for NMT based on cosine similarity in embedding space, using DistilBERT (Sanh et al., 2019) for efficiency. In contrast, we use VAMPIRE, and focus on augmenting TAPT data for text classification tasks. Khandelwal et al. (2020) introduced $k$ NN-LMs that allows easy domain adaptation of pretrained LMs by simply adding a datastore per domain and no further training; an alternative to integrate domain information in an LM. Our study of human-curated data §5.1 is related to focused crawling (Chakrabarti et al., 1999) for collection of suitable data, especially with LM reliance (Remus and Biemann, 2016).

What is a domain? Despite the popularity of domain adaptation techniques, most research and practice seems to use an intuitive understanding of domains. A small body of work has attempted to address this question (Lee, 2001; Eisenstein et al., 2014; van der Wees et al., 2015; Plank, 2016; Ruder et al., 2016, among others). For instance, Aharoni and Goldberg (2020) define domains by implicit clusters of sentence representations in pretrained LMs. Our results show that dapt and tapt complement each other, which suggests a spectra of domains defined around tasks at various levels of granularity (e.g., Amazon reviews for a specific product, all Amazon reviews, all reviews on the web, the web).

