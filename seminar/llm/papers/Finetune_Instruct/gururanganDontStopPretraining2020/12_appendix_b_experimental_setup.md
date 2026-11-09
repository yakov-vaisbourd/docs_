# B Experimental Setup

Preprocessing for DAPT The unlabeled corpus in each domain was pre-processed prior to language model training. Abstracts and body paragraphs from biomedical and computer science articles were used after sentence splitting using scispaCy (Neumann et al., 2019). We used summaries and full text of each news article, and the entire body of review from Amazon reviews. For both news and reviews, we perform sentence splitting using spaCy (Honnibal and Montani, 2017).

Training details for DAPT We train ROBERTA on each domain for 12.5K steps. We focused on matching all the domain dataset sizes (see Table 1) such that each domain is exposed to the same amount of data as for 12.5K steps it is trained for. Amazon reviews contain more documents, but each is shorter. We used an effective batch size of 2048 through gradient accumulation, as recommended in Liu et al. (2019). See Table 13 for more hyperparameter details.

Training details for TAPT We use the same pretraining hyperparameters as DAPT, but we artificially augmented each dataset for TAPT by randomly masking different tokens across epochs, using the masking probability of 0.15 . Each dataset was trained for 100 epochs. For tasks with less than 5K examples, we used a batch size of 256 through gradient accumulation. See Table 13 for more hyperparameter details.

Optimization We used the Adam optimizer (Kingma and Ba, 2015), a linear learning rate scheduler with 6\% warm-up, a maximum learning rate of 0.0005. When we used a batch size of 256, we

\begin{table}
\begin{tabular}[t]{|l|l|l|l|l|l|l|l|l|}
\hline & DAPT Domains (if applicable) & Tasks & Model & DAPT & TAPT & DAPT + TAPT & $k \mathrm{NN}$ TAPT & CuratedTAPT \\
\hline This Paper & biomedical \& computer science papers, news, reviews & 8 classification tasks & ROBERTA & ✓ & ✓ & ✓ & ✓ & ✓ \\
\hline Aharoni and Goldberg (2020) & - & NMT & DistilBERT + Transformer NMT & - & - & - & similar & - \\
\hline Alsentzer et al. (2019) & clinical text & NER, NLI, de-identification & (Bio)BERT & ✓ & - & - & - & - \\
\hline Chakrabarty et al. (2019) & opinionated claims from Reddit & claim detection & ULMFIT & ✓ & ✓ & - & - & - \\
\hline Chronopoulou et al. (2019) & - & 5 classification tasks & ULMFIT ${ }^{\dagger}$ & - & similar & - & - & - \\
\hline Han and Eisenstein (2019) & - & NER in historical texts & ELMo, BERT & - & ✓ & - & - & - \\
\hline Howard and Ruder (2018) & - & 6 classification tasks & ULMFIT & - & ✓ & - & - & - \\
\hline Khandelwal et al. (2020) & - & language modeling & Transformer LM & - & - & - & similar & - \\
\hline Lee et al. (2019) & biomedical papers & NER, QA, relation extraction & BERT & ✓ & - & - & - & - \\
\hline Logeswaran et al. (2019) & - & zero-shot entity linking in Wikia & BERT & - & ✓ & - & - & - \\
\hline Mitra et al. (2020) & - & commonsense QA & BERT & - & ✓ & - & - & - \\
\hline Phang et al. (2018) & - & GLUE tasks & ELMo, BERT, GPT & - & ✓ & - & - & - \\
\hline Radford et al. (2018) & - & NLI, QA, similarity, classification & GPT & - & similar & - & - & - \\
\hline Sun et al. (2019) & sentiment, question, topic & 7 classification tasks & BERT & ✓ & ✓ & - & - & - \\
\hline Swayamdipta et al. (2019) & - & NER, parsing, classification & ELMo & - & similar & - & - & - \\
\hline Xu et al. (2019a) & reviews & RC, aspect extract., sentiment classification & BERT & ✓ & ✓ & ✓ & - & - \\
\hline Xu et al. (2019b) & restaurant reviews, laptop reviews & conversational RC & BERT & ✓ & ✓ & - & - & - \\
\hline
\end{tabular}

**Table 11: Overview of prior work across strategies for continued pre-training summarized in Table 10. ULMFIT is pretrained on English Wikipedia; ULMFIT ${ **^{\dagger}$ on English tweets; ELMo on the 1B WordBenchmark (newswire; Chelba et al., 2014); GPT on BookCorpus; BERT on English Wikipedia and BookCorpus. In comparison to these pretraining corpora, RoBERTA's pretraining corpus is substantially more diverse (see Appendix §A.1).}
\end{table}
used a maximum learning rate of 0.0001 , as recommended in Liu et al. (2019). We observe a high variance in performance between random seeds when fine-tuning RoBERTA to HyperPartisan, because the dataset is extremely small. To produce final results on this task, we discard and resample degenerate seeds. We display the full hyperparameter settings in Table 13.

Implementation Our LM implementation uses the HuggingFace transformers library (Wolf et al., 2019) ${ }^{11}$ and PyTorch XLA for TPU compatibility. ${ }^{12}$ Each adaptive pretraining exper-

\footnotetext{
${ }^{11}$ https://github.com/huggingface/ transformers
${ }^{12}$ https://github.com/pytorch/xla
}
iment was performed on a single v3-8 TPU from Google Cloud. ${ }^{13}$ For the text classification tasks, we used AllenNLP (Gardner et al., 2018). Following standard practice (Devlin et al., 2019) we pass the final layer [CLS] token representation to a task-specific feedforward layer for prediction.

