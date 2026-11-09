# 4 Experiments

## 4.1 Tasks and Datasets

We conduct evaluations of our Self-Rag and diverse baselines on a range of downstream tasks, holistically evaluating outputs with metrics designed to assess overall correctness, factuality, and fluency. Throughout these experiments, we conduct zero-shot evaluations, where we provide instructions describing tasks without few-shot demonstrations (Wei et al., 2022; Sanh et al., 2022). Details of our experiments' settings, including test-time instructions, are available in the Appendix Section B.1.

Closed-set tasks include two datasets, i.e., a fact verification dataset about public health (PubHealth; Zhang et al. 2023) and a multiple-choice reasoning dataset created from scientific exams (ARC-

Challenge; Clark et al. 2018). We use accuracy as an evaluation metric and report on the test set. We aggregate the answer probabilities of target classes for both of these datasets (Appendix Section B.2).

Short-form generations tasks include two open-domain question answering (QA) datasets, PopQA (Mallen et al., 2023) and TriviaQA-unfiltered (Joshi et al., 2017), where systems need to answer arbitrary questions about factual knowledge. For PopQA, we use the long-tail subset, consisting of 1,399 rare entity queries whose monthly Wikipedia page views are less than 100. As the TriviaQA-unfiltered (open) test set is not publicly available, we follow prior work's validation and test split (Min et al., 2019; Guu et al., 2020), using 11,313 test queries for evaluation. We evaluate performance based on whether gold answers are included in the model generations instead of strictly requiring exact matching, following Mallen et al. (2023); Schick et al., (2023).

Long-form generation tasks include a biography generation task (Min et al., 2023) and a long-form QA task ALCE-ASQA Gao et al. (2023); Stelmakh et al. (2022). We use FactScore (Min et al., 2023) to evaluate biographies, and we use official metrics of correctness (str-em), fluency based on MAUVE (Pillutla et al., 2021), and citation precision and recall (Gao et al., 2023) for ASQA. ${ }^{5}$

## 4.2 Baselines

Baselines without retrievals. We evaluate strong publicly available pre-trained LLMs, Llama2 ${ }_{7 \mathrm{~B}, 13 \mathrm{~B}}$ (Touvron et al., 2023), instruction-tuned models, Alpaca ${ }_{7 \mathrm{~B}, 13 \mathrm{~B}}$ (Dubois et al., 2023) (our replication based on Llama2); and models trained and reinforced using private data, Chat-GPT (Ouyang et al., 2022) and Llama2-chat ${ }_{13 \mathrm{~b}}$. For instruction-tuned LMs, we use the official system prompt or instruction format used during training if publicly available. We also compare our method to concurrent work, $\mathrm{CoVE}_{65 \mathrm{~B}}$ (Dhuliawala et al., 2023), which introduces iterative prompt engineering to improve the factuality of LLM generations.

Baselines with retrievals. We evaluate models augmented with retrieval at test time or during training. The first category includes standard RAG baselines, where an LM (Llama2, Alpaca) generates output given the query prepended with the top retrieved documents using the same retriever as in our system. It also includes Llama2-FT, where Llama2 is fine-tuned on all training data we use without the reflection tokens or retrieved passages. We also report the result of retrieval-augmented baselines with LMs trained with private data: Ret-ChatGPT and Ret-Llama2-chat, which deploy the same augmentation technique above, as well as perplexity.ai, an InstructGPT-based production search system. The second category includes concurrent methods that are trained with retrieved text passages, i.e., SAIL (Luo et al., 2023) to instruction-tune an LM on the Alpaca instruction-tuning data with top retrieved documents inserted before instructions, and Toolformer (Schick et al., 2023) to pre-train an LM with API calls (e.g., Wikipedia APIs) ${ }^{6}$

## 4.3 Experimental settings

Training data and settings. Our training data consists of diverse instruction-following input-output pairs. In particular, we sample instances from Open-Instruct processed data (Wang et al., 2023) and knowledge-intensive datasets (Petroni et al., 2021; Stelmakh et al., 2022; Mihaylov et al., 2018). In total, we use 150k instruction-output pairs. We use Llama2 7B and 13B (Touvron et al., 2023) as our generator base LM, and we use Llama2 7B as our base critic LM. For the retriever model $\mathcal{R}$, we use off-the-shelf Contriever-MS MARCO (Izacard et al., 2022a) by default and retrieve up to ten documents for each input. More training details are in the Appendix Section B. 1

Inference settings. As a default configuration, we assign the weight terms IsREL, ISSUP, ISUSE values of 1.0, 1.0 and 0.5, respectively. To encourage frequent retrieval, we set the retrieval threshold to 0.2 for most tasks and to 0 for ALCE (Gao et al., 2023) due to citation requirements. We speed up inference using vllm (Kwon et al., 2023). At each segment level, we adopt a beam width of 2. For a token-level generation, we use greedy decoding. By default, we use the top five documents from Contriever-MS MARCO (Izacard et al., 2022a); for biographies and open-domain QA, we use additional top five documents retrieved by a web search engine, following Luo et al. (2023); for ASQA, we use the author-provided top 5 documents by GTR-XXL (Ni et al., 2022) across all baselines for a fair comparison.

\footnotetext{
${ }^{5}$ https://github.com/princeton-nlp/ALCE
${ }^{6}$ We report numbers using the results reported in the paper as the implementations are not available.
}

\begin{table}

**Table 2: Overall experiment results on six tasks. Bold numbers indicate the best performance among non-proprietary models, and gray-colored bold text indicates the best proprietary model when they outperforms all non-proprietary models. * indicates concurrent or recent results reported by concurrent work. - indicates numbers that are not reported by the original papers or are not applicable. Models are sorted based on scale. FS, em, rg, mau, prec, rec denote FactScore (factuality); str-em, rouge (correctness); MAUVE (fluency); citation precision and recall, respectively.**
\begin{tabular}[t]{|l|l|l|l|l|l|l|l|l|l|l|}
\hline \multirow[b]{2}{*}{LM} & \multicolumn{2}{|c|}{Short-form} & \multicolumn{2}{|l|}{Closed-set} & \multicolumn{6}{|c|}{Long-form generations (with citations)} \\
\hline & PopQA (acc) & TQA (acc) & Pub (acc) & ARC (acc) & Bio (FS) & (em) & (rg) & ASQA (mau) & (pre) & (rec) \\
\hline \multicolumn{11}{|c|}{LMs with proprietary data} \\
\hline Llama2-c ${ }_{13 \mathrm{~b}}$ & 20.0 & 59.3 & 49.4 & 38.4 & 55.9 & 22.4 & 29.6 & 28.6 & - & - \\
\hline Ret-Llama2- $\mathrm{c}_{13 \mathrm{в}}$ & 51.8 & 59.8 & 52.1 & 37.9 & 79.9 & 32.8 & 34.8 & 43.8 & 19.8 & 36.1 \\
\hline ChatGPT & 29.3 & 74.3 & 70.1 & 75.3 & 71.8 & 35.3 & 36.2 & 68.8 & - & - \\
\hline Ret-ChatGPT & 50.8 & 65.7 & 54.7 & 75.3 & - & 40.7 & 39.9 & 79.7 & 65.1 & 76.6 \\
\hline Perplexity.ai & - & - & - & - & 71.2 & - & - & - & - & - \\
\hline \multicolumn{11}{|c|}{Baselines without retrieval} \\
\hline Llama2 ${ }_{7 \mathrm{~B}}$ & 14.7 & 30.5 & 34.2 & 21.8 & 44.5 & 7.9 & 15.3 & 19.0 & - & - \\
\hline Alpaca ${ }_{7}$ & 23.6 & 54.5 & 49.8 & 45.0 & 45.8 & 18.8 & 29.4 & 61.7 & - & - \\
\hline Llama2 ${ }_{13}$ & 14.7 & 38.5 & 29.4 & 29.4 & 53.4 & 7.2 & 12.4 & 16.0 & - & - \\
\hline Alpaca ${ }_{13}$ b & 24.4 & 61.3 & 55.5 & 54.9 & 50.2 & 22.9 & 32.0 & 70.6 & - & - \\
\hline $\mathrm{CoVE}_{65 \mathrm{~B}}$ * & - & - & - & - & 71.2 & - & - & - & - & - \\
\hline \multicolumn{11}{|c|}{Baselines with retrieval} \\
\hline Toolformer* ${ }_{6 \mathrm{~B}}$ & - & 48.8 & - & - & - & - & - & - & - & - \\
\hline Llama2 ${ }_{7 \mathrm{~b}}$ & 38.2 & 42.5 & 30.0 & 48.0 & 78.0 & 15.2 & 22.1 & 32.0 & 2.9 & 4.0 \\
\hline Alpaca ${ }_{7}$ & 46.7 & 64.1 & 40.2 & 48.0 & 76.6 & 30.9 & 33.3 & 57.9 & 5.5 & 7.2 \\
\hline Llama2-FT ${ }_{7 \mathrm{~B}}$ & 48.7 & 57.3 & 64.3 & 65.8 & 78.2 & 31.0 & 35.8 & 51.2 & 5.0 & 7.5 \\
\hline SAIL* ${ }_{7 \mathrm{~B}}$ & - & - & 69.2 & 48.4 & - & - & - & - & - & - \\
\hline Llama2 ${ }_{13 \mathrm{~b}}$ & 45.7 & 47.0 & 30.2 & 26.0 & 77.5 & 16.3 & 20.5 & 24.7 & 2.3 & 3.6 \\
\hline Alpaca ${ }_{13}$ & 46.1 & 66.9 & 51.1 & 57.6 & 77.7 & 34.8 & 36.7 & 56.6 & 2.0 & 3.8 \\
\hline Our Self-RAG 7B & 54.9 & 66.4 & 72.4 & 67.3 & 81.2 & 30.0 & 35.7 & 74.3 & 66.9 & 67.8 \\
\hline Our Self-Rag 13b & 55.8 & 69.3 & 74.5 & 73.1 & 80.2 & 31.7 & 37.0 & 71.6 & 70.3 & 71.3 \\
\hline
\end{tabular}
\end{table}

