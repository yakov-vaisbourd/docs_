# 4 Analysis

## 4.1 Experiment 1

## 4.1.1 Datasets

We selected two datasets in the one million token range, each representative of corpora that users may encounter in their real-world activities:

Podcast transcripts. Public transcripts of Behind the Tech with Kevin Scott, a podcast featuring conversations between Microsoft CTO Kevin Scott and various thought leaders in science and technology (Scott, 2024). This corpus was divided into 1669 × 600-token text chunks, with 100-token overlaps between chunks (~1 million tokens).

News articles. A benchmark dataset comprised of news articles published from September 2013 to December 2023 in a range of categories, including entertainment, business, sports, technology, health, and science (Tang and Yang, 2024). The corpus is divided into $3197 \times 600$-token text chunks, with 100-token overlaps between chunks (~1.7 million tokens).

## 4.1.2 Conditions

We compared six conditions including GraphRAG at four different graph community levels (C0, C1, C2, C3), a text summarization method that applies our map-reduce approach directly to source texts (TS), and a vector RAG "semantic search" approach (SS):
\begin{itemize}
\item[-] CO. Uses root-level community summaries (fewest in number) to answer user queries.
\item[-] C1. Uses high-level community summaries to answer queries. These are sub-communities of C0, if present, otherwise C0 communities projected downwards.
\item[-] C2. Uses intermediate-level community summaries to answer queries. These are subcommunities of C1, if present, otherwise C1 communities projected downwards.
\item[-] C3. Uses low-level community summaries (greatest in number) to answer queries. These are sub-communities of C2, if present, otherwise C2 communities projected downwards.
\item[-] TS. The same method as in Section 3.1.6, except source texts (rather than community summaries) are shuffled and chunked for the map-reduce summarization stages.
\item[-] SS. An implementation of vector RAG in which text chunks are retrieved and added to the available context window until the specified token limit is reached.
\end{itemize}

The size of the context window and the prompts used for answer generation are the same across all six conditions (except for minor modifications to reference styles to match the types of context information used). Conditions only differ in how the contents of the context window are created.

The graph index supporting conditions C0-C3 was created using our generic prompts for entity and relationship extraction, with entity types and few-shot examples tailored to the domain of the data.

## 4.1.3 Configuration

We used a fixed context window size of 8k tokens for generating community summaries, community answers, and global answers (explained in Appendix C). Graph indexing with a 600 token window (explained in Section A.2) took 281 minutes for the Podcast dataset, running on a virtual machine (16GB RAM, Intel(R) Xeon(R) Platinum 8171M CPU @ 2.60GHz) and using a public OpenAI endpoint for gpt-4-turbo (2M TPM, 10k RPM).

We implemented Leiden community detection using the graspologic library (Chung et al., 2019). The prompts used to generate the graph index and global answers can be found in Appendix E, while the prompts used to evaluate LLM responses against our criteria can be found in Appendix F. A full statistical analysis of the results presented in the next section can be found in Appendix G.

## 4.2 Experiment 2

To validate the comprehensiveness and diversity results from Experiment 1, we implemented claimbased measures of these qualities. We use the definition of a factual claim from Ni et al. (2024), which is "a statement that explicitly presents some verifiable facts." For example, the sentence "California and New York implemented incentives for renewable energy adoption, highlighting the broader importance of sustainability in policy decisions" contains two factual claims: (1) California implemented incentives for renewable energy adoption, and (2) New York implemented incentives for renewable energy adoption.

To extract factual claims, we used Claimify (Metropolitansky and Larson, 2025), an LLM-based method that identifies sentences in an answer containing at least one factual claim, then decomposes these sentences into simple, self-contained factual claims. We applied Claimify to the answers generated under the conditions from Experiment 1. After removing duplicate claims from each answer, we extracted 47,075 unique claims, with an average of 31 claims per answer.

We defined two metrics, with higher values indicating better performance:
\begin{itemize}
\item[1.] Comprehensiveness: Measured as the average number of claims extracted from the answers generated under each condition.
\item[2.] Diversity: Measured by clustering the claims for each answer and calculating the average number of clusters.
\end{itemize}

For clustering, we followed the approach described by Padmakumar and He (2024), which involved using Scikit-learn's implementation of agglomerative clustering (Pedregosa et al., 2011). Clusters were merged through "complete" linkage, meaning they were combined only if the maximum distance between their farthest points was less than or equal to a predefined distance threshold. The distance metric used was 1 - ROUGE-L. Since the distance threshold influences the number of clusters, we report results across a range of thresholds.

