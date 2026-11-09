# 3 Methods

Overview of RAPTOR Building on the idea that long texts often present subtopics and hierarchical structures (Cao \& Wang, 2022; Dong et al., 2023b), RAPTOR addresses the issue of semantic depth and connection in reading by building a recursive tree structure that balances broader thematic comprehension with granular details and which allows nodes to be grouped based on semantic similarity not just order in the text.

Construction of the RAPTOR tree begins with segmenting the retrieval corpus into short, contiguous texts of length 100, similar to traditional retrieval augmentation techniques. If a sentence exceeds the 100-token limit, we move the entire sentence to the next chunk, rather than cutting it mid-sentence. This preserves the contextual and semantic coherence of the text within each chunk. These texts are then embedded using SBERT, a BERT-based encoder (multi-qa-mpnet-base-cos-v1) (Reimers \& Gurevych, 2019). The chunks and their corresponding SBERT embeddings form the leaf nodes of our tree structure.

To group similar text chunks, we employ a clustering algorithm. Once clustered, a Language Model is used to summarize the grouped texts. These summarized texts are then re-embedded, and the cycle of embedding, clustering, and summarization continues until further clustering becomes infeasible, resulting in a structured, multi-layered tree representation of the original documents. An important aspect of RAPTOR is its computational efficiency. The system scales linearly in terms of both build time and token expenditure, making it suitable for processing large and complex corpora. For a comprehensive discussion on RAPTOR's scalability, please refer to the Appendix A.

For querying within this tree, we introduce two distinct strategies: tree traversal and collapsed tree. The tree traversal method traverses the tree layer-by-layer, pruning and selecting the most relevant nodes at each level. The collapsed tree method evaluates nodes collectively across all layers to find the most relevant ones.

Clustering Algorithm Clustering plays a key role in building the RAPTOR tree, organizing text segments into cohesive groups. This step groups related content together, which helps the subsequent retrieval process.

One of the unique aspects of our clustering approach is the use of soft clustering, where nodes can belong to multiple clusters without requiring a fixed number of clusters. This flexibility is essential because individual text segments often contain information relevant to various topics, thereby warranting their inclusion in multiple summaries.

Our clustering algorithm is based on Gaussian Mixture Models (GMMs), an approach that offers both flexibility and a probabilistic framework. GMMs assume that data points are generated from a mixture of several Gaussian distributions.

Given a set of $N$ text segments, each represented as a $d$-dimensional dense vector embedding, the likelihood of a text vector, $\mathbf{x}$, given its membership in the $k^{\text {th }}$ Gaussian distribution, is denoted by \(P(\mathbf{x} \mid k)=\mathcal{N}\left(\mathbf{x} ; \mu_{k}, \boldsymbol{\Sigma}_{k}\right)\). The overall probability distribution is a weighted combination \(P(\mathbf{x})=\) \(\sum_{k=1}^{K} \pi_{k} \mathcal{N}\left(\mathbf{x} ; \mu_{k}, \boldsymbol{\Sigma}_{k}\right)\), where $\pi_{k}$ signifies the mixture weight for the $k^{\text {th }}$ Gaussian distribution.
The high dimensionality of vector embeddings presents a challenge for traditional GMMs, as distance metrics may behave poorly when used to measure similarity in high-dimensional spaces (Aggarwal et al., 2001). To mitigate this, we employ Uniform Manifold Approximation and Projection (UMAP), a manifold learning technique for dimensionality reduction (McInnes et al., 2018). The number of nearest neighbors parameter, n_neighbors, in UMAP determines the balance between the preservation of local and global structures. Our algorithm varies n_neighbors to create a hierarchical clustering structure: it first identifies global clusters and then performs local clustering within these global clusters. This two-step clustering process captures a broad spectrum of relationships among the text data, from broad themes to specific details.

Should a local cluster's combined context ever exceed the summarization model's token threshold, our algorithm recursively applies clustering within the cluster, ensuring that the context remains within the token threshold.

To determine the optimal number of clusters, we employ the Bayesian Information Criterion (BIC) for model selection. BIC not only penalizes model complexity but also rewards goodness of fit (Schwarz, 1978). The BIC for a given GMM is \(B I C=\ln (N) k-2 \ln (\hat{L})\), where $N$ is the number of text segments (or data points), $k$ is the number of model parameters, and $\hat{L}$ is the maximized value of the likelihood function of the model. In the context of GMM, the number of parameters $k$ is a function of the dimensionality of the input vectors and the number of clusters.

With the optimal number of clusters determined by BIC, the Expectation-Maximization algorithm is then used to estimate the GMM parameters, namely the means, covariances, and mixture weights.

While the Gaussian assumption in GMMs may not perfectly align with the nature of text data, which often exhibits a sparse and skewed distribution, our empirical observations suggest that it offers an effective model for our purpose. We run an ablation comparing GMM Clustering with summarizing contiguous chunks and provide details in Appendix B.

Model-Based Summarization After clustering the nodes using Gaussian Mixture Models, the nodes in each cluster are sent to a language model for summarization. This step allows the model to transform large chunks of text into concise, coherent summaries of the selected nodes. For our experiments, we use gpt-3.5-turbo to generate the summaries. The summarization step condenses the potentially large volume of retrieved information into a manageable size. We provide statistics on the compression due to the summarization in Appendix C and the prompt used for summarization in Appendix D.

While the summarization model generally produces reliable summaries, a focused annotation study revealed that about 4\% of the summaries contained minor hallucinations. These did not propagate to parent nodes and had no discernible impact on question-answering tasks. For an in-depth analysis of hallucinations, refer to the appendix E.

Querying In this section, we elaborate on the two querying mechanisms employed by RAPTOR: tree traversal and collapsed tree. These methods offer unique ways of traversing the multi-layered RAPTOR tree to retrieve relevant information, each with its own advantages and trade-offs. We provide the pseudocode of both methods in Appendix F. Note that we embed all nodes using SBERT.

The tree traversal method first selects the top-k most relevant root nodes based on their cosine similarity to the query embedding. The children of these selected nodes are considered at the next layer and the top-k nodes are selected from this pool again based on their cosine similarity to the query vector. This process is repeated until we reach the leaf nodes. Finally, the text from all selected nodes is concatenated to form the retrieved context. The algorithm's steps are outlined below:
\begin{itemize}
\item[1.] Start at the root layer of the RAPTOR tree. Compute the cosine similarity between the query embedding and the embeddings of all nodes present at this initial layer.
\item[2.] Choose the top- $k$ nodes based on the highest cosine similarity scores, forming the set $S_{1}$.
\end{itemize}

![Figure](./sarthiRAPTORRecursiveAbstractive2024_images/4beeda7d-949d-4d45-a072-4861cfa71615-05_439_1229_339_441.jpg)

**Figure 2: Illustration of the tree traversal and collapsed tree retrieval mechanisms. Tree traversal starts at the root level of the tree and retrieves the top- $k$ (here, top-1) node(s) based on cosine similarity to the query vector. At each level, it retrieves the top- $k$ node(s) from the child nodes of the previous layer's top- $k$. Collapsed tree collapses the tree into a single layer and retrieves nodes until a threshold number of tokens is reached, based on cosine similarity to the query vector. The nodes on which cosine similarity search is performed are highlighted in both illustrations.**

\begin{itemize}
\item[3.] Proceed to the child nodes of the elements in set $S_{1}$. Compute the cosine similarity between the query vector and the vector embeddings of these child nodes.
\item[4.] Select the top $k$ child nodes with the highest cosine similarity scores to the query, forming the set $S_{2}$.
\item[5.] Continue this process recursively for $d$ layers, producing sets $S_{1}, S_{2}, \ldots, S_{d}$.
\item[6.] Concatenate sets $S_{1}$ through $S_{d}$ to assemble the relevant context to the query.
\end{itemize}

By adjusting the depth $d$ and the number of nodes $k$ selected at each layer, the tree traversal method offers control over the specificity and breadth of the information retrieved. The algorithm starts with a broad outlook by considering the top layers of the tree and progressively focuses on finer details as it descends through the lower layers.

The collapsed tree approach offers a simpler way to search for relevant information by considering all nodes in the tree simultaneously, as depicted in Figure 2. Instead of going layer-by-layer, this method flattens the multi-layered tree into a single layer, essentially bringing all the nodes onto the same level for comparison. The steps for this method are outlined below:
\begin{itemize}
\item[1.] First, collapse the entire RAPTOR tree into a single layer. This new set of nodes, denoted as $C$, contains nodes from every layer of the original tree.
\item[2.] Next, calculate the cosine similarity between the query embedding and the embeddings of all nodes present in the collapsed set $C$.
\item[3.] Finally, pick the top- $k$ nodes that have the highest cosine similarity scores with the query. Keep adding nodes to the result set until you reach a predefined maximum number of tokens, ensuring you don't exceed the model's input limitations.
\end{itemize}

We tested both approaches on 20 stories from the QASPER dataset. Figure 3 shows the performance of tree traversal with different top- sizes and collapsed tree with different maximum token numbers. The collapsed tree approach consistently performs better. We believe collapsed tree retrieval is better due to offering greater flexibility than tree traversal; i.e., by searching through all the nodes simultaneously, it retrieves information that is at the correct level of granularity for a given question. In comparison, while using tree traversal with the same values of $d$ and $k$, the ratio of nodes from each level of the tree will be constant. So, the ratio of higher-order thematic information to granular details will remain the same regardless of the question.

One drawback, however, of the collapsed tree approach is that it requires cosine similarity search to be performed on all nodes in the tree. However, this can be made more efficient with fast $k$-nearest neighbor libraries such as FAISS (Johnson et al., 2019).

![Figure](./sarthiRAPTORRecursiveAbstractive2024_images/4beeda7d-949d-4d45-a072-4861cfa71615-06_505_642_436_376.jpg)

**Figure 3: Comparison of querying methods. Results on 20 stories from the QASPER dataset using tree traversal with different top-k values, and collapsed tree with different context lengths. Collapsed tree with 2000 tokens produces the best results, so we use this querying strategy for our main results.**

Overall, given the collapsed tree approach's greater flexibility and its superior performance on the subset of the QASPER dataset, this is the querying approach with which we proceed. Specifically, we use the collapsed tree with 2000 maximum tokens, which approximately equates to retrieving the top-20 nodes. Using a token-based approach ensures the context does not exceed model context constraints as token counts can vary across nodes. For experiments with the UnifiedQA model, we provide 400 tokens of context, as UnifiedQA has a max context length of 512 tokens. We provide the same amount of tokens of context to RAPTOR and to the baselines.

Qualitative Study We conduct a qualitative analysis to understand the benefits of RAP-TOR's retrieval process compared to Dense Passage Retrieval (DPR) methods. Our study focuses on thematic, multi-hop questions using a 1500-word Cinderella fairytale. As illustrated in Figure 4, RAPTOR's tree-based retrieval allows it to choose nodes from different tree layers, matching the question's detail level. This approach often yields more relevant and comprehensive information for downstream tasks than DPR. For a detailed discussion and examples, including the text retrieved by both RAPTOR and DPR for specific questions, please refer to the appendix G.

