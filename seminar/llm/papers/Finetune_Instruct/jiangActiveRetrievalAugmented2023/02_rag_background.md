# 2 Retrieval Augmented Generation

We formally define single-time retrieval augmented generation and propose the framework of active retrieval augmented generation.

## 2.1 Notations and Definitions

Given a user input $\boldsymbol{x}$ and a document corpus $\mathcal{D}=$ $\left\{\boldsymbol{d}_{i}\right\}_{i=1}^{|\mathcal{D}|}$ (such as all Wikipedia articles), the goal of retrieval augmented LMs is to generate the answer $\boldsymbol{y}=\left[\boldsymbol{s}_{1}, \boldsymbol{s}_{2}, \ldots, \boldsymbol{s}_{m}\right]=\left[w_{1}, w_{2}, \ldots, w_{n}\right]$ containing $m$ sentences or $n$ tokens leveraging information retrieved from the corpus.

In retrieval augmented LM, the LM typically pairs with a retriever that can retrieve a list of documents \(\mathcal{D}_{\boldsymbol{q}}=\operatorname{ret}(\boldsymbol{q})\) for a query $\boldsymbol{q}$; the LM conditions on both the user input $\boldsymbol{x}$ and retrieved documents $\mathcal{D}_{\boldsymbol{q}}$ to generate the answer. Since we focus on examining various methods of determining when and what to retrieve, we follow existing methods (Ram et al., 2023; Trivedi et al., 2022) to prepend the retrieved documents before the user input to aid future generation for both baselines and our method for fair comparisons: \(\boldsymbol{y}=\operatorname{LM}\left(\left[\mathcal{D}_{\boldsymbol{q}}, \boldsymbol{x}\right]\right)\), where $[\cdot, \cdot]$ is concatenation following the specified order.

## 2.2 Single-time Retrieval Augmented Generation

The most common choice is to directly use the user input as the query for retrieval and generate the complete answer at once \(\boldsymbol{y}=\operatorname{LM}\left(\left[\mathcal{D}_{\boldsymbol{x}}, \boldsymbol{x}\right]\right)\).

## 2.3 Active Retrieval Augmented Generation

To aid long-form generation with retrieval, we propose active retrieval augmented generation. It is a generic framework that actively decides when and what to retrieve through the generation process, resulting in the interleaving of retrieval and generation. Formally, at step \(t(t \geq 1)\), the retrieval query $\boldsymbol{q}_{t}$ is formulated based on both the user input $\boldsymbol{x}$ and previously generated output $\boldsymbol{y}_{<t}=\left[\boldsymbol{y}_{0}, \ldots, \boldsymbol{y}_{t-1}\right]$ :
$$
\boldsymbol{q}_{t}=\operatorname{qry}\left(\boldsymbol{x}, \boldsymbol{y}_{<t}\right),
$$
where \(\operatorname{qry}(\cdot)\) is the query formulation function. At the beginning \((t=1)\), the previous generation is empty ( $\boldsymbol{y}_{<1}=\emptyset$ ), and the user input is used as the initial query ( $\boldsymbol{q}_{1}=\boldsymbol{x}$ ). Given retrieved documents $\mathcal{D}_{\boldsymbol{q}_{t}}$, LMs continually generate the answer until the next retrieval is triggered or reaches the end:
$$
\boldsymbol{y}_{t}=\operatorname{LM}\left(\left[\mathcal{D}_{\boldsymbol{q}_{t}}, \boldsymbol{x}, \boldsymbol{y}_{<t}\right]\right),
$$
where $\boldsymbol{y}_{t}$ represents the generated tokens at the current step $t$, and the input to LMs is the concatenation of the retrieved documents $\mathcal{D}_{\boldsymbol{q}_{t}}$, the user input $\boldsymbol{x}$, and the previous generation $\boldsymbol{y}_{<t}$. We discard previously retrieved documents $\cup_{t^{\prime}<t} \mathcal{D}_{\boldsymbol{q}_{t^{\prime}}}$ and only use the retrieved documents from the current step to condition the next generation to prevent reaching the input length limit of LMs.

