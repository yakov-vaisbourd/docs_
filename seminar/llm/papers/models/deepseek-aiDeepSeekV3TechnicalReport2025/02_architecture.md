# 2. Architecture

We first introduce the basic architecture of DeepSeek-V3, featured by Multi-head Latent Attention (MLA) (DeepSeek-AI, 2024c) for efficient inference and DeepSeekMoE (Dai et al., 2024) for economical training. Then, we present a Multi-Token Prediction (MTP) training objective, which we have observed to enhance the overall performance on evaluation benchmarks. For other minor details not explicitly mentioned, DeepSeek-V3 adheres to the settings of DeepSeek-V2 (DeepSeek-AI, 2024c).

## 2.1. Basic Architecture

The basic architecture of DeepSeek-V3 is still within the Transformer (Vaswani et al., 2017) framework. For efficient inference and economical training, DeepSeek-V3 also adopts MLA and DeepSeekMoE, which have been thoroughly validated by DeepSeek-V2. Compared with DeepSeek-V2, an exception is that we additionally introduce an auxiliary-loss-free load balancing

![Figure](./deepseek-aiDeepSeekV3TechnicalReport2025_images/7bf953a3-e8a0-4bc9-9115-d38b75ce977a-07_1265_1571_289_248.jpg)

**Figure 2 | Illustration of the basic architecture of DeepSeek-V3. Following DeepSeek-V2, we adopt MLA and DeepSeekMoE for efficient inference and economical training.**

strategy (Wang et al., 2024a) for DeepSeekMoE to mitigate the performance degradation induced by the effort to ensure load balance. Figure 2 illustrates the basic architecture of DeepSeek-V3, and we will briefly review the details of MLA and DeepSeekMoE in this section.

## 2.1.1. Multi-Head Latent Attention

For attention, DeepSeek-V3 adopts the MLA architecture. Let $d$ denote the embedding dimension, $n_{h}$ denote the number of attention heads, $d_{h}$ denote the dimension per head, and $\mathbf{h}_{t} \in \mathbb{R}^{d}$ denote the attention input for the $t$-th token at a given attention layer. The core of MLA is the low-rank joint compression for attention keys and values to reduce Key-Value (KV) cache during inference:
$$
\begin{aligned}
\boxed{\mathbf{c}_{t}^{K V}} & =W^{D K V} \mathbf{h}_{t} \\
{\left[\mathbf{k}_{t, 1}^{C} ; \mathbf{k}_{t, 2}^{C} ; \ldots ; \mathbf{k}_{t, n_{h}}^{C}\right]=\mathbf{k}_{t}^{C} } & =W^{U K} \mathbf{c}_{t}^{K V} \\
\mathbf{k}_{t}^{R} & =\operatorname{RoPE}\left(W^{K R} \mathbf{h}_{t}\right), \\
\mathbf{k}_{t, i} & =\left[\mathbf{k}_{t, i}^{C} ; \mathbf{k}_{t}^{R}\right], \\
{\left[\mathbf{v}_{t, 1}^{C} ; \mathbf{v}_{t, 2}^{C} ; \ldots ; \mathbf{v}_{t, n_{h}}^{C}\right]=\mathbf{v}_{t}^{C} } & =W^{U V} \mathbf{c}_{t}^{K V},
\end{aligned}
$$
where $\mathbf{c}_{t}^{K V} \in \mathbb{R}^{d_{c}}$ is the compressed latent vector for keys and values; \(d_{c}\left(\ll d_{h} n_{h}\right)\) indicates the KV compression dimension; $W^{D K V} \in \mathbb{R}^{d_{c} \times d}$ denotes the down-projection matrix; $W^{U K}, W^{U V} \in \mathbb{R}^{d_{h} n_{h} \times d_{c}}$ are the up-projection matrices for keys and values, respectively; $W^{K R} \in \mathbb{R}^{d_{h}^{R} \times d}$ is the matrix used to produce the decoupled key that carries Rotary Positional Embedding (RoPE) (Su et al., 2024); RoPE(•) denotes the operation that applies RoPE matrices; and $[\cdot ; \cdot]$ denotes concatenation. Note that for MLA, only the blue-boxed vectors (i.e., $\mathbf{c}_{t}^{K V}$ and $\mathbf{k}_{t}^{R}$ ) need to be cached during generation, which results in significantly reduced KV cache while maintaining performance comparable to standard Multi-Head Attention (MHA) (Vaswani et al., 2017).

For the attention queries, we also perform a low-rank compression, which can reduce the activation memory during training:
$$
\begin{aligned}
\mathbf{c}_{t}^{Q} & =W^{D Q} \mathbf{h}_{t} \\
{\left[\mathbf{q}_{t, 1}^{C} ; \mathbf{q}_{t, 2}^{C} ; \ldots ; \mathbf{q}_{t, n_{h}}^{C}\right]=\mathbf{q}_{t}^{C} } & =W^{U Q} \mathbf{c}_{t}^{Q} \\
{\left[\mathbf{q}_{t, 1}^{R} ; \mathbf{q}_{t, 2}^{R} ; \ldots ; \mathbf{q}_{t, n_{h}}^{R}\right]=\mathbf{q}_{t}^{R} } & =\operatorname{RoPE}\left(W^{Q R} \mathbf{c}_{t}^{Q}\right) \\
\mathbf{q}_{t, i} & =\left[\mathbf{q}_{t, i}^{C} ; \mathbf{q}_{t, i}^{R}\right]
\end{aligned}
$$
where $\mathbf{c}_{t}^{Q} \in \mathbb{R}^{d_{c}^{\prime}}$ is the compressed latent vector for queries; \(d_{c}^{\prime}\left(\ll d_{h} n_{h}\right)\) denotes the query compression dimension; $W^{D Q} \in \mathbb{R}^{d_{c}^{\prime} \times d}, W^{U Q} \in \mathbb{R}^{d_{h} n_{h} \times d_{c}^{\prime}}$ are the down-projection and up-projection matrices for queries, respectively; and $W^{Q R} \in \mathbb{R}^{d_{h}^{R} n_{h} \times d_{c}^{\prime}}$ is the matrix to produce the decoupled queries that carry RoPE.

Ultimately, the attention queries \(\left(\mathbf{q}_{t, i}\right)\), keys \(\left(\mathbf{k}_{j, i}\right)\), and values \(\left(\mathbf{v}_{j, i}^{C}\right)\) are combined to yield the final attention output $\mathbf{u}_{t}$ :
$$
\begin{aligned}
\mathbf{o}_{t, i} & =\sum_{j=1}^{t} \operatorname{Softmax}_{j}\left(\frac{\mathbf{q}_{t, i}^{T} \mathbf{k}_{j, i}}{\sqrt{d_{h}+d_{h}^{R}}}\right) \mathbf{v}_{j, i^{\prime}}^{C} \\
\mathbf{u}_{t} & =W^{O}\left[\mathbf{o}_{t, 1} ; \mathbf{o}_{t, 2} ; \ldots ; \mathbf{o}_{t, n_{h}}\right]
\end{aligned}
$$
where $W^{O} \in \mathbb{R}^{d \times d_{h} n_{h}}$ denotes the output projection matrix.

## 2.1.2. DeepSeekMoE with Auxiliary-Loss-Free Load Balancing

Basic Architecture of DeepSeekMoE. For Feed-Forward Networks (FFNs), DeepSeek-V3 employs the DeepSeekMoE architecture (Dai et al., 2024). Compared with traditional MoE architectures like GShard (Lepikhin et al., 2021), DeepSeekMoE uses finer-grained experts and isolates some experts as shared ones. Let $\mathbf{u}_{t}$ denote the FFN input of the $t$-th token, we compute the FFN output $\mathbf{h}_{t}^{\prime}$ as follows:
$$
\begin{aligned}
\mathbf{h}_{t}^{\prime} & =\mathbf{u}_{t}+\sum_{i=1}^{N_{s}} \mathrm{FFN}_{i}^{(s)}\left(\mathbf{u}_{t}\right)+\sum_{i=1}^{N_{r}} g_{i, t} \mathrm{FFN}_{i}^{(r)}\left(\mathbf{u}_{t}\right), \\
g_{i, t} & =\frac{g_{i, t}^{\prime}}{\sum_{j=1}^{N_{r}} g_{j, t}^{\prime}}, \\
g_{i, t}^{\prime} & = \begin{cases}s_{i, t}, & s_{i, t} \in \operatorname{Topk}\left(\left\{s_{j, t} \mid 1 \leqslant j \leqslant N_{r}\right\}, K_{r}\right), \\
0, & \text { otherwise }\end{cases} \\
s_{i, t} & =\operatorname{Sigmoid}\left(\mathbf{u}_{t}^{T} \mathbf{e}_{i}\right),
\end{aligned}
$$
where $N_{s}$ and $N_{r}$ denote the numbers of shared experts and routed experts, respectively; \(\operatorname{FFN}_{i}^{(s)}(\cdot)\) and \(\operatorname{FFN}_{i}^{(r)}(\cdot)\) denote the $i$-th shared expert and the $i$-th routed expert, respectively; $K_{r}$ denotes the number of activated routed experts; $g_{i, t}$ is the gating value for the $i$-th expert; $s_{i, t}$ is the token-to-expert affinity; $\mathbf{e}_{i}$ is the centroid vector of the $i$-th routed expert; and Topk \((\cdot, K)\) denotes the set comprising $K$ highest scores among the affinity scores calculated for the $t$-th token and all routed experts. Slightly different from DeepSeek-V2, DeepSeek-V3 uses the sigmoid function to compute the affinity scores, and applies a normalization among all selected affinity scores to produce the gating values.

Auxiliary-Loss-Free Load Balancing. For MoE models, an unbalanced expert load will lead to routing collapse (Shazeer et al., 2017) and diminish computational efficiency in scenarios with expert parallelism. Conventional solutions usually rely on the auxiliary loss (Fedus et al., 2021; Lepikhin et al., 2021) to avoid unbalanced load. However, too large an auxiliary loss will impair the model performance (Wang et al., 2024a). To achieve a better trade-off between load balance and model performance, we pioneer an auxiliary-loss-free load balancing strategy (Wang et al., 2024a) to ensure load balance. To be specific, we introduce a bias term $b_{i}$ for each expert and add it to the corresponding affinity scores $s_{i, t}$ to determine the top-K routing:
$$
g_{i, t}^{\prime}= \begin{cases}s_{i, t}, & s_{i, t}+b_{i} \in \operatorname{Topk}\left(\left\{s_{j, t}+b_{j} \mid 1 \leqslant j \leqslant N_{r}\right\}, K_{r}\right), \\ 0, & \text { otherwise } .\end{cases}
$$
Note that the bias term is only used for routing. The gating value, which will be multiplied with the FFN output, is still derived from the original affinity score $s_{i, t}$. During training, we keep monitoring the expert load on the whole batch of each training step. At the end of each step, we will decrease the bias term by $\gamma$ if its corresponding expert is overloaded, and increase it by $\gamma$ if its corresponding expert is underloaded, where $\gamma$ is a hyper-parameter called bias update speed. Through the dynamic adjustment, DeepSeek-V3 keeps balanced expert load during training, and achieves better performance than models that encourage load balance through pure auxiliary losses.

Complementary Sequence-Wise Auxiliary Loss. Although DeepSeek-V3 mainly relies on the auxiliary-loss-free strategy for load balance, to prevent extreme imbalance within any single sequence, we also employ a complementary sequence-wise balance loss:
$$
\begin{aligned}
\mathcal{L}_{\text {Bal }} & =\alpha \sum_{i=1}^{N_{r}} f_{i} P_{i} \\
f_{i}=\frac{N_{r}}{K_{r} T} \sum_{t=1}^{T} \mathbb{1} & \left(s_{i, t} \in \operatorname{Topk}\left(\left\{s_{j, t} \mid 1 \leqslant j \leqslant N_{r}\right\}, K_{r}\right)\right), \\
s_{i, t}^{\prime} & =\frac{s_{i, t}}{\sum_{j=1}^{N_{r}} s_{j, t}}, \\
P_{i} & =\frac{1}{T} \sum_{t=1}^{T} s_{i, t}^{\prime}
\end{aligned}
$$
where the balance factor $\alpha$ is a hyper-parameter, which will be assigned an extremely small value for DeepSeek-V3; \(\mathbb{1}(\cdot)\) denotes the indicator function; and $T$ denotes the number of tokens in a sequence. The sequence-wise balance loss encourages the expert load on each sequence to be balanced.

![Figure](./deepseek-aiDeepSeekV3TechnicalReport2025_images/7bf953a3-e8a0-4bc9-9115-d38b75ce977a-10_702_1571_291_248.jpg)

**Figure 3 | Illustration of our Multi-Token Prediction (MTP) implementation. We keep the complete causal chain for the prediction of each token at each depth.**

Node-Limited Routing. Like the device-limited routing used by DeepSeek-V2, DeepSeek-V3 also uses a restricted routing mechanism to limit communication costs during training. In short, we ensure that each token will be sent to at most $M$ nodes, which are selected according to the sum of the highest $\frac{K_{r}}{M}$ affinity scores of the experts distributed on each node. Under this constraint, our MoE training framework can nearly achieve full computation-communication overlap.

No Token-Dropping. Due to the effective load balancing strategy, DeepSeek-V3 keeps a good load balance during its full training. Therefore, DeepSeek-V3 does not drop any tokens during training. In addition, we also implement specific deployment strategies to ensure inference load balance, so DeepSeek-V3 also does not drop tokens during inference.

## 2.2. Multi-Token Prediction

Inspired by Gloeckle et al. (2024), we investigate and set a Multi-Token Prediction (MTP) objective for DeepSeek-V3, which extends the prediction scope to multiple future tokens at each position. On the one hand, an MTP objective densifies the training signals and may improve data efficiency. On the other hand, MTP may enable the model to pre-plan its representations for better prediction of future tokens. Figure 3 illustrates our implementation of MTP. Different from Gloeckle et al. (2024), which parallelly predicts $D$ additional tokens using independent output heads, we sequentially predict additional tokens and keep the complete causal chain at each prediction depth. We introduce the details of our MTP implementation in this section.

MTP Modules. To be specific, our MTP implementation uses $D$ sequential modules to predict $D$ additional tokens. The $k$-th MTP module consists of a shared embedding layer \(\operatorname{Emb}(\cdot)\), a shared output head OutHead (•), a Transformer block \(\operatorname{TRM}_{k}(\cdot)\), and a projection matrix $M_{k} \in \mathbb{R}^{d \times 2 d}$. For the $i$-th input token $t_{i}$, at the $k$-th prediction depth, we first combine the representation of the $i$-th token at the \((k-1)\)-th depth $\mathbf{h}_{i}^{k-1} \in \mathbb{R}^{d}$ and the embedding of the \((i+k)\)-th token \(\operatorname{Emb}\left(t_{i+k}\right) \in \mathbb{R}^{d}\)
with the linear projection:
$$
\mathbf{h}_{i}^{\prime k}=M_{k}\left[\operatorname{RMSNorm}\left(\mathbf{h}_{i}^{k-1}\right) ; \operatorname{RMSNorm}\left(\operatorname{Emb}\left(t_{i+k}\right)\right)\right],
$$
where $[\because \cdot]$ denotes concatenation. Especially, when $k=1, \mathbf{h}_{i}^{k-1}$ refers to the representation given by the main model. Note that for each MTP module, its embedding layer is shared with the main model. The combined $\mathbf{h}_{i}^{\prime k}$ serves as the input of the Transformer block at the $k$-th depth to produce the output representation at the current depth $\mathbf{h}_{i}^{k}$ :
$$
\mathbf{h}_{1: T-k}^{k}=\operatorname{TRM}_{k}\left(\mathbf{h}_{1: T-k}^{\prime k}\right),
$$
where $T$ represents the input sequence length and ${ }_{i: j}$ denotes the slicing operation (inclusive of both the left and right boundaries). Finally, taking $\mathbf{h}_{i}^{k}$ as the input, the shared output head will compute the probability distribution for the $k$-th additional prediction token $P_{i+1+k}^{k} \in \mathbb{R}^{V}$, where $V$ is the vocabulary size:
$$
P_{i+k+1}^{k}=\text { OutHead }\left(\mathbf{h}_{i}^{k}\right) .
$$
The output head OutHead(•) linearly maps the representation to logits and subsequently applies the \(\operatorname{Softmax}(\cdot)\) function to compute the prediction probabilities of the $k$-th additional token. Also, for each MTP module, its output head is shared with the main model. Our principle of maintaining the causal chain of predictions is similar to that of EAGLE (Li et al., 2024b), but its primary objective is speculative decoding (Leviathan et al., 2023; Xia et al., 2023), whereas we utilize MTP to improve training.

MTP Training Objective. For each prediction depth, we compute a cross-entropy loss $\mathcal{L}_{\text {MTP }}^{k}$ :
$$
\mathcal{L}_{\mathrm{MTP}}^{k}=\operatorname{CrossEntropy}\left(P_{2+k: T+1}^{k}, t_{2+k: T+1}\right)=-\frac{1}{T} \sum_{i=2+k}^{T+1} \log P_{i}^{k}\left[t_{i}\right],
$$
where $T$ denotes the input sequence length, $t_{i}$ denotes the ground-truth token at the $i$-th position, and $P_{i}^{k}\left[t_{i}\right]$ denotes the corresponding prediction probability of $t_{i}$, given by the $k$-th MTP module. Finally, we compute the average of the MTP losses across all depths and multiply it by a weighting factor $\lambda$ to obtain the overall MTP loss $\mathcal{L}_{\text {MTP }}$, which serves as an additional training objective for DeepSeek-V3:
$$
\mathcal{L}_{\mathrm{MTP}}=\frac{\lambda}{D} \sum_{k=1}^{D} \mathcal{L}_{\mathrm{MTP}}^{k} .
$$

MTP in Inference. Our MTP strategy mainly aims to improve the performance of the main model, so during inference, we can directly discard the MTP modules and the main model can function independently and normally. Additionally, we can also repurpose these MTP modules for speculative decoding to further improve the generation latency.

