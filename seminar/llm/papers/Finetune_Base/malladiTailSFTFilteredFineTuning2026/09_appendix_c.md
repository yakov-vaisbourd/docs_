# C Details for Graph Navigation Experiments

In this section, we describe the task, training details, and results for the graph navigation experiments presented in Figure 3. The setting is closely related to that of Chen et al. (2026a); we present all details for completeness but highlight where our experiments diverge from theirs.

# C. 1 Graph Reasoning Task and Data

Following Chen et al. (2026a), we use a path-following task in a directed acyclic graph (DAG) as an abstraction for reasoning problems. The setup builds on a long line of work using synthetic tasks to understand phenomena in language modeling and serves as a minimal, expressive, yet flexible setting to develop interventions like TailSFT.

The task is path-following a directed acyclic graph. Each prompt $x \in \mathcal{X}$ encodes a graph $G$ along with a source node $s$ and a target node $t$, such that \(x=(G, s, t)\), and each response $y \in \mathcal{Y}$ ideally encodes an $s \rightarrow t$ path via a sequence of vertices \(\left(s, v_{1}, \ldots, v_{n}, t\right)\). In our experiments we fix a single prompt distribution \(\mu \in \Delta(\mathcal{X})\) for both pre-training and SFT stages, however we use different data-collection policies \(\pi_{\text {pre }}, \pi_{\text {SFT }}: Y \rightarrow \Delta(\mathcal{X})\), both of which map prompts to (distributions over) $s \rightarrow t$ paths.

All graphs $G$ are directed layered graphs with 10 total layers and with edges only between consecutive layers. The source vertex $s$ is the only vertex in layer $L=1$ and the target vertex $t$ is the only vertex in layer 10. The subsequent 8 layers have 4 vertices each. For each of layers $L \in\{1, \ldots, 9\}$, a subset of the vertices in that layer are called passable. The edge structure is such that every passable node in layer $\ell$ has directed edges to all nodes in layer $\ell+1$. Nodes that are not passable have no out-going edges. Thus a valid $s \rightarrow t$ path consists of 10 total nodes \(\left(s, v_{1}, \ldots, v_{8}, t\right)\) where $v_{\ell}$ is in layer $\ell$ and each $v_{\ell}$ must be passable.

Graphs are parameterized by a number $k \in\{0, \ldots, 8\}$ denoting the number of layers with two passable nodes; the remaining layers have exactly one passable node. Given $k$ we choose the layers with two passable nodes at random. Choosing the passable nodes is more intricate. We identify every vertex with a number $\in\{0, \ldots, 99\}$, such that in each of the intermediate layers, at least one vertex is odd and at least one vertex is even. When there is just one passable vertex in a layer, it is chosen uniformly at random. When there are two passable vertices, they are chosen at random such that one is odd and one is even. Thus, there are $2^{k}$ valid $s \rightarrow t$ paths in each graph. We use $k=3$ for all experiments.

As described above, pre-training and SFT share the same prompt distribution. These are graphs of the above structure with vertex names assigned randomly (subject to the aforementioned even/odd restrictions in each layer), with passable nodes assigned randomly, and with layers with two passable nodes selected randomly. For pre-training, the data-collection policy $\pi_{\text {pre }}$ selects an $s \rightarrow t$ path uniformly at random from all $2^{k}$ valid paths.

For SFT, the data-collection policy is more complex. First the policy computes a certain "cryptographic" function of the input graph to determine if the graph belongs to shard0 or shard1 (specifically, the cryptographic function is the and of the parity of the vertices layers 1-5 and the parity of the vertices in layers 6-10). The shard determines a certain rule for how the policy chooses vertices in the layers where there are two passable nodes. Specifically in shard0 the policy alternates between agreeing with the parity of the target vertex $t$ and disagreeing with the parity of target node $t$, such that it agrees with the target vertex parity in the first layer with two passable nodes, disagrees in the second, and so on. In shard1 the policy does exactly the opposite, it disagrees with the target vertex parity in the first layer with two passable nodes, agrees in the second, and so on.

Intuitively, learning the pre-training policy's response distribution is relatively easy as it only requires local rules such as identifying all passable nodes in the subsequent layer and choosing one at random. Pre-training on this distribution teaches the model to understand the input format and the high-level task. On the other hand, learning the SFT policy's response distribution is very challenging, as it requires identifying global structure both to determine the shard and to determine which passable vertex to select in each layer.

To measure the performance of the trained model, we observe that the SFT policy is deterministic (for any input graph $x$, there is a single path in the support of the SFT policy's distribution \(\pi_{\mathrm{SFT}}(\cdot \mid x)\) ) and set the reward to be 1 if and only if the path chosen by the trained model matches the path selected by the SFT policy. Since learning the SFT policy's response distribution is challenging, it is corresponding quite challenging to achieve high reward in this task.

Inputs and outputs are represented as follows. First, all vertices, including the source and target, are assigned a number in $\{0, \ldots, 99\}$. Then, the input prompt is represented as an edge list followed by the source and target nodes, formatted as:
$$
x: \quad \text { u_1 v_1 | u_2 v_2 | ... | u_k v_k / s t = }
$$
where $u_{i}, v_{i}$ are the numerical values assigned to the vertices in the $i$ th edge. We use the delimiter characters $\mid$, /, and = to separate edges from each other, the edge list from the source and target vertices, and the input from the response, respectively. Responses $y$ are formatted as:
$$
y: \quad \text { v_1 } \quad \text { v_2 } \quad \text { v_2 } \quad \text { v_3 } \ldots . \mathrm{v}_{-} 9 \quad \text { v_10. }
$$

# C. 2 Training Details

We use the same numerical tokenizer and transformer architecture as Chen et al. (2026a). For tokenization, each node is $v$ is tokenized as its numerical value and the special characters are tokenized as 100, 101, 102. The architecture is a GPT2-style transformer model with 4 heads, 6 transformer blocks and a 384 dimensional embeddings. We used absolute positional encodings. We always use the Adam optimizer with a fixed learning $\left(1 \times 10^{-4}\right.$ during pre-training and $5 \times 10^{-6}$ during SFT) and a batch size of 128. Pre-training operates in an offline training setting with a fixed pool of 256,000 graphs while SFT operates in an online training setting with new graphs generated on-the-fly for each batch. We pre-train for 200k training steps and SFT for 50k training steps with evaluation every 200 steps.

![Figure](./malladiTailSFTFilteredFineTuning2026_images/b0c2e9de-14c4-464c-9bb3-a4879161b9bf-24_502_1652_235_233.jpg)

**Figure 5: Detailed results of hyperparameter sweep for TailSFT with absolute clipping in the synthetic graph navigation experiment.**

![Figure](./malladiTailSFTFilteredFineTuning2026_images/b0c2e9de-14c4-464c-9bb3-a4879161b9bf-24_495_1647_846_235.jpg)

**Figure 6: Detailed results of hyperparameter sweep for TailSFT with quantile clipping in the synthetic graph navigation experiment.**

# C. 3 Additional Experimental Results

In Figure 3 we show the cross entropy, pass@1, and pass@8 for each of four methods: vanilla SFT, TailSFT with absolute clipping, TailSFT with quantile clipping, and TailSFT with offset quantile clipping using the pre-trained model as the reference. For the TailSFT variants we sweep over clipping hyperparameter and select the best hyperparameter based on pass@8 performance at the end of training. The specific hyperparameter selected are 0.05 for absolute, 0.4 for quantile, and 0.35 for offset quantile.

In Figs. 5 to 7, we show detailed results of the hyperparameter sweep for the three TailSFT variants. In all cases, we sweep the clipping hyperparameter in the range $0.05 \times\{1 \ldots, 19\}$. For all variants clipping at value 0 corresponds to vanilla SFT as no sequences are dropped. We highlight two observations. First, absolute clipping is much more sensitive to the clipping hyperparameter than quantile or offset quantile variants. This is rather intuitive, as setting an absolute threshold for the cross-entropy loss requires understanding how the distribution of per-example cross-entropy losses evolves during training. On the other hand, setting the threshold as a per-batch quantile (either on the loss or on the offset loss) is much easier and these methods are much more robust. Indeed, the second observation is that for quantile-variants of TailSFT, setting the threshold too small recovers baseline SFT performance and setting the threshold too large does not impact pass@8 performance, but rather slows training convergence. This is also intuitive, as a large clipping parameter still focuses training on the harder examples but sacrifices computational efficiency as fewer per-sample gradients are utilized.

![Figure](./malladiTailSFTFilteredFineTuning2026_images/b0c2e9de-14c4-464c-9bb3-a4879161b9bf-25_505_1654_233_233.jpg)

**Figure 7: Detailed results of hyperparameter sweep for TailSFT with offset quantile clipping in the synthetic graph navigation experiment.**

