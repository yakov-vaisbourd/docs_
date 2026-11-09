# Training and Adaptation

## Next-Token Prediction and Cross-Entropy Loss

The standard LLM training objective. Given a context prefix $\mathbf{z}_{1:n}$, the model autoregressively maximises the likelihood of each ground-truth token:

$$
\begin{aligned}
P(\mathbf{x}_{n+1:L} \mid \mathbf{z}_{1:n}) = \prod_{l=n+1}^{L} P_\theta(\mathbf{x}_l \mid \mathbf{x}_{1:l-1},\,\mathbf{z}_{1:n})
\end{aligned}
$$

$$
\begin{aligned}
\mathcal{L}_{CE} = -\sum_{l=n+1}^{L} \mathbf{y}_l \log P(\mathbf{x}_l \mid \mathbf{z}_{1:n})
\end{aligned}
$$

Numerical values can also be predicted this way by treating them as text tokens, though this is architecturally fragile: the model must learn to associate token sequences with numerical magnitude, and the loss treats ``3.9'' and ``4.1'' as equally wrong as ``3.9'' and ``9.1''.

## Fine-Tuning and Visual Instruction Tuning

Full fine-tuning updates all model weights on a task-specific dataset. Visual instruction tuning is the MLLM variant: the model is fine-tuned on (image/video, question, answer) triples so it learns to follow visual instructions. Common two-stage strategy: (1) train only the cross-modality projector on large captioning data to align visual features to the LLM token space; (2) fine-tune the full model (or most of it) on a smaller, task-specific instruction dataset.

## Catastrophic Forgetting

When a model is fine-tuned on new data, its performance on previously learned tasks degrades. A key motivation for inference-time adaptation methods such as RAG. In continual learning, the trade-off is: fine-tuning gives higher peak performance on the new domain but risks forgetting; RAG gives lower peak but monotonically improves as the database grows.

## LoRA - Low-Rank Adaptation

A parameter-efficient fine-tuning method. Instead of updating all weights $W$, only a small low-rank decomposition $\Delta W = A B^T$ ($A \in \mathbb{R}^{d \times r}$, $B \in \mathbb{R}^{d \times r}$, $r \ll d$) is trained and added to the frozen base weights. Typical LoRA trains $\approx 1\%$ of parameters. Key advantage: the base LLM weights are never modified, so general capabilities are preserved. Adapters can be merged at inference (zero latency overhead) or swapped per domain.

## Contrastive Learning and CLIP

Visual encoders such as LanguageBind and CLIP are pre-trained by matching images/videos to their text descriptions. Positive pairs (image, matching caption) are pulled together in embedding space; negatives are pushed apart. The result is a language-aligned visual representation that can be projected into an LLM's token space via a lightweight MLP projector.

## Triplet Loss and Metric Learning
Given an anchor $\mathbf{a}$, a positive sample $\mathbf{p}$ (similar), and a negative $\mathbf{n}$ (dissimilar):

$$
\begin{aligned}
\mathcal{L}_{Tri}(\mathbf{a},\mathbf{p},\mathbf{n}) = \max\!\left(\|\mathbf{a}-\mathbf{p}\|_2 - \|\mathbf{a}-\mathbf{n}\|_2 + \text{margin},\;0\right)
\end{aligned}
$$

Used to train retrieval embeddings so that semantically similar inputs cluster together. Quality depends entirely on how positive/negative pairs are defined - using a cheap proxy such as TF-IDF text similarity introduces failure modes from synonym blindness and annotation-style variance.

