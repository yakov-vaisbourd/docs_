# Architecture

## Multi-Modal Large Language Models (MLLMs)

LLMs extended to accept non-text modalities (images, video, audio) as additional token sequences. Three components are composed: a frozen visual encoder, a cross-modality projector, and a pre-trained LLM backbone.

**Visual encoder:** A Vision Transformer (ViT) pre-trained with a contrastive objective (CLIP, SigLIP, LanguageBind) maps an image or video to a sequence of patch embeddings $\mathbf{V} \in \mathbb{R}^{N_v \times d_v}$, where $N_v$ is the number of visual tokens and $d_v$ is the encoder hidden size. The encoder is pre-trained to align its embedding space with natural language, so its features are already partially language-compatible before any multimodal training.

**Cross-modality projector:** A lightweight module maps $\mathbf{V}$ into the LLM's token space $\mathbb{R}^{N_v \times d_{llm}}$. Common choices: (1) a two-layer MLP (used in LLaVA), (2) a Querying Transformer (Q-Former in BLIP-2) that compresses $N_v$ patch tokens into a smaller fixed set of learned query tokens via cross-attention, reducing the context budget consumed by the visual input. The projected tokens $\hat{\mathbf{V}}$ are then concatenated with the text token sequence and processed by the LLM as a single flat sequence.

**LLM backbone:** A decoder-only causal LM (e.g., Vicuna, LLaMA, Mistral). It attends to the concatenated $[\hat{\mathbf{V}};\ \mathbf{T}]$ sequence and autoregressively generates the response.

**Alignment training.** Bridging the representation gap between a visual encoder and an LLM is done in two stages:

**Stage 1 - Feature alignment:** Only the projector weights are trained; the visual encoder and LLM are frozen. The training signal is large-scale image-caption pairs: the model must predict the caption given the projected visual tokens. This forces the projector to translate visual features into tokens the frozen LLM can interpret. No instruction-following ability is acquired here.

**Stage 2 - Visual instruction tuning:** The projector remains trainable; the LLM backbone is either fully unfrozen or updated via LoRA. Training uses a curated set of (image, instruction, response) triples - e.g., GPT-4V-generated conversations about images (LLaVA-Instruct), visual QA datasets, or chart/document understanding data. The model learns to follow instructions grounded in visual content. The visual encoder is often kept frozen throughout both stages to preserve its pre-trained alignment and avoid overfitting.

**Cross-attention vs.\ token-concatenation.** The token-concatenation approach above (LLaVA-style) feeds visual tokens directly into the causal attention layers of the LLM. An alternative is to introduce dedicated cross-attention layers interleaved with the LLM's self-attention layers (Flamingo-style): the LLM's text stream attends to visual features via cross-attention at each layer, with the visual encoder and cross-attention weights trained together. Cross-attention architectures support arbitrarily many images without linearly expanding the context window, but require architectural modification of the backbone.

**Modality-specific encoders.** Audio and video extend the same pipeline: a temporal or spectrogram encoder (e.g., a video ViT operating on sampled frames, or a Whisper encoder for audio) replaces or supplements the image ViT, and a per-modality projector maps into the shared LLM token space.

## Autoregressive Generation

The model generates output tokens one at a time, each conditioned on all previous tokens. Because generation is sequential, inference latency scales with output length. Batched inference (vLLM, TGI) and quantisation (INT8/FP16) are standard mitigation techniques.

## KV Cache

In transformer self-attention every token attends to all previous tokens. During autoregressive generation, naively recomputing the key ($K$) and value ($V$) projections for all prior tokens at each step would cost $O(L)$ per new token and $O(L^2)$ total for a sequence of length $L$.

The KV cache stores the $K$ and $V$ projections for all previously processed tokens. Each new token computes only its own projections and attends to the cached ones, reducing per-step cost to $O(1)$ in the sequence length. The query ($Q$) projection is not cached because it is only needed at the current step.

**Prompt caching:** If the same prefix (system prompt, retrieved context, few-shot examples) is reused across many requests, the KV cache for that prefix can be computed once and reused, eliminating redundant prefill computation across calls.

**Beam search:** All $C$ continuations branching from the same beam state $s_i$ share the KV cache of the common prefix (computed once). Only tokens generated after the branch point require fresh computation. See \S*sec-pav*.

**Memory cost:** Cache size grows as $O(L \cdot n_{\text{heads}} \cdot d_{\text{head}} \cdot n_{\text{layers}})$ per sequence. At long context lengths or large batch sizes this becomes the dominant memory consumer, motivating grouped-query attention (GQA) and multi-query attention (MQA), which reduce the number of distinct K/V heads.

## Context Window

The maximum number of tokens a model can attend to in one forward pass. Each ICL example or retrieved document consumes tokens from this budget. For Vicuna 1.5 7B the limit is 4096 tokens; modern open-source models (Llama 3.1, Qwen 2.5) support 128k tokens. Context window size is the primary architectural constraint on how many ICL examples can be used.

