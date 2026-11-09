# Autoregressive Generation Cycle: Token IDs → Latent Space → Logits

This diagram illustrates the complete cycle of token generation in transformer-based LLMs, emphasizing that:
- **Token IDs** (integers) are the discrete outputs generated at each step
- **Latent vectors** (embeddings) are the continuous representations that flow through the transformer
- **Attention and MLPs** operate entirely in latent space
- The model generates discrete tokens autoregressively by sampling from logits, which are computed from latent representations

```mermaid
graph TD
    Start["Start of sequence<br/>(or prior generated token)"]
    
    Start --> EmbedStep["Embedding Layer<br/>(ID → vector lookup)"]
    
    EmbedStep --> LatentIn["Latent vectors<br/>(embedded token IDs)"]
    
    LatentIn --> AttentionQ["Transformer Block<br/>Query, Key, Value<br/>(projections of latents)"]
    
    AttentionQ --> Attention["Attention<br/>softmax(QK^T/√d) · V<br/>(operates on latent vectors)"]
    
    Attention --> LatentAfterAttn["New latent vectors<br/>(attention output)"]
    
    LatentAfterAttn --> MLP["MLP Layer<br/>(operates on latents)"]
    
    MLP --> LatentAfterMLP["Updated latent vectors<br/>(MLP output)"]
    
    LatentAfterMLP --> MoreBlocks{"More<br/>blocks?"}
    
    MoreBlocks -->|Yes| AttentionQ
    MoreBlocks -->|No| LMHead["LM Head<br/>(linear projection)"]
    
    LMHead --> Logits["Logits over vocabulary<br/>(V scores, one per token ID)"]
    
    Logits --> Softmax["Softmax"]
    
    Softmax --> Distribution["Probability distribution<br/>over V tokens"]
    
    Distribution --> Sample["Sampling Strategy<br/>(greedy / top-k / top-p / temperature)"]
    
    Sample --> TokenID["Selected token ID<br/>(discrete output)"]
    
    TokenID --> Feedback["Feed back as<br/>next input"]
    
    Feedback --> EmbedStep
    
    TokenID --> Output["Add to output<br/>sequence"]
    
    style LatentIn fill:#e8f4f8
    style LatentAfterAttn fill:#e8f4f8
    style LatentAfterMLP fill:#e8f4f8
    style Attention fill:#f0e8f8
    style MLP fill:#f0e8f8
    style TokenID fill:#f8ede8
    style Logits fill:#f8ede8
    style Distribution fill:#f8ede8
```

## Key Points

| Component | Input | Output | Space |
|-----------|-------|--------|-------|
| Embedding | Token ID (integer) | Latent vector | Continuous |
| Attention | Latent vectors | Latent vectors | Continuous |
| MLP | Latent vectors | Latent vectors | Continuous |
| LM Head | Latent vectors | Logits | Continuous |
| Sampling | Logits (probabilities) | Token ID (integer) | Discrete |

**The token ID is the only discrete object.** Everything else—embeddings, attention weights, layer norm, MLP outputs, logits—is continuous and lives in latent space. Attention never sees token IDs; it only sees and combines embedded latent vectors based on their similarity in that space.

The feedback loop closes when a newly sampled token ID is embedded and fed back as input to the next forward pass.
