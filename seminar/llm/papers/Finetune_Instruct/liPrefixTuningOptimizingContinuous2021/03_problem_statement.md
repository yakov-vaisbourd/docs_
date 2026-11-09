# 3 Problem Statement

Consider a conditional generation task where the input is a context $x$ and the output $y$ is a sequence of tokens. We focus on two tasks, shown in Figure 2 (right): In table-to-text, $x$ corresponds to a linearized data table and $y$ is a textual description; in summarization, $x$ is an article and $y$ is a short summary.

## 3.1 Autoregressive LM

Assume we have an autoregressive language model \(p_{\phi}(y \mid x)\) based on the Transformer (Vaswani et al., 2017) architecture (e.g., GPT-2; Radford et al., 2019) and parametrized by $\phi$. As shown in Figure 2 (top), let $z=[x ; y]$ be the concatenation of $x$ and $y$; let $\mathrm{X}_{\mathrm{idx}}$ denote the sequence of indices that corresponds to $x$, and $\mathrm{Y}_{\text {idx }}$ denote the same for $y$.

The activation at time step $i$ is $h_{i} \in \mathbb{R}^{d}$, where \(h_{i}=\left[h_{i}^{(1)} ; \cdots ; h_{i}^{(n)}\right]\) is a concatenation of all activation layers at this time step, and \(h_{i}^{(j)}\) is the activation of the $j$-th Transformer layer at time step $i .{ }^{1}$

The autoregressive Transformer model computes $h_{i}$ as a function of $z_{i}$ and the past activations in its left context, as follows:
$$
h_{i}=\operatorname{LM}_{\phi}\left(z_{i}, h_{<i}\right),
$$
where the last layer of $h_{i}$ is used to compute the distribution for the next token: \(p_{\phi}\left(z_{i+1} \mid h_{\leq i}\right)=\) \(\operatorname{softmax}\left(W_{\phi} h_{i}^{(n)}\right)\) and $W_{\phi}$ is a pretrained matrix that map \(h_{i}^{(n)}\) to logits over the vocabulary.

## 3.2 Encoder-Decoder Architecture

We can also use an encoder-decoder architecture (e.g., BART; Lewis et al., 2020) to model \(p_{\phi}(y \mid x)\), where $x$ is encoded by the bidirectional encoder, and the decoder predicts $y$ autoregressively (conditioned on the encoded $x$ and its left context). We use the same indexing and activation notation, as shown in Figure 2 (bottom). $h_{i}$ for all $i \in \mathrm{X}_{\mathrm{idx}}$ is computed by the bidirectional Transformer encoder; $h_{i}$ for all $i \in \mathrm{Y}_{\text {idx }}$ is computed by the autoregressive decoder using the same equation (1).

## 3.3 Method: Fine-tuning

In the fine-tuning framework, we initialize with the pretrained parameters $\phi$. Here $p_{\phi}$ is a trainable language model distribution and we perform gradient updates on the following log-likelihood objective:
$$
\max _{\phi} \log p_{\phi}(y \mid x)=\sum_{i \in \mathrm{Y}_{\mathrm{idx}}} \log p_{\phi}\left(z_{i} \mid h_{<i}\right) .
$$

