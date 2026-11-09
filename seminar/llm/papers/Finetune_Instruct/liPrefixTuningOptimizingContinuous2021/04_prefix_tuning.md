# 4 Prefix-Tuning

We propose prefix-tuning as an alternative to fine-tuning for conditional generation tasks. We first provide intuition in §4.1 before defining our method formally in §4.2.

\footnotetext{
\({ }^{1} h_{i}^{(n)}\) is composed of a key-value pair. In GPT-2, the dimension of each key and value is 1024.
}

![Figure](./liPrefixTuningOptimizingContinuous2021_images/83a5514b-cc14-4b9b-9dd4-6bbee02942d4-04_530_1590_224_242.jpg)

**Figure 2: An annotated example of prefix-tuning using an autoregressive LM (top) and an encoder-decoder model (bottom). The prefix activations $\forall i \in \mathrm{P**_{\mathrm{idx}}, h_{i}$ are drawn from a trainable matrix $P_{\theta}$. The remaining activations are computed by the Transformer.}

## 4.1 Intuition

Based on intuition from prompting, we believe that having a proper context can steer the LM without changing its parameters. For example, if we want the LM to generate a word (e.g., Obama), we can prepend its common collocations as context (e.g., Barack), and the LM will assign much higher probability to the desired word. Extending this intuition beyond generating a single word or sentence, we want to find a context that steers the LM to solve an NLG task. Intuitively, the context can influence the encoding of $x$ by guiding what to extract from $x$; and can influence the generation of $y$ by steering the next token distribution. However, it's non-obvious whether such a context exists. Natural language task instructions (e.g., "summarize the following table in one sentence") might guide an expert annotator to solve the task, but fail for most pretrained LMs. ${ }^{2}$ Data-driven optimization over the discrete instructions might help, but discrete optimization is computationally challenging.

Instead of optimizing over discrete tokens, we can optimize the instruction as continuous word embeddings, whose effects will be propagated upward to all Transformer activation layers and rightward to subsequent tokens. This is strictly more expressive than a discrete prompt which requires matching the embedding of a real word. Meanwhile, this is less expressive than intervening all layers of the activations (§7.2), which avoids long-range dependencies and includes more tunable parameters. Prefix-tuning, therefore, optimizes all layers of the prefix.

\footnotetext{
${ }^{2}$ In our preliminary experiments, GPT-2 and BART fail in this setting; the only exception is GPT-3.
}

## 4.2 Method

Prefix-tuning prepends a prefix for an autoregressive LM to obtain $z=[$ PrefiX $; x ; y]$, or prepends prefixes for both encoder and encoder to obtain $z=\left[\right.$ Prefix; $x ;$ Prefix $\left.^{\prime} ; y\right]$, as shown in Figure 2. Here, $\mathrm{P}_{\mathrm{idx}}$ denotes the sequence of prefix indices, and we use $\left|\mathrm{P}_{\text {idx }}\right|$ to denote the length of the prefix.

We follow the recurrence relation in equation (1), except that the prefix are free parameters. Prefix-tuning initializes a trainable matrix $P_{\theta}$ (parametrized by $\theta$ ) of dimension \(\left|\mathrm{P}_{\mathrm{idx}}\right| \times \operatorname{dim}\left(h_{i}\right)\) to store the prefix parameters.
$$
h_{i}= \begin{cases}P_{\theta}[i,:], & \text { if } i \in \mathrm{P}_{\mathrm{idx}} \\ \operatorname{LM}_{\phi}\left(z_{i}, h_{<i}\right), & \text { otherwise }\end{cases}
$$
The training objective is the same as equation (2), but the set of trainable parameters changes: the language model parameters $\phi$ are fixed and the prefix parameters $\theta$ are the only trainable parameters.

Here, $h_{i}$ (for all $i$ ) is a function of the trainable $P_{\theta}$. When $i \in \mathrm{P}_{\mathrm{idx}}$, this is clear because $h_{i}$ copies directly from $P_{\theta}$. When $i \notin \mathrm{P}_{\mathrm{idx}}, h_{i}$ still depends on $P_{\theta}$, because the prefix activations are always in the left context and will therefore affect any activations to its right.

## 4.3 Parametrization of $P_{\theta$}

Empirically, directly updating the $P_{\theta}$ parameters leads to unstable optimization and a slight drop in performance. ${ }^{3}$ So we reparametrize the matrix \(P_{\theta}[i,:]=\operatorname{MLP}_{\theta}\left(P_{\theta}^{\prime}[i,:]\right)\) by a smaller matrix \(\left(P_{\theta}^{\prime}\right)\) composed with a large feedforward neural network \(\left(\mathrm{MLP}_{\theta}\right)\). Note that $P_{\theta}$ and $P_{\theta}^{\prime}$ has the same rows

\footnotetext{
${ }^{3}$ We find in preliminary experiments that directly optimizing the prefix is very sensitive to the learning rate and initialization.
}
dimension (i.e. the prefix length), but different columns dimension. ${ }^{4}$ Once training is complete, these reparametrization parameters can be dropped, and only the prefix \(\left(P_{\theta}\right)\) needs to be saved.

