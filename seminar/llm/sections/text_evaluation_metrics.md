# Text Evaluation Metrics

## TF-IDF

Term Frequency-Inverse Document Frequency assigns each word $w$ in document $d$ a score that rewards words distinctive to that document. Given a corpus of $D$ documents:

$$
\begin{aligned}
\mathrm{TF}(w, d) = \frac{\text{count of } w \text{ in } d}{|d|} \qquad
    \mathrm{IDF}(w) = \log\frac{D}{|\{d : w \in d\}|}
\end{aligned}
$$

$$
\begin{aligned}
\mathrm{TF\text{-}IDF}(w, d) = \mathrm{TF}(w, d) \cdot \mathrm{IDF}(w)
\end{aligned}
$$

Each document is represented as a sparse vector over the full vocabulary, with dimension $w$ holding $\mathrm{TF\text{-}IDF}(w,d)$ (zero if $w$ absent). Two documents are compared by cosine similarity of these vectors.

**Example.** Suppose the corpus has 1000 documents, and ``photosynthesis'' appears in only 2. Then $\mathrm{IDF} = \log(500) \approx 6.2$, so a single occurrence yields a high score. ``The'' appears in all 1000 documents: $\mathrm{IDF} = \log(1) = 0$, contributing nothing regardless of frequency.

**Limitations.** Purely lexical: ``purchase'' and ``buy'' are orthogonal vocabulary dimensions. Sensitive to annotation style - two authors describing the same event with different vocabulary produce near-zero cosine similarity.

## BLEU (2002)

**Modified n-gram precision.** Let $C$ be the generated text and $R$ the set of references. For each n-gram $g$, the clipped count caps occurrences in $C$ at the maximum count in any reference: $\mathrm{Count}_\mathrm{clip}(g) = \min(\mathrm{Count}(g,C),\, \max_{r \in R}\mathrm{Count}(g,r))$. The modified n-gram precision is:

$$
\begin{aligned}
p_n = \frac{\displaystyle\sum_{g \in C} \mathrm{Count}_\mathrm{clip}(g)}{\displaystyle\sum_{g \in C} \mathrm{Count}(g, C)}
\end{aligned}
$$

The clipping prevents inflating precision by repeating a single common word.

**Brevity penalty.** Short outputs have high precision trivially. BLEU multiplies by $\mathrm{BP} = \min(1,\, e^{1 - r/c})$, where $c = |C|$ and $r = |R|$. The final score is:

$$
\begin{aligned}
\mathrm{BLEU}\text{-}N = \mathrm{BP} \cdot \exp\!\left(\frac{1}{N}\sum_{n=1}^{N} \log p_n\right) = \mathrm{BP} \cdot \left(\prod_{n=1}^{N} p_n\right)^{1/N}
\end{aligned}
$$

Standard usage is $N=4$. The geometric mean collapses to zero if any $p_n = 0$, so the score can only be non-zero if every n-gram order has at least one match.

**Example.** Generated: ``the cat sat on the mat''. Reference: ``the cat is on the mat''.
- $p_1$: matched 1-grams are \{the, cat, on, the, mat\}; clipped count for ``the'' is $\min(2,2)=2$. Total matched = 5, total generated = 6. $p_1 = 5/6$.
- $p_2$: matched 2-grams are ``the cat'', ``on the'', ``the mat''. $p_2 = 3/5$.
- $p_3$: only ``on the mat'' matches. $p_3 = 1/4$.
- $p_4$: no 4-gram matches. $p_4 = 0$, so BLEU-4 = 0.

One wrong word at the wrong position collapses the 4-gram score entirely.

**Limitations.** Precision-only view of quality (brevity penalty is a coarse recall proxy). All n-grams weighted equally. No synonym handling.

## METEOR (2005)

Directly addresses BLEU's lack of recall and synonymy. Alignment between $C$ and reference $R$ proceeds in three passes in order of preference: (1) exact token match, (2) stemmed match (``running'' $\leftrightarrow$ ``run''), (3) WordNet synonym match. Each token in $C$ is aligned to at most one token in $R$. Let $m$ be the number of aligned tokens:

$$
\begin{aligned}
P = \frac{m}{|C|} \qquad R = \frac{m}{|R|}
\end{aligned}
$$

$$
\begin{aligned}
F_\alpha = \frac{P \cdot R}{\alpha P + (1-\alpha)R} \qquad (\alpha = 0.9,\text{ recall-weighted})
\end{aligned}
$$

A fragmentation penalty discounts many short non-contiguous matched chunks versus one contiguous match. Let $ch$ be the number of matched chunks:

$$
\begin{aligned}
\mathrm{METEOR} = F_\alpha \cdot \left(1 - \gamma\left(\frac{ch}{m}\right)^\beta\right) \qquad (\gamma=0.5,\ \beta=3)
\end{aligned}
$$

**Example.** Generated: ``the vehicle brakes''. Reference: ``the car stops''. BLEU-1 = 1/3 (only ``the'' matches). METEOR additionally aligns ``brakes'' $\to$ ``stops'' via WordNet, giving $m=2$, $P = 2/3$, $R = 2/3$, and a much higher score.

**Limitation.** WordNet is a general English resource; domain-specific paraphrases are missed.

## CIDEr (2015)

Designed for captioning tasks where multiple reference captions are available. Applies TF-IDF weighting to n-grams (not individual words) and scores against the full consensus set of references.

For each n-gram $\omega$ of order $n$, compute its TF-IDF weight in generated caption $c_i$ and reference $s_{ij}$ as a vector $\mathbf{g}^n(\cdot)$ over the n-gram vocabulary. The CIDEr-$n$ score is the average cosine similarity over all $m$ references:

$$
\begin{aligned}
\mathrm{CIDEr}_n(c_i, S_i) = \frac{1}{m} \sum_{j=1}^{m} \frac{\mathbf{g}^n(c_i) \cdot \mathbf{g}^n(s_{ij})}{\|\mathbf{g}^n(c_i)\|\; \|\mathbf{g}^n(s_{ij})\|}
\end{aligned}
$$

The final score averages over $n = 1,\ldots,4$ (uniform weights), and is conventionally multiplied by 10:

$$
\begin{aligned}
\mathrm{CIDEr}(c_i, S_i) = \frac{10}{4}\sum_{n=1}^{4} \mathrm{CIDEr}_n(c_i, S_i)
\end{aligned}
$$

Typical scores are in the range 0-200+.

**Example.** In a corpus of 10{,}000 captions, ``the car moves'' appears in thousands - near-zero IDF, near-zero contribution. ``Pedestrian crossing ahead'' appears in 20 captions - high IDF, dominates the score when matched. BLEU would count both n-grams equally.

**Limitation.** Still purely lexical. A caption that paraphrases common corpus phrases while being factually wrong can score high; a correct but unusually worded caption can score low.

## Common Failure and Modern Alternatives

BLEU, METEOR, and CIDEr all measure surface text similarity, not semantic or factual correctness. A fluent but wrong answer outscroes a correct but unusually worded one if the former better matches corpus vocabulary. Recent work replaces or supplements them with LLM-as-judge evaluation using structured rubrics, and with task-specific accuracy metrics such as exact-match or verifiable answer checking.

