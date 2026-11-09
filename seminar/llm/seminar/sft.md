---
marp: true
theme: sft
paginate: true
math: katex
---

<!-- Slide 1 -->
<!-- _class: title-slide -->

<div class="kicker">Large Language Models</div>

# Supervised Fine-Tuning
# (SFT)

<div class="byline">Teaching LLMs to Formulate Text &amp; Follow Instructions<br/>Yakov &middot; September 2026</div>

---

<!-- Slide 2 -->
<!-- _class: center-body -->
# Two Purposes of SFT

<div class="mini-cols mini-cols-lg">
  <div class="mini-col">
    <div class="mini-title">Alignment SFT</div>
    <p class="mini-note">Base completion model &rarr; general instruction-following assistant.</p>
    <p class="small-note">Teaches the model to act on instructions, not just continue text. Foundational, one-time.</p>
  </div>
  <div class="mini-col">
    <div class="mini-title">Domain / Application SFT</div>
    <p class="mini-note">Instruction-tuned model &rarr; specialized behavior for one product.</p>
    <p class="small-note">Already follows instructions — this adds <em>your</em> schema, vocabulary, and edge cases.</p>
  </div>
</div>

<div class="landscape-note">Same mechanism both times (masked next-token loss on instruction data) — just a different starting checkpoint and goal.</div>

---


<!-- Slide 3 -->****
<!-- _class: center-body -->
# The LLM Alignment Pipeline

<div class="flow-steps">
  <div class="step">Pre-training<br/><small>(Base Model)</small></div>
  <div class="arrow">&rarr;</div>
  <div class="step highlight">SFT<br/><small>(Instruction Model)</small></div>
  <div class="arrow">&rarr;</div>
  <div class="step">RLHF / DPO<br/><small>(Human Preferences)</small></div>
</div>

- **Pre-training:** web-scale raw knowledge and language structure acquisition.
- **SFT:** teaching interaction formats, tasks, and system behaviors.
- **RLHF/DPO:** micro-tuning behavior to meet safety standards and human feedback.

---

<!-- Slide 4 -->
# Resource Cost Across the Pipeline

<div class="decode-table-wrap">
  <table class="decode-table">
    <thead>
      <tr><th>Stage</th><th>Data</th><th>Training Compute</th></tr>
    </thead>
    <tbody>
      <tr><td>Pre-training</td><td>14.8T tokens (DeepSeek-V3) / 15.6T tokens (Llama 3 405B) — scraped web text, ~free to collect</td><td>2,664K H800-hrs &rarr; $5.328M <small>(DeepSeek-V3)</small></td></tr>
      <tr><td>Post-training<br/><small>(SFT + RLHF/DPO)</small></td><td>1.5M curated instructions <small>(DeepSeek-V3)</small>; 6 iterative rounds of human annotation + rejection sampling <small>(Llama 3)</small></td><td>5K H800-hrs &rarr; $0.01M <small>(DeepSeek-V3)</small></td></tr>
    </tbody>
  </table>
</div>

<div class="landscape-note">&gt;500&times; cheaper than pre-training, by DeepSeek-V3's own reported figures — but note both labs report SFT and RLHF/DPO together as one "post-training" budget, not split apart: production pipelines interleave multiple rounds of both (Llama 3 ran six) rather than running each stage once.</div>

---

<!-- Slide 5 -->
# Pre-training vs. SFT (The Mindset Shift)

<div class="verdict-grid paired">
  <div class="verdict-pair">
    <div class="verdict-box bad">
      <div class="verdict-label">Base LLM — Text Completer</div>
      <p><strong>Prompt:</strong> Write a python function for quicksort.</p>
      <p><strong>Completion:</strong> Write a python function for mergesort.<br/>Write a python function for binary search.<br/>Write a python function for&hellip;</p>
    </div>
    <p class="verdict-clarification"><em>Chaos pattern:</em> treats the prompt as one item in a worksheet, not a request — keeps completing the list instead of answering it.</p>
  </div>
  <div class="verdict-pair">
    <div class="verdict-box good">
      <div class="verdict-label">SFT LLM — Helpful Assistant</div>
      <p><strong>Prompt:</strong> Write a python function for quicksort.</p>
      <p><strong>Completion:</strong><br/><code>def quicksort(arr): ... return arr</code></p>
    </div>
    <p class="verdict-clarification"><em>Alignment pattern:</em> transitions from pure raw completion to playing the persona of an assistant.</p>
  </div>
</div>

---

<!-- Slide 6 -->
# Domain SFT in Practice: An Autonomous-Driving Copilot

<div class="verdict-grid paired">
  <div class="verdict-pair">
    <div class="verdict-box bad">
      <div class="verdict-label">General Instruct Model</div>
      <p><strong>Prompt:</strong> Pedestrian at crosswalk distracted; truck in bike lane 20m ahead.</p>
      <p><strong>Completion:</strong> Moderate risk — distracted pedestrian near the crosswalk, truck blocking the bike lane ahead. Recommend slowing down.</p>
    </div>
    <p class="verdict-clarification"><em>Limitation:</em> correct but free-form prose — not parseable by a planner.</p>
  </div>
  <div class="verdict-pair">
    <div class="verdict-box good">
      <div class="verdict-label">Domain-SFT'd Driving Copilot</div>
      <p><strong>Prompt:</strong> (same)</p>
      <p><strong>Completion:</strong><br/>
      <code>{"actors": [</code><br/>
      <code>&nbsp;&nbsp;{"type": "VRU_pedestrian", "risk": "high"},</code><br/>
      <code>&nbsp;&nbsp;{"type": "static_obstacle", "risk": "medium"}</code><br/>
      <code>], "action": "reduce speed; widen clearance"}</code></p>
    </div>
    <p class="verdict-clarification"><em>Result:</em> same instruction-following base model, fine-tuned on the AV stack's own scenario logs to speak its schema and vocabulary.</p>
  </div>
</div>

<div class="landscape-note">The model didn't need to be taught to follow instructions here — it already could. Domain SFT only had to teach it <em>this</em> team's risk taxonomy and output format.</div>

---

<!-- Slide 7 -->
# SFT Data: Not Free, Not Web-Scale

<div class="mini-cols mini-cols-lg">
  <div class="mini-col">
    <div class="mini-title">Pre-training Data</div>
    <p class="mini-note">Scraped from the open web — Common Crawl, books, code repos.</p>
    <p class="small-note">Trillions of tokens, collected essentially for free. Quality is uneven; the model learns language and knowledge from sheer volume.</p>
  </div>
  <div class="mini-col">
    <div class="mini-title">SFT Data</div>
    <p class="mini-note">Written or generated on purpose — by humans, or by a stronger "teacher" model, then filtered.</p>
    <p class="small-note">Thousands to low hundreds-of-thousands of examples. Every example costs real annotation or generation-plus-review effort.</p>
  </div>
</div>

<div class="decode-table-wrap">
  <table class="decode-table">
    <thead><tr><th>Dataset</th><th>Size</th><th>How it was made</th></tr></thead>
    <tbody>
      <tr><td>LIMA</td><td>1,000</td><td>Hand-curated by experts</td></tr>
      <tr><td>Dolly</td><td>15,000</td><td>Human-written by employees</td></tr>
      <tr><td>Alpaca</td><td>52,000</td><td>Self-Instruct — generated by GPT, lightly filtered</td></tr>
      <tr><td>Llama 2 (SFT)</td><td>~27,000</td><td>Vendor-curated, reported as "quality over quantity"</td></tr>
      <tr><td>OpenAssistant</td><td>~161,000</td><td>Crowd-sourced conversations</td></tr>
    </tbody>
  </table>
</div>

<div class="landscape-note">Orders of magnitude smaller than pre-training corpora — but rarely as small as LIMA's 1k in production. The next two slides explain what "curated" actually means, and why less can be more.</div>

---

<!-- Slide 8 -->
# The Anatomy of an Instruction Example

**ChatML / OpenAI JSON template structure**

```json
[
  {"role": "system", "content": "You are a helpful coding assistant."},
  {"role": "user", "content": "How do I reverse a list in Python?"},
  {"role": "assistant", "content": "You can reverse a list using `list.reverse()` or slicing `[::-1]`."}
]
```

<div class="formula-box">[Prompt] + [Assistant] &rarr; Loss</div>

<p class="slide-caption"><strong>The Gold Target</strong> — loss is calculated ONLY on the assistant response.</p>

- **System:** defines boundaries/tone.
- **User:** input instructions — part of the context, not the target.
- **Assistant:** the training target — this is what the loss actually optimizes.

---

<!-- Slide 9 -->
# Quality Over Quantity: The LIMA Effect

<div class="lima-chart">
  <div class="lima-row"><div class="lima-row-label">Curated LIMA (1k)</div><div class="lima-row-bar good">~high win rate</div></div>
  <div class="lima-row"><div class="lima-row-label">Web Scraps (100k+)</div><div class="lima-row-bar bad">~lower win rate</div></div>
</div>

<div class="frame counts">
  <div class="frame-label">Definition: Human Win Rate</div>
  Blind A/B tests where human experts compare two anonymous model outputs for the same prompt. If Model A wins 75/100 matchups, its win rate is 75%.
</div>

<div class="landscape-note"><strong>LIMA (Less Is More for Alignment):</strong> 1,000 meticulously filtered, high-quality instruction responses consistently outperform massive, multi-thousand-sample datasets full of noise and repetitive patterns — the "not too little, not too cheap" data from two slides ago, taken to its extreme.</div>

---

<!-- Slide 10 -->
# Training: Same Loss, Adapted Rules

<div class="frame">
  <code>&lt;|user|&gt; What is 2+2? &lt;|assistant|&gt;</code> <strong class="gold-answer"><code>The answer is 4.</code></strong>
</div>

<div class="formula-box">

$$
\mathcal{L} = -\sum_{t=i}^{N} \log P(x_t \mid x_{<t})
$$

</div>

<p class="slide-caption"><em>Identical to pre-training's next-token loss — where i is the first token of the assistant's answer</em></p>

- **No new objective:** SFT is the same cross-entropy next-token prediction as pre-training.
- **Masking:** the loss is only computed on assistant tokens — the model isn't penalized for "predicting" the prompt it was given.
- **Chat template tokens** (e.g. `<|im_start|>`/`<|im_end|>`) mark where each turn begins and ends, so the mask — and the model's own stop condition — has something reliable to key on.
- **Learning rate:** dropped 1-2 orders of magnitude vs. pre-training (~2e-5 full fine-tuning, ~1e-4 LoRA) — we're adapting a trained model, not learning from scratch.

---

<!-- Slide 11 -->
# Prompt Engineering vs. Domain SFT

<div class="decode-table-wrap">
  <table class="decode-table">
    <thead>
      <tr><th>Metric</th><th>Prompt Engineering (Few-Shot)</th><th>Supervised Fine-Tuning (SFT)</th></tr>
    </thead>
    <tbody>
      <tr><td>Upfront Cost</td><td>$0 — immediate setup via API</td><td>High — data collection + GPU rent</td></tr>
      <tr><td>Cost Per Request</td><td>High — paying for examples every time</td><td>Minimal — short, clean prompts</td></tr>
      <tr><td>Format Reliability</td><td>Weak — model can drift or forget rules</td><td>Hardcoded — baked into model weights</td></tr>
    </tbody>
  </table>
</div>

<div class="landscape-note">Mostly a choice against <em>domain</em> SFT specifically (recall slide 3) — an already instruction-following model can often be steered by examples instead of retrained. Prompt engineering can also approximate alignment itself (few-shot, chat-style examples in context can make a base model act instruction-following) — but you pay for it on every request, and it's fragile; SFT bakes the behavior into the weights once.</div>

---

<!-- Slide 12 -->
# A Third Axis: In-Context Learning &amp; RAG

<div class="flow-steps">
  <div class="step">Query</div>
  <div class="arrow">&rarr;</div>
  <div class="step highlight">Retrieve<br/><small>relevant docs / examples</small></div>
  <div class="arrow">&rarr;</div>
  <div class="step">Inject into<br/>Context</div>
  <div class="arrow">&rarr;</div>
  <div class="step">Model Answers<br/><small>no weight update</small></div>
</div>

- **In-context learning:** few-shot examples or instructions placed directly in the prompt — no training, forgotten the moment the request ends.
- **RAG (Retrieval-Augmented Generation):** automates *what* goes into that context, pulling relevant passages from a knowledge base instead of hand-picking examples.

<div class="landscape-note">SFT bakes behavior into the weights — durable, cheap per request, but paid for upfront. RAG/ICL instead change what's in the context window — cheap to update (edit the knowledge base, not retrain), but you pay the retrieval and longer-context cost on every single request. Rule of thumb: reach for RAG when what's changing is <em>knowledge</em>; reach for SFT when what's changing is <em>behavior</em>.</div>

---

<!-- Slide 13 -->
# Full Fine-Tuning vs. PEFT (LoRA)

<div class="mini-cols mini-cols-lg">
  <div class="mini-col">
    <div class="mini-title">Full Fine-Tuning (FT)</div>
    <div class="delta-grid">
      <div class="delta-cell">&Delta;&theta;</div><div class="delta-cell">&Delta;&theta;</div><div class="delta-cell">&Delta;&theta;</div><div class="delta-cell">&Delta;&theta;</div>
      <div class="delta-cell">&Delta;&theta;</div><div class="delta-cell">&Delta;&theta;</div><div class="delta-cell">&Delta;&theta;</div><div class="delta-cell">&Delta;&theta;</div>
      <div class="delta-cell">&Delta;&theta;</div><div class="delta-cell">&Delta;&theta;</div><div class="delta-cell">&Delta;&theta;</div><div class="delta-cell">&Delta;&theta;</div>
    </div>
    <p class="mini-note">Every parameter updates. High risk of catastrophic forgetting; massive VRAM for gradients + optimizer states.</p>
  </div>
  <div class="mini-col">
    <div class="mini-title">PEFT — Variant, Not Alternative</div>
    <div class="lora-diagram">
      <div class="lora-frozen">Frozen</div>
      <div class="lora-op">+</div>
      <div class="lora-ba-col">
        <div class="lora-A">A</div>
      </div>
      <div class="lora-B end">B</div>
    </div>
    <p class="mini-note">Still SFT — same masked CE loss, same data. Only a tiny fraction of parameters (the LoRA matrices) actually train; the base model stays frozen.</p>
  </div>
</div>

---

<!-- Slide 14 -->
# LoRA Inside the Transformer

<div class="transformer-block">
  <div class="tb-layer">
    <div class="tb-layer-label">Multi-Head Self-Attention</div>
    <div class="tb-matrices">
      <span class="tb-matrix lora">W<sub>q</sub></span>
      <span class="tb-matrix lora">W<sub>k</sub></span>
      <span class="tb-matrix lora">W<sub>v</sub></span>
      <span class="tb-matrix lora">W<sub>o</sub></span>
    </div>
  </div>
  <div class="tb-layer">
    <div class="tb-layer-label">Feed-Forward (MLP)</div>
    <div class="tb-matrices">
      <span class="tb-matrix">W<sub>up</sub></span>
      <span class="tb-matrix">W<sub>down</sub></span>
    </div>
  </div>
</div>

<div class="formula-box">

$$
W_{\mathrm{new}} = W_0 + \frac{\alpha}{r}(B \times A), \quad A \in \mathbb{R}^{r \times d},\ B \in \mathbb{R}^{d \times r}
$$

</div>

- **Targeted, not global:** each adapted matrix gets its own low-rank pair (A, B) — LoRA is applied per-weight-matrix, not once to the whole model.
- **Attention first:** W<sub>q</sub>/W<sub>k</sub>/W<sub>v</sub>/W<sub>o</sub> are the standard targets — they control *what the model attends to*, which is usually enough to teach a new format or behavior.
- **MLP is optional:** the feed-forward matrices are larger (higher-dimensional) and hold more of the model's factual/world knowledge — adapting them costs more but helps when the task needs new knowledge pathways, not just re-weighted attention.

<p class="slide-caption">Highlighted = typical LoRA targets. Frozen weights throughout.</p>

---

<!-- Slide 15 -->
<!-- _class: title-slide -->

<div class="kicker">Archive</div>

# Archived From Here Down

<div class="byline">These slides are frozen for reference — not to be edited.<br/>New material is inserted above, between the working front section and this marker.</div>

---

<!-- Slide 16 -->
# Prompt Engineering vs. SFT: The Core Tradeoff

<div class="decode-table-wrap">
  <table class="decode-table">
    <thead>
      <tr><th>Metric</th><th>Prompt Engineering (Few-Shot)</th><th>Supervised Fine-Tuning (SFT)</th></tr>
    </thead>
    <tbody>
      <tr><td>Upfront Cost</td><td>$0 — immediate setup via API</td><td>High — data collection + GPU rent</td></tr>
      <tr><td>Cost Per Request</td><td>High — paying for examples every time</td><td>Minimal — short, clean prompts</td></tr>
      <tr><td>Format Reliability</td><td>Weak — model can drift or forget rules</td><td>Hardcoded — baked into model weights</td></tr>
    </tbody>
  </table>
</div>

<div class="landscape-note">Optimization path: cheap, ad-hoc customization (prompting) trades off against low per-request cost and durable behavior (SFT) as usage scales.</div>

---

<!-- Slide 17 -->
# The Anatomy of an Instruction Dataset

**ChatML / OpenAI JSON template structure**

```json
[
  {"role": "system", "content": "You are a helpful coding assistant."},
  {"role": "user", "content": "How do I reverse a list in Python?"},
  {"role": "assistant", "content": "You can reverse a list using `list.reverse()` or slicing `[::-1]`."}
]
```

<div class="formula-box">[Prompt] + [Assistant] &rarr; Loss</div>

<p class="slide-caption"><strong>The Gold Target</strong> — loss is calculated ONLY on the assistant response.</p>

- **System:** defines boundaries/tone.
- **User:** input instructions.
- **Assistant:** the training target.

---

<!-- Slide 18 -->
# Quality Over Quantity: The LIMA Effect

<div class="lima-chart">
  <div class="lima-row"><div class="lima-row-label">Curated LIMA (1k)</div><div class="lima-row-bar good">~high win rate</div></div>
  <div class="lima-row"><div class="lima-row-label">Web Scraps (100k+)</div><div class="lima-row-bar bad">~lower win rate</div></div>
</div>

<div class="frame counts">
  <div class="frame-label">Definition: Human Win Rate</div>
  Blind A/B tests where human experts compare two anonymous model outputs for the same prompt. If Model A wins 75/100 matchups, its win rate is 75%.
</div>

<div class="landscape-note"><strong>LIMA (Less Is More for Alignment):</strong> a tiny set of 1,000 meticulously filtered, high-quality instruction responses consistently outperforms massive, multi-thousand-sample datasets full of noise and repetitive patterns.</div>

---

<!-- Slide 19 -->
# Data Synthetic Generation (Self-Instruct)

<div class="flow-steps">
  <div class="step">Teacher Model<br/><small>(e.g. GPT-4)</small></div>
  <div class="arrow">&rarr;</div>
  <div class="step">Raw Prompt &amp;<br/>Response Generation</div>
  <div class="arrow">&rarr;</div>
  <div class="step">Filtering<br/><small>Heuristics &amp; LLM-eval</small></div>
  <div class="arrow">&rarr;</div>
  <div class="step highlight">Student Model<br/><small>SFT Engine</small></div>
</div>

<div class="landscape-note">The student model's outputs can loop back to seed the next generation round (dashed feedback in the original pipeline).</div>

- **Pipeline:** seed tasks &rarr; generation loops &rarr; deduplication checks &rarr; target training.
- **Pitfall:** risk of data echo chambers — repeating systemic hallucination footprints of the teacher.

---

<!-- Slide 20 -->
# The Necessity of Chat Templates

**The Challenge: Text Completion vs. Interaction**
- **Base model behavior:** predicts the next word in a sequence. Without structure, it treats a prompt as a *continuation* of text (e.g., finishing your question).
- **The goal:** turn an open-ended generator into a controlled *dialogue agent*.

<div class="verdict-grid">
  <div class="verdict-box good">
    <div class="verdict-label">Expectation (Controlled)</div>
    <code>User: 4+4?</code><br/><code>Assistant: 8</code><br/><code>[STOP TOKEN]</code>
  </div>
  <div class="verdict-box bad">
    <div class="verdict-label">Reality (Uncontrolled — Role Bleeding)</div>
    <code>User: 4+4?</code><br/><code>Assistant: 8</code><br/><strong><code>User: And what is 5+5?</code><br/><code>Assistant: 10</code></strong><br/><code>[STOP TOKEN]</code>
  </div>
</div>

---

<!-- Slide 21 -->
# The Solution: Chat Templates

**Enforcing boundaries with special tokens**

<div class="verdict-box good">
  <div class="verdict-label">Model respects the structure</div>
  <code>&lt;|im_start|&gt;User: What is 4+4?&lt;|im_end|&gt;</code><br/><code>&lt;|im_start|&gt;Assistant: 8&lt;|im_end|&gt;</code>
</div>

- **Containment:** the model is confined to the marked scope.
- **Control:** the `<|im_end|>` tag serves as an explicit "Stop" signal for the tokenizer.

<div class="frame step">
  <div class="frame-label">Crucial technical step</div>
  Tags must be registered as <strong>Special Tokens</strong> to prevent fragmentation.<br/>
  Fix: <code>tokenizer.add_special_tokens()</code> &rarr; <code>model.resize_token_embeddings()</code>.
</div>

---

<!-- Slide 22 -->
# Standard SFT Loss Function

<div class="frame">
  <code>&lt;|user|&gt; What is 2+2? &lt;|assistant|&gt;</code> <strong class="gold-answer"><code>The answer is 4.</code></strong>
</div>

<div class="formula-box">

$$
\mathcal{L} = -\sum_{t=i}^{N} \log P(x_t \mid x_{<t})
$$

</div>

<p class="slide-caption"><em>where i is the first token of the assistant's answer</em></p>

- **Masked multi-turn loss:** only the assistant's response contributes to the loss.
- **Gradient flow:** backpropagation is constrained to target tokens, forcing the model to learn answer generation, not prompt replication.

---

<!-- Slide 23 -->
# Full Fine-Tuning vs. PEFT

<div class="mini-cols">
  <div class="mini-col">
    <div class="mini-title">Full Fine-Tuning (FT)</div>
    <div class="delta-grid">
      <div class="delta-cell">&Delta;&theta;</div><div class="delta-cell">&Delta;&theta;</div><div class="delta-cell">&Delta;&theta;</div><div class="delta-cell">&Delta;&theta;</div>
      <div class="delta-cell">&Delta;&theta;</div><div class="delta-cell">&Delta;&theta;</div><div class="delta-cell">&Delta;&theta;</div><div class="delta-cell">&Delta;&theta;</div>
      <div class="delta-cell">&Delta;&theta;</div><div class="delta-cell">&Delta;&theta;</div><div class="delta-cell">&Delta;&theta;</div><div class="delta-cell">&Delta;&theta;</div>
    </div>
    <p class="mini-note">Modifies every single parameter. High risk of catastrophic forgetting. Massive VRAM requirements to maintain gradients and optimizer states.</p>
  </div>
  <div class="mini-col">
    <div class="mini-title">PEFT (LoRA)</div>
    <div class="lora-diagram">
      <div class="lora-frozen">Frozen</div>
      <div class="lora-op">+</div>
      <div class="lora-ba-col">
        <div class="lora-A">A</div>
      </div>
      <div class="lora-B end">B</div>
    </div>
    <p class="mini-note"><strong>Strategy:</strong> train only a tiny fraction of parameters. Base model remains locked (99%+ frozen). Injects trainable matrices A, B: &Delta;W = AB.</p>
  </div>
</div>

---

<!-- Slide 24 -->
# Inside LoRA (Low-Rank Adaptation)

<div class="formula-box">

$$
W_{\mathrm{new}} = W_0 + \Delta W = W_0 + \frac{\alpha}{r}(B \times A)
$$

</div>

<div class="lora-diagram">
  <div class="lora-frozen mini">W<sub>0</sub> &isin; &#8477;<sup>d&times;d</sup></div>
  <div class="lora-op">+</div>
  <div class="lora-A mini">A &isin; &#8477;<sup>d&times;r</sup></div>
  <div class="lora-op">&times;</div>
  <div class="lora-B mini">B &isin; &#8477;<sup>r&times;d</sup></div>
</div>

<div class="decode-table-wrap">
  <table class="decode-table">
    <thead><tr><th>Method</th><th>Trainable Params / Layer</th></tr></thead>
    <tbody>
      <tr><td>Full FT</td><td>~16.8M (d=4096)</td></tr>
      <tr><td>LoRA (r=8)</td><td>~65K &mdash; &asymp;256&times; reduction</td></tr>
    </tbody>
  </table>
</div>

- **Assumption:** target adjustments &Delta;W feature highly reduced intrinsic rank.
- **Setup:** A initializes Gaussian; B initializes at zero &rarr; &Delta;W = 0 at step zero.

---

<!-- Slide 25 -->
# Memory Layout: Quantization &amp; QLoRA

<div class="quant-bars">
  <div class="quant-bar fp32">FP32</div>
  <div class="quant-bar fp16">FP16</div>
  <div class="quant-bar nf4">NF4</div>
</div>
<div class="quant-bar-labels">
  <span class="quant-bar-label">4 Bytes</span>
  <span class="quant-bar-label">2 Bytes</span>
  <span class="quant-bar-label">0.5 Bytes</span>
</div>

**VRAM footprint for a standard 7B parameter model:**

<div class="decode-table-wrap">
  <table class="decode-table">
    <thead><tr><th>Precision Mode</th><th>Data Format</th><th>Weights Footprint</th><th>Total VRAM for SFT</th></tr></thead>
    <tbody>
      <tr><td>Full Precision</td><td>FP32</td><td>~28 GB</td><td>~112 GB (incl. states)</td></tr>
      <tr><td>Half Precision</td><td>FP16 / BF16</td><td>~14 GB</td><td>~40-50 GB</td></tr>
      <tr><td>QLoRA</td><td>NF4 (NormalFloat4)</td><td>~4.5 GB</td><td>&lt; 8 GB (consumer-grade GPU)</td></tr>
    </tbody>
  </table>
</div>

---

<!-- Slide 26 -->
# Hyperparameters That Actually Matter

<div class="mini-cols">
  <div class="mini-col">
    <div class="mini-title">Learning Rate</div>
    <div class="lr-bar"></div>
    <p class="small-note"><strong>Cosine decay schedule</strong> — high &rarr; low.<br/>~1e-4 for LoRA runs.<br/>~2e-5 for full FT runs.<br/>Magnitudes lower than pre-training rates.</p>
  </div>
  <div class="mini-col">
    <div class="mini-title">Batch Size</div>
    <div class="batch-boxes">
      <div class="batch-box"></div><div class="batch-box"></div><div class="batch-box"></div><div class="batch-box"></div>
    </div>
    <p class="small-note"><strong>Effective batch size:</strong> samples processed before updating weights (target: 32-128).<br/><strong>Gradient accumulation:</strong> sum gradients over mini-batches when GPU is too small, then update once.</p>
  </div>
  <div class="mini-col">
    <div class="mini-title">Context Strategy</div>
    <div class="context-strip"><div class="seg-full"></div><div class="seg-used"></div><div class="seg-pad"></div></div>
    <p class="small-note"><strong>Padding:</strong> wastes tensor space via zero allocation.<br/><strong>Packing:</strong> chains short strings sequentially, separated by EOS tokens, to maximize sequence efficiency.</p>
  </div>
</div>

---

<!-- Slide 27 -->
# Fine-tuning Challenges: Overfitting &amp; Forgetting

<div class="mini-cols">
  <div class="mini-col">
    <div class="mini-title">1. Overfitting</div>
    <p class="small-note">Model memorizes noise instead of patterns. Train loss keeps falling; validation loss falls, then turns back up — that turning point is where to stop.</p>
    <p class="small-note"><strong>Solution:</strong> Early Stopping</p>
  </div>
  <div class="mini-col">
    <div class="mini-title">2. Catastrophic Forgetting</div>
    <p class="small-note">New skills override base general intelligence.</p>
    <div class="stack-bar-wrap">
      <div class="stack-bar">
        <div class="stack-seg anchor">10% Web Tokens</div>
        <div class="stack-seg sft">90% SFT Dataset</div>
      </div>
    </div>
    <p class="small-note"><strong>Mitigation:</strong> Data Rehearsal</p>
  </div>
</div>

---

<!-- Slide 28 -->
# How to Evaluate an SFT Model?

<div class="bottleneck-grid">
  <div class="bottleneck-tile">
    <div class="bottleneck-title">Benchmarks</div>
    <div class="bottleneck-sub">MMLU, GSM8K, HumanEval<br/><em>(rigid metrics / ground-truth match)</em></div>
  </div>
  <div class="bottleneck-tile">
    <div class="bottleneck-title">LLM-as-a-Judge</div>
    <div class="bottleneck-sub">GPT-4 Evaluation, MT-Bench<br/><em>(relative quality checks using strong models)</em></div>
  </div>
  <div class="bottleneck-tile">
    <div class="bottleneck-title">Human Evaluation</div>
    <div class="bottleneck-sub">Side-by-side A/B testing<br/><em>(gold standard verification; resource heavy)</em></div>
  </div>
</div>

<div class="landscape-note"><strong>Legacy text metrics (ROUGE / BLEU):</strong> mostly ineffective for abstract alignment tracking — language answers change shape without affecting accuracy.</div>
