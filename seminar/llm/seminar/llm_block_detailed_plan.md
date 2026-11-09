# LLM Block — Detailed Session Plan (Lectures 6-9)

Companion to [[syllabus]]. Expands the four LLM sessions (assigned to Oryan, Yakov, Itamar, Itay) into a
lecture-by-lecture run sheet: sub-topics with time budgets, prerequisite reading, and pointers into the
vault's existing [[llm-main]] notes so speakers don't duplicate research that's already written up.

**Session length:** ~90 min total. **No hands-on labs** — per the RL block's actual experience, the
scheduled coding labs never happened in practice, and there's no plan to insist on them here either. All
90 minutes go to theory/discussion, which also resolves the pacing squeeze flagged below for Itay's
41-slide draft and the SFT deck: with the lab time folded back in, there's room for the full decks plus
live Q&A without needing to cut content. The **pacing flags** on Lectures 6 and 8 are left in as
"don't rush past this" notes for delivery pacing, not as pressure to cut slides.

---

## Lecture 6 — The Foundation (Itay)

**Goal:** Understand what goes into a model before any training happens — data, tokenization, and
(per Itay's current draft, see [[presentation]]) a full walkthrough of embeddings, attention, and
positional encoding as well.

> **Status:** Itay already has a working draft at `4_docs/llm/seminar/presentation.md` (41 slides). It
> goes well beyond "data + tokenization" — it also covers token embeddings, the full self-attention
> mechanism (bottleneck motivation, Q/K/V, scaled dot-product + causal masking, softmax, weighted-sum
> accumulation, multi-head attention and why concatenation beats averaging), absolute positional
> encoding with a worked numeric example and its extrapolation limitation, and decoding/detokenization
> strategies (greedy, temperature, top-k, top-p). This is a lot of ground for one ~60 min slot — see the
> **pacing flag** below and the note on Lecture 7 overlap.

### Theory breakdown as currently drafted (~35-40 min if tightly paced)
1. **Data pipeline** (~10 min) — raw crawler data → FineWeb-style filtering pipeline → the wider dataset
   landscape (pretraining / instruction-tuning / preference / eval / multimodal). Matches
   [[data_preparation_and_tokenization]] § *Pretraining Data Collection and Filtering* and § *Data
   Mixture, Upsampling, and Epochs* — draft slides 2-6.
2. **Tokenization** (~10 min) — the character-vs-word tradeoff, Byte-Pair Encoding merge loop with a
   worked `low/lower/newest/widest` example, chat-template special tokens, live Tiktokenizer demo.
   Matches [[data_preparation_and_tokenization]] §§ *Choosing a Tokenization Granularity* through
   *Vocabulary Size Tradeoff* — draft slides 7-16.
3. **Embeddings** (~3 min) — token ID → learned lookup-table row, why static embeddings alone can't
   disambiguate a word like "train" or "bank" — draft slides 17-20.
4. **Self-attention + multi-head attention** (~12-15 min) — Q/K/V, scaled dot-product, causal masking,
   softmax, value accumulation, then why one head isn't enough and how multi-head splits the embedding
   depth (not the sentence) — draft slides 20-30. This is architecturally identical to what a "core
   attention" section of Lecture 7 would otherwise cover, so treat draft slides 20-30 as the canonical
   source and have Itamar build on top of it rather than re-deriving it.
5. **Positional encoding** (~5 min) — why order matters, absolute (sinusoidal) PE with a worked
   `pos=1, dmodel=4096` example, its extrapolation limitation past the trained context length. RoPE is
   named as the alternative but not derived in the current draft. **Decision:** RoPE is not Itamar's to
   pick up (see Lecture 7 — it's been removed from there to avoid the lecture sprawling). It's either a
   **nice-to-have bonus slide for Itay** here (a few sentences: rotate Q/K by an angle proportional to
   position, generalizes better to unseen lengths than absolute PE's fixed table) if he wants to close
   the loop on the alternative he already names, or it's simply **skipped** for the seminar — leaving
   RoPE as a named-but-undetailed alternative is a perfectly fine place to stop.
6. **Decoding & detokenization** (~5 min) — greedy / temperature / top-k / top-p compared, then
   detokenization as BPE's mirror image. Draft slides 36-39.

### Pacing flag
41 content slides is a lot for a 35-40 minute theory budget. Suggest one of:
- **Cut scope:** move positional encoding (and/or decoding strategies) into Lecture 7, so Lecture 6
  stays to data → tokenization → embeddings → attention.
- **Keep scope, compress delivery:** treat the numeric BPE walkthrough (draft slides 10-13) and the
  numeric PE calculation (draft slide 33) as "read the slide" material rather than live-derived, and
  spend the saved time on attention, which is the conceptually hardest part.
Either way, flag this to Itay before the run-through rehearsal rather than discovering it live.

### Pre-reading for the speaker
- [[presentation]] — Itay's own draft, already the primary source
- [[data_preparation_and_tokenization]] (whole file)
- [[architecture]] for how the attention/embedding pieces connect to the rest of the stack

---

## Lecture 7 — The Architecture (Itamar)

**Goal:** Assemble the pieces Lecture 6 introduced (embeddings, attention, positional encoding) into a
full Transformer block and a working autoregressive model, and cover what Lecture 6 explicitly leaves
open.

> **Overlap note:** because Itay's draft (see above) already derives self-attention, multi-head
> attention, and absolute positional encoding in detail, Itamar should **not** re-derive those from
> scratch. Open with a ~3-5 min recap of "here's the attention output Lecture 6 built" and spend the
> lecture's weight on assembly and the topics Lecture 6 left as open items. **RoPE is deliberately not
> included here** — it's Itay's to pick up as an optional bonus in Lecture 6, or to skip entirely (see
> the Lecture 6 decision note above). Keeping it out of Lecture 7 is part of what makes room for the
> additions below.

### Core theory breakdown (~35-40 min)
1. **Recap: from attention output to a block** (~5 min) — one paragraph bridging Lecture 6's attention
   output to "now what?" — sets up the rest of the lecture.
2. **Transformer block assembly** (~10 min) — attention + MLP + residual connections + layernorm,
   stacking into a stack of blocks, LM head projecting back to vocab logits.
3. **Autoregressive generation loop + practical constraints** (~10-15 min) — next-token prediction
   objective, sampling loop, and the two mechanisms that make long-context inference tractable:
   - **KV Cache** — avoid recomputing key/value projections for past tokens on every step.
   - **Context Window** — the fixed limit on tokens attended to, and why it matters for cost/latency,
     and why it's the same limit that caused absolute PE's extrapolation problem in Lecture 6.
   Both already written up in [[architecture]] §§ *KV Cache*, *Context Window*, *Autoregressive
   Generation*. There's also a visual walkthrough at [[autoregressive_generation_cycle]] worth pulling
   diagrams from.

With the 90 min session and no labs, the core breakdown above only fills part of the slot. Rather than
padding it, extend it with the additions below — split into what's worth teaching solidly vs. what's
worth a passing mention, so Itamar can dial the depth to however much time is actually left after the
core material and discussion.

### Recommended additions (~15-20 min) — direct extensions of the KV cache thread above
These answer the question the core material raises but doesn't resolve: "KV cache is a real constraint —
so what do production models actually do about it?" Concrete, and reuses the same K/V framing already on
the table.
1. **Multi-Query Attention (MQA) and Grouped-Query Attention (GQA)** — share Key/Value projections
   across multiple (or all) query heads instead of giving every head its own K/V, shrinking the cache
   proportionally. GQA is the practical middle ground (used in Llama 3, Mistral) between full
   multi-head attention's quality and MQA's aggressive memory savings.
2. **Sliding-window attention** (Mistral) — bound cache growth by only attending to the last *W* tokens
   instead of the full context, trading some long-range recall for a cache that stops growing past a
   fixed size.

### Nice-to-have additions (a few sentences each, only if time allows)
Treat these as a closing "here's what's being explored beyond what we just covered" pointer — not
derivations. Cutting any (or all) of them costs nothing structurally.
1. **Multi-head Latent Attention (MLA)** — DeepSeek-V2/V3's approach: compress K/V into a shared
   low-rank latent vector before caching, reconstructing per-head K/V at attention time. More cache
   savings than GQA, more complex to explain — and a nice callback since DeepSeek-R1 already appears in
   Lecture 9.
2. **State-Space Models (Mamba / S4 / S6)** — the leading non-attention alternative to the Transformer
   block this lecture just assembled: linear (not quadratic) sequence cost, no KV cache at all in the
   traditional sense. Enough to say "this is being explored as a real alternative," not a derivation.
3. **Mixture-of-Experts (MoE)** — sparse activation (e.g. Mixtral): route each token to a subset of
   "expert" MLP sub-networks instead of running the full dense MLP every time, the dominant way frontier
   models scale parameter count without scaling compute per token proportionally.

### Pre-reading for the speaker
- [[presentation]] (Lecture 6 draft — read this first so the recap lands as a continuation, not a repeat)
- [[architecture]] (whole file)
- [[autoregressive_generation_cycle]] for the generation-loop diagram.

---

## Lecture 8 — Refinement / SFT (Yakov)

**Goal:** Go from a raw document-completer to something that behaves like an assistant.

> **Status:** there's already a full 15-slide Beamer draft at `4_docs/llm/seminar/sft.tex` ("Supervised
> Fine-Tuning: Teaching LLMs to Formulate Text & Follow Instructions"). It's considerably more detailed
> than what I originally sketched below, and covers several topics I hadn't scoped in at all. See the
> **comparison** subsection for the topic-by-topic diff.

### What `sft.tex` covers vs. what I originally proposed

| # | `sft.tex` slide | In my original plan? |
| :-- | :--- | :--- |
| 2 | Pre-training vs. SFT mindset shift (base "chaos" completion vs. aligned assistant completion, side by side) | Yes — folded into "Why SFT is needed," but sft.tex's concrete before/after example is stronger than my one-liner. |
| 3 | Full alignment pipeline (Pre-training → **SFT** → RLHF/DPO), with SFT's data scale ($10^4$-$10^5$ examples) vs. pre-training's ($10^{12}$+ tokens) | Partially — I gestured at this as "motivates Lecture 9" but didn't frame the three-stage pipeline explicitly or give the data-scale contrast. |
| 4 | Prompt engineering vs. SFT: cost/tradeoff table (upfront cost, cost-per-request, format reliability) | **No** — not in my plan at all. Useful framing for "why bother fine-tuning instead of just prompting?" |
| 5 | Anatomy of an instruction dataset — ChatML/JSON role structure, and explicitly which tokens get loss ("Gold Target") | Yes, but shallower — I had "chat data formatting" as one bullet; sft.tex shows the actual JSON and marks the loss-bearing span. |
| 6 | **LIMA effect** — quality-over-quantity, human win-rate curve, 1k curated examples beating 100k+ noisy ones | **No** — not in my plan. Significant, citable result (LIMA paper) worth adding. |
| 7 | Synthetic data generation (Self-Instruct: teacher model → generate → filter → student/SFT), with the "echo chamber" pitfall | **No** — not in my plan at all. |
| — | Chat template necessity: "role bleeding" failure mode (model keeps generating fake User/Assistant turns) and the fix (special tokens + `tokenizer.add_special_tokens()` / `model.resize_token_embeddings()`) | Partially — I mentioned special tokens marking turn boundaries but not the failure mode they prevent, nor the concrete implementation step. |
| 9 | SFT loss function, explicit: $\mathcal{L} = -\sum_{t=i}^{N}\log P(x_t\mid x_{<t})$, masked so only assistant tokens contribute | Yes, but I described it loosely as "same cross-entropy loss as pretraining" — sft.tex is more precise (masked multi-turn loss). |
| 10-11 | Full FT vs. PEFT/LoRA, with the actual math ($W = W_0 + \frac{\alpha}{r}BA$) and a concrete parameter-count comparison (16.8M vs. 65K for $d{=}4096, r{=}8$) | Yes, matches my LoRA section, but sft.tex has the worked numbers already — reuse them directly. |
| 12 | **Quantization & QLoRA** — FP32/FP16/NF4 memory footprint table, down to <8GB VRAM on consumer GPUs | **No** — not in my plan. Directly relevant to the Colab-free-tier hands-on lab (explains *why* the LoRA lab is Colab-feasible at all). |
| 13 | **Hyperparameters that matter** — learning rate (cosine decay, LoRA vs. full-FT magnitudes), effective batch size / gradient accumulation, packing vs. padding | **No** — not in my plan. Practical detail that would help whoever runs the hands-on lab debug a real training run. |
| 14 | Overfitting (train/val loss divergence, early stopping) *and* catastrophic forgetting (data rehearsal — mixing in ~10% pretraining-style data) | Partially — I mentioned catastrophic forgetting as a concept but not its concrete mitigation (rehearsal), and hadn't mentioned overfitting/early-stopping at all. |
| 15 | Evaluating an SFT model specifically — benchmarks vs. LLM-as-judge vs. human eval, and why ROUGE/BLEU don't work here | Not in Lecture 8 — I'd placed general evaluation content in **Lecture 9** ([[evaluation_paradigms]]). Worth deciding whether SFT-specific evaluation stays here or gets folded into 9 to avoid repeating benchmarks/LLM-as-judge twice. |

**Net effect:** `sft.tex` is meaningfully broader than my original 5-point outline — it adds the
prompting-vs-SFT tradeoff, LIMA/data-quality, synthetic data generation, chat-template failure modes,
QLoRA/quantization, and concrete hyperparameter guidance. It's also likely too much for a 30-35 min
theory slot on its own (15 dense slides), so the same kind of pacing conversation flagged for Lecture 6
applies here — Yakov should decide what to cut or compress rather than presenting all 15 slides at full
depth.

### Theory breakdown — revised to track `sft.tex` (~35-40 min)
1. **Pre-training vs. SFT mindset shift + the 3-stage pipeline** (~5 min) — sft.tex slides 2-3. Frame SFT
   as the middle stage between pre-training's raw-knowledge acquisition and RLHF/DPO's preference tuning.
2. **Why not just prompt-engineer instead?** (~5 min) — sft.tex slide 4's cost/reliability tradeoff table.
   Good hook before justifying the cost of fine-tuning at all.
3. **Instruction dataset anatomy + loss masking** (~7 min) — ChatML/JSON structure, which tokens get
   loss, chat-template role-bleeding failure mode and its fix. sft.tex slides 5, "Necessity of Chat
   Templates," 8-9. Matches [[training_and_adaptation]] § *Fine-Tuning and Visual Instruction Tuning*.
4. **Data quality: LIMA + synthetic generation** (~7 min) — LIMA's quality-over-quantity result, then
   Self-Instruct's teacher→generate→filter→student loop and its echo-chamber risk. sft.tex slides 6-7.
5. **Parameter-efficient fine-tuning: LoRA + QLoRA** (~10 min) — full FT vs. LoRA, the $W_0 + \frac{\alpha}{r}BA$
   math with sft.tex's worked parameter counts, then quantization (FP32/FP16/NF4) and why QLoRA is what
   makes it possible to fine-tune a 7B+ model on a single consumer GPU at all. sft.tex slides 10-12;
   matches [[training_and_adaptation]] § *LoRA - Low-Rank Adaptation*.
6. **Training practicalities + failure modes** (~5-8 min, compress if time-constrained) — learning rate /
   batch size / packing-vs-padding, overfitting + early stopping, catastrophic forgetting + data
   rehearsal. sft.tex slides 13-14; matches [[training_and_adaptation]] § *Catastrophic Forgetting*.
7. **Limitations of SFT** (~3 min) — imitates demonstrations, doesn't optimize for what users actually
   prefer; motivates Lecture 9 (RLHF). Not explicit in sft.tex but a natural closing bridge — keep it.
8. **RAG as an alternative to (re)training for knowledge** (~5-7 min) — **not in `sft.tex`, added at
   Yakov's request.** Pretraining bakes knowledge into weights (frozen, stale the moment training ends,
   expensive to update); SFT and RLHF only reshape *behavior/format*, not the model's factual knowledge
   — neither pipeline is a good way to add or update facts. Retrieval-Augmented Generation sidesteps this
   entirely: instead of baking knowledge into weights, retrieve relevant documents at inference time and
   place them in context, so the "knowledge base" can be updated instantly (swap the index) with zero
   retraining and no catastrophic-forgetting risk. Frame it as a **third axis** alongside the
   prompting-vs-SFT tradeoff from step 2: prompting controls *behavior* cheaply but not *knowledge*,
   SFT/RLHF control *behavior* durably but are expensive and static, RAG controls *knowledge freshness*
   cheaply without touching weights at all — and the three are complementary, not mutually exclusive
   (a RAG-based assistant is typically also SFT'd for format/behavior). Existing writeup at
   [[inference_techniques]] § *Retrieval-Augmented Generation (RAG)* — reuse directly rather than
   re-deriving.

**Cut candidate if time-constrained:** sft.tex slide 15 (SFT-specific evaluation — benchmarks / LLM-as-
judge / human eval) substantially overlaps [[evaluation_paradigms]], which Lecture 9 already covers in
depth. Recommend dropping it here (or reducing to one bridging sentence) rather than covering
benchmarks/LLM-as-judge twice in the block.

### Pre-reading for the speaker
- `4_docs/llm/seminar/sft.tex` (own draft — primary source; render it to check current content since it
  changes independently of this plan)
- [[training_and_adaptation]] (whole file — LoRA and catastrophic-forgetting sections map almost 1:1)
- [[inference_techniques]] § *Retrieval-Augmented Generation (RAG)* — for the new closing section

---

## Lecture 9 — Alignment & Future (Oryan)

**Goal:** State-of-the-art post-training — preference optimization and reasoning models.

### Theory breakdown (~35-40 min)
1. **Why SFT isn't enough** (~5 min) — recap from Lecture 8; imitation vs. preference optimization.
2. **RLHF pipeline** (~15 min) — reward model training from human preference pairs, then PPO against the
   reward model; connect back to Block 1's PPO lecture (Lecture 4) rather than re-deriving PPO from
   scratch. Relevant background: [[inference_techniques]] and [[evaluation_paradigms]] § *Human
   Evaluation*, § *LLM-as-Judge* (judging model outputs is the same mechanism reward models formalize).
3. **Reasoning / "thinking" models** (~10 min) — inference-time compute, chain-of-thought as a learned
   behavior rather than a prompting trick, process-level reward signals. There's substantial existing
   depth here at [[agentic_workflows]] on **Process Advantage Verifiers (PAVs)** — step-level rewards for
   math reasoning, bounded beam search, training PAVs, and RL with dense rewards. Worth condensing that
   into a simplified "why reward the process, not just the outcome" narrative for the seminar.
4. **Evaluating alignment/reasoning progress** (~5 min) — benchmarks, perplexity's limits as a metric,
   LLM-as-judge. [[evaluation_paradigms]] (whole file is a compact fit for this slice).

### Pre-reading for the speaker
- [[agentic_workflows]] (PAV section — heaviest math in the block, budget extra prep time)
- [[evaluation_paradigms]] (whole file)

---

## Cross-cutting notes for the block

- **Reuse, don't re-derive:** every lecture above has a matching section already written up in
  `4_docs/llm/sections/`, and Lecture 6 additionally has Itay's own slide draft at [[presentation]].
  Speakers should treat these as first drafts to adapt into slides/talking points, not sources to
  independently re-research — flag gaps back into the notes if something's missing rather than
  duplicating it elsewhere. Itamar in particular should read Itay's draft before preparing Lecture 7 to
  avoid re-deriving attention/positional-encoding material that's already covered.
- **Continuity across lectures:** 6→7→8→9 is a strict prerequisite chain (data/tokenization → architecture
  → SFT → alignment). Recommend a 5-minute recap at the start of 7, 8, 9 tying back to the prior lecture's
  punchline.
- **No hands-on labs:** per [[syllabus]]'s speaker guidelines, each lecture originally scoped a coding
  lab (Colab notebooks, pre-trained weights for slow training runs, etc.). Since those didn't happen in
  practice for the RL block, this plan drops them for the LLM block too — all four lectures are theory +
  discussion only. If the syllabus itself still lists labs, treat that as stale relative to actual
  practice rather than something to reconcile here.
