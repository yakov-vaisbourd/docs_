# SFT Presentation — Working Tracker

Companion to [[llm_block_detailed_plan]] (Lecture 8 section) and the live deck, now
`seminar/tex/pres/sft_pres.tex` (LaTeX/Beamer, Atlas theme). This file is our shared scratchpad
while building the talk: goal, current state, open decisions, and a running log of what we've
settled on.

> **Workflow note (updated 2026-09-06): the deck switched from Marp/Markdown to LaTeX/Beamer.**
> The working file is now `seminar/tex/pres/sft_pres.tex` — **`sft.md` is no longer edited and is
> effectively retired/frozen.** Yakov describes the change he wants, Claude edits the `.tex` file
> directly. The Beamer deck uses the `Atlas` theme (`../styles/beamerthemeAtlas.sty`) plus local
> `tcolorbox` environments (`warmbox`/`coolbox`/`goodbox`/`badbox`/`formulabox`/`noteframe`) defined
> at the top of `sft_pres.tex` itself — there is no separate external-CSS-file split like the old
> Marp `marp-theme/sft.css` had; styling helpers live in the preamble of the `.tex` file. All of the
> content/structure history below (slide numbering, decisions log, etc.) describes the old Marp
> deck's evolution up through 2026-09-06; see the "2026-09-06 — Switched to LaTeX/Beamer" decisions-
> log entry for how that content maps onto the new file, and treat slide *numbers* below as
> Marp-era history, not current fact — check the `.tex` file directly for current slide order.

> **Workflow note: archive boundary — status unclear after the LaTeX switch (flagging, not
> resolved).** In the old Marp deck there was a title-slide marker ("Archived From Here Down") that
> split it into a working front section (slides 1-14) and a frozen archive (slides 15-28, the
> deck's original topics 2-8 content). As of 2026-09-06, `sft_pres.tex` contains **14 frames only**
> and no archive marker or archived slides — it matches the old front section 1:1 (see mapping in
> the 2026-09-06 decisions-log entry) but the archived material (old topics/slides 16-28: full
> chat-template demo, synthetic data generation, QLoRA, training practicalities, overfitting/
> forgetting, evaluation) does not currently exist anywhere in the `.tex` file. Whether that
> archived content should eventually be ported into the `.tex` file (as new frames, if/when the
> open decisions below get resolved) or was intentionally left behind is **not yet decided — ask
> Yakov before assuming either way.** Until further notice, treat `sft_pres.tex` as having no
> archive/frozen section at all: every frame in it is presumed live/working.

> **Maintenance note (for whichever session is reading this):** this file is the single source of
> truth for where the SFT talk stands. Keep it current *proactively* — after any decision, slide
> edit, or scope change, update the relevant section (state table, decisions log, next steps)
> without waiting to be asked. A new session should be able to read just this file, plus `sft.md`
> and the plan, and pick up the work fluently with no other context. When in doubt, over-record
> reasoning here rather than leave it implicit — the "why" behind a decision matters as much as the
> decision itself for picking this back up cold.
>
> **Session start-up:** reading this file is for context only. At the start of a session, do not
> proactively surface the open decisions, propose resolutions, or otherwise push toward action on
> them (e.g. firing off a batch of questions about slide 16 / topics 7-8 / pacing) until Yakov gives
> an explicit instruction for that session. Load the context silently, then wait.

## Goal

Turn `sft.md` into a **detailed but well-organized** ~90-min-block talk on SFT that:
- Covers real depth (this is a seminar, not a surface-level overview) —
- ...but fits a **~35-40 min theory slot** (the block's 90 min also holds Q&A/discussion and ties
  into Lectures 6/7/9), so density has to come from *organization*, not just cramming slides in.
- Is explainable end-to-end by Yakov in that window — every slide earns its minutes.

## Current state (as of 2026-09-06, revised same day after an out-of-session edit)

**The deck is `seminar/tex/pres/sft_pres.tex` (LaTeX/Beamer, Atlas theme) — `sft.md` is retired.**
The file was edited again after the LaTeX-switch entry below was written (not through this
session — found changed on disk when this session re-read it); this "Current state" section has
been rewritten to match what's actually in the file now. Two structural changes since the last
write-up:
1. **Reordered again:** "Training: Same Loss, Adapted Rules" moved from late in the deck (after
   SFT Data / LIMA) to right after the Domain SFT example, ahead of the Pipeline slide. Both the
   Domain SFT slide and the Training slide are now wrapped in `\atlasslide{...}` (recordable for a
   map-return), and a **map-return frame revisiting the Training slide's sidebar map is now
   actually present** right after it — this is the first live use of the `\atlasmapreturn`/
   `\atlasmapcoverbelow` progressive-reveal system described below; it was "not yet wired into any
   frame" as of the last write-up.
2. **Several slides now carry a red author-note (`\note{\color{red} ...}`)** flagging unfinished
   research/revision work Yakov wants to do before the talk is done — distinct from the plain
   speaker notes (black text) that explain delivery. See the per-slide list below for what each one
   flags; these are new open items, not yet reflected in "Open decisions" until this pass.

Actual `.tex` frame order now: (1) title, (2) Two Purposes of SFT, (3) Alignment SFT mindset-shift
example, (4) Domain SFT example (autonomous-driving copilot — `\atlasslide`-wrapped), (5)
Training: Same Loss, Adapted Rules (`\atlasslide`-wrapped), (6) map-return frame revisiting (5)'s
sidebar, (7) The LLM Alignment Pipeline, (8) Anatomy of an Instruction Example (now has a full
delivery speaker note — added this session, see decisions log), (9) **DeepSeek-V3: Quantity vs.
Quality, by the Numbers** (rebuilt 2026-09-07 from the old "Resource Cost Across the Pipeline",
then re-rebuilt same day into a DeepSeek-only case study; red note resolved), (10) **What
"Curated" Actually Means** (rebuilt 2026-09-07 from the old "SFT Data: Not Free, Not Web-Scale",
100% Llama-3-sourced; its unverified dataset table dropped), (11) LIMA
Effect (red note: needs references for its claims, plus a Google paper on less-fine-tuning-helps-
RL to fold in), (12) Prompt Engineering vs. Domain SFT, (13) A Third Axis: ICL & RAG (red note:
pull from an earlier RAG-paper deck, redo diagrams in Excalidraw), (14) Full FT vs. PEFT (LoRA)
(red note: is LoRA's lower expressiveness a regularization benefit or a cost — unresolved, needs
research), (15) LoRA Inside the Transformer (red note: also cover SVD-based methods, find a survey,
check Dima's presentation for other points to include). 15 pieces of content across 16 rendered
frames (frame 6 is the map-return, revisiting frame 5). The prose below still describes each piece
of *content* accurately (and is cross-referenced by the old Marp/early-LaTeX slide numbers in the
mapping table and decisions log), but treat every slide-number reference below as **content
identity, not current position** — check `sft_pres.tex` directly for actual frame order. The old
Marp deck's frozen archive (old slides 15-28) was **not** carried over into the `.tex` file — see
the archive-boundary workflow note above. Content descriptions, old-numbering order:
1 title, 2 "Two Purposes of SFT" (alignment vs. domain/application), 3 "The LLM Alignment Pipeline"
(3-stage pipeline diagram) — **note:** this tracker previously (through 2026-08-30) documented 2 as
the pipeline and 3 as "Two Purposes"; that was stale, the file has actually had them in this order
for a while (found 2026-09-06, not changed, see decisions log), 4 "Resource Cost Across the
Pipeline" (**new 2026-09-06** — see decisions log), 5 the alignment-SFT mindset-shift example, 6 a
domain-SFT example (autonomous-driving copilot emitting a structured risk/action schema) — 1-2 and
5-6 are the pre-existing approved core.
Then, added 2026-08-30 as part of the **"Technicalities & Why" / "Alternatives & Variants"
restructure** (see decisions log): 7 "SFT Data: Not Free, Not Web-Scale" (pretraining vs. SFT data
cost/scale, quantified against real dataset sizes), 8 "The Anatomy of an Instruction Example"
(ChatML roles + loss-only-on-assistant-tokens), 9 "Quality Over Quantity: The LIMA Effect", 10
"Training: Same Loss, Adapted Rules" (masked CE loss = pretraining's loss, chat-template tokens,
lower learning rate) — this closes out **Part A, the technical "why"**. Then **Part B,
alternatives & variants**: 11 "Prompt Engineering vs. Domain SFT" (reframed to explicitly scope
itself to domain SFT, resolving former Open decision #5 as a fresh slide rather than an archived-
slide edit; landscape-note corrected 2026-09-06, see decisions log), 12 "A Third Axis: In-Context
Learning & RAG", 13 "Full Fine-Tuning vs. PEFT (LoRA)", 14 "LoRA Inside the Transformer" (new —
names the actual weight matrices LoRA targets: Wq/Wk/Wv/Wo in attention, optionally the MLP
up/down projections).

Mapped against the plan's 8-topic breakdown ([[llm_block_detailed_plan]] § Lecture 8, "Theory
breakdown — revised to track sft.tex") — note the mapping is now against the **front-section
slides that are actually presented**, not the archive:

| Plan topic | Slide(s) in `sft.md` (front section) | Status |
| :-- | :-- | :-- |
| 1. Pre-training vs. SFT mindset + 3-stage pipeline | 2, 3, 5 | Done |
| — Two purposes of SFT (alignment vs. domain) + domain example | 2, 6 | Done |
| — Resource cost across pipeline stages (data volume, training compute) | 4 → "Two Data Regimes" | Done — new 2026-09-06, rebuilt 2026-09-07 as a verified quantity-side table (DeepSeek-V3 Table 1 + token counts); not in original plan |
| — Data cost/scale: pretraining (cheap, web-scale) vs. SFT (curated, quantified) | 7 → "What 'Curated' Actually Means" | Done — new framing, not in original plan. Rebuilt 2026-09-07 as the quality-side companion: the two pipelines' actual filtering steps from Llama 3 §3.1/§4.2.3; the old unverified dataset-size table dropped |
| 3. Instruction dataset anatomy + loss masking + chat templates | 8, 10 | Done — condensed; full chat-template role-bleeding demo (archived 20-21) not recreated, only mentioned in passing on slide 10 |
| 4. Data quality: LIMA | 9 | Done — synthetic-generation half (archived 19) not recreated, not requested |
| 2. Why not just prompt-engineer? | 11 | Done — now explicitly scoped to domain SFT |
| — In-context learning + RAG as a further alternative | 12 | Done — new, covers plan topic 8's "RAG as third axis (knowledge vs. behavior)" framing |
| 5. PEFT: LoRA (as a variant of SFT, not an alternative to it) | 13-14 | Done — slide 14 adds transformer-architecture precision (which weight matrices) not in the archived version. QLoRA/quantization (archived 25) not recreated, not requested |
| 6. Training practicalities + failure modes | — | **Not in front section** — only in archive (26-27); not part of this restructure |
| 7. Limitations of SFT (bridge to Lecture 9) | — | **Still missing** |
| Cut candidate: SFT-specific evaluation | — | **Not in front section** — only in archive (28) |

**Net:** the deck is now organized as **Part A (slides 7-10): the technical mechanics and why they
work this way** (data cost/quality, dataset anatomy, LIMA, masked CE loss), followed by **Part B
(slides 11-14): alternatives and variants** (prompt engineering, ICL+RAG, LoRA as a cheaper variant
of SFT with real transformer-architecture detail), with the new **slide 4** giving a real-numbers
resource-cost view of the pipeline right where the 3 stages are first introduced. Topic 6 (training
practicalities/failure modes) and topic 7 (limitations of SFT, bridge to Lecture 9) are not yet
represented in the front section — still open. Pacing continues to be explicitly deferred per
Yakov's 2026-08-30 call (see Open decisions #4) — the deck is now well past the original 35-40 min
budget in raw slide count and this has not been revisited.

## Reference: the Atlas "map-return" system (sidebar image, cover above/below)

This section exists so that when Yakov says **"the sidebar image"** or **"cover above/below"**, the
meaning is unambiguous without re-deriving it from scratch. Mechanism lives in
`seminar/tex/styles/beamerthemeAtlas.sty`; content lives in `seminar/tex/pres/sft_pres.tex` and
`seminar/tex/pres/resources/`.

**What a "map-return" frame is:** a special slide that revisits an earlier slide (e.g. the Domain
SFT example) by sliding a vertical spine rightward, covering most of the page with a dimmed,
cropped reproduction of that earlier slide, and opening up a plain-white panel on the left —
**this white left panel is "the sidebar."** It shows a mind-map image (built in Excalidraw) meant
to orient the audience ("here's where we are in the bigger picture") before returning to the
detailed slide underneath.

**Key macros (all in `beamerthemeAtlas.sty`):**
- `\atlasslide{title}{\boxreg}{anchorname}{body}` — replaces `\begin{frame}{title}...\end{frame}`
  for any slide that will later be revisited via a map-return frame. Records the slide's title/body
  so it can be reproduced pixel-identically later.
- `\atlasmapreturn[coverage]{map title}{map content}{title}{\boxreg}{anchorname}` — the actual
  map-return frame. `coverage` (default 0.8) is what fraction of the page width the sidebar takes up.
  **"map content"** (3rd argument) is exactly the sidebar image — whatever TikZ/`\includegraphics`
  content goes there is what fills the white sidebar panel. The last three arguments must be the
  exact same `title`/`\boxreg`/`anchorname` passed to `\atlasslide` for the slide being revisited, so
  the reproduction is pixel-identical, not hand-copied.
- `\atlashighlight{x}{y}{r}` — draws a ring at coordinate `(x,y)` (cm, relative to the map content's
  own local TikZ origin) with radius `r`, to mark "you are here" on the mind-map image.
- `\atlasmapcover{x1}{y1}{x2}{y2}` — draws an opaque white rectangle over part of the sidebar image,
  in the same local coordinate system as `\atlashighlight`. **This is "cover"** — it hides a region of
  the mind-map so only part of it is visible at a given point in the talk.
- `\atlasmapcoverbelow{y}` / `\atlasmapcoverabove{y}` — convenience wrappers over `\atlasmapcover`
  for the common case of a horizontal curtain: hide everything below/above height `y` (extends
  ±1000cm left/right/up/down so it always fully covers regardless of image size). **These are "cover
  below" / "cover above."**

**The progressive-reveal workflow this enables (added 2026-09-06; now live in the deck — see
"Current state" above):** the Training slide's map-return frame uses exactly this pattern
(`\atlasmapcoverbelow{-2.2}` at coverage `0.6`).
instead of exporting a separate mind-map image per stage of the talk, keep ONE master Excalidraw
mind map (`resources/map.excalidraw`, auto-exported to `resources/map.png` — see next subsection),
lay its components out in the order you intend to reveal them (e.g. top-to-bottom), and in each
`\atlasmapreturn` call wrap the `\includegraphics` in a `tikzpicture` with a `\atlasmapcoverbelow{y}`
(or `...above`) call whose `y` shrinks (reveals more) at each successive map-return frame in the
deck:
```latex
\atlasmapreturn[0.6]{}{%
  \begin{tikzpicture}
    \node[anchor=north west, inner sep=0pt] (img) {\includegraphics[width=\atlaspanelinnerwidth]{map}};
    \atlasmapcoverbelow{-6}   % only the top ~6cm of the diagram is visible at this stage
  \end{tikzpicture}
}{\domainSftTitle}{\domainSftBox}{domainSftBodyAnchor}
```
Caveat: this only suits a *directional* reveal (a curtain sliding one way) — revealing disconnected
regions out of spatial order needs multiple `\atlasmapcover` calls (one per hidden region) instead of
a single moving edge. Finding the right `y` per stage is trial-and-error against the actual image,
same as positioning `\atlashighlight` rings.

**How `resources/map.png` actually gets (re)generated — not manual, not any Excalidraw
extension's doing:** `.vscode/settings.json` (project root) defines a custom LaTeX Workshop build
recipe, `"pdflatex ➞ excalidraw-auto"`, whose first step (`excalidraw-compile`) runs
`find . -iname '*.excalidraw'` and calls the `excalidraw-export` CLI (globally installed via nvm,
package `@moona3k/excalidraw-export`) on every match, overwriting each file's same-named `.png`
— e.g. `map.excalidraw` → `map.png` — **before** `pdflatex` even runs. So every LaTeX build silently
re-exports every `.excalidraw` file in the project. This CLI has **no frame-scoping support** (its
own custom renderer, not the official Excalidraw library — confirmed by reading
`@moona3k/excalidraw-export`'s source, `src/index.js`/`src/cli.js`): it always exports the entire
scene, so Excalidraw's own "Frames" feature can't be used to select a subset for a given map-return
stage. (This is exactly why the cover-above/below approach above was chosen over per-frame Excalidraw
exports — it works with a single always-fully-exported master image instead of needing per-stage
files.)

## Open decisions (to resolve together)

0. **NEW (2026-09-07) — Review the FLAN / Self-Instruct slides and the two flagged additions,
   just added this session, not yet reviewed:** the two new frames ("Where Does Instruction Data
   Come From? FLAN's Task-Diversity Approach" and "Bootstrapping Instructions from the Model
   Itself: Self-Instruct") need a normal content pass like every other slide in the deck. Also
   decide whether to keep or delete the two flagged/optional pieces (search `FLAGGED` in
   `sft_pres.tex`): the one-line InstructGPT citation added to the already-reviewed Pipeline slide,
   and the whole extra frame "After SFT: Why RL Still Helps" placed right after LIMA. See the
   matching decisions-log entry above for full detail on what was added and why.
1. **Evaluation slide:** still only in the archive (slide 28, unchanged). Cut entirely, compress to
   one bridging sentence, or bring a version into the front section? Plan's default recommendation
   is cut/compress to avoid repeating benchmarks/LLM-as-judge material Lecture 9 already covers.
2. **Limitations of SFT (topic 7):** still not added anywhere. Add as a short closing slide?
3. **Training practicalities + failure modes (topic 6):** was "Done" in the old archive-only
   structure (archived slides 26-27) but has no front-section equivalent after the 2026-08-30
   restructure. Worth pulling forward like the other topics were, or intentionally left as an
   archive-only aside?
4. **Overall pacing:** deck is now at 28 slides (14 in the front section alone) against the original
   ~35-40 min target. **Yakov's call (2026-08-30, reaffirmed while adding the Part A/B restructure):
   don't worry about this while building out content — compress later if the final deck runs long.**
5. ~~**Prompt-engineering slide reframing**~~ — **Resolved 2026-08-30.** Done via the new slide 11,
   which explicitly scopes itself to domain SFT, rather than editing the archived slide 16.
6. **Red author-notes flagging unfinished revisions (found 2026-09-06, out-of-session edit — see
   decisions log). One of the five is now resolved; four remain:**
   - ~~Resource Cost Across the Pipeline: too verbose/unorganized, needs validated numbers.~~
     **Resolved 2026-09-07** — the slide was rebuilt as "Two Data Regimes" (plus a new companion,
     "What 'Curated' Actually Means") with every figure verified against the two papers in
     `papers/models/`; red note replaced by a delivery note carrying the citations. See decisions log.
   - LIMA Effect: needs a reference for the LIMA claims; also wants to fold in a Google paper
     showing less fine-tuning at this stage improves RL success, plus more papers generally
     (mathpix-convert and pull in main claims) for this slide and the next one.
   - A Third Axis: ICL & RAG: pull material from an earlier RAG-paper slide deck; redo diagrams in
     Excalidraw instead of the current Goodnotes versions.
   - Full FT vs. PEFT (LoRA): whether LoRA's lower expressiveness is a regularization benefit or a
     cost is unresolved — needs to be checked, and whether it's even the most important point to
     make here vs. other candidates.
   - LoRA Inside the Transformer: should also cover SVD-based (and other) low-rank methods, not
     just LoRA; wants a survey/comparison reference; also wants to check Dima's presentation for
     other points to include.

## Decisions log

*(most recent first)*

- **2026-09-07 — Added two new frames (FLAN, Self-Instruct) plus two flagged/optional additions,
  based on the papers in `papers/Finetune_Base/` — NOT YET REVIEWED by Yakov.** Session task: fill
  the gap between "The LLM Alignment Pipeline" and "Quality Over Quantity: The LIMA Effect" with
  the history of where Alignment-SFT instruction data itself comes from, using the papers indexed
  under `papers/papers_index.md` § Finetune Base. Plan approved via plan mode
  (`~/.claude-personal/plans/between-the-slides-starting-abundant-stream.md`) before editing.
  **Changes made to `sft_pres.tex`:**
  1. **New frame, "Where Does Instruction Data Come From? FLAN's Task-Diversity Approach"**
     (Wei et al., 2022) — inserted right after "The Anatomy of an Instruction Example," before
     DeepSeek-V3. Speaker note mentions Flan-PaLM/Flan-T5 (Chung et al., 2022) as later
     continuation work (same finding at 1.8K tasks/540B params) rather than giving it its own slide.
  2. **New frame, "Bootstrapping Instructions from the Model Itself: Self-Instruct"** (Wang et al.,
     2023) — inserted right after the FLAN frame. Framed as the direct ancestor of the Llama 3
     generate$\to$filter$\to$dedup pipeline on the next-but-one slide.
  3. **Flagged, optional one-line citation added to the already-reviewed "The LLM Alignment
     Pipeline" frame** — credits InstructGPT (Ouyang et al., 2022) as the paper that introduced the
     3-stage pipeline. Wrapped in `% >>> FLAGGED -- OPTIONAL ... <<<` comment markers in the `.tex`
     so it's trivial to find and delete without touching the diagram/bullets.
  4. **Flagged, optional new frame, "After SFT: Why RL Still Helps"** (SFT Memorizes, RL
     Generalizes — Chu et al., 2025) — inserted right after LIMA, before "Prompt Engineering vs.
     Domain SFT," as a bridge into whatever RLHF/DPO content follows. Also wrapped in
     `% >>> FLAGGED ... <<<` markers. Its speaker note mentions TailSFT (Malladi et al., 2026) as
     recent continuation work (filters SFT's already-fit sequences to preserve pass@K for
     subsequent RL) rather than giving it its own slide.
  **Verification done:** `latexmk -pdf` compiles clean (exit 0, 19 pages now vs. 15 before —
  4 new/flagged frames added, no new overfull-box warnings beyond the pre-existing unrelated one).
  **Verification NOT done:** Yakov has not reviewed the new slide content, wording, visual fit, or
  whether he wants the two flagged items kept — see the new Open decision below. Search `FLAGGED`
  in `sft_pres.tex` to find both optional pieces.

- **2026-09-07 — Frame 10: unified both chains as TikZ, shrunk to `\tiny`, and nudged the "$\times$6
  rounds" label right of the brace.** Two follow-ups on the brace change (previous entry): (1) the
  label initially sat on top of the brace tip --- moved via `xshift` 22pt $\to$ 32pt so it clears the
  brace. (2) the left column was still plain-text `$\big\Downarrow$` arrows while the right column
  used TikZ `Stealth` arrows --- rebuilt the left column as its own TikZ node-chain (same `arr`
  style, same `text width=3.1cm`, same `node distance`) so both sides render identical arrowheads,
  and dropped both chains' font from `\scriptsize`/mixed to a uniform `\tiny` per Yakov's request
  for a slightly smaller size. Recompiled clean, no new overflow.
- **2026-09-07 — Frame 10: right column's arrow-chain rebuilt in TikZ so "$\times$6 rounds" sits
  beside the chain with a curly brace spanning all four steps** (Rejection sampling / Clean &
  topic-tag / Quality$\times$difficulty score / Semantic dedup), instead of stacked below them as
  plain text. Uses `decorations.pathreplacing` (already loaded) plus a newly added `arrows.meta`
  tikz library (preamble) for the down-arrows between nodes. Left column stays plain-text (no brace
  needed --- it has no repeated-rounds annotation). Hit one error along the way: `\scriptsize` isn't
  a valid bare tikz option key --- fixed by using `font=\scriptsize` in the tikzpicture options.
  Recompiled clean, no new overflow; re-rendered to confirm the brace aligns against all four nodes.
- **2026-09-07 — Frame 10: converted both columns from prose+bullets to a terse arrow-chain of
  labels.** Yakov felt the slide (after the warmbox/note pass, previous entry) still had too much
  text for a slide. Replaced each column's itemized bullet list (full-sentence descriptions) with a
  vertical chain of short labels separated by $\Downarrow$: left is Dedup $\to$ Heuristic filters
  $\to$ fastText/RoBERTa classifiers; right is Rejection sampling $\to$ Clean \& topic-tag $\to$
  Quality$\times$difficulty score $\to$ Semantic dedup $\to$ $\times$6 rounds. All the dropped
  explanatory detail was already living in the `\note{}` from the previous entry --- re-worded that
  note (still per-term, `\scriptsize`) to read "top to bottom" in the same order as the slide's
  chain, so notes and slide stay matched term-for-term (also merged the note's separate "rule-based
  cleaning" and "topic classifier" explanations into one "Clean \& topic-tag" entry to mirror the
  merged slide label). Recompiled clean, no new overflow.
- **2026-09-07 — Frame 10 ("What 'Curated' Actually Means"): both columns wrapped in `warmbox`
  blocks, and a full glossary `\note{}` added.** Yakov wanted the two columns visually boxed (both
  in the warm/AtlasWarm style already used for the column headers --- this slide has no cool/warm
  split since it's Llama-only, so both sides share one accent color) and wanted the notes to define
  every technical term the slide names but doesn't explain (MinHash, n-gram/dirty-word heuristics,
  token-distribution KL, fastText/RoBERTa, rejection sampling K=10--30, topic classifier, RM quality
  score, Instag difficulty score, semantic dedup, six rounds) --- this frame previously had no
  `\note{}` at all. Wrapping in boxes plus the new note overflowed badly on the first pass (slide
  body +31pt, notes pane +100pt) --- fixed by dropping the frame from `\small`/`\footnotesize` to
  `\footnotesize`/`\scriptsize` throughout the boxes, trimming inter-line `\\[]` spacing, shortening
  `\landscapenote`'s attribution, and rewriting the note to one compact `\scriptsize` definition per
  term instead of full sentences. Recompiled clean and re-rendered to confirm both boxes and the
  notes pane fit.
- **2026-09-07 — Frame 9 tweak: dropped the Total row, explained Context extension.** Yakov didn't
  want the Total row (it added a number without adding insight) and wanted the "Context extension"
  stage-row to actually say what that stage does, since "(same corpus, longer sequences)" was a
  placeholder, not a citation. Replaced with the real mechanism from DeepSeek-V3 \S4.3: YaRN applied
  after pre-training, two phases of 1000 steps each, first extending 4K$\to$32K then 32K$\to$128K,
  same learning rate as the end of pre-training. Added a one-line on-slide caption ("no new data,
  just longer sequences --- the model re-learns to attend over a longer window before it's taught to
  follow instructions") plus a matching, shortened citation in the note. Recompiled clean.
- **2026-09-07 — Split frames 9-10 so each is a single-paper case study instead of a DeepSeek/Llama
  blend.** Trigger: while discussing what each paper does/doesn't report (DeepSeek gives absolute
  SFT volume + a generation/verification story but no filtering pipeline or category mix; Llama 3
  gives a detailed filtering pipeline + category mix but no absolute SFT count or cost), Yakov asked
  to stop mixing the two papers on frame 9 and instead have each slide reflect a different paper's
  strength: sizes/cost from DeepSeek-V3, curation mechanics from Llama 3.
  - **Frame 9** renamed **"DeepSeek-V3: Quantity vs. Quality, by the Numbers"** (was "Two Data
    Regimes"), subtitle changed to *"A case study from one published model report --- DeepSeek-V3."*
    The 4-column multirow Report/Stage/Data/Compute table (previous entry) is replaced by a plain
    3-column **Stage / Data / Compute** table restricted to DeepSeek-V3's own three reported stages
    --- Pre-training, Context extension, Post-training (SFT+RL) --- plus a Total row, all still from
    Table 1 (\S1). Llama 3 numbers and the separate "For scale" model-specs table were dropped from
    this frame entirely; DeepSeek's own specs (MoE, 671B/37B, 128K context/vocab) now appear as one
    small `\scriptsize` "at a glance" line under the table instead of a second table, since comparing
    against a second model no longer applies. `\usepackage{multirow}` is now unused on this frame
    (left loaded; still harmless) since there is only one model to group rows for.
  - **Frame 10** ("What 'Curated' Actually Means") was already 100% Llama-3-sourced content (§3.1 +
    §4.2.3), so no data changed --- only added a matching subtitle, *"A case study from one published
    model report --- Llama 3,"* and trimmed the landscape note's attribution line (now just
    "§3.1 (pre-training) and §4.2.3 (post-training)" since the subtitle already names the paper).
  - Fixed a duplicate `\vspace{2pt}` left over from the subtitle edit and tightened both columns'
    itemize spacing (`itemsep`/`topsep` 1pt/3pt $\to$ 0pt/1pt) to clear a resulting 5-9pt `Overfull
    \vbox` on frame 10. Recompiled clean (only the pre-existing unrelated line-189 overfull hbox
    remains) and re-rendered both pages at 150dpi to confirm.
- **2026-09-07 — Follow-up polish on frame 9 ("Two Data Regimes"): per-model row attribution, title
  reworked to name the two source reports, and a new companion table of model specs.** Three
  requests in sequence, all same session as the frame's rebuild (previous entry):
  1. Yakov wanted the frame title to signal it's a two-model case study drawn from actual reports.
     Titling the frame itself that way made it wrap to two lines and collide with the table below —
     fixed by keeping the frame title as plain **"Two Data Regimes"** and adding a small italic
     subtitle line under it instead: *"A case study from two published model reports ---
     DeepSeek-V3 \& Llama 3."* (matches the deck's existing `\slidecaption`-style convention, e.g.
     frame 5's caption line).
  2. The table didn't make clear which report each number came from. Restructured from a
     2-column (Pre-training / SFT) layout to a 4-column **Report / Stage / Data / Compute** table,
     with `\multirow` grouping each report's two stage-rows under its own (color-coded) name —
     DeepSeek-V3 in `AtlasWarm`, Llama 3 405B in `AtlasCool`. This also surfaced, honestly, that
     Llama 3 has no absolute SFT-post-training numbers at all (mix is percentages only, no compute
     figure reported) — shown as explicit `not reported` / `\% mix only` cells rather than left
     blank or omitted, so the gap in the record is visible instead of papered over. Needed adding
     `\usepackage{multirow}` (not previously loaded).
  3. Added a second, smaller **"For scale"** table underneath giving the two models' own specs, so
     the reader can judge whether the data/compute contrast is confounded by model size. Verified
     against the same two papers: architecture (DeepSeek-V3 MoE via DeepSeekMoE+MLA vs. Llama 3
     dense Transformer), total/active params (671B/37B vs. 405B/405B --- all active for a dense
     model), and context window \& vocabulary (both 128K tokens, confirmed coincidental --- different
     tokenizers landing on the same round number, not a shared design choice).
  **Iteration to fit the frame:** the first attempt (2-column model-specs table plus the original
  wording) overflowed vertically off the slide (`Overfull \vbox` up to 105pt/2.4cm too tall) and
  also overflowed the **notes pane** on its own (a separate, easy-to-miss failure mode — the note
  text box has no dynamic height, so the compiler warns via a generic `Overfull \vbox` at
  `\end{frame}` and any note text past the frame's height is silently cut off in the rendered PDF,
  not just visually cramped). Fixed by: removing a stray `\parbox` that was forcing "DeepSeek-V3" /
  "Llama 3 405B" to wrap awkwardly across two lines inside the multirow cells; tightening
  `\arraystretch` on both tables (1.0 and 0.95); collapsing the model-specs table from 5 rows to 2
  by merging architecture+params and context+vocabulary into single rows; and substantially
  trimming the frame's `\note{}` (cut redundant asides, shortened three paragraphs) since that was
  the actual remaining overflow once the slide body itself fit. Confirmed clean via three more
  `pdflatex` passes plus a final render-and-read of both the slide and its notes pane at 150dpi —
  zero overfull boxes left except the pre-existing, unrelated one at line 189 (a different frame).
  **Practical note for future frames with heavy notes:** always render and check the *notes pane*
  alongside the slide body when compiling near a page's vertical limit — a fully fitting slide can
  still have its speaker note truncated with no visual sign in the main content.
- **2026-09-07 — Rebuilt frames 9-10 around a single verified quantity$\to$quality story: "Resource
  Cost Across the Pipeline" $\to$ **"Two Data Regimes"**, and "SFT Data: Not Free, Not Web-Scale"
  $\to$ **"What 'Curated' Actually Means".** Trigger: Yakov was unhappy with both — the first too
  verbose/unorganized (its own red author-note said so), the second's top half fine but its bottom
  dataset table LLM-generated and unverified; plus real repetition between them (both stated the
  data-scale gap) and no through-line. His ask: show the quantity-vs-quality difference between this
  stage and the previous one, **based only on verified information**.
  **Story now:** pre-training and SFT run the same next-token objective on data from opposite
  regimes — pre-training buys capability with *scale* (bulk automatic filtering), SFT buys behavior
  with *selection* (every example individually judged). Frame 9 sizes the gap; frame 10 shows what
  closing it costs in curation effort.
  **Frame 9 (numbers only):** a 3-column table, Data + Compute rows, pre-training vs. SFT columns.
  All figures from the two papers in `papers/models/`, verified in-file: DeepSeek-V3 Table 1
  (`01_introduction.md:11-20`) — 2664K H800-hrs / \$5.328M pre-training vs. 5K H800-hrs / \$0.01M
  post-training, at the paper's stated \$2/H800-GPU-hour; 14.8T pre-training tokens
  (`01_introduction.md:9`, `04_pre_training.md:8`); 1.5M SFT instruction instances
  (`05_post_training.md:5`); Llama 3's 15.6T tokens / 405B params / 3.8e25 FLOPs / up to 16K H100s
  (`01_introduction.md:12`, `03_pre_training.md:148`). Landscape note states the ratio
  ($\sim$500$\times$ less compute, $\sim$10^7 fewer items) rather than asking anyone to read digits.
  The old inline caveat paragraph was cut down to **one `\scriptsize` line** (Yakov's call): DeepSeek's
  post-training figure covers SFT *and* RL together; Llama 3 reports its SFT mix only as percentages
  with no absolute count. The red author-note is **resolved and replaced** by a plain delivery note
  carrying the full citations (paper + section) so the numbers can be defended live.
  **Frame 10 (the curation contrast):** the two provenance blurbs at the top of the old slide are
  **kept verbatim** as the two column headers (Yakov confirmed that half was fine); under each sits
  that side's *actual* filtering pipeline, both from Llama 3, as the evidence for the blurb above it.
  Left (`03_pre_training.md:16-30`) — URL/document(MinHash)/line dedup, heuristic filters (n-gram
  coverage, dirty-word counts, token-distribution KL), fastText/RoBERTa quality classifiers; labelled
  *bulk, automatic, corpus-level*. Right (`04_post_training.md:97,105-116,54`) — sample K=10-30 per
  prompt with the RM picking the best, rule-based cleaning, topic classifier, quality score (RM top
  quartile) $\times$ difficulty score (Instag), semantic dedup, **repeated over six rounds**;
  labelled *per-example, model-judged*. Landscape note quotes Llama 3's own framing
  (`01_introduction.md:11`) verbatim and attributes both columns to Llama 3 §3.1/§4.2.3 so it doesn't
  read as a cross-paper splice.
  **Dropped entirely:** the LIMA 1k / Dolly 15k / Alpaca 52k / Llama 2 ~27k / OpenAssistant ~161k
  table and its caption. Those numbers were LLM-generated and never verified, and **none of those
  papers exist in `papers/`** — nothing in that table was citable from this repo. Yakov's explicit
  call was to drop rather than verify or replace; the DeepSeek 1.5M figure plus the LIMA frame right
  after already carry the scale point with real sources.
  **Downstream fix:** the LIMA frame's landscape note said *"the 'not too little, not too cheap' data
  from two slides ago"* — that phrase came from the deleted caption, so the callback would have
  dangled; reworded to *"the curation pipeline from two slides ago."*
  **Mechanics/verification:** rewrote both frames in place, no new preamble macros (reused
  `\landscapenote`, `AtlasWarm`/`AtlasCool`/`NoteGray`, `booktabs`, the deck's existing
  `columns`+`itemize` idiom). Dropped an initial `leftmargin=*` on the itemizes to match the deck's
  other lists. `pdflatex` compiles clean at **15 pages** (unchanged — two frames replaced by two),
  no new overfull boxes; both pages rendered to PNG and checked visually, twice (first pass had the
  Data cell's "·" separator dangling at a line break and an `\hfill` that didn't take in the
  landscape note — both fixed). **Note on the deck's look:** bulletless `itemize` is deck-wide
  convention (frame 5 renders the same way), not a bug on these frames.
  **Bonus finding, not acted on:** Llama 3 `04_post_training.md:24` states SFT uses "a standard cross
  entropy loss on the target tokens (while masking loss on prompt tokens)", LR 1e-5 over 8.5K-9K
  steps — a direct citation for the Anatomy/Training frames' masking claim, currently uncited there.
  Worth adding to those frames' speaker notes; deliberately out of scope for this change.
- **2026-09-07 — Slide-by-slide content review of `sft_pres.tex` started; reviewed through frame 8
  ("The Anatomy of an Instruction Example") so far, no gaps found yet in frames 1-8.** Frames 1-7
  (title, Two Purposes, Alignment SFT example, Domain SFT example, Training: Same Loss/Adapted
  Rules, its map-return, The LLM Alignment Pipeline) passed with no changes. Frame 8 got one
  content fix: **replaced the "`[Prompt] + [Assistant]` $\rightarrow$ Loss" formula box with a TikZ
  role diagram** — Yakov flagged that the equation-style formula doesn't actually communicate the
  masking idea (reads as if both operands feed the result equally). New diagram: three role blocks
  (gray System, gray User, colored Assistant), a brace under System+User labeled "Prompt — context
  only, no gradient," and an arrow to a "Loss $\mathcal{L}$" box that starts *only* at the Assistant
  block — so the absence of an arrow out of System/User carries the "no gradient from the prompt"
  point visually, not just in the caption. Iterated once more on Yakov's "make it slightly smaller"
  follow-up (node sizes 0.85cm/1.7-2.1cm → 0.6cm/1.3-1.6cm, font `\small`→`\footnotesize`, brace
  label `\footnotesize`→`\scriptsize`, Loss box distance 1.15cm→0.85cm). Required adding
  `decorations.pathreplacing` to the `\usetikzlibrary{...}` call (line 9) for the brace decoration —
  first use of a brace anywhere in the deck. Updated the frame's speaker note to describe the new
  diagram (was describing the old formula box) and to explicitly call out that the arrow starting
  only at Assistant is itself part of the point, not just the caption underneath it. Verified via
  `pdflatex` compile (15 pages, no errors) + rendered slide 8 to PNG and visually checked both
  passes. **This review pass is ongoing — frames 9 onward (Resource Cost through LoRA Inside the
  Transformer, frames 9-15) not yet reviewed this pass;** see Next steps.
- **2026-09-07 — `excalidraw-export` now auto-starts a persistent background daemon, reused across
  every repo, instead of launching a fresh browser on every invocation.** Follow-up to the same-day
  batching fix below — batching helped within one build (all files in one `excalidraw-export` call)
  but every separate call (each LaTeX build, each manual re-export) still paid Chromium's ~300ms
  launch/bundle-load cost from scratch. `~/.local/share/excalidraw-export/daemon.mjs` now keeps one
  browser + one ready Excalidraw export page alive over a Unix socket
  (`~/.local/share/excalidraw-export/daemon.sock`); `render.mjs` (the CLI) tries it first, auto-spawns
  it on first use if not running, and falls back to a one-shot standalone browser automatically if the
  daemon can't come up — so correctness never depends on it. Idle-shuts-down after 10 minutes; `excalidraw-export
  --daemon-stop` stops it manually; `--no-daemon` forces standalone. Verified: cold start (no daemon)
  ~4.3s including daemon spawn, subsequent single-file calls ~0.4-3.4s (varies with diagram size —
  see caveat below), stop/restart clean, and this project's own `map.png` export still correct through
  the shared daemon.
  **Found and fixed two real bugs while building this** (not just slow, actually broken): (1) Node's
  `net.Server` defaults to `allowHalfOpen: false`, so the daemon was auto-closing its own writable side
  the instant the client's request half-close (FIN) arrived — meaning every response the daemon tried
  to send after finishing the (necessarily async) export was silently dropped; fixed with
  `{ allowHalfOpen: true }` on the server. (2) the client's socket had a single `setTimeout` meant only
  to detect "no daemon listening," but it applied to the *entire* connection lifetime — so any export
  taking longer than that timeout (most of them) got treated as "daemon unreachable" mid-response even
  though the daemon was alive and correctly working; fixed by using a timer scoped only to the connect
  phase, with no timeout at all on the response wait. Both were caught by testing end-to-end (raw
  socket probes, not just trusting exit codes) rather than assuming success from a fast return time.
  **Caveat carried over from the batching entry, restated because it's the ceiling on this fix too:**
  the daemon removes fixed per-invocation overhead; it does not make Chromium's screenshot/PNG-encoding
  time (roughly proportional to output pixel count) any faster. A large diagram at `--scale 3` is
  still a multi-second rasterize even on a warm daemon — a lower `--scale` for that specific diagram
  is the next lever if it's still too slow, not further daemon work. See `path_adjust_ml`'s
  `EXCALIDRAW-WORKFLOW.md` for the full write-up.
- **2026-09-07 — `excalidraw-export` sped up: batches multiple files into one browser launch
  instead of one browser per file.** Yakov noticed exports felt slow compared to instant manual
  export from the Excalidraw app. Root cause found by profiling: launching headless Chromium + the
  bundled Excalidraw renderer costs a fixed ~300ms, paid **per invocation** — the old `.vscode/settings.json`
  wiring (both here and in `path_adjust_ml`) shelled out to `excalidraw-export` once per `.excalidraw`
  file in a loop, so a 6-diagram repo paid that cost 6 times for no reason. Fixed at the tool level
  (`~/.local/share/excalidraw-export/render.mjs`): it now accepts multiple input paths in one call and
  reuses a single browser (a dedicated long-lived page for the JS export step, a fresh short-lived page
  per file only for the final rasterize/screenshot step) across all of them. Updated this project's
  `.vscode/settings.json` to pass every found path to **one** `excalidraw-export` invocation via
  `find -print0 | xargs -0 excalidraw-export --scale 3` instead of a `while read` loop calling it once
  per file — verified end-to-end (`seminar/tex/pres/resources/map.png` still exports correctly).
  **Not fully solved — a real remaining cost, not a bug:** rasterizing to PNG at `--scale 3` takes time
  roughly proportional to output pixel count (Chromium's screenshot/PNG-encoding cost, not diagram
  complexity) — a large diagram at scale 3 can be tens of megapixels and take a couple of seconds to
  encode even by itself. This project only has one `.excalidraw` file so the batching fix mattered less
  here than in `path_adjust_ml` (6 files, ~18s to ~10s); if this project's own build ever feels slow
  despite the fix, a lower `--scale` for that specific diagram is the next lever, not further tool
  changes. See `path_adjust_ml`'s `EXCALIDRAW-WORKFLOW.md` for the full write-up.
- **2026-09-07 — Found (in a sibling session, on `path_adjust_ml`) that `map.png`'s handwritten-looking
  text is actually rendering in a generic system font, not Excalidraw's real Virgil font — root cause
  applies equally to this project's own `map.png`/`.vscode/settings.json` wiring, not yet fixed here.**
  Root cause: `excalidraw-export` (`@moona3k/excalidraw-export`) embeds the Virgil font as a base64
  `@font-face` in the SVG it generates, but rasterizes with `@resvg/resvg-js`, which does not honor
  CSS `@font-face`/data-URI fonts — only fonts resolvable via its own system-font database
  (`loadSystemFonts: true`). So every PNG this tool produces silently falls back to a generic
  cursive/system font for all text, while the hand-drawn shapes (vector paths from `roughjs`) still
  render correctly — meaning **every `map.png` regenerated by the `excalidraw-compile` build step in
  this project's `.vscode/settings.json` has had non-handwritten text**, not just the one investigated
  in `path_adjust_ml`.
  **Fix built in `path_adjust_ml`** (`docs/tex/styles/excalidraw-export/`, see that repo's
  `EXCALIDRAW-WORKFLOW.md`): a local Node/Puppeteer tool that calls Excalidraw's own `exportToSvg`
  (via `@excalidraw/utils`) inside headless Chromium instead of `resvg` — same rendering path a manual
  export from the Excalidraw app/VS Code extension/Obsidian plugin uses, so fonts match exactly. Fully
  offline after a one-time `npm install`/`npm run build` (Puppeteer's headless-Chromium download is the
  only network step; export itself makes no network calls). Same CLI shape as the old tool
  (`node render.mjs <input.excalidraw> [-o path] [--svg] [--scale n] [--no-background]`).
  **Applied same day (2026-09-07), then superseded same day by a global install (see next
  paragraph):** copied `path_adjust_ml`'s `docs/tex/styles/excalidraw-export/` tool into this
  project as `seminar/tex/styles/excalidraw-export/`, pointed `.vscode/settings.json` at it. This
  per-repo-copy approach was then replaced — see below — so that copy no longer exists in this repo.
  **Final form:** the tool's source now lives once, machine-wide, at
  `~/.local/share/excalidraw-export/` (outside any project repo), installed as a global `excalidraw-export`
  command via `npm install -g .` from that source directory — a drop-in replacement for the old
  `@moona3k/excalidraw-export` npm package under the **same command name**, so no per-repo copy is
  needed at all (mirrors how the old tool worked: one global install, every repo's build wiring just
  calls `excalidraw-export`). `path_adjust_ml`'s copy was removed the same way once this existed.
  This project's `.vscode/settings.json` `excalidraw-compile` step now calls the global
  `excalidraw-export` binary (with the same `BIN=$(command -v excalidraw-export || echo
  "$HOME/.nvm/versions/node/v22.17.0/bin/excalidraw-export")` fallback pattern the old wiring used,
  for shells/tasks that don't inherit nvm's `PATH`), and the `find ... -print0 | while IFS= read -r
  -d ''` loop (not `for f in $(find ...)`, which silently breaks on this project's absolute path
  containing a space, `2-1_Projects (current)`) — verified by running the exact substituted command
  by hand. Re-ran the export once by hand afterward: `seminar/tex/pres/resources/map.png` now renders
  its text in Excalidraw's real handwritten (Virgil) font — confirmed visually against the previous
  (system-font) version. Going forward this happens automatically on every LaTeX build via the
  existing recipe, same as before the switch, just pointed at the new global tool.
  **Updating the tool going forward:** edit files directly under `~/.local/share/excalidraw-export/`
  (not tracked in this project's or `path_adjust_ml`'s git repo — it's shared, machine-local tooling);
  `npm install -g .` on a local path symlinks rather than copies, so edits to `render.mjs`/`entry.js`
  there take effect immediately without reinstalling — re-run `npm run build` there after editing
  `entry.js` specifically (needs re-bundling). See `path_adjust_ml`'s
  `docs/tex/styles/EXCALIDRAW-WORKFLOW.md` for the full write-up (canonical doc for this tool; this
  project doesn't duplicate it).
- **2026-09-06 — Found the deck restructured and annotated with red author-notes since the last
  tracker write-up (edit happened outside this session — discovered when the `.tex` file was
  re-read mid-session).** Two changes: (1) "Training: Same Loss, Adapted Rules" moved to right
  after the Domain SFT example (both now `\atlasslide`-wrapped) and ahead of the Pipeline slide,
  with a map-return frame now actually revisiting the Training slide's sidebar map — the first live
  use of the progressive-reveal system documented in the "Reference" section above (previously
  built but unused); (2) five slides (Resource Cost, LIMA Effect, ICL & RAG, Full FT vs. PEFT,
  LoRA Inside the Transformer) each gained a red-colored `\note{}` — distinct from the plain
  (black) delivery notes like the one added to the Anatomy slide below — flagging revision/research
  TODOs Yakov left for himself (see Open decisions #6 for the full list). Rewrote "Current state"
  and the map-return reference section to match; did not touch any of the flagged slides' content
  or notes themselves, this pass was purely descriptive.
- **2026-09-06 — Added a speaker note to the "Anatomy of an Instruction Example" slide.** Yakov
  asked for notes explaining what to explain and the purpose of each sentence/word on the slide
  (previously had no `\note{}` at all). Added a `\footnotesize` note covering: the slide's overall
  purpose (show the physical JSON format + set up loss-masking for the next slide's formula), what
  each ChatML role (system/user/assistant) means and why it's included, how the flattening into one
  token sequence connects to the next slide's $x_1,\dots,x_N$ notation, the formula box
  (`[Prompt]+[Assistant]→Loss`) as the compressed mechanical rule, the "Gold Target" caption as the
  supervised-learning-label framing, and the bottom bullets as talking points (not a re-read of the
  code). No slide-body content changed, only the note.
- **2026-09-06 — Expanded the "Training: Same Loss, Adapted Rules" slide's speaker note with an
  explicit breakdown of the loss formula's notation.** Yakov asked what's literally plugged into
  $\mathcal{L}=-\sum_{t=i}^{N}\log P(x_t\mid x_{<t})$ (currently shown with bare probability
  notation, no explanation of what each symbol is). Added a paragraph at the top of the slide's
  existing `\note{}` (didn't create a separate note-slide/frame — Beamer's per-frame `\note{}`
  already covers this) explaining: $x_1,\dots,x_N$ is the full tokenized example (system prompt +
  chat-template tokens + user turn + assistant turn, concatenated as the model sees it at
  inference); $x_t$/$x_{<t}$ are the token at position $t$ and everything before it; $P(x_t\mid
  x_{<t})$ is literally the model's own softmax output at position $t$, not a separate quantity;
  $N$/$i$ are sequence length and the index of the first assistant token (why the sum only covers
  assistant tokens = the masking bullet below it). Initial version also traced the formula through
  the slide's own "2+2" worked example token-by-token, but Yakov flagged the note as overwhelmed
  with text, spilling over the notes-pane page (`\setbeameroption{show notes on second screen=right}`
  — narrow pane) — trimmed to one tight paragraph, dropped the token-by-token trace (the formula
  explanation plus the slide's own worked example already cover it).
  **Follow-up same day:** Yakov wanted the note framed explicitly as annotation-vs-prediction
  rather than just naming the existing symbols — rewrote to introduce $\bar{x}_t$ (the ground-truth
  annotation/label from the training example) and $\mathcal{N}(x_{<t})$ (the model's predicted
  distribution given input $x_{<t}$), then ties back to the slide's own $P(x_t\mid x_{<t})$ notation
  via $P(x_t\mid x_{<t}) := \mathcal{N}(x_{<t})[\bar{x}_t]$. Also switched the entire `\note{}` block
  to `\tiny` (was default note-font size) since it still didn't fit at the previous length/size.
  **Follow-up same day:** `\tiny` read as too small — bumped to `\footnotesize` and trimmed the
  wording slightly further (dropped a few redundant words/examples) to keep it fitting at the
  larger size. No frame count change — still 14 frames in `sft_pres.tex`.
- **2026-09-06 — Added `\atlasmapcover`/`\atlasmapcoverbelow`/`\atlasmapcoverabove` to
  `beamerthemeAtlas.sty`, enabling progressive-reveal map-return frames off a single master
  Excalidraw image.** Trigger: Yakov wanted to show the mind-map sidebar (see new "Reference: the
  Atlas map-return system" section above) gradually — more components exposed at each successive
  map-return frame — without maintaining a separate exported image per stage. Considered and ruled
  out: (a) manually re-exporting the whole canvas per stage, (b) one `.excalidraw` file per stage
  (works, but duplicates elements across files), (c) extending the `excalidraw-export` CLI
  (`@moona3k/excalidraw-export`, confirmed via reading its source to have no frame-filtering
  support) to add per-Excalidraw-frame export. Yakov's proposed alternative, adopted: keep one master
  image, overlay a white rectangle in the `.sty` that shrinks slide-to-slide to reveal more of the
  same diagram — no new export tooling, single source image, reuses `\atlashighlight`'s existing
  local-TikZ-coordinate convention. Purely additive to `beamerthemeAtlas.sty` (new macros only,
  nothing existing changed); not yet wired into any actual frame in `sft_pres.tex`. Also discovered
  and documented while investigating this (see the new reference section): `map.png`'s apparent
  auto-sync with `map.excalidraw` is **not** any Excalidraw VS Code extension's doing — it's
  `.vscode/settings.json`'s custom `excalidraw-compile` LaTeX Workshop build step, which re-exports
  every `.excalidraw` file in the project via the `excalidraw-export` CLI on every build.
- **2026-09-06 — Switched the deck from Marp/Markdown to LaTeX/Beamer.** Yakov's call: the live
  deck is now `seminar/tex/pres/sft_pres.tex`, not `sft.md`. All 14 front-section slides from the
  Marp deck (see "Current state" below, which still describes their content) were ported over, with
  **one reorder**: the Domain SFT example slide now sits right after the Alignment SFT mindset-
  shift example, ahead of the Pipeline and Resource Cost slides (see "Current state" for the full
  actual order) — the two worked examples are now back-to-back instead of split across the
  pipeline/cost overview. Ported using Beamer/TikZ/tcolorbox equivalents of the old HTML/CSS
  components:
  `warmbox`/`coolbox` (↔ the old `mini-cols`/domain-example color coding), `goodbox`/`badbox`
  (↔ `verdict-box`), `formulabox` (↔ boxed formula), `noteframe` (↔ the LIMA definition box),
  `\landscapenote`/`\slidecaption` macros (↔ `.landscape-note`/caption text classes), and TikZ
  `tikzpicture`s replacing the hand-rolled flow-step/diagram divs (pipeline diagram, resource-cost
  table, LIMA win-rate bar chart, ICL/RAG pipeline diagram, full-FT-vs-LoRA parameter-matrix
  diagrams, LoRA-in-transformer weight-matrix diagram). Styling helpers (box environments, colors,
  macros) live directly in `sft_pres.tex`'s preamble — there is no separate external theme-CSS file
  this time (unlike the old Marp `marp-theme/sft.css` split). The old Marp deck's *frozen archive*
  (slides 15-28) was **not** ported — it does not exist anywhere in the `.tex` file; see the
  archive-boundary workflow note above, still unresolved. **`sft.md` itself was not deleted, just
  stopped being edited** — it remains on disk as a historical/frozen artifact of the old workflow,
  same status as its own former archive section. Going forward: all slide edits happen in
  `sft_pres.tex` directly; this tracker's per-slide "Current state" descriptions and decisions-log
  history below still accurately describe *content*, just not the file format it now lives in.
- **2026-09-06 — Added new slide 4, "Resource Cost Across the Pipeline," right after the pipeline
  slide (now slide 3).** Yakov wanted the pipeline slide (or one right after it) to carry resource
  requirements per stage — data size, data generation cost, training cost, data quality — and
  pointed at the two papers in `4_docs/llm/papers/models/` (DeepSeek-V3 Technical Report 2025,
  Llama 3 Herd 2024) as the reference. Researched both:
  **DeepSeek-V3** publishes an actual stage-by-stage $/GPU-hour breakdown (`01_introduction.md`):
  Pre-Training 2,664K H800-hrs / $5.328M, Context Extension 119K-hrs / $0.238M, Post-Training
  (SFT+RL combined) 5K-hrs / $0.01M, Total 2,788K-hrs / $5.576M — pretraining on 14.8T tokens
  (`04_pre_training.md`); SFT itself used 1.5M curated instruction instances, generated via
  R1-distillation for reasoning data and DeepSeek-V2.5-generation-plus-human-verification for
  non-reasoning data, 2 epochs (`05_post_training.md`).
  **Llama 3** has no comparable cost table in the extracted text, but gives data-side detail instead
  (`01_introduction.md`, `03_pre_training.md`, `04_post_training.md`): 405B model pretrained on
  15.6T tokens at 3.8×10^25 FLOPs on up to 16K H100s; post-training ran **six iterative rounds** of
  human annotation + rejection sampling + preference tuning (SFT and DPO interleaved, not run once);
  Table 7 gives SFT data composition by category (% of examples, avg. tokens) though not a total
  example count; the code-SFT subset alone used 2.7M synthetic examples.
  **Slide content:** a 2-row `decode-table` (Pre-training vs. Post-training [SFT+RLHF/DPO]) with
  Data and Training Compute columns, mixing both papers' numbers, plus a `landscape-note` stating the
  >500x cost gap and flagging the caveat that neither paper splits SFT cost from RLHF/DPO cost —
  real pipelines interleave both across multiple rounds rather than running each once. Reused
  existing `decode-table-wrap`/`decode-table`/`landscape-note` CSS classes only — no new theme CSS.
  **Mechanics:** inserted as new slide 4 (old slides 4-27 renumbered to 5-28 via a scripted sed pass
  over the `<!-- Slide N -->` comments, processed highest-number-first to avoid collisions); archive
  marker moved from slide 14 to 15. This tracker's "Current state" section, mapping table, and Open
  decisions section were rewritten to match the new numbering — see below for the numbering-mismatch
  finding this also cleaned up.
  **Also resolved while rewriting "Current state":** the 2026-08-30-vs-actual slide 2/3 order
  mismatch flagged earlier today (tracker said 2=pipeline/3="Two Purposes"; file actually has
  2="Two Purposes"/3=pipeline) — confirmed the file's order is correct and long-standing, tracker
  text was simply never updated to match; corrected in place, no slide content changed. The stray
  `****` on slide 3's `<!-- Slide 3 -->` HTML comment is still present, still untouched, still just
  a cosmetic artifact, doesn't affect rendering.
- **2026-09-06 — Corrected slide 10's prompt-engineering-vs-alignment claim.** Trigger: while fixing
  slide 2 (below), Yakov corrected the underlying claim that had leaked onto both slides — prompt
  engineering (few-shot / chat-style in-context examples) *can* approximate alignment behavior in a
  base model; it's not strictly domain-SFT-only. The tradeoff is cost/durability (pay per request,
  fragile) vs. SFT baking it into the weights once — not "impossible." Slide 10's `landscape-note`
  rewritten from "It's not a substitute for alignment SFT: a base model has no instruction-following
  behavior to prompt-engineer in the first place" to the more accurate framing above. Yakov also
  confirmed the general claim that prompt engineering / LoRA / RAG are *mostly* alternatives to
  domain SFT specifically — that's still right and is what Part B (slides 10-13) is for — but per the
  next entry, it's premature to raise on slide 2.
- **2026-09-06 — Trimmed slide 2 ("Two Purposes of SFT") for verbosity, then removed its
  prompt-engineering aside entirely.** First pass: Yakov flagged the slide as too wordy; shortened
  both columns' `small-note` text (dropped the "shown on the next slide" / "shown a couple slides
  from now" cross-references, which were also stale — see numbering note below) and condensed the
  `landscape-note`. Second pass, same session: the condensed `landscape-note` still said "Prompt
  engineering only substitutes for the second kind" — Yakov flagged this as too early to mention
  (alternatives are Part B's job, slides 10-13) and, separately, not fully accurate (see slide-10
  entry above). Line removed; `landscape-note` now just states the shared mechanism, no alternatives
  mentioned at all on slide 2.
  **Numbering discrepancy found while doing this, not yet resolved:** this tracker's "Current state"
  section (last updated 2026-08-30) documents the front-section order as 2=pipeline, 3="Two Purposes
  of SFT", but the actual `sft.md` file has them swapped — slide 2 is "Two Purposes of SFT" and slide
  3 is "The LLM Alignment Pipeline". Slides 4 (mindset-shift example) and 5 (domain example) match
  what's documented. Also noticed slide 3's `<!-- Slide 3 -->` HTML comment has a stray `****` suffix
  on that line. Neither was touched — flagging for Yakov to confirm which order is intended before
  the tracker's slide map and numbering get corrected.
- **2026-08-30 — Major restructure: split the front section into "Part A: Technicalities & Why"
  (slides 6-9, new) and "Part B: Alternatives & Variants" (slides 10-13, new), pushing the archive
  marker from slide 6 to slide 14.** Trigger: Yakov wanted (a) prompt engineering discussed later,
  as one of several alternatives rather than right after the mindset-shift examples; (b) in-context
  learning + RAG added as another alternative; (c) LoRA/SVD low-rank fine-tuning treated as a
  *variant* of SFT (not an alternative to it) with more precision on how it actually attaches to the
  transformer architecture; (d) the instruction-dataset anatomy, the LIMA effect, and the fact that
  fine-tuning uses the same CE loss as pretraining (with masking + a lower learning rate) all pulled
  into the front section as the technical core, ahead of the alternatives/variants discussion. All
  of this content already existed in the archive in some form (slides 15-27 after renumbering) —
  per Yakov's explicit instruction this round, **nothing in the archive was touched**; every new
  slide is freshly authored content in the front section, reusing archived material only as a
  starting point where relevant.
  **New slides, in order (see Current state above for full descriptions):** 6 data cost/scale
  (new content — not previously in the deck in any form), 7 anatomy of an instruction example
  (re-authored from archived 16), 8 LIMA effect (re-authored from archived 17), 9 same-CE-loss +
  chat-template-tokens + lower-LR (condensed from archived 19-21; the elaborate role-bleeding demo
  on archived 19 was *not* recreated — Yakov's ask was specifically about the loss/LR fact, kept
  concise), 10 prompt engineering (re-authored from archived 15, now explicitly scoped to domain SFT
  — this also resolves former Open decision #5, which had been deferred), 11 in-context learning +
  RAG (new content, covers plan topic 8), 12 full-FT-vs-PEFT (re-authored from archived 22), 13 LoRA
  inside the transformer (new — names Wq/Wk/Wv/Wo as the standard LoRA targets and the MLP
  up/down-projections as the optional, costlier target; archived slide 23's generic W₀ box diagram
  didn't have this level of architectural detail).
  **Numbers Yakov asked me to fill in, worth double-checking:** slide 6's dataset-size table (LIMA
  1K, Dolly 15K, Alpaca 52K, Llama 2 SFT ~27K, OpenAssistant ~161K) and slide 9's CE-loss/LR framing
  (same objective as pretraining, masked to assistant tokens, ~2e-5 full FT / ~1e-4 LoRA — these LR
  figures were already in the archived deck at slide 25, not new).
  **New CSS added to `marp-theme/sft.css`:** `.transformer-block`, `.tb-layer`, `.tb-layer-label`,
  `.tb-matrices`, `.tb-matrix`, `.tb-matrix.lora` (pill-badge weight-matrix diagram for slide 13).
  **Not done / explicitly out of scope this round:** synthetic data generation (archived 18), the
  full chat-template necessity/solution demo (archived 19-20), QLoRA/quantization (archived 24),
  training practicalities and hyperparameters (archived 25), overfitting/forgetting (archived 26),
  and evaluation (archived 27) — none of these were pulled into the front section; see Open
  decisions #2-3 for what's still outstanding. Topic 7 (limitations of SFT) still has no slide
  anywhere, front section or archive.
- **2026-08-30 — Fixed slide 5's "bad" completion text.** The general-instruct-model
  completion read "Moderate risk identified, but prose format — can't be parsed" —
  self-referential meta-commentary a model wouldn't actually produce (the "can't be
  parsed" point already belongs to, and duplicates, the `verdict-clarification` line
  below the box). Replaced with a short realistic prose answer: "Moderate risk —
  distracted pedestrian near the crosswalk, truck blocking the bike lane ahead.
  Recommend slowing down."
- **2026-08-30 — Fixed slide 3 box text size and slide 4/5 box-height mismatch.**
  Slide 3: the two top `mini-cols` boxes read too small (17px/16px/15px) next to the
  landscape-note below (28px). Added an opt-in `.mini-cols-lg` modifier class (title
  19&rarr;23px, mini-note 16&rarr;19px, small-note 15&rarr;18px) applied only to slide 3's
  `mini-cols` div, so archived slides 14/17/18 (same `.mini-cols` base class) keep their
  original sizing untouched.
  Slides 4 & 5: the two `verdict-box`es (bad/good) were rendering at different heights
  even when the *shorter*-captioned one had less box content — root cause was
  `.verdict-pair` being an independent flex column per side, so `.verdict-box`'s
  `flex:1` absorbed whatever vertical space its own sibling `.verdict-clarification`
  paragraph didn't use, with no coupling between the two pairs' box heights. Fixed by
  adding an opt-in `.paired` modifier on `.verdict-grid` (applied to slides 4 & 5 only)
  that switches the grid to `display:grid` with `grid-auto-flow:column` over an
  explicit 2-row/2-column track, and sets `.verdict-pair` to `display:contents` so the
  box and clarification become direct grid items — row 1 (both boxes) and row 2 (both
  clarifications) now each size to their tallest cell and stretch both columns to
  match. Slide 11 (archived, uses bare `.verdict-box` pairs with no `.verdict-pair`
  wrapper and no `.paired` class) is unaffected and keeps its original flex layout.
- **2026-08-30 — Added "Two Purposes of SFT" (new slide 3) and a domain-SFT example (new slide 5),
  bracketing the existing alignment-SFT mindset-shift slide (pushed to slide 4).** Trigger: Yakov
  noticed that the archived prompt-engineering-vs-SFT slide (now slide 7) doesn't actually compare
  against SFT in general — prompt engineering can't realistically substitute for *alignment* SFT
  (teaching a base model to recognize instructions at all; you'd have to hand-write the equivalent
  of the training data at inference time, every call), it's only a real alternative to *domain/
  application* SFT (further specializing an already-instruction-following model for one product's
  schema/vocabulary/edge cases). Slide 3 now names this split explicitly so the reader has the
  distinction before hitting the alignment example (slide 4) and the domain example (slide 5).
  Domain example chosen: an autonomous-driving copilot — general instruct model answers a risk
  question in free prose (not consumable by a planner); the domain-SFT'd version emits a structured
  JSON schema (actor list + risk levels + recommended action) instead, same underlying
  instruction-following ability, just taught the team's own format. Both new slides reuse existing
  CSS classes only (`mini-cols`/`mini-col` for the concept slide, `verdict-grid`/`verdict-box` for
  the domain example, matching slide 4's pattern) — no new theme CSS needed.
  **Deliberately not done as part of this change:** slide 7 (prompt engineering) was NOT reframed to
  explicitly scope itself to domain SFT — Yakov said this "can be avoided for now" (see Open
  decisions #5). Pacing was also explicitly waved off — see Open decisions #4.
- **2026-08-30 — Inserted an archive-boundary marker slide (new slide 4); old slides 4-16 renumbered
  to 5-17, content unchanged.** Yakov's call: slides 1-3 are approved and stay the active working
  front of the deck; everything from the old slide 4 onward is now a frozen archive that should not
  be edited going forward. New slide 4 is a title-slide-styled divider ("Archived From Here Down")
  stating the boundary. **Going forward:** all new content (topics 7/8, any further restructuring)
  gets inserted as new slides between slide 3 and this marker — pushing the marker and the archived
  slides further down the deck — rather than edited into the archived slides in place. See the
  workflow note near the top of this file for the durable version of this rule.
- **2026-08-30 — Fixed the text-completer example on slide 3 (was flagged weak same day, see entry
  below).** Replaced the "bad" completion's trivia-pivot (Tony Hoare/1959 history) with the
  worksheet/list-continuation example floated in the earlier entry: the base model responds to
  "Write a python function for quicksort" by continuing with "Write a python function for
  mergesort. Write a python function for binary search. Write a python function for…" — i.e. it
  treats the prompt as one item in a worksheet rather than a request. Caption line updated to match
  ("treats the prompt as one item in a worksheet, not a request — keeps completing the list instead
  of answering it"). Rationale: the old example wasn't actually the statistically likely
  continuation of that prompt (code is), so it undercut its own "predicts next likely tokens"
  framing; the list-completion version is a more faithful illustration of raw text-completion
  behavior on an instruction-shaped prompt.
- **2026-08-30 — Fixed slide 2 vertical spacing; confirmed the external-theme-CSS reload tradeoff.**
  Slide 2 (`center-body`) had its content vertically centered via `margin: auto` on all non-h1
  children, which sat too low. Tried removing the auto margins entirely (content jumped to sit
  right under the heading — too high), then added a fixed `margin-top: 120px` +
  `margin-bottom: 24px` on `section.center-body h1 + *` (targets just the flow-steps div
  immediately after the h1 on slide 2) to land it in between, with a proper gap before the bullet
  list. **Confirmed a plugin limitation while doing this:** the Obsidian Marp Slides plugin only
  live-reloads on edits to the open `.md` file — it does not watch files in the theme folder
  (`marp-theme/sft.css`), so any CSS-only change requires closing/reopening the preview to see it.
  Yakov's call: live with it (close/reopen after CSS edits) rather than move CSS back into `sft.md`
  or chase a plugin setting/fix for this.
- **2026-08-30 — REVERTED the macro-preprocessor workflow below.** Yakov's call: "it's too much" —
  switching between a `.src.md` source file, a compile step, and the generated `.md` added more
  friction than it removed. Deleted `sft.src.md` and `expand_marp_macros.ps1`; reverted slide 3 back
  to plain `verdict-grid`/`verdict-box` HTML (matching slides 8/9); removed the now-unused
  `.mbox`/`.mbox-label`/`.mbox-center`/`.mcols`/`.mcol` rules from `marp-theme/sft.css`. **Kept:** the
  external theme CSS split (`marp-theme/sft.css`, `theme: sft`) — that part wasn't in question, only
  the macro layer on top of it. **Going forward:** `sft.md` is the only file to edit; Yakov describes
  the change in plain language, Claude edits the HTML/CSS directly. Do not re-propose a macro/DSL/
  source-file preprocessing approach for this deck unless Yakov brings it up again himself.
- **2026-08-30 — Built a macro preprocessor (`\box`/`\cols`) so Yakov can avoid hand-writing HTML for
  the deck's repeated colored-box/side-by-side components; piloted on slide 3. REVERTED same day, see
  entry above — kept here only for the "why we tried and why it didn't stick" context.**
  Source lives in `4_docs/llm/seminar/sft.src.md`; `sft.md` is now generated from it (see the
  Workflow note at the top of this file). Script: `1_resources/Scripts/expand_marp_macros.ps1`.
  **Why PowerShell and not Python:** first wrote it in Python, but this machine has no real Python
  install — only Windows' Store-install stub aliases (`python`/`python3` both fail with "Python was
  not found; run without arguments to install from the Microsoft Store"), and no Node.js either. Note this
  likely also affects the `/mathpix` workflow (`1_resources/Scripts/mathpix_convert.py`,
  `split_md.py`), which also don't currently exist in `1_resources/Scripts/` despite being documented
  in the root `CLAUDE.md` — worth checking next time that command is used. PowerShell was already
  guaranteed available, so the macro expander was ported there instead of asking Yakov to install a
  runtime for this one thing.
  **Macro syntax (v1):**
  ```
  \box[bg=#hex, border=#hex, label="Some Label", align=center]{
  Markdown content — **bold**, *italic*, `code`, lists, $math$ all work,
  as long as blank lines separate paragraphs (the script wraps content in
  blank lines so Marp's HTML-block parsing lets normal Markdown through).
  }

  \cols{
  \box[...]{ ... }
  }{
  \box[...]{ ... }
  }
  ```
  `\cols` takes 2–3 `{...}` blocks side by side; `\box`/`\cols` can nest; brace-matching is depth-aware
  so LaTeX braces in math (e.g. `\sum_{t=i}^{N}`) inside a box are safe. New CSS added to
  `marp-theme/sft.css`: `.mbox`, `.mbox-label`, `.mbox-center`, `.mcols`, `.mcol`.
  **Why not the plugin's built-in `:::` container syntax instead:** ruled out already (see entry
  below) — it only supports one hardcoded class name, can't do per-component styling.
  **Scope:** only slide 3's verdict-grid was converted, as a pilot — same "convert one, verify
  rendering in Obsidian, then decide whether to roll out further" approach used for the `center-body`
  CSS fix earlier. The other 15 slides' components (frame, verdict-box, mini-cols, bottleneck-grid,
  landscape-note, etc.) are NOT yet converted and still contain raw HTML in `sft.src.md` — converting
  those to `\box`/`\cols` is the natural next step once Yakov confirms slide 3 looks right and the
  syntax feels good to write. The deck's bespoke one-off diagrams (flow-steps arrows, lora-diagram,
  quant-bars, delta-grid, stack-bar, batch-boxes, context-strip, lima-chart) are intentionally OUT of
  scope for `\box`/`\cols` — they're custom visuals, not a reusable "colored box" pattern, so
  genericizing them wouldn't actually be simpler than the HTML already there.
- **2026-08-30 — Moved all CSS out of `sft.md` into `4_docs/llm/seminar/marp-theme/sft.css`
  (a custom Marp theme, `/* @theme sft */`).** `sft.md` frontmatter now just says `theme: sft` instead of
  embedding ~355 lines of `style:`. Reason: Yakov isn't comfortable authoring raw HTML/CSS and wanted the
  styling definitions out of the file he actually edits.
  **Action needed (one-time, in Obsidian):** Settings → Community plugins → Marp Slides → set
  **Theme Path** to `4_docs/llm/seminar/marp-theme`. The plugin loads *every file* directly inside that
  folder as theme CSS (checked its source — `main.js`, `onOpen` handler) for both live preview and export,
  so that folder must contain only `sft.css` and nothing else, ever.
  Investigated whether the plugin's bundled `markdown-it-container` support (`::: name ... :::` syntax)
  could replace the repeated `<div class="...">` component wrappers (verdict-box, mini-cols, etc.) with
  simpler syntax — confirmed by reading the plugin's `main.js` that it registers the container plugin with
  a single hardcoded name, `"container"` (`marp.use(markdownItContainer, "container")`), so every `:::`
  block renders as generic `<div class="container">` regardless of what word follows `:::` — it is not a
  per-component templating mechanism. Ruled out; not worth building on. A separate HTML-snippet "cookbook"
  file (copy-paste reference, no new syntax) is the fallback plan for reducing the authoring burden on the
  remaining inline HTML — not yet built.

- **2026-08-30 — Reordered slides 2-3.** Pipeline slide ("The LLM Alignment Pipeline") now leads (slide 2);
  mindset-shift/example slide ("Pre-training vs. SFT") follows (slide 3). Rationale: position SFT in the
  pipeline first, then land the before/after example as "here's what that stage buys you." Both slides
  otherwise judged solid regardless of how topics 7/8/evaluation shake out.
- **2026-08-30 — Flagged the text-completer example (now slide 3) as weak, not yet fixed.** The "bad"
  completion has the base model launch into quicksort's invention history (Tony Hoare, 1959) — unrealistic;
  that prompt phrasing is common in tutorials/exercises, so a base model would more likely just emit code or
  a docstring, not pivot to trivia. Reads as a strawman rather than the real failure mode. Better
  alternative floated: base model continuing with a *list* of similar exercises ("Write a python function
  for mergesort...") to show it's completing a worksheet, not answering — still **not yet applied**,
  Yakov wants to hold off changing it for now.

## Next steps

- [ ] Resolve the remaining open decisions above (evaluation slide fate; add Limitations of SFT;
  whether/how to pull training-practicalities content in) — add as new frames in `sft_pres.tex`.
  The old archive-marker convention doesn't currently apply (see archive-boundary workflow note —
  status unresolved after the LaTeX switch). Prompt-engineering reframing and RAG are done as of
  2026-08-30; resource-cost slide is done as of 2026-09-06 (see decisions log).
- [ ] Work through the four remaining red author-notes (Open decisions #6): find references for
  LIMA's numbers/claims, locate the Google RL/fine-tuning paper, pull RAG-paper material and
  redo its diagrams in Excalidraw, resolve the LoRA-expressiveness-as-regularization question, and
  cover SVD-based methods + check Dima's presentation on the LoRA-in-transformer slide.
- [ ] Once scope is locked, re-time the full deck slide-by-slide against the 35-40 min budget (
  explicitly deferred for now per Yakov's 2026-08-30 call — do not raise pacing again until asked).
- [ ] Continue the in-progress slide-by-slide content review — frames 1-10 reviewed
  (2026-09-07, see decisions log; frame 8's formula box replaced with a role diagram, frames 9-10
  fully rebuilt), frames 11-15 (LIMA Effect, Prompt Engineering vs. Domain SFT, A Third Axis:
  ICL & RAG, Full FT vs. PEFT, LoRA Inside the Transformer) still to go.
- [ ] Cite the masking claim on the Anatomy/Training frames from Llama 3
  `04_post_training.md:24` ("a standard cross entropy loss on the target tokens, while masking loss
  on prompt tokens"; LR 1e-5 over 8.5K-9K steps) — found 2026-09-07, out of scope at the time.
- [ ] Build a companion notebook with coding examples to illustrate key concepts from the talk.
  Candidates: (1) loss masking (mask prompt tokens, train only on assistant — supports Anatomy/
  Training slides), (2) LoRA weight injection (low-rank update B @ A added to frozen W — supports
  LoRA slides), (3) chat template application (JSON messages → tokenized sequence). Priority: loss
  masking has highest signal-to-noise ratio and directly demonstrates the core SFT training claim.
