# Progress — RL Seminar, Part 2 (session handoff file)

> **For Claude, at the start of a session:** read this whole file first. It is the single source of truth for where the work stands.
> 1. Use **Where we are** and **Next steps** to decide what to do. Don't re-derive them by reading the whole deck.
> 2. Use the **File map** to find things. All paths are relative to the repo root `/home/yakov/Documents/docs_/`.
> 3. Follow **Conventions** and **Decisions** when editing the deck.
> 4. **Before the session ends** (or when the user says "update progress"): update **Where we are**, the **Status** table, and **Next steps**; add any new decision to **Decisions**; add one line to the **Session log**. Keep the file short: replace stale text, don't pile on more.

---

## Where we are

_Last updated: 2026-11-09_

- **Task:** building the Beamer deck `seminar/RL_algos/seminar/tex/pres_part2/rl_pres_part2.tex`.
- **Storyline:** CPI → TRPO → GAE → PPO (done), then a bridge to LLM alignment: InstructGPT/RLHF → DPO → GRPO → RLOO.
- **New companion document (2026-11-09):** `seminar/RL_algos/seminar/tex/doc/rl_notes.tex`, ~56 pages. It holds the content of **both** decks (Part 1 + Part 2), with more explanation, plots and boxes. **Workflow: organize the material in this document first, then decide what goes on the slides.**
- **Current focus:** the companion document is drafted end to end (CPI, TRPO, GAE, PPO, LLMs, appendix). It is not yet reviewed against the papers.
- **Open bug in the deck:** in Part 2, `\againframe[label=policy-advantage-recap]{policy-advantage}` does not create a label (the first `[...]` of `\againframe` is the overlay spec), so the "Back to the policy advantage recap" button in the appendix proof is a dead link.

## Next steps

Ordered list; the top item is what to do next.

1. [ ] Read through `doc/rl_notes.tex` and mark what to keep, cut or expand (the document is the place to organize content).
2. [ ] Review the document's new explanations against the papers (the content copied from the decks was already checked; added text and toy plots are ours).
3. [ ] Decide the slide list for Part 2 from the document, including the LLM part (Part 1's deck already has drafted LLM frames that can be reused).
4. [ ] Fix the dead `policy-advantage-recap` link in the Part 2 deck.

---

## File map

| What | Path |
|---|---|
| **The deck (main file)** | `seminar/RL_algos/seminar/tex/pres_part2/rl_pres_part2.tex` |
| Scratch/test copy of the deck preamble (~650 lines; not the main file) | `seminar/RL_algos/seminar/tex/pres_part2/t1.tex` |
| Images used in the deck (`\graphicspath` includes `resources/`) | `seminar/RL_algos/seminar/tex/pres_part2/resources/` |
| Excalidraw figure drafts (not yet in the deck), with index | `seminar/RL_algos/seminar/tex/pres_part2/figures_drafts/README.md` |
| Extra reference material | `seminar/RL_algos/seminar/tex/pres_part2/reference/` (`docs/`, `papers/`, `sft/`) |
| Beamer theme (`Atlas`) and shared images | `seminar/RL_algos/seminar/tex/styles/` |
| Part 1 deck (earlier talk; covers CPI → TRPO → GAE/PPO → RLHF/DPO/GRPO/RLOO) | `seminar/RL_algos/seminar/tex/pres_part1/rl_pres_part1.tex` |
| **Companion document (main file)** — both decks as a document | `seminar/RL_algos/seminar/tex/doc/rl_notes.tex` |
| Document parts: macros/boxes, how-to, CPI, TRPO, GAE, PPO, LLMs, appendix | `seminar/RL_algos/seminar/tex/doc/parts/00_macros.tex` … `06_appendix.tex` |
| Document style (shared) | `seminar/RL_algos/seminar/tex/styles/document-style.sty` |
| Paper copies (Markdown) | `seminar/RL_algos/papers/` |
| Paper figures | `seminar/RL_algos/papers/*_images/` |
| Older RL notes + original source list | `seminar/RL_algos/rl-main.md` |
| LLM part of the seminar (separate thread) | `seminar/llm/` |

**Build:** from `seminar/RL_algos/seminar/tex/pres_part2/`, run `pdflatex rl_pres_part2.tex` (twice, so the navigation and TOC update). The preamble loads the theme from `../styles/`, so it must be built from that folder.

**Build the document:** from `seminar/RL_algos/seminar/tex/doc/`, run `pdflatex rl_notes.tex` twice (it loads `../styles/document-style.sty` and images from `../pres_part2/resources/`). VS Code's LaTeX Workshop auto-builds this file on save; a second build running at the same time corrupts `rl_notes.aux`/`.toc`. When building by hand (or as Claude), use `-output-directory=<scratch folder>`; if the build fails with `\@writefiwritefile`, delete `rl_notes.aux` and `rl_notes.toc` and rebuild.

## Sources

Paper paths are relative to `seminar/RL_algos/papers/`.

| # | Key | Paper | Local copy | In talk? |
|---|---|---|---|---|
| 1 | Kakade2002 | Approximately Optimal Approximate Reinforcement Learning | `approximately_optimal_approximate_RL.md` | Yes (Part I, CPI) |
| 2 | TRPO | Trust Region Policy Optimization | `trust_region_policy_optimization.md` | Yes |
| 3 | GAE | High-Dimensional Continuous Control Using Generalized Advantage Estimation | `gae.md` | Yes |
| 4 | PPO | Proximal Policy Optimization Algorithms | `proximal_policy_optimization_algorithms.md` | Yes |
| 5 | SAC | Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL with a Stochastic Actor | `sac.md` | **Dropped** (see Decisions) |
| 6 | RLHF | Deep Reinforcement Learning from Human Preferences | `rlhf.md` | Yes (LLM part) |
| 7 | DPO | Direct Preference Optimization: Your Language Model is Secretly a Reward Model | `dpo/` (main file `dpo.md`, plus per-section files `02_…`–`14_…`) | Yes (LLM part) |
| 8 | RLOO | Back to Basics: Revisiting REINFORCE-Style Optimization for Learning from Human Feedback in LLMs | `REINFORCE_leave_1_out/` (per-section files `01_introduction.md`–`10_appendix_a_additional_results.md`) | Yes (LLM part) |
| 9 | DeepSeekMath | DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models (GRPO) | `deepseekmat_pushing_the_limits_of_mathematical_reasoning_in_open_language_models.md` | Yes (LLM part) |
| 10 | InstructGPT | Training Language Models to Follow Instructions with Human Feedback (Ouyang et al. 2022, arXiv 2203.02155) | `2203.02155v1.pdf` (PDF only; read with `pdftotext`). A Markdown copy exists at `seminar/llm/papers/Finetune_Base/ouyangTrainingLanguageModels2022.md` | Bridge only (2–3 slides) |

---

## Status

Line numbers drift, so use the `\section` titles and frame `label=`s to find things in the deck.

| Part (deck `\section`) | State | Notes / key frame labels |
|---|---|---|
| Conservative Policy Iteration | Drafted | Gaps 1–2, PDL (`pdl`), policy advantage (`policy-advantage`), improvement bound (`improvement-bound`), CPI algorithm (`cpi-alg`), summary (`cpi-summary`) |
| Recap: Conservative Policy Iteration | Drafted | `two-lanes`, `pa-gradient` |
| Trust Region Policy Optimization | Drafted | `cpi-to-trpo`, `trpo-thm1`, IS (`is-trpo`), CG + line search, `trpo-recap`, `trpo-noexplore` |
| Generalized Advantage Estimation | Drafted | TD(λ) for advantages, actor–critic |
| Proximal Policy Optimization | Drafted | Clipped surrogate, PPO in practice, TRPO vs. PPO |
| Experiments | Drafted | TRPO MuJoCo/Atari, GAE γ/λ, PPO ablation, PPO vs. others, "Whole Story", "One Line per Paper" |
| Appendix | Drafted | Proofs (`proof-pa-1`), Bertsekas & Tsitsiklis comparison (`bt-compare`), Gibbs policy, Hoeffding, IS (`is-cpi`), `trpo-continuous`, `trpo-connections` |
| LLM bridge: InstructGPT / RLHF | Not started in Part 2 | Drafted in the Part 1 deck and in the document (Part V) |
| DPO | Not started in Part 2 | Drafted in the Part 1 deck and in the document |
| RLOO | Not started in Part 2 | Drafted in the Part 1 deck and in the document |
| GRPO (DeepSeekMath) | Not started in Part 2 | Drafted in the Part 1 deck and in the document |
| **Companion document** `doc/rl_notes.tex` | Drafted | Parts I–V + appendices A–L; results placed next to each method (the deck groups them in "Experiments") |

States: Not started → Outlined → Drafted → Reviewed → Final.

---

## Conventions

- **Math in `.tex` files:** use the `gather` environment (amsmath), not other display delimiters (from `CLAUDE.md`).
- **Notation:** reuse the macros defined in the deck preamble (e.g. `\E`, `\PA`) instead of writing new ones.
- **Frame labels:** give important frames a `label=` so they can be cross-referenced and found from this file.
- **Equations on slides** should be checked against the local paper copies, not written from memory.
- **Layout rule (very important, for the document and the slides):** the phrasing must be *visually aligned with the content*. Use plots, boxes, itemize, numbered steps, tables — whatever fits — so that the partition of the ideas is visible on the page at a glance. Avoid big blocks of text.
  - In the document this means: one idea per box type (`defbox` definition, `thmbox` result, `keybox` take-away, `warnbox` limitation, `algbox` algorithm, `notebox` remark), `pair` for side-by-side contrasts, numbered steps for derivations, a plot or diagram where a picture explains it, and short grey `\src{}` lines for sources. Box and color macros are in `doc/parts/00_macros.tex`.
  - Colors follow the decks: warm (copper) = value side / lane A "how far", cool (blue) = policy side / lane B "where".

## Decisions & observations

Choices that guide how the talk is built. Add new ones at the bottom with a date; mark reversed ones as ~~struck~~ rather than deleting them.

- (pre-2026-11-09) **SAC dropped:** it is off-policy and Q-based, doesn't build on CPI/TRPO, and doesn't feed the LLM thread.
- (pre-2026-11-09) **InstructGPT is a bridge, not a section:** 2–3 slides on the SFT → reward model → PPO pipeline and the Eq. 2 KL-regularized objective (including the "PPO-ptx" pretraining term), set up as DPO's starting point.
- (2026-11-09) **Document first, slides second:** content is organized in `doc/rl_notes.tex`; the slide selection is decided from it afterwards.
- (2026-11-09) **In the document, results sit next to their method** (TRPO, GAE and PPO results at the end of each part), unlike the deck's single "Experiments" section.

## Session log

One line per session, newest first.

- 2026-11-09 (2): Wrote the companion document `doc/rl_notes.tex` (both decks, with plots and boxes); added the layout rule; found the dead `policy-advantage-recap` link in the deck.
- 2026-11-09: Created this handoff file; mapped the deck, papers and folders; recorded status from the deck outline.
