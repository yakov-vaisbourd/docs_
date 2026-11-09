# Reinforcement Learning Algorithms

## List of Sources

Papers (local copies in `papers/`):

1. [Kakade2002] Approximately Optimal Approximate Reinforcement Learning — `papers/approximately_optimal_approximate_RL.md`
2. [TRPO] Trust Region Policy Optimization — `papers/trust_region_policy_optimization.md`
3. [GAE] High-Dimensional Continuous Control Using Generalized Advantage Estimation — `papers/gae.md`
4. [PPO] Proximal Policy Optimization Algorithms — `papers/proximal_policy_optimization_algorithms.md`
5. [SAC] Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL with a Stochastic Actor — `papers/sac.md` — **dropped from the talk** (off-policy, Q-based; doesn't build on CPI/TRPO or feed the LLM thread)
6. [RLHF] Deep Reinforcement Learning from Human Preferences — `papers/rlhf.md`
7. [DPO] Direct Preference Optimization: Your Language Model is Secretly a Reward Model — `papers/dpo/`
8. [RLOO] Back to Basics: Revisiting REINFORCE-Style Optimization for Learning from Human Feedback in LLMs — `papers/REINFORCE_leave_1_out/`
9. [DeepSeekMath] DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models (GRPO) — `papers/deepseekmat_pushing_the_limits_of_mathematical_reasoning_in_open_language_models.md`
10. [InstructGPT] Training Language Models to Follow Instructions with Human Feedback (Ouyang et al. 2022, arXiv 2203.02155) — `papers/2203.02155v1.pdf` (PDF only, no Markdown copy; read with `pdftotext`). Used as a short bridge (2–3 slides): the SFT → reward model → PPO pipeline, and its Eq. 2 KL-regularized objective (with the "PPO-ptx" pretraining term), which is DPO's starting point.

Personal notes (**read-only — never modify anything in this folder**):

- [ModernRL] Modern RL notes — `/home/yakov/Documents/Vaulty/4_docs/rl/papers/modern_rl/`
  - `modern_rl.md` — main summary (background, CPI, TRPO derivations; includes the discussion of $D$ vs. the restart distribution $\mu$).
  - `papers_md/` — another copy of the papers; `resources/` — drawings and the algorithm map.

## Presentation

- Decks: `seminar/tex/pres_part1/rl_pres_part1.tex` (first talk, presented up to the end of CPI) and `seminar/tex/pres_part2/rl_pres_part2.tex` (second talk, 90 min: CPI recap → TRPO → GAE → PPO → Experiments; **Part III / LLMs dropped**). Paper PDFs for figure crops: `seminar/tex/pres_part2/reference/papers/` (TRPO, PPO from arXiv).
- Drawings: `seminar/tex/pres_part2/resources/<name>.excalidraw.png` (PNG with the Excalidraw scene embedded, edited directly by the user). Include them with `\excalidraw[opts]{<name>}` (defined in the deck preamble); every build picks up the latest drawing. Part I sources figure: `papers_part1`. Later sources figures:
  - `papers_part2` (GAE, PPO): opens Part II. Re-saved by the user as a real PNG, but with the original text, so two corrections are still open: GAE was a follow-up to TRPO, not "published parallel"; TRPO used a hard KL constraint, not a "KL penalty". The PPO equation is also missing (rendered SVG in the session scratchpad).
  - `papers_part3` (RLHF, InstructGPT, DPO, GRPO, RLOO): opens Part III. The user fills in the InstructGPT row by hand (Ouyang et al. 2022, description, Eq. 2). The other rows still have the old, uncorrected text.
- Previous SFT deck (reference only): `seminar/tex/pres_part2/reference/sft/sft_pres.tex`

## Presentation Plan

- Audience has some RL background but is not expert. No background section up front; reminders go directly in the main body where they are needed.
- **Normalization:** the deck uses Kakade & Langford's $(1-\gamma)$-normalized $V, Q, d$, so the policy gradient carries $\frac{1}{1-\gamma}$ (reminder slide); the paper's Sec. 3.2 formula omits it. TRPO slides use TRPO's unnormalized $\eta, \rho_\pi$ (stated on its first slide).
- **Overlays to reduce load:** when a slide needs all its content but feels crowded, build it in steps (`\only`/`\uncover`) so the figure and the text grow together, one idea per click. Slide 4 has 3 steps, slide 5 (Notation) has 5 steps, the two reminder slides and Gap 1 have 2 steps each, the Gap 1 tightness slide has 3, the TD/MC slide has 4, the four SGD slides have 3, 4, 4 and 4, the Gap 2 covariate-shift slide has 5, the roadmap after the two fixes has 4, the TRPO importance-sampling slide has 3, and App. 83 (CPI vs. SGD) has 2 (its notes are split over the two steps), so the PDF has 35 more pages than frames; slide numbers in the table count frames. Notation pattern: equations stack on the left (current one in black, earlier ones grey, macro `\eqstep{k}{...}`), and the right side shows a small figure plus one line for the current equation.
- **Visual alignment:** each slide's layout should show its content's structure, not just state it. Prefer a figure or diagram that carries the argument (curves, flows, comparisons), with a few short labeled steps next to it, over paragraphs in a box. Keep text minimal and put the details and paper quotes in the speaker notes. Example: slide 4 ($D$) shows $V_\pi(s)$ curves crossing over $D(s)$ weight bars.
- **Color code (Parts II–III):** warm = value side (critic, advantage estimate, reward model), cool = policy side, slate = frozen/reference pieces (e.g. $\pi_{\text{ref}}$).
- **Paper title slides:** every paper gets the CPI/TRPO layout (blue title panel left, "Setting & Assumptions" right). New ones use the preamble macro `\paperpanel[title size]{title}{authors}{venue}{what it introduces}`.
- **Color code (Part I):** warm (`AtlasWarm`, `warmbox`, `warmformulabox`) = approximate value function methods and what fixes them (conservative update); cool (`AtlasCool`, `coolbox`, `formulabox`) = policy gradients and what fixes them (restart distribution $\mu$). `badbox`/`goodbox` stay for verdicts.
- Left side panel (`\atlasmapframe`) shows the overall map (`pres_map.excalidraw.png`) on selected slides, to tie each part back to the big picture.

## Content Outline (partial, in progress)

### 1. CPI — Kakade & Langford 2002 [Kakade2002]

- **Opening (3 slides):** (a) Part I sources figure (`papers_part1`); (b) split slide — left half the paper's title, right half the setting/assumptions: finite $S, A$, rewards in $[0,R]$, discount $\gamma$, unknown $P$ (model-free), start distribution $D$ — **no $\mu$ yet**; (c) the start distribution $D$ — figure ($\pi^*$ on top everywhere; $\pi_1,\pi_2$ from a restricted class cross; $D(s)$ bars weight them) + three short steps; why it is part of the objective (an exact $\pi^*$ is best at every state, so $D$ would not matter; approximate policies (two causes: restricted class $\{\pi_\theta\}$, and estimation error since $P$ is unknown, e.g. CPI ends with $\mathrm{OPT}(\mathbb{A}_{\pi,\mu})<2\varepsilon$) are not comparable state-by-state, so we need one number — the $D$-weighted $\eta_D$; within a restricted class the best policy depends on the weighting, Sec. 2–3), and where it comes from (the application's real starting conditions — not a design or theory choice) [ModernRL, "D vs. μ" section and the $\rho_0$ remark]. The restart distribution $\mu$ is introduced only at "The Two Fixes".
- **Motivation (Sec. 3):** three questions — (1) is there a performance measure that improves every step? (2) how hard is it to verify an update improves it? (3) what performance after a reasonable number of updates?
- **Order for each family: reminder (with an example) → gap.** Reminder 1 (warm): approximate policy iteration loop $\pi \to \tilde Q \approx Q_\pi \to$ greedy $\pi'$; example: Q-learning with $\tilde Q_w$ (the max hides the greedy step; DQN with a neural net). Then **How is $\tilde Q$ estimated?** (4 steps, Notation-style layout): TD(0) and MC as the two extremes, $n$-step targets in between, TD($\lambda$) averaging them ($\lambda=0$ TD(0), $\lambda=1$ MC); figure shows how many real rewards each target uses before bootstrapping, plus the bias–variance axis. Background from Sutton & Barto, foreshadows GAE. Reminder 2 (cool): loop $\pi_\theta \to$ rollouts $\to \hat g \approx \nabla\eta_D \to \theta \mathrel{+}= \alpha\hat g$, gradient formula from ref [8]; example: REINFORCE with the Gibbs policy (same class as Ex. 2). Then four SGD slides (source: the user's notes in `seminar/tex/pres_part2/resources/temp.excalidraw`): (i) **PG = stochastic gradient ascent**, a mapping table (objective, root, sample, update, noise $\eta_k$; twist: samples depend on $\theta$, the log-derivative trick keeps $\hat g$ unbiased, with a "Why?" button to App. 82); the convergence slide defines $\mathcal H_k$ (history $w_0,x_0,\dots,x_{k-1}$) under the unbiasedness condition; (ii) **convergence, convex smooth case**: curvature $0<c_1\le\nabla^2 f\le c_2$, unbiased noise, finite variance + $\sum a_k=\infty,\ \sum a_k^2<\infty$, each with its role, giving $w_k\to w^*$ a.s.; (iii) **sample complexity** $N(\epsilon,\delta)$ (first $N$ such that the bound holds for all $k\ge N$), with a ladder: convex smooth $O(\sigma^2/(c_1^2\epsilon^2\delta))$ → non-convex smooth (stationary point, $O(\sigma^2/\epsilon^4)$, for a randomly chosen iterate $w_R$, not the last one) → non-convex non-smooth (weaker); PG with neural nets lives in the lower rows, but the convex case still shows what drives $N$; (iv) **signal-to-noise**: $n\gtrsim\sigma^2/\|\nabla J\|^2$; with $\|\nabla J\|\ge c_1\|w-w^*\|$ this becomes $\sigma^2/(c_1\epsilon)^2$, the convex-row $N$ (so $1/c_1^2$ is the signal-to-noise cost); flat regions or tiny gradients → intractable, leading into Ex. 1–2. The examples and the TD/MC slide are ours, not the paper's.
- **Gap 1 — approximate value functions (Sec. 3.1):** no improvement guarantee. With $\varepsilon = \max_s |\tilde V(s) - V_\pi(s)|$ and $\pi'$ greedy w.r.t. $\tilde V$, the bound from Bertsekas & Tsitsiklis, *Neuro-Dynamic Programming* (ref [3]) only says the policy can't get much worse: $V_{\pi'}(s) \ge V_\pi(s) - \frac{2\gamma\varepsilon}{1-\gamma}$.
- **Gap 2 — policy gradient (Sec. 3.2):** exploration / sample-complexity problems, shown with two examples:
  - Ex. 1 (Fig. 3.1): chain MDP; random walk takes $\approx 10^{15}$ steps to reach the goal for $n=50$, so gradient estimates are zero (right magnitude, no direction).
  - Ex. 2 (Fig. 3.2–3.3): two-state MDP with Gibbs policy; learning at $i$ starves state $j$ ($\rho(j)$ drops to $10^{-7}$), giving a long plateau.
- **Solution 1 — conservative update (Sec. 4):** mixture $\pi_{\text{new}} = (1-\alpha)\pi + \alpha\pi'$, $\alpha \in [0,1]$ acting as a step size.
- **Solution 2 — restart distribution (Def. 2.1, Sec. 4):** optimize $\eta_\mu$ with an "exploratory" $\mu$ instead of $\eta_D$.
- **Policy advantage (Sec. 4.1):** $\mathbb{A}_{\pi,\mu}(\pi') = E_{s \sim d_{\pi,\mu}} E_{a \sim \pi'}[A_\pi(s,a)]$; first-order agreement: $\Delta\eta_\mu = \frac{\alpha}{1-\gamma}\mathbb{A}_{\pi,\mu}(\pi') + O(\alpha^2)$.
- **Theory:**
  - Thm. 4.1: improvement lower bound (first-order term minus $\alpha^2$ penalty); Cor. 4.2: choice of $\alpha$.
  - Thm. 4.4: improves every step, stops within $O(R^2/\varepsilon^2)$ updates.
  - Lemma 6.1: performance difference lemma — the bridge to TRPO.
  - Cor. 4.5 / Thm. 6.2: distance from optimum, depends on how well $\mu$ covers $d_{\pi^*}$.

### 2. TRPO — Schulman et al. 2015 [TRPO]

- **Mixture $\alpha$ → general policies (Sec. 3, Thm. 1):** replace $\alpha$ with $D_{\mathrm{TV}}^{\max}(\pi_{\text{old}}, \pi_{\text{new}})$; $\eta(\pi_{\text{new}}) \ge L_{\pi_{\text{old}}}(\pi_{\text{new}}) - \frac{4\epsilon\gamma}{(1-\gamma)^2}\alpha^2$ (proof via coupling, App. A).
- **TV → KL:** $D_{\mathrm{TV}}^2 \le D_{\mathrm{KL}}$ gives $\eta(\tilde\pi) \ge L_\pi(\tilde\pi) - C\,D_{\mathrm{KL}}^{\max}(\pi,\tilde\pi)$; Alg. 1 = MM (minorize–maximize) with monotone improvement.
- **Penalty → constraint (Sec. 4):** theoretical $C$ gives tiny steps, so use a trust region: $\max_\theta L_{\theta_{\text{old}}}(\theta)$ s.t. $D_{\mathrm{KL}}^{\max} \le \delta$.
- **Max KL → mean KL:** $D^{\max}$ has one constraint per state, so replace it with $\bar D_{\mathrm{KL}}^{\rho} = E_{s\sim\rho}[D_{\mathrm{KL}}]$, which can be sampled.
- **Sampling (Sec. 5):** $\sum_s \rho \to E_{s \sim \rho_{\text{old}}}$; $\sum_a \to$ importance sampling with $q$; $A \to Q$; single-path vs. vine.
- **Solver (Sec. 6, App. C) — included in the talk:**
  - Linear approximation of $L$, quadratic approximation of $\bar D_{\mathrm{KL}}$ (Fisher matrix $A$) → natural-gradient direction $s \approx A^{-1}g$.
  - Conjugate gradient with Fisher-vector products $A = J^\top M J$ (no full matrix); $k=10$ iterations, ~10% of data.
  - Max step $\beta = \sqrt{2\delta / s^\top A s}$, then backtracking line search on the true $L$ and KL constraint.
- **Connections (Sec. 7):** vanilla PG, natural PG and policy iteration as special cases of the same problem.

### 3. Part II — GAE + PPO [GAE], [PPO]

- Bridge: "Where TRPO leaves us": where $\hat A_t$ comes from (MC, high variance) → GAE; how the update is solved (CG, one step per batch) → PPO.
- GAE: title; which $\Psi_t$ multiplies the score (6 choices, advantage best; $\gamma$ as a variance knob); TD($\lambda$) for advantages (same figure as the Part I TD/MC slide plus the $-V(s_t)$ baseline); two knobs (response function $\chi$, reward shaping with $\Phi=V$, $\gamma$ biased even with exact $V$, $\lambda$ only if $V$ is wrong); GAE + TRPO algorithm with the trust-region value fit.
- PPO: title; back to the CPI surrogate ($L^{CPI}$ is named after CPI) and three ways to stay close (constraint, penalty, clip); clipped objective (Fig. 1 redrawn, $\min$ = pessimistic bound); Table 1 ablation as a bar chart + adaptive-KL rule; algorithm with $L^{CLIP+VF+S}$, truncated GAE, MuJoCo hyperparameters; TRPO vs. PPO table with kept/given-up boxes.

### 4. Part III — RL for LLMs: "what can we remove?" [RLHF], [InstructGPT], [DPO], [DeepSeekMath], [RLOO]

- Setting: text generation as an MDP (state = prompt + prefix, action = token, deterministic known dynamics, reward at EOS) and the bandit view; $D$ = prompts (callback to Part I).
- Roadmap: RLHF with PPO needs four models (policy, reference, reward model, critic); timeline of what each paper removes.
- Human preferences (Christiano 2017): title; three asynchronous processes; Bradley–Terry reward model (logistic in the reward difference, Elo).
- InstructGPT: title (bandit environment, API prompts); three-step recipe (SFT 13k, RM 33k with $K$=4–9 rankings, PPO 31k); Eq. 2 objective with KL and ptx, 1.3B preferred over 175B.
- DPO: title; 4-step derivation (objective → tilted optimum → reward from policy → $Z$ cancels in BT); gradient and its weighting; what is removed vs. costs (offline, coverage — our echo of CPI Thm 6.2).
- GRPO (DeepSeekMath): title; group-relative advantages; PPO vs. GRPO pipeline (Fig. 4 redrawn) and objective (KL in the loss, k3 estimator in notes; single update per batch ⇒ ratio = 1, noted).
- RLOO: title; which PPO layers RLHF needs (critic/GAE, clipping, per-token, crossed out one by one); leave-one-out baseline and the algebra $R_i-\bar R=\frac{k-1}{k}(R_i-b_i)$ (ours) linking it to GRPO.
- Closing: "What did we remove?" table; "The thread: from CPI to LLMs" (surrogate, stay close, baseline, where states come from — our synthesis).

### Second talk (`rl_pres_part2.tex`) — storyline
Based on two separate CPI problems that the talk follows throughout: **lane A, update size** (mixture α → TRPO KL trust region → PPO clipping) and **lane B, state coverage** (restart μ — dropped by TRPO/GAE/PPO, which all sample s0∼ρ0; the vine resets only to visited states; IS reweights actions, not states). GAE's λ is a bias–variance knob for Â, **not** exploration. Part I frames are unchanged in the file (frames 1–31). New structure:
- **Recap** (new section): two-lanes figure; "The Policy Advantage Is a Gradient" (main-flow version of the appendix proof, θ:=α); `\againframe` of Thm 4.1, Cor 4.2, CPI algorithm, CPI summary.
- **TRPO**: paper → "From CPI to TRPO: What Changes" (table + lanes: A kept, B dropped) → identity → Thm 1 (TV definition + overlap figure, button to the continuous-spaces backup) → TV→KL/MM ("what the theory covers") → five simplifications, each with the `\trpostep{k}` progress strip and the `\simpfooter{why}{what changes}{cost}` footer: (1) penalty→constraint, (2) max→mean KL, IS tool slide, (3) sums→samples, sampling schemes, (4) search direction, (5) CG + line search (merged) → algorithm → recap → "What TRPO Does Not Fix: Where the Data Come From".
- **GAE**: sources, "Where TRPO Leaves Us" (3 threads: Â, states [still open], solver), paper, "What GAE Fixes — and What It Does Not" (estimate dots: MC wide/unbiased, TD tight/biased), Ψ choice, TD(λ), two knobs, "Actor–Critic: Policy and Value Networks".
- **PPO**: paper, "PPO's Goal: Many SGD Steps per Batch" (timeline + r_t running away), back to the CPI surrogate, clipped objective (step 4: wrong-way moves keep their gradient), "What Clipping Replaces" (α → KL ≤ δ → clip ε; no KL guarantee), PPO in practice, TRPO vs PPO.
- **Experiments** (all numbers in one place): TRPO MuJoCo (Fig. 4 crop `trpo_fig4_mujoco.png`), TRPO Atari (Table 1 redrawn), GAE (Figs. 2–3, `gae_fig2_sweep.jpg`, `gae_fig3_3d.jpg`), PPO ablation (Table 1 bars), PPO vs others (Fig. 3 crop `ppo_fig3_mujoco.png` + Atari Table 2), "The Whole Story" grid, "One Line per Paper".
- Appendix additions: continuous S, A (`trpo-continuous`) and the related-algorithms table (`trpo-connections`), both with Back buttons.

## Slide Status

Draft v1 in `seminar/tex/pres_part2/rl_pres.tex` (77 slides + 7 appendix), to be revised:

**Temporary:** a "[TEMP] Review Before the Talk" slide sits right after the Gap 1 tightness slide (review notes in red). It is not counted in the numbers below; delete it after the review.

| Slides | Content |
|---|---|
| 1 | Title |
| 2–4 | Part I opening: sources figure, CPI title + assumptions, the start distribution $D$ |
| 5–26 | CPI: notation, three questions, value-function reminder (Q-learning), TD(0)↔MC targets, Gap 1, Gap 1 bound is tight (our two-state example: $s$ with stay/go, absorbing $G$ with reward $c=2\varepsilon$; $\tilde V$ off by $\varepsilon$ in opposite directions ⇒ greedy stays forever, loss $=\frac{2\gamma\varepsilon}{1-\gamma}$; notes: B&T Example 6.4 is tightness of the API bound, one-step form is Singh & Yee 1994), policy-gradient reminder, PG = SGD (4 slides), Gap 2 (Ex. 1, Ex. 2), Gap 2 as covariate shift (train $d_{\pi,D}$ vs. target $d_{\pi^*,D}$, no overlap; differences: unknown target, learner moves its own data, reweighting fails — IS weights exponential; → change the data source, $\mu$; our framing for an audience that knows covariate shift), two fixes (1 conservative update, warm, left; 2 restart $\mu$, cool, right; equal heights), roadmap (fixes → results → three questions: conservative update → improvement bound → Q1; restarts → CPI estimation → Q2 and speed of Q3; coverage → perf. difference lemma / Thm 6.2 → quality of Q3; our framing), policy advantage, Thm 4.1 / Cor 4.2 (gray lines: $\mathbb{A}$ is estimable, $\varepsilon$ replaced by $R$, so $\alpha\le(1-\gamma)/4$, tiny steps), algorithm + Thm 4.4 (`label=cpi-alg`; "How?" button on the $G_\varepsilon$ footnote to App. 84), performance difference lemma, Thm 6.2 (read as a covariate-shift bound: test risk ≤ density ratio × train risk $\mathrm{OPT}(\mathbb{A}_{\pi,\mu})$, defined in a footnote on the CPI algorithm slide; uniform $\mu$ caps the ratio at $|S|/(1-\gamma)$), summary (`label=cpi-summary`; one line "Like non-convex SGD, but Thm 6.2 turns stationary into near-optimal" with a "CPI vs. SGD" button to App. 83) |
| 27 | TRPO title + setting & assumptions (same layout as the CPI slide): finite $S,A$ in the proofs (says continuous also holds, conditions later), $\gamma\in(0,1)$, $r(s)$, unknown dynamics, start distribution $\rho_0$ (= CPI's $D$) with no restart $\mu$ (vine needs state resets), unnormalized $\eta$, any differentiable $\pi_\theta$ |
| 28–42 | TRPO: identity + $L_\pi$, Thm 1, TV→KL + MM, continuous $S,A$ (slide 30, placed after Thm 1 and $D_{KL}^{\max}$ so everything it cites is already shown): footnote 1 ("replace sums with integrals") holds — no proof uses finiteness. Left: why it carries over (perf. difference, maximal coupling has mass $1-D_{TV}$ on $a=\tilde a$ also for densities, perturbation proof with Markov operators). Right: hidden conditions (bounded $\epsilon$, max → sup over all states, stochastic policies, measurability). Footnote-style remark: often vacuous in continuous control (Gaussian actions + $-\|u\|^2$ reward in Swimmer ⇒ $\epsilon=\infty$); practical algorithm drops the bound anyway, penalty→constraint, max→mean KL, sampling ("Turning Sums Into Samples"), importance sampling (3 steps: identity $\mathbb{E}_p[f]=\mathbb{E}_q[wf]$ → variance $\frac1n\mathrm{Var}_q[wf]$, rare huge weights when $q$ is far from $p$ → in TRPO $w=\pi_\theta/\pi_{\theta_{\text{old}}}$, trust region keeps $w\approx1$, only actions reweighted; figure: $p,q$ densities + weight bars; notes from the vault `statistics.md` and `modern_rl.md`), single path vs. vine, solver (3 slides), algorithm, connections, recap |
| 43 | The Big Picture: full map (`map`) |
| 44–55 | Part II: sources (`papers_part2`), where TRPO leaves us, GAE title, which $\Psi_t$, TD($\lambda$) for advantages, two knobs, GAE + TRPO, PPO title, back to the CPI surrogate, clipped surrogate, clip vs. KL ablation, PPO in practice, TRPO vs. PPO |
| 56–77 | Part III: sources (`papers_part3`), text generation as RL, four models / what can we remove, Human Preferences (title, loop, Bradley–Terry), InstructGPT (title, recipe, objective), DPO (title, derivation, gradient, gains/costs), DeepSeekMath/GRPO (title, group advantages, PPO vs. GRPO), RLOO (title, PPO layers, leave-one-out), what did we remove, the thread from CPI to LLMs |
| App. 78–80 | Optional, after `\appendix`: proof (3 slides) that $\partial\eta_\mu/\partial\alpha\vert_{\alpha=0}=\frac{1}{1-\gamma}\mathbb{A}_{\pi,\mu}(\pi')$ (from the user's notes in `resources/temp.excalidraw`). Slide 1 stresses that the mixture family is a one-parameter policy class, so the policy gradient formula applies with $\theta:=\alpha$ (warm/cool bridge); slide 3 ("$\alpha$ and $\theta$ play the same role") links this to TRPO Eq. 4 (first-order match for any $\theta$) and Thm. 1 (step size $\alpha \to D_{\mathrm{TV}}^{\max}$) — the user's main reason to show the proof. Reached by the "Proof" button on the policy advantage slide (`label=policy-advantage`); the last proof slide has a "Back" button. Future optional slides go here the same way (`\hyperlink` + `\beamergotobutton` / `\beamerreturnbutton`). |
| App. 81 | Gibbs (softmax) policy: bar figure (scores $\theta_{sa}$ → softmax probabilities, one state, 3 actions), definition, why "Gibbs" (Boltzmann), always stochastic, rich class, score $\mathbf{1}[a=b]-\pi(b;s)$. Reached by the "Gibbs policy" button on the policy-gradient reminder (`label=pg-reminder`); "Back" returns to its step 2 (`pg-reminder<2>`). |
| App. 82 | Why the log-derivative trick is unbiased: trajectory figure ($\pi_\theta$ terms in cool, $D$ and $P$ in grey), $p_\theta(\tau)$ as a product, $\nabla p=p\nabla\log p$ gives $\nabla\eta_D=\E[R\,\nabla\log p_\theta]$, log turns the product into a sum so $D$ and $P$ drop out; box: $\E[\hat g]=\nabla\eta_D$ exactly, no need to know $P$. Notes: Williams 1992, reward-to-go and baselines stay unbiased ($\E[\nabla\log\pi]=0$), lead to GAE. Reached by the "Why?" button on PG = SGA (`label=pg-sga`); "Back" returns to its step 3 (`pg-sga<3>`). |
| App. 83 | CPI through the SGD lens (our synthesis): table SGD (non-convex, smooth) vs. CPI — direction (greedy $G_\varepsilon$ = best direction of the linearization), step (mixture = Frank–Wolfe), step size ($\alpha\propto\hat{\mathbb A}$, like $1/L$), noise (averaged out vs. each step accurate by Hoeffding), stationarity ($\|\nabla J(w_R)\|\le\epsilon$ random iterate vs. $\mathrm{OPT}(\mathbb A)<2\varepsilon$ last iterate), cost, global (✗ vs. ✓ Thm 6.2). Notes derive: (1) slope toward $\pi'$ is $\mathbb A/(1-\gamma)$, so CPI is Frank–Wolfe on the table $\pi(a;s)$, $\mathrm{OPT}(\mathbb A)$ is the Frank–Wolfe gap (Vieillard, Pietquin & Geist 2019, Sec. 3; earlier Scherrer & Geist 2014, Shani et al. 2019; rate: Lacoste-Julien 2016); (2) Gibbs gradient $d\,\pi(a;s)A/(1-\gamma)$ vs. natural gradient $\pi\propto\pi e^{\beta A}$ (Kakade 2001), CPI is the hard-greedy version; (3) stochastic Frank–Wolfe vs. SGD, gradient domination (Agarwal et al. 2021). "Back" returns to `cpi-summary`. |
| App. 84 | Building $G_\varepsilon$ by regression (Sec. 7.1, `label=greedy-regression`): pipeline figure (μ-restart data $s\sim d_{\pi,\mu}$, uniform $a$, rollout $\hat Q$ → regression $f_\pi\approx A_\pi$ → greedy $\arg\max_a f_\pi$), loss $\E_{s\sim d_{\pi,\mu}}\max_a|A_\pi-f_\pi|\le\varepsilon/2$, our three-step proof that this gives $\mathbb{A}(\pi')\ge\mathrm{OPT}-\varepsilon$, box: $\ell_1$ over states vs. $\ell_\infty$ of greedy DP. Notes: fitting $Q$ suffices, link to squared loss (MSE $\le(\varepsilon/(2n_a))^2$), sample cost open, Lagoudakis & Parr 2003, Deep CPI. "Back" returns to `cpi-alg`. |

Open items:
- Map frames: `TODO(map)` placeholders at the start of each part, waiting for `pres_map`.
- Figures for CPI Ex. 1 / Ex. 2 are simplified TikZ redraws (paper figure links expired); Ex. 2 is schematic.
- Sources figure `papers_part1`: Safe Policy Iteration (Pirotta et al. 2013) dropped from the talk — its figure text didn't match the paper (its bound is L1/TV with $\Delta A$, no KL). The user removes it from the drawing; the TRPO entry should take over the "mixtures → any policy" role, with TRPO's Eq. 9: $\eta(\tilde\pi)\ge L_\pi(\tilde\pi)-C\,D_{KL}^{\max}(\pi,\tilde\pi)$, $C=\frac{4\epsilon\gamma}{(1-\gamma)^2}$, $\epsilon=\max_{s,a}\lvert A_\pi(s,a)\rvert$.
- Parts II and III drafted (v1). Order in Part III is GRPO then RLOO (chronological, Feb 2024, and "remove more": GRPO keeps PPO's clip, RLOO drops it).
- Beamer pitfall: `#1` in a TikZ style defined inside a frame breaks the build (also with `##1` when the frame has overlays). Define parametrized styles in the preamble; Parts II–III use the preamble style `cbox=<color>` plus local shape-only styles.
- Beamer pitfall: an `\uncover` around a whole table row breaks `\\`/`\bottomrule`; uncover cell by cell (`\pgrow`, `\sumrow`).
- Beamer pitfall: a braced group right after `\begin{frame}{title}` is read as a (hidden) subtitle — start such a line with `\noindent`. This was the "empty space" on the first TRPO content slide (fixed).

## Notes

<!-- To be filled as we discuss the contents. -->
