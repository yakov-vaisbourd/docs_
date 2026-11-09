# Excalidraw figure drafts (not used in the deck)

Open the files in VS Code with ExcaliMath. Equations are ExcaliMath equation elements, so double-click one to edit its LaTeX. Style: hand-drawn, Atlas palette (navy, copper, plum, washes). The equations were checked against the local paper copies in `papers/`.

## A. One overview per paper
- `01_paper_cpi_overview`: the problem → fix A (mixture α) / fix B (restarts μ) → Thm 4.1 / Cor 4.2 → algorithm loop
- `02_paper_trpo_overview`: CPI bound → Thm 1 (TV) → KL form → 5 practical steps → eq. 14 + trust-region sketch
- `03_paper_gae_overview`: TD residual → GAE sum → γ/λ knobs → λ=0 vs λ=1 → TRPO loop
- `04_paper_ppo_overview`: ratio, L^CLIP, clip plots, iteration loop, L^{CLIP+VF+S}
- `05_paper_cards`: four cards (problem / idea / key equation)

## B. All papers
- `06_all_timeline`: 2001–2017 timeline with lanes A and B
- `07_all_two_lanes`: lane A carried forward, lane B dropped (icons)
- `08_all_matrix`: step control / advantage / optimizer / state source × 4 papers
- `09_all_step_control`: α → KL ≤ δ → clip ε (by construction / solver / objective)
- `10_all_surrogate_forms`: L_π (eq. 3) → IS form (eq. 14) → L^CPI → L^CLIP
- `11_all_guarantee_vs_cost`: guarantee vs. practicality trajectory
- `12_all_problem_relay`: each paper fixes one problem and hands one on

## C. Key ideas
- `13_idea_mm_bound`: minorize–maximize picture, monotone improvement
- `14_idea_mixture_simplex`: mixtures on the simplex; a mixture leaves the neural-net family
- `15_idea_gain_vs_penalty`: linear gain vs. quadratic penalty, α*
- `16_idea_penalty_vs_constraint`: C ≈ 4·10⁴ε → tiny step vs. trust-region step
- `17_idea_max_vs_mean_kl`: per-state KL bars, max vs. mean
- `18_idea_importance_sampling`: q vs π_θ, sample weights; only actions are reweighted
- `19_idea_trust_region`: contours, KL ellipse vs. Euclidean ball, Fisher
- `20_idea_cg_line_search`: CG direction, β, backtracking
- `21_idea_single_path_vs_vine`: both start from ρ0 (the vine is not μ)
- `22_idea_bias_variance_lambda`: target plots for λ = 0 / 0.95 / 1
- `23_idea_lambda_weights`: k-step estimators and their geometric weights
- `24_idea_clip_plots`: L^CLIP for Â>0 / Â<0, flat vs. "wrong way kept" regions
- `25_idea_ratio_drift`: r_t over epochs, L^CPI vs. L^CLIP
- `26_idea_trpo_vs_ppo_batches`: one step per batch vs. K epochs of minibatches
- `27_idea_trust_region_not_exploration`: visited region, trust region, unreached states, μ

## D. Concepts (+1 result)
- `28_concept_rl_loop`: agent–environment loop, η(π)
- `29_concept_v_q_advantage`: Q bars, V line, signed advantages
- `30_concept_discount_horizon`: γ^t curves, 1/(1−γ)
- `31_concept_state_visitation`: heat maps d_π old vs. new
- `32_concept_performance_difference`: TRPO eq. 1 with advantages along a trajectory
- `33_concept_tv_vs_kl`: overlap = 1 − D_TV, KL, Pinsker-type bound
- `34_concept_policy_gradient`: probabilities before/after by the sign of Â
- `35_concept_mc_vs_td`: full return vs. one-step bootstrap
- `36_concept_actor_critic`: actor and critic networks, data flow
- `37_concept_natural_gradient`: parameter space vs. policy space
- `38_result_ppo_ablation`: PPO Table 1 as a bar chart
