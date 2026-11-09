# Agentic Workflows

## Process Advantage Verifiers (PAVs) for LLM Reasoning
This subsection is based on (Setlur et al., 2025).

### Context: Improving Math Reasoning via Step-Level Rewards

Solving mathematical problems often requires a sequence of logical reasoning steps. Instead of generating an entire response at once, we can control the LLM's generation step-by-step using demarcation tokens - special delimiters (typically newlines) that mark the end of each reasoning step. This enables the model to generate line-by-line, allowing intermediate steps to be scored before committing to the next action. The entire response becomes a structured sequence: problem statement, step 1 (terminated by demarcation token), step 2 (terminated by demarcation token), and so on, culminating in a final answer.

This step-level structure enables training process reward models (PRMs) that assign feedback at each intermediate step rather than only after completion. However, traditional approaches (outcome reward models or ORMs) provide sparse feedback only after the full trace, making credit assignment difficult. Even automated PRMs measuring absolute likelihood of future success ($Q$-values) have shown limited improvements because they fail to distinguish progress made by individual steps. This paper introduces a paradigm shift: rewarding progress measured by advantages - the change in success probability before and after a step - under a carefully chosen prover policy distinct from the base policy.

Formally, the autoregressive generation of reasoning steps is framed as a Markov decision process. A state $s_h$ comprises the problem $x$ and all prior steps up to step $h-1$. An action $a_h$ is a complete reasoning step (a sequence of tokens terminated by a demarcation token, typically a newline). Transitions are deterministic: $s_{h+1}$ is the concatenation $(s_h, a_h)$. The outcome is determined by matching the final answer against ground truth using a regex evaluator.

### Bounded Beam Search: Algorithm and Mechanics

Given the step-level structure enabled by demarcation tokens, we can apply bounded beam search at inference to explore multiple candidate solution paths in parallel, evaluating each step-by-step rather than committing to a single greedy path. The algorithm maintains a fixed set of $B$ partial candidate solutions (the beam), and at each reasoning step:

**1. Stopping:** The demarcation token (newline) is configured as a stop sequence for the LLM inference call. The model generates tokens autoregressively as usual, but halts as soon as it emits this token. The result is one complete reasoning step $a_h$ - a variable-length sequence of tokens ending at the newline - without the model attempting to continue to the next step.

**2. Branching:** From each of the $B$ current beam states $\{s_i\}_{i=1}^B$, launch $C$ independent inference calls, each conditioned on the same prefix $s_i = (x, a_1, \ldots, a_{h-1})$. Each call samples one step until the stop sequence, yielding $C$ candidate next steps $\{a_{i,j}\}_{j=1}^C$ and $B \times C$ new candidate states $\{(s_i, a_{i,j})\}$. The $C$ calls from the same prefix are independent with different random seeds; they share the same key-value cache (In transformer inference, every token attends to all previous tokens via attention. The key ($K$) and value ($V$) projections for each prior token must be recomputed at every new generation step unless cached. The KV-cache stores these projections so they are computed once per token and reused, reducing per-step cost from $O(L)$ to $O(1)$ in the sequence length $L$. Here, all $C$ branches share the KV-cache of the common prefix $s_i$, so only the diverging tokens from position $h$ onward require fresh computation.) for the prefix tokens (computed once), and only diverge in generation from position $h$ onward.

**3. Scoring and Pruning:** The PAV (or verifier) scores all $B \times C$ candidate states. Each candidate is scored by appending a designated scoring token to its prefix and reading the corresponding logits. Only the top $B$ states by score are retained; the rest are discarded. The surviving states form the new beam and the next branching step begins from them.

This bounds computation while exploring diverse paths—unlike greedy decoding (single path) or sampling all possibilities (exponential explosion), beam search trades off between solution diversity and computational cost. The beam size $B$ is the beam budget, a hard computational constraint that defines the ranking problem: given more candidate steps than budget, which ones deserve advancement?

Beam search is exclusively an inference technique. During RL training the paper uses REINFORCE: the policy samples complete rollouts via standard autoregressive generation, and the PAV's effective reward $Q^\pi + \alpha A^\mu$ is applied as a step-level signal for policy gradient updates. The exploration mechanism is conceptually analogous—both direct the policy toward steps that make progress—but the algorithm differs: beam search prunes a candidate set explicitly, while REINFORCE reinforces good steps implicitly through gradient updates.

### Process Advantages Under a Prover Policy

Given the beam budget constraint, how should we score and rank candidate steps? This section develops the answer through the lens of advantages.

**The Q-value problem:** Using $Q^\pi(s_h, a_h)$ to score steps fails to decouple the quality of an action from the baseline promise of its originating state. With finite beam budget, this conflates state quality with step quality. Consider:

$$
\begin{aligned}
\text{State 1:} \quad V^\pi(s_1) = 0.8, \quad Q^\pi(s_1, a_1) = 0.7 \implies A^\pi(s_1, a_1) = -0.1 \\
\text{State 2:} \quad V^\pi(s_2) = 0.3, \quad Q^\pi(s_2, a_2) = 0.6 \implies A^\pi(s_2, a_2) = +0.3
\end{aligned}
$$

Selecting by $\max Q^\pi$ retains state 1 ($0.7 > 0.6$), yet this action regresses the solution path. Advantages reverse the ranking, directing search toward steps making progress regardless of initial state quality.

**Advantages as progress:** Progress is isolated by the advantage function:

$$
\begin{aligned}
A^\pi(s_h, a_h) = Q^\pi(s_h, a_h) - V^\pi(s_h)
\end{aligned}
$$

This measures the marginal contribution of a step relative to the state's baseline. Using advantages in beam search and RL rewards directed exploration: the algorithm selects steps that increase success likelihood regardless of initial state quality.

**The mathematical collapse:** A critical issue arises when attempting to use $A^\pi$ as a dense step-level reward alongside sparse outcome rewards. The base policy's own advantages sum to zero in expectation: $\mathbb{E}_{a \sim \pi}[A^\pi(s,a)] = 0$. This mathematical property causes the policy gradient to collapse: the step-level advantages cancel out, reducing the effective training signal to outcome-only optimization - no improvement over sparse ORMs. Standard off-policy methods (target networks, Actor-Critic) do not address this because they estimate $V^\pi$ (the value of the current policy), whereas we need a fundamentally different source of advantage signal.

**The prover policy solution:** The key insight is to measure advantages under a distinct prover policy $\mu$, not the base policy $\pi$:

$$
\begin{aligned}
A^\mu(s_h, a_h) = Q^\mu(s_h, a_h) - V^\mu(s_h)
\end{aligned}
$$

Now the expectation is no longer zero, and the policy gradient receives a dense, non-trivial signal at every step. However, the choice of $\mu$ is critical. A weak prover fails to solve from most intermediate states; a too-strong prover succeeds from any state, making all advantages zero since $Q^\mu(s, a) \approx 1$ and $V^\mu(s) \approx 1$ for any action. The ideal prover is *complementary* - capable enough to distinguish good and bad steps taken by $\pi$, yet weak enough that poor steps degrade its success rate. Best-of-$K$ policies (sampling $K$ responses and selecting the highest-scored one) serve as practical complementary provers; empirically, $K=4$ balances distinguishability and coverage.

**Weaker provers can improve stronger bases:** Counterintuitively, a prover weaker than the base policy can still provide useful advantage signals if it has high variance across different steps. This is not imitation learning (which caps student performance at teacher ability) but rather reward shaping: the method rewards *relative progress* rather than absolute action correctness, allowing a weaker model to guide exploration in a stronger one.

### Training PAVs

Process advantage verifiers distill step-level advantages into a learnable model. The pipeline is:

**1. ORM baseline:** Train an outcome reward model on complete reasoning traces, yielding sparse binary labels (correct or incorrect).

**2. Monte-Carlo rollouts:** For each intermediate prefix, sample multiple completions from the prover policy $\mu$ and compute the average success rate to estimate $Q^\mu$ and advantage $A^\mu$.

**3. PAV training:** Fine-tune a pre-trained LLM (e.g., Gemma 9B) to predict $A^\mu$ by reading the prefix and outputting a continuous score at a designated token position.

### Inference with Beam Search

To apply bounded beam search for inference, use the trained PAV verifier to score candidates at the pruning step: the PAV predicts advantages $A^\mu$ for all $B \times C$ candidate steps in real-time, and the beam is pruned to the top $B$ by advantage score. This directs exploration toward steps that increase success likelihood. Result: inference search achieves $>8\%$ accuracy improvement and $1.5\times$-$5\times$ compute efficiency gain over ORMs.

### Reinforcement Learning with Dense Rewards

The PAV is used as a dense reward signal during RL. The effective reward combines outcome evaluation and process advantages:

$$
\begin{aligned}
\text{Effective Reward} = Q^\pi(s_h, a_h) + \alpha A^\mu(s_h, a_h)
\end{aligned}
$$

Policy gradient updates directly supervise the base policy to generate steps that increase future success likelihood. This provides immediate step-level feedback, dramatically accelerating credit assignment. Empirical results show $5\times$-$6\times$ sample efficiency gain and $>6\%$ accuracy improvement over outcome-only RL.

