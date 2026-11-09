# 3 Natural Language Program Synthesis using LLMs

We consider a task specified by a dataset \(\mathcal{D}_{\text {train }}=\{(Q, A)\}\) of input/output demonstrations sampled from population $\mathcal{X}$, and a prompted model $\mathcal{M}$. The goal of natural language program synthesis is to find a single instruction $\rho$ such that, when $\mathcal{M}$ is prompted with the concatenation $[\rho ; Q]$ of instruction and a given input, $\mathcal{M}$ produces the corresponding output $A$. More formally, we frame this as an optimization problem, where we seek instruction $\rho$ that maximizes the expectation of some per-sample score \(f(\rho, Q, A)\) over possible \((Q, A)\) :
$$
\rho^{\star}=\underset{\rho}{\arg \max } f(\rho)=\underset{\rho}{\arg \max } \mathbb{E}_{(Q, A)}[f(\rho, Q, A)]
$$
Note that in general, $Q$ may be the empty string, such that we are optimizing $\rho$ as a prompt that directly produces outputs $\{A\}$. While this task has been widely attempted by humans, we have little knowledge of how compatible any particular instruction is with model $\mathcal{M}$. Thus, we propose to treat this human-intractable question as a black-box optimization process guided by LLMs. Our algorithm, APE, uses LLMs in each of two key components, proposal and scoring. As shown in Figure 1 and summarized in Algorithm 1, APE first proposes a few candidate prompts, and then filters/refines the candidate set according to a chosen score function, ultimately choosing the instruction with the highest score. We discuss options for proposal and scoring next.

## 3.1 Initial Proposal Distributions

Due to the infinitely large search space, finding the right instruction can be extremely difficult, which has rendered natural language program synthesis historically intractable. Recent progress in NLP has shown language models are very good at generating diverse natural language text. Therefore, we
\begin{lstlisting}[mathescape=true]
Algorithm 1 Automatic Prompt Engineer (APE)
Require: \(\mathcal{D}_{\text {train }} \leftarrow\{(Q, A)\}_{n}\) : training examples, $f: \rho \times \mathcal{D} \mapsto \mathbb{R}$ : score function
Use LLM to sample instruction proposals $\mathcal{U} \leftarrow\left\{\rho_{1}, \ldots, \rho_{m}\right\}$. (See Section 3.1)
while not converged do
Choose a random training subset $\widetilde{\mathcal{D}}_{\text {train }} \subset \mathcal{D}_{\text {train }}$.
for all $\rho$ in $\mathcal{U}$ do
Evaluate score on the subset \(\widetilde{s} \leftarrow f\left(\rho, \widetilde{\mathcal{D}}_{\text {train }}\right)\) (See Section 3.2)
end for
Filter the top k% of instructions with high scores $\mathcal{U}_{k} \subset \mathcal{U}$ using $\left\{\widetilde{s}_{1}, \ldots, \widetilde{s}_{m}\right\}$
Update instructions $\mathcal{U} \leftarrow \mathcal{U}_{k}$ or use LLM to resample \(\mathcal{U} \leftarrow \operatorname{resample}\left(\mathcal{U}_{k}\right)\) (See Section 3.3)
end while
Return instruction with the highest score \(\rho^{\star} \leftarrow \arg \max _{\rho \in \mathcal{U}_{k}} f\left(\rho, \mathcal{D}_{\text {train }}\right)\)
\end{lstlisting}

consider leveraging a pretrained LLM to propose a good set $\mathcal{U}$ of candidate solutions that will guide our search procedure. While random samples from LLMs are unlikely to produce the desired \((Q, A)\) pairs, we can instead ask the LLM to approximately infer the most likely instructions with a high score, given the input/output demonstrations; i.e., to approximately sample from \(P\left(\rho \mid \mathcal{D}_{\text {train }}, f(\rho)\right.\) is high \()\).

Forward Mode Generation We consider two approaches to generate high-quality candidates from \(P\left(\rho \mid \mathcal{D}_{\text {train }}, f(\rho)\right.\) is high \()\). First, we adopt an approach based on "forward" mode generation by translating this distribution \(P\left(\rho \mid \mathcal{D}_{\text {train }}, f(\rho)\right.\) is high) into words. For example, in our instruction induction experiments (Subsection 4.1), we follow Honovich et al. (2022) and prompt the LLM using Figure 2 (Top).

Reverse Mode Generation Although the "forward" model works out of the box for most of the pretrained LLMs, translating \(P\left(\rho \mid \mathcal{D}_{\text {train }}, f(\rho)\right.\) is high) into words requires custom engineering across different tasks. This is because while instructions are typically found in the beginning of passages, the "forward" model only generates text from left to right, which requires the instruction to be predicted at the end of the prompt. Therefore, we desire a more flexible approach such that the instruction can be anywhere in the text. To address this, we consider "reverse" mode generation, which uses an LLM with infilling capabilities-e.g., T5 (Raffel et al., 2020), GLM (Du et al., 2022), and InsertGPT (Bavarian et al., 2022)-to infer the missing instructions. Our "reverse" model directly samples from \(P\left(\rho \mid \mathcal{D}_{\text {train }}, f(\rho)\right.\) is high) by filling in the blank. We show an example of the such template in Figure 2 (Middle).
Customized Prompts Note that depending on the score function being used, there may exist more appropriate prompts than the samples above. For example, in our TruthfulQA experiments, we start

Forward Generation Template
I gave a friend an instruction and five inputs. The friend read the instruction and wrote an output for every one of the inputs. Here are the input-output pairs:

Input: $\left[Q_{1}\right]$ Output: $\left[A_{1}\right]$ Input: $\left[Q_{2}\right]$ Output: $\left[A_{2}\right]$ ⋯

The instruction was <COMPLETE>

Reverse Generation Template
I instructed my friend to <INSERT>.
The friend read the instruction and wrote an output for every one of the inputs. Here are the input-output pairs:

Input: $\left[Q_{1}\right]$ Output: $\left[A_{1}\right]$ Input: $\left[Q_{2}\right]$ Output: $\left[A_{2}\right]$ …

Template for TruthfulQA
Professor Smith was given the following instructions: <INSERT>

Here are the Professor's responses:
Input: $\left[Q_{1}\right]$ Output: $\left[A_{1}\right]$ Input: $\left[Q_{2}\right]$ Output: $\left[A_{2}\right]$ ... with the human-designed instructions from the original dataset (Lin et al., 2022) and ask the the "reverse" model to propose initial instruction samples that fit the missing context (Figure 2 (Bottom)).

Figure 2: Prompts for LLMs

## 3.2 Score Functions

To cast our problem as black-box optimization, we choose a score function that accurately measures the alignment between the dataset and the data the model generates. In our instruction induction experiments, we consider two potential score functions, described below. In the TruthfulQA experiments, we focused primarily on automated metrics proposed in Lin et al. (2022), similar to the execution accuracy. In each case, we evaluate the quality of a generated instruction using Equation (1), and take the expectation over a held-out test dataset $\mathcal{D}_{\text {test }}$.

Execution accuracy First, we consider evaluating the quality of an instruction $\rho$ using the execution accuracy metric proposed by Honovich et al. (2022), which we denote as $f_{\text {exec }}$. In most cases,
execution accuracy is simply defined as the 0-1 loss, \(f(\rho, Q, A)=\mathbb{1}[\mathcal{M}([\rho ; Q])=A]\). On some tasks, execution accuracy takes into account invariants; e.g., it may be an order invariant set matching loss, as described in Appendix A of Honovich et al. (2022).

Log probability We further consider a softer probabilistic score function, which we hypothesize might improve optimization by providing a more fine-grained signal when searching over low-quality instruction candidates. In particular, we consider the log probability of the desired answer given the instruction and question under the target model $\mathcal{M}$, which on a per sample basis, is \(\log P(A \mid[\rho ; Q])\).

Efficient score estimation Estimating the score by computing the score over the entire training dataset for all instruction candidates can be expensive. To reduce the computation cost, we adopt a filtering scheme where a promising candidate receives more computation resources while a lowquality candidate receives less computation. It can be achieved by using a multi-stage computation strategy on lines 2-9 Algorithm 1 We first evaluate all candidates with a small subset of the training dataset. For the candidates with a score greater than a certain threshold, we sample and evaluate a new non-overlapping subset from the training dataset to update the moving average of the score. Then, we repeat this process until a small set of candidates is left, which are evaluated on the entire training dataset. This adaptive filtering scheme significantly improves the computation efficiency by keeping the exact computation costs for the high-quality samples and drastically reducing the computation costs for low-quality candidates. We note that a similar score estimation scheme has been used in previous works (Li et al., 2022; Maclaurin \& Adams, 2015).

## 3.3 Iterative Proposal Distributions

Despite our attempt to directly sample high-quality initial instruction candidates, it could be the case that the method described in Subsection 3.1 fails to produce a good proposal set $\mathcal{U}$, either because it lacks of diversity or does not contain any candidates with a suitably high score. In case of such challenges, we explore an iterative process for resampling $\mathcal{U}$.

Iterative Monte Carlo Search Instead of only sampling from the initial proposal, we consider exploring the search space locally around the current best candidates. This allows us to generate new instructions that are more likely to be successful. We call this variant iterative APE. At each stage, we evaluate a set of instructions and filter out candidates with low scores. Then, an LLM is asked to generate new instructions similar to those with high scores. We provide the prompt used for resampling in Figure 3 Figure 6 (Right) shows that although this approach improves the overall quality of the proposal set $\mathcal{U}$, the highest scoring instruction tends to remain the same with more stages. We conclude iterative generation provides marginal improvement over the relative simplicity and effectiveness of the generative process described in Subsection 3.1 Therefore, we use APE without iterative search as default unless otherwise stated.

