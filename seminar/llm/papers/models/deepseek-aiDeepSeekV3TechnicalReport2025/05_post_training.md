# 5. Post-Training

## 5.1. Supervised Fine-Tuning

We curate our instruction-tuning datasets to include 1.5M instances spanning multiple domains, with each domain employing distinct data creation methods tailored to its specific requirements.

Reasoning Data. For reasoning-related datasets, including those focused on mathematics, code competition problems, and logic puzzles, we generate the data by leveraging an internal DeepSeek-R1 model. Specifically, while the R1-generated data demonstrates strong accuracy, it suffers from issues such as overthinking, poor formatting, and excessive length. Our objective is to balance the high accuracy of R1-generated reasoning data and the clarity and conciseness of regularly formatted reasoning data.

To establish our methodology, we begin by developing an expert model tailored to a specific domain, such as code, mathematics, or general reasoning, using a combined Supervised FineTuning (SFT) and Reinforcement Learning (RL) training pipeline. This expert model serves as a data generator for the final model. The training process involves generating two distinct types of SFT samples for each instance: the first couples the problem with its original response in the format of <problem, original response>, while the second incorporates a system prompt
alongside the problem and the R1 response in the format of <system prompt, problem, R1 response>.

The system prompt is meticulously designed to include instructions that guide the model toward producing responses enriched with mechanisms for reflection and verification. During the RL phase, the model leverages high-temperature sampling to generate responses that integrate patterns from both the R1-generated and original data, even in the absence of explicit system prompts. After hundreds of RL steps, the intermediate RL model learns to incorporate R1 patterns, thereby enhancing overall performance strategically.

Upon completing the RL training phase, we implement rejection sampling to curate highquality SFT data for the final model, where the expert models are used as data generation sources. This method ensures that the final training data retains the strengths of DeepSeek-R1 while producing responses that are concise and effective.

Non-Reasoning Data. For non-reasoning data, such as creative writing, role-play, and simple question answering, we utilize DeepSeek-V2.5 to generate responses and enlist human annotators to verify the accuracy and correctness of the data.

SFT Settings. We fine-tune DeepSeek-V3-Base for two epochs using the SFT dataset, using the cosine decay learning rate scheduling that starts at $5 \times 10^{-6}$ and gradually decreases to $1 \times 10^{-6}$. During training, each single sequence is packed from multiple samples. However, we adopt a sample masking strategy to ensure that these examples remain isolated and mutually invisible.

## 5.2. Reinforcement Learning

## 5.2.1. Reward Model

We employ a rule-based Reward Model (RM) and a model-based RM in our RL process.

Rule-Based RM. For questions that can be validated using specific rules, we adopt a rulebased reward system to determine the feedback. For instance, certain math problems have deterministic results, and we require the model to provide the final answer within a designated format (e.g., in a box), allowing us to apply rules to verify the correctness. Similarly, for LeetCode problems, we can utilize a compiler to generate feedback based on test cases. By leveraging rule-based validation wherever possible, we ensure a higher level of reliability, as this approach is resistant to manipulation or exploitation.

Model-Based RM. For questions with free-form ground-truth answers, we rely on the reward model to determine whether the response matches the expected ground-truth. Conversely, for questions without a definitive ground-truth, such as those involving creative writing, the reward model is tasked with providing feedback based on the question and the corresponding answer as inputs. The reward model is trained from the DeepSeek-V3 SFT checkpoints. To enhance its reliability, we construct preference data that not only provides the final reward but also includes the chain-of-thought leading to the reward. This approach helps mitigate the risk of reward hacking in specific tasks.

## 5.2.2. Group Relative Policy Optimization

Similar to DeepSeek-V2 (DeepSeek-AI, 2024c), we adopt Group Relative Policy Optimization (GRPO) (Shao et al., 2024), which foregoes the critic model that is typically with the same size as the policy model, and estimates the baseline from group scores instead. Specifically, for each question $q$, GRPO samples a group of outputs $\left\{o_{1}, o_{2}, \cdots, o_{G}\right\}$ from the old policy model $\pi_{\theta_{\text {old }}}$ and then optimizes the policy model $\pi_{\theta}$ by maximizing the following objective:
$$
\begin{aligned}
& \mathcal{J}_{G R P O}(\theta)=\mathbb{E}\left[q \sim P(Q),\left\{o_{i}\right\}_{i=1}^{G} \sim \pi_{\theta_{o l d}}(O \mid q)\right] \\
& \qquad \frac{1}{G} \sum_{i=1}^{G}\left(\min \left(\frac{\pi_{\theta}\left(o_{i} \mid q\right)}{\pi_{\theta_{o l d}}\left(o_{i} \mid q\right)} A_{i}, \operatorname{clip}\left(\frac{\pi_{\theta}\left(o_{i} \mid q\right)}{\pi_{\theta_{o l d}}\left(o_{i} \mid q\right)}, 1-\varepsilon, 1+\varepsilon\right) A_{i}\right)-\beta \mathbb{D}_{K L}\left(\pi_{\theta} \| \pi_{r e f}\right)\right), \\
& \mathbb{D}_{K L}\left(\pi_{\theta} \| \pi_{r e f}\right)=\frac{\pi_{r e f}\left(o_{i} \mid q\right)}{\pi_{\theta}\left(o_{i} \mid q\right)}-\log \frac{\pi_{r e f}\left(o_{i} \mid q\right)}{\pi_{\theta}\left(o_{i} \mid q\right)}-1,
\end{aligned}
$$
where $\varepsilon$ and $\beta$ are hyper-parameters; $\pi_{\text {ref }}$ is the reference model; and $A_{i}$ is the advantage, derived from the rewards $\left\{r_{1}, r_{2}, \ldots, r_{G}\right\}$ corresponding to the outputs within each group:
$$
A_{i}=\frac{r_{i}-\operatorname{mean}\left(\left\{r_{1}, r_{2}, \cdots, r_{G}\right\}\right)}{\operatorname{std}\left(\left\{r_{1}, r_{2}, \cdots, r_{G}\right\}\right)} .
$$

We incorporate prompts from diverse domains, such as coding, math, writing, role-playing, and question answering, during the RL process. This approach not only aligns the model more closely with human preferences but also enhances performance on benchmarks, especially in scenarios where available SFT data are limited.

## 5.3. Evaluations

## 5.3.1. Evaluation Settings

Evaluation Benchmarks. Apart from the benchmark we used for base model testing, we further evaluate instructed models on IFEval (Zhou et al., 2023), FRAMES (Krishna et al., 2024), LongBench v2 (Bai et al., 2024), GPQA (Rein et al., 2023), SimpleQA (OpenAI, 2024c), C-SimpleQA (He et al., 2024), SWE-Bench Verified (OpenAI, 2024d), Aider ${ }^{1}$, LiveCodeBench (Jain et al., 2024) (questions from August 2024 to November 2024), Codeforces ${ }^{2}$, Chinese National High School Mathematics Olympiad (CNMO 2024) ${ }^{3}$, and American Invitational Mathematics Examination 2024 (AIME 2024) (MAA, 2024).

Compared Baselines. We conduct comprehensive evaluations of our chat model against several strong baselines, including DeepSeek-V2-0506, DeepSeek-V2.5-0905, Qwen2.5 72B Instruct, LLaMA-3.1 405B Instruct, Claude-Sonnet-3.5-1022, and GPT-4o-0513. For the DeepSeek-V2 model series, we select the most representative variants for comparison. For closed-source models, evaluations are performed through their respective APIs.

Detailed Evaluation Configurations. For standard benchmarks including MMLU, DROP, GPQA, and SimpleQA, we adopt the evaluation prompts from the simple-evals framework ${ }^{4}$.

\footnotetext{
${ }^{1}$ https://aider.chat
${ }^{2}$ https://codeforces.com
${ }^{3}$ https://www.cms.org.cn/Home/comp/comp/cid/12.html
${ }^{4}$ https://github.com/openai/simple-evals
}

We utilize the Zero-Eval prompt format (Lin, 2024) for MMLU-Redux in a zero-shot setting. For other datasets, we follow their original evaluation protocols with default prompts as provided by the dataset creators. For code and math benchmarks, the HumanEval-Mul dataset includes 8 mainstream programming languages (Python, Java, Cpp, C\#, JavaScript, TypeScript, PHP, and Bash) in total. We use CoT and non-CoT methods to evaluate model performance on LiveCodeBench, where the data are collected from August 2024 to November 2024. The Codeforces dataset is measured using the percentage of competitors. SWE-Bench verified is evaluated using the agentless framework (Xia et al., 2024). We use the "diff" format to evaluate the Aider-related benchmarks. For mathematical assessments, AIME and CNMO 2024 are evaluated with a temperature of 0.7, and the results are averaged over 16 runs, while MATH-500 employs greedy decoding. We allow all models to output a maximum of 8192 tokens for each benchmark.

\begin{table}
\begin{tabular}[t]{|l|l|l|l|l|l|l|l|l|}
\hline & Benchmark (Metric) & DeepSeek V2-0506 & DeepSeek V2.5-0905 & Qwen2.5 72B-Inst. & LLaMA-3.1 405B-Inst. & Claude-3.5-Sonnet-1022 & GPT-4o 0513 & DeepSeek V3 \\
\hline \multirow{3}{*}{} & Architecture & MoE & MoE & Dense & Dense & - & - & MoE \\
\hline & \# Activated Params & 21B & 21B & 72B & 405B & - & - & 37B \\
\hline & \# Total Params & 236B & 236B & 72B & 405B & - & - & 671B \\
\hline \multirow{9}{*}{\begin{tabular}[t]{l}
English \\
English
\end{tabular}} & MMLU (em) & 78.2 & 80.6 & 85.3 & 88.6 & 88.3 & 87.2 & 88.5 \\
\hline & MMLU-Redux (em) & 77.9 & 80.3 & 85.6 & 86.2 & 88.9 & 88.0 & 89.1 \\
\hline & MMLU-Pro (em) & 58.5 & 66.2 & 71.6 & 73.3 & 78.0 & 72.6 & 75.9 \\
\hline & DROP (3-shot F1) & 83.0 & 87.8 & 76.7 & 88.7 & 88.3 & 83.7 & 91.6 \\
\hline & IF-Eval (Prompt Strict) & 57.7 & 80.6 & 84.1 & 86.0 & 86.5 & 84.3 & 86.1 \\
\hline & GPQA-Diamond (Pass@1) & 35.3 & 41.3 & 49.0 & 51.1 & 65.0 & 49.9 & 59.1 \\
\hline & SimpleQA (Correct) & 9.0 & 10.2 & 9.1 & 17.1 & 28.4 & 38.2 & 24.9 \\
\hline & FRAMES (Acc.) & 66.9 & 65.4 & 69.8 & 70.0 & 72.5 & 80.5 & 73.3 \\
\hline & LongBench v2 (Acc.) & 31.6 & 35.4 & 39.4 & 36.1 & 41.0 & 48.1 & 48.7 \\
\hline \multirow{7}{*}{Code} & HumanEval-Mul (Pass@1) & 69.3 & 77.4 & 77.3 & 77.2 & 81.7 & 80.5 & 82.6 \\
\hline & LiveCodeBench (Pass@1-cot) & 18.8 & 29.2 & 31.1 & 28.4 & 36.3 & 33.4 & 40.5 \\
\hline & LiveCodeBench (Pass@1) & 20.3 & 28.4 & 28.7 & 30.1 & 32.8 & 34.2 & 37.6 \\
\hline & Codeforces (Percentile) & 17.5 & 35.6 & 24.8 & 25.3 & 20.3 & 23.6 & 51.6 \\
\hline & SWE Verified (Resolved) & - & 22.6 & 23.8 & 24.5 & 50.8 & 38.8 & 42.0 \\
\hline & Aider-Edit (Acc.) & 60.3 & 71.6 & 65.4 & 63.9 & 84.2 & 72.9 & 79.7 \\
\hline & Aider-Polyglot (Acc.) & - & 18.2 & 7.6 & 5.8 & 45.3 & 16.0 & 49.6 \\
\hline \multirow{3}{*}{Math} & AIME 2024 (Pass@1) & 4.6 & 16.7 & 23.3 & 23.3 & 16.0 & 9.3 & 39.2 \\
\hline & MATH-500 (em) & 56.3 & 74.7 & 80.0 & 73.8 & 78.3 & 74.6 & 90.2 \\
\hline & CNMO 2024 (Pass@1) & 2.8 & 10.8 & 15.9 & 6.8 & 13.1 & 10.8 & 43.2 \\
\hline \multirow{3}{*}{Chinese} & CLUEWSC (em) & 89.9 & 90.4 & 91.4 & 84.7 & 85.4 & 87.9 & 90.9 \\
\hline & C-Eval (em) & 78.6 & 79.5 & 86.1 & 61.5 & 76.7 & 76.0 & 86.5 \\
\hline & C-SimpleQA (Correct) & 48.5 & 54.1 & 48.4 & 50.4 & 51.3 & 59.3 & 64.8 \\
\hline
\end{tabular}

**Table 6 | Comparison between DeepSeek-V3 and other representative chat models. All models are evaluated in a configuration that limits the output length to 8K. Benchmarks containing fewer than 1000 samples are tested multiple times using varying temperature settings to derive robust final results. DeepSeek-V3 stands as the best-performing open-source model, and also exhibits competitive performance against frontier closed-source models.**
\end{table}

## 5.3.2. Standard Evaluation

Table 6 presents the evaluation results, showcasing that DeepSeek-V3 stands as the bestperforming open-source model. Additionally, it is competitive against frontier closed-source models like GPT-4o and Claude-3.5-Sonnet.

English Benchmarks. MMLU is a widely recognized benchmark designed to assess the performance of large language models, across diverse knowledge domains and tasks. DeepSeek-V3 demonstrates competitive performance, standing on par with top-tier models such as LLaMA-3.1-405B, GPT-4o, and Claude-Sonnet 3.5, while significantly outperforming Qwen2.5 72B. Moreover, DeepSeek-V3 excels in MMLU-Pro, a more challenging educational knowledge benchmark, where it closely trails Claude-Sonnet 3.5. On MMLU-Redux, a refined version of MMLU with corrected labels, DeepSeek-V3 surpasses its peers. In addition, on GPQA-Diamond, a PhD-level evaluation testbed, DeepSeek-V3 achieves remarkable results, ranking just behind Claude 3.5 Sonnet and outperforming all other competitors by a substantial margin.

In long-context understanding benchmarks such as DROP, LongBench v2, and FRAMES, DeepSeek-V3 continues to demonstrate its position as a top-tier model. It achieves an impressive 91.6 F1 score in the 3-shot setting on DROP, outperforming all other models in this category. On FRAMES, a benchmark requiring question-answering over 100k token contexts, DeepSeek-V3 closely trails GPT-4o while outperforming all other models by a significant margin. This demonstrates the strong capability of DeepSeek-V3 in handling extremely long-context tasks. The long-context capability of DeepSeek-V3 is further validated by its best-in-class performance on LongBench v2, a dataset that was released just a few weeks before the launch of DeepSeek V3. On the factual knowledge benchmark, SimpleQA, DeepSeek-V3 falls behind GPT-4o and Claude-Sonnet, primarily due to its design focus and resource allocation. DeepSeek-V3 assigns more training tokens to learn Chinese knowledge, leading to exceptional performance on the C-SimpleQA. On the instruction-following benchmark, DeepSeek-V3 significantly outperforms its predecessor, DeepSeek-V2-series, highlighting its improved ability to understand and adhere to user-defined format constraints.

Code and Math Benchmarks. Coding is a challenging and practical task for LLMs, encompassing engineering-focused tasks like SWE-Bench-Verified and Aider, as well as algorithmic tasks such as HumanEval and LiveCodeBench. In engineering tasks, DeepSeek-V3 trails behind Claude-Sonnet-3.5-1022 but significantly outperforms open-source models. The open-source DeepSeek-V3 is expected to foster advancements in coding-related engineering tasks. By providing access to its robust capabilities, DeepSeek-V3 can drive innovation and improvement in areas such as software engineering and algorithm development, empowering developers and researchers to push the boundaries of what open-source models can achieve in coding tasks. In algorithmic tasks, DeepSeek-V3 demonstrates superior performance, outperforming all baselines on benchmarks like HumanEval-Mul and LiveCodeBench. This success can be attributed to its advanced knowledge distillation technique, which effectively enhances its code generation and problem-solving capabilities in algorithm-focused tasks.

On math benchmarks, DeepSeek-V3 demonstrates exceptional performance, significantly surpassing baselines and setting a new state-of-the-art for non-o1-like models. Specifically, on AIME, MATH-500, and CNMO 2024, DeepSeek-V3 outperforms the second-best model, Qwen2.5 72B, by approximately 10\% in absolute scores, which is a substantial margin for such challenging benchmarks. This remarkable capability highlights the effectiveness of the distillation technique from DeepSeek-R1, which has been proven highly beneficial for non-o1-like models.

Chinese Benchmarks. Qwen and DeepSeek are two representative model series with robust support for both Chinese and English. On the factual benchmark Chinese SimpleQA, DeepSeek-V3 surpasses Qwen2.5-72B by 16.4 points, despite Qwen2.5 being trained on a larger corpus compromising 18T tokens, which are 20\% more than the 14.8T tokens that DeepSeek-V3 is

\begin{table}
\begin{tabular}[t]{|l|l|l|}
\hline Model & Arena-Hard & AlpacaEval 2.0 \\
\hline DeepSeek-V2.5-0905 & 76.2 & 50.5 \\
\hline Qwen2.5-72B-Instruct & 81.2 & 49.1 \\
\hline LLaMA-3.1 405B & 69.3 & 40.5 \\
\hline GPT-4o-0513 & 80.4 & 51.1 \\
\hline Claude-Sonnet-3.5-1022 & 85.2 & 52.0 \\
\hline DeepSeek-V3 & 85.5 & 70.0 \\
\hline
\end{tabular}

**Table 7 | English open-ended conversation evaluations. For AlpacaEval 2.0, we use the lengthcontrolled win rate as the metric.**
\end{table}
pre-trained on.
On C-Eval, a representative benchmark for Chinese educational knowledge evaluation, and CLUEWSC (Chinese Winograd Schema Challenge), DeepSeek-V3 and Qwen2.5-72B exhibit similar performance levels, indicating that both models are well-optimized for challenging Chinese-language reasoning and educational tasks.

## 5.3.3. Open-Ended Evaluation

In addition to standard benchmarks, we also evaluate our models on open-ended generation tasks using LLMs as judges, with the results shown in Table 7. Specifically, we adhere to the original configurations of AlpacaEval 2.0 (Dubois et al., 2024) and Arena-Hard (Li et al., 2024a), which leverage GPT-4-Turbo-1106 as judges for pairwise comparisons. On Arena-Hard, DeepSeek-V3 achieves an impressive win rate of over 86\% against the baseline GPT-4-0314, performing on par with top-tier models like Claude-Sonnet-3.5-1022. This underscores the robust capabilities of DeepSeek-V3, especially in dealing with complex prompts, including coding and debugging tasks. Furthermore, DeepSeek-V3 achieves a groundbreaking milestone as the first open-source model to surpass 85\% on the Arena-Hard benchmark. This achievement significantly bridges the performance gap between open-source and closed-source models, setting a new standard for what open-source models can accomplish in challenging domains.

Similarly, DeepSeek-V3 showcases exceptional performance on AlpacaEval 2.0, outperforming both closed-source and open-source models. This demonstrates its outstanding proficiency in writing tasks and handling straightforward question-answering scenarios. Notably, it surpasses DeepSeek-V2.5-0905 by a significant margin of 20\%, highlighting substantial improvements in tackling simple tasks and showcasing the effectiveness of its advancements.

## 5.3.4. DeepSeek-V3 as a Generative Reward Model

We compare the judgment ability of DeepSeek-V3 with state-of-the-art models, namely GPT-4o and Claude-3.5. Table 8 presents the performance of these models in RewardBench (Lambert et al., 2024). DeepSeek-V3 achieves performance on par with the best versions of GPT-4o-0806 and Claude-3.5-Sonnet-1022, while surpassing other versions. Additionally, the judgment ability of DeepSeek-V3 can also be enhanced by the voting technique. Therefore, we employ DeepSeekV3 along with voting to offer self-feedback on open-ended questions, thereby improving the effectiveness and robustness of the alignment process.

\begin{table}
\begin{tabular}[t]{|l|l|l|l|l|l|}
\hline Model & Chat & Chat-Hard & Safety & Reasoning & Average \\
\hline GPT-4o-0513 & 96.6 & 70.4 & 86.7 & 84.9 & 84.7 \\
\hline GPT-4o-0806 & 96.1 & 76.1 & 88.1 & 86.6 & 86.7 \\
\hline GPT-4o-1120 & 95.8 & 71.3 & 86.2 & 85.2 & 84.6 \\
\hline Claude-3.5-sonnet-0620 & 96.4 & 74.0 & 81.6 & 84.7 & 84.2 \\
\hline Claude-3.5-sonnet-1022 & 96.4 & 79.7 & 91.1 & 87.6 & 88.7 \\
\hline DeepSeek-V3 & 96.9 & 79.8 & 87.0 & 84.3 & 87.0 \\
\hline DeepSeek-V3 (maj@6) & 96.9 & 82.6 & 89.5 & 89.2 & 89.6 \\
\hline
\end{tabular}

**Table 8 | Performances of GPT-4o, Claude-3.5-sonnet and DeepSeek-V3 on RewardBench.**
\end{table}

\begin{table}
\begin{tabular}[t]{|l|l|l|l|l|}
\hline \multirow{2}{*}{Model} & \multicolumn{2}{|l|}{LiveCodeBench-CoT} & \multicolumn{2}{|c|}{MATH-500} \\
\hline & Pass@1 & Length & Pass@1 & Length \\
\hline DeepSeek-V2.5 Baseline & 31.1 & 718 & 74.6 & 769 \\
\hline DeepSeek-V2.5 +R1 Distill & 37.4 & 783 & 83.2 & 1510 \\
\hline
\end{tabular}

**Table 9 | The contribution of distillation from DeepSeek-R1. The evaluation settings of LiveCodeBench and MATH-500 are the same as in Table 6**
\end{table}

## 5.4. Discussion

## 5.4.1. Distillation from DeepSeek-R1

We ablate the contribution of distillation from DeepSeek-R1 based on DeepSeek-V2.5. The baseline is trained on short CoT data, whereas its competitor uses data generated by the expert checkpoints described above.

Table 9 demonstrates the effectiveness of the distillation data, showing significant improvements in both LiveCodeBench and MATH-500 benchmarks. Our experiments reveal an interesting trade-off: the distillation leads to better performance but also substantially increases the average response length. To maintain a balance between model accuracy and computational efficiency, we carefully selected optimal settings for DeepSeek-V3 in distillation.

Our research suggests that knowledge distillation from reasoning models presents a promising direction for post-training optimization. While our current work focuses on distilling data from mathematics and coding domains, this approach shows potential for broader applications across various task domains. The effectiveness demonstrated in these specific areas indicates that long-CoT distillation could be valuable for enhancing model performance in other cognitive tasks requiring complex reasoning. Further exploration of this approach across different domains remains an important direction for future research.

## 5.4.2. Self-Rewarding

Rewards play a pivotal role in RL, steering the optimization process. In domains where verification through external tools is straightforward, such as some coding or mathematics scenarios, RL demonstrates exceptional efficacy. However, in more general scenarios, constructing a feedback mechanism through hard coding is impractical. During the development of DeepSeek-V3, for these broader contexts, we employ the constitutional AI approach (Bai et al., 2022), leveraging the voting evaluation results of DeepSeek-V3 itself as a feedback source. This method has
produced notable alignment effects, significantly enhancing the performance of DeepSeek-V3 in subjective evaluations. By integrating additional constitutional inputs, DeepSeek-V3 can optimize towards the constitutional direction. We believe that this paradigm, which combines supplementary information with LLMs as a feedback source, is of paramount importance. The LLM serves as a versatile processor capable of transforming unstructured information from diverse scenarios into rewards, ultimately facilitating the self-improvement of LLMs. Beyond self-rewarding, we are also dedicated to uncovering other general and scalable rewarding methods to consistently advance the model capabilities in general scenarios.

## 5.4.3. Multi-Token Prediction Evaluation

Instead of predicting just the next single token, DeepSeek-V3 predicts the next 2 tokens through the MTP technique. Combined with the framework of speculative decoding (Leviathan et al., 2023; Xia et al., 2023), it can significantly accelerate the decoding speed of the model. A natural question arises concerning the acceptance rate of the additionally predicted token. Based on our evaluation, the acceptance rate of the second token prediction ranges between 85\% and 90\% across various generation topics, demonstrating consistent reliability. This high acceptance rate enables DeepSeek-V3 to achieve a significantly improved decoding speed, delivering 1.8 times TPS (Tokens Per Second).

