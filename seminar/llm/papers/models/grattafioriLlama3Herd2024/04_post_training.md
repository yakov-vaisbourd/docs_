# 4 Post-Training

We produce the aligned Llama 3 models by applying several rounds of post-training, ${ }^{6}$ or aligning the model with human feedback (Ouyang et al., 2022; Rafailov et al., 2024) on top of a pre-trained checkpoint. Each round of post-training involves supervised finetuning (SFT) followed by Direct Preference Optimization (DPO; Rafailov et al., 2024) on examples collected either via human annotations or generated synthetically. Our post-training modeling and data approaches are described in Sections 4.1 and 4.2 respectively. We further detail custom data curation strategies to improve the reasoning, coding, factuality, multilingual, tool use, long context, and precise instruction following in Section 4.3.

## 4.1 Modeling

The backbone of our post-training strategy is a reward model and a language model. We first train a reward model on top of the pre-trained checkpoint using human-annotated preference data (see Section 4.1.2). We then finetune pre-trained checkpoints with supervised finetuning (SFT; see Section 4.1.3), and further align the checkpoints with Direct Preference Optimization (DPO; see Section 4.1.4). This process is illustrated in Figure 7. Unless otherwise noted, our modeling procedure applies to Llama 3 405B, and we refer to Llama 3 405B as Llama 3 for simplicity.

## 4.1.1 Chat Dialog Format

To tune LLMs for human-AI interaction, we need to define a chat dialog protocol for the model to understand human instructions and perform conversational tasks. Compared to its predecessor, Llama 3 has new capabilities such as tool use (Section 4.3.5) which may require generating multiple messages and sending

\footnotetext{
${ }^{6}$ We use the term "post-training" to refer to any model training that happens outside of pre-training.
}
them to different locations (e.g., user, ipython) within a single dialog turn. To support this, we design a new multi-message chat protocol which uses various special header and termination tokens. The header tokens are used to indicate the source and destination of each message in a conversation. Similarly, the termination tokens indicate when it is the time to alternate between human and AI to speak.

## 4.1.2 Reward Modeling

We train a reward model (RM) covering different capabilities on top of the pre-trained checkpoint. The training objective is the same as Llama 2 except that we remove the margin term in the loss, as we observe diminishing improvements after data scaling. Following Llama 2, we use all of our preference data for reward modeling after filtering out samples with similar responses. In addition to standard preference pair of (chosen, rejected) response, annotations also create a third "edited response" for some prompts, where the chosen response from the pair is further edited for improvement (see Section 4.2.1). Hence, each preference ranking sample has two or three responses with clear ranking (edited $>$ chosen $>$ rejected). We concatenate the prompt and multiple responses into a single row during training with responses randomly shuffled. This is an approximation to the standard scenario of putting the responses in separate rows and computing the scores, but in our ablations, this approach improves training efficiency without a loss in accuracy.

## 4.1.3 Supervised Finetuning

The reward model is then used to perform rejection sampling on our human annotation prompts, the details of which are described in Section 4.2. Together with this rejection-sampled data and other data sources (including synthetic data), we finetune the pre-trained language model using a standard cross entropy loss on the target tokens (while masking loss on prompt tokens). More details about the data mix can be found in Section 4.2. We refer to this stage as supervised finetuning (SFT; Wei et al., 2022a; Sanh et al., 2022; Wang et al., 2022b), even though many of the training targets are model-generated. Our largest models are finetuned with a learning rate of $10^{-5}$ over the course of 8.5K to 9K steps. We found these hyperparameter settings to work well across different rounds and data mixes.

## 4.1.4 Direct Preference Optimization

We further train our SFT models with Direct Preference Optimization (DPO; Rafailov et al., 2024) for human preference alignment. For training, we primarily use the most recent batches of preference data collected using the best performing models from the previous alignment rounds. As a result, our training data conforms better to the distribution of the policy model that is being optimized in each round. We also explored on-policy algorithms such as PPO (Schulman et al., 2017), but found that DPO required less compute for large-scale models and performed better, especially on instruction following benchmarks like IFEval (Zhou et al., 2023). For Llama 3, we use a learning rate of $10^{-5}$ and set the $\beta$ hyper-parameter to be 0.1. In addition, we apply the following algorithmic modifications to DPO:
\begin{itemize}
\item[-] Masking out formatting tokens in DPO loss: We mask out special formatting tokens including header and termination tokens (described in Section 4.1.1) from both chosen and rejected responses in the loss to stabilize DPO training. We observe that having these tokens contribute to the loss may lead to undesired model behaviors such as tail repetition or abruptly generating termination tokens. We hypothesize that this is due to the contrastive nature of the DPO loss - the presence of common tokens in both chosen and rejected responses leads to a conflicting learning objective as the model needs to increase and reduce the likelihood of these tokens simultaneously.
\item[-] Regularization with NLL loss: We add an additional negative log-likelihood (NLL) loss term with a scaling coefficient of 0.2 on the chosen sequences, similar to Pang et al. (2024). This helps further stabilize DPO training by maintaining desired formatting for generation and preventing the decrease of log probability of chosen responses (Pang et al., 2024; Pal et al., 2024).
\end{itemize}

## 4.1.5 Model Averaging

Finally, we average models obtained from experiments using various versions of data or hyperparameters at each RM, SFT, or DPO stage (Izmailov et al., 2019; Wortsman et al., 2022; Li et al., 2022).

\begin{table}
\begin{tabular}[t]{|l|l|l|l|l|l|}
\hline Dataset & \% of comparisons & Avg. \# turns per dialog & Avg. \# tokens per example & Avg. \# tokens in prompt & Avg. \# tokens in response \\
\hline General English & 81.99\% & 4.1 & 1,000.4 & 36.4 & 271.2 \\
\hline Coding & 6.93\% & 3.2 & 1,621.0 & 113.8 & 462.9 \\
\hline Multilingual & 5.19\% & 1.8 & 1,299.4 & 77.1 & 420.9 \\
\hline Reasoning and tools & 5.89\% & 1.6 & 707.7 & 46.6 & 129.9 \\
\hline Total & 100\% & 3.8 & 1,041.6 & 44.5 & 284.0 \\
\hline
\end{tabular}

**Table 6 Statistics of human preference data. We list statistics of the internally collected human preference data used for Llama 3 alignment. We ask annotators to perform multi-turn dialogues with the models and make comparisons among responses at each turn. In post-processing, we split each dialogue to multiple examples at a turn level. Each example consists of a prompt (including previous dialog if available) and a response (e.g., chosen or rejected response).**
\end{table}

## 4.1.6 Iterative Rounds

Following Llama 2, we apply the above methods in six rounds. In each cycle, we collect new preference annotations and SFT data, sampling synthetic data from the latest models.

## 4.2 Post-training Data

The post-training data composition plays a critical role in the usefulness and behavior of language models. In this section, we discuss our human annotation procedures and preference data collection (Section 4.2.1), the composition of our SFT data (Section 4.2.2), and methods for data quality control and cleaning (Section 4.2.3).

## 4.2.1 Preference Data

Our preference data annotation process is similar to Llama 2. We deploy multiple models for annotation after each round and sample two responses from two different models for each user prompt. These models can be trained with different data mixes and alignment recipes, allowing for different capability strength (e.g., code expertise) and increased data diversity. We ask annotators to rate the strength of their preference by categorizing it into one of four levels, based on how much more they prefer the chosen response over the rejected one: significantly better, better, slightly better, or marginally better. We also incorporate an editing step after preference ranking to encourage annotators to further improve the preferred response. Annotators edit the chosen response directly or prompt the model with feedback to refine its own response. Consequently, a portion of our preference data has three responses ranked (edited > chosen > rejected).

In Table 6, we report the statistics of preference annotations that we use for Llama 3 training. General English covers multiple subcategories such as knowledge-based question and answering or precise instruction-following, which fall outside the scope of specific capabilities. Compared to Llama 2, we observe an increase in the average length of prompt and response, suggesting that we train Llama 3 on more complex tasks. In addition, we implement a quality analysis and human evaluation process to rigorously assess the data collected, allowing us to refine our prompts and provide systematic, actionable feedback to annotators. For example, as Llama 3 improves after each round, we increase prompt complexity accordingly to target areas where the model lags.

In each round of post-training, we use all the preference data that is available at the time for reward modeling, while only using the latest batches from various capabilities for DPO training. For both reward modeling and DPO, we use samples that are labeled as the chosen response being significantly better or better than the rejected counterpart for training and discard samples with similar responses.

## 4.2.2 SFT Data

Our finetuning data is largely comprised of the following sources:
\begin{itemize}
\item[-] Prompts from our human annotation collection with rejection-sampled responses.
\item[-] Synthetic data targeting specific capabilities (see Section 4.3 for more details).
\end{itemize}

\begin{table}
\begin{tabular}[t]{|l|l|l|l|l|l|}
\hline Dataset & \% of examples & Avg. \# turns & Avg. \# tokens & Avg. \# tokens in context & Avg. \# tokens in final response \\
\hline General English & 52.66\% & 6.3 & 974.0 & 656.7 & 317.1 \\
\hline Code & 14.89\% & 2.7 & 753.3 & 378.8 & 374.5 \\
\hline Multilingual & 3.01\% & 2.7 & 520.5 & 230.8 & 289.7 \\
\hline Exam-like & 8.14\% & 2.3 & 297.8 & 124.4 & 173.4 \\
\hline Reasoning and tools & 21.19\% & 3.1 & 661.6 & 359.8 & 301.9 \\
\hline Long context & 0.11\% & 6.7 & 38,135.6 & 37,395.2 & 740.5 \\
\hline Total & 100\% & 4.7 & 846.1 & 535.7 & 310.4 \\
\hline
\end{tabular}

**Table 7 Statistics of SFT data. We list internally collected SFT data used for Llama 3 alignment. Each SFT example consists of a context (i.e., all conversation turns except the last one) and a final response.**
\end{table}
\begin{itemize}
\item[-] Small amounts of human-curated data (see Section 4.3 for more details).
\end{itemize}

As our post-training rounds progress, we develop stronger Llama 3 variants that we use to collect larger datasets that cover a wide range of complex capabilities. In this section, we discuss the details for the rejection-sampling procedure and overall composition of our final SFT datamix.

Rejection sampling. During rejection sampling (RS), for each prompt collected during human annotation (Section 4.2.1) we sample $K$ (typically between 10 and 30) outputs from the latest chat model policy (usually the best performing checkpoint from the previous post-training iteration, or the best performing checkpoint for a particular capability) and use our reward model to select the best candidate, consistent with Bai et al. (2022). In later rounds of post-training, we introduce system prompts to steer RS responses to conform with desirable tone, style, or formatting, which might be different for different capabilities.

To increase the efficiency of rejection sampling, we adopt PagedAttention (Kwon et al., 2023). PagedAttention enhances memory efficiency through dynamic key-value cache allocation. It supports arbitrary output lengths by dynamically scheduling requests based on the current cache capacity. Unfortunately, this carries the risk of swap-out when running out of memory. To eliminate such swap overhead, we define a maximum output length and perform a request only if sufficient memory is available to fit an output with that length. PagedAttention also enables us to share the key-value cache pages for a prompt across all corresponding outputs. Together, this leads to a throughput improvement of over 2× during rejection sampling.

Overall data composition. Table 7 shows data statistics for each broad category of our "helpfulness" mix. While SFT and preference data contain overlapping domains, they are curated differently, yielding distinct count statistics. In Section 4.2.3 we describe techniques for categorizing topic, complexity, and quality of our data samples. In each round of post-training, we adjust our overall data mix carefully across these axes to tune performance across a wide range of benchmarks. Our final data mix epochs multiple times on some high quality sources and downsamples others.

## 4.2.3 Data Processing and Quality Control

Given that most of our training data is model-generated, it requires careful cleaning and quality control.
Data cleaning. In the early rounds, we observed a number of undesirable patterns common in our data, such as excessive use of emojis or exclamation points. Therefore, we implement a series of rule-based data removal and modification strategies to filter or clean problematic data. For example, to mitigate overly-apologetic tonal issues, we identify overused phrases (such as "I'm sorry" or "I apologize") and carefully balance the proportion of such samples in our dataset.

Data pruning. We also apply a collection of model-based techniques to remove low-quality training samples and improve overall model performance:
\begin{itemize}
\item[-] Topic classification: We first finetune Llama 3 8B into a topic classifier, and perform inference over all data to classify it into both coarsely-grained buckets ("mathematical reasoning") and fine-grained
\end{itemize}
buckets ("geometry and trigonometry").
\begin{itemize}
\item[-] Quality scoring: We use both reward model and Llama-based signals to obtain a quality score for each sample. For an RM-based score, we consider data that is in the top quartile of RM scores as high quality. For a Llama-based score, we prompt Llama 3 checkpoint to rate each sample on a three-point scale for general English data (accuracy, instruction following, and tone/presentation) and a two-point scale for coding data (bug identification and user intention), and consider samples that obtain the maximum score as high quality. The RM and Llama-based scores have high disagreement rates, and we find that combining these signals yield the best recall on our internal test set. Ultimately, we select examples that are marked as high quality by the RM or the Llama-based filter.
\item[-] Difficulty scoring: Because we are also interested in prioritizing examples that are more complex for the model, we score data using two measures of difficulty: Instag (Lu et al., 2023) and Llama-based scoring. For Instag, we prompt Llama 3 70B to perform intention tagging of SFT prompts, where more intentions implies more complexity. We also prompt Llama 3 to measure the difficulty (Liu et al., 2024c) of dialogs on a three-point scale.
\item[-] Semantic deduplication: Finally, we perform semantic deduplication (Abbas et al., 2023; Liu et al., 2024c). We first cluster complete dialogs using RoBERTa (Liu et al., 2019b) and within each cluster sort them by quality score × difficulty score. We then do greedy selection by iterating through all sorted examples, and only keeping the ones that have maximum cosine similarity less than a threshold to the examples seen so far in the cluster.
\end{itemize}

## 4.3 Capabilities

We highlight special efforts to improve performance for specific capabilities such as code (Section 4.3.1), multilinguality (Section 4.3.2), math and reasoning (Section 4.3.3), long context (Section 4.3.4), tool use (Section 4.3.5), factuality (Section 4.3.6), and steerability (Section 4.3.7).

## 4.3.1 Code

LLMs for code have received significant attention since the release of Copilot and Codex (Chen et al., 2021). Developers are now widely using these models to generate code snippets, debug, automate tasks, and improve code quality. For Llama 3, we target improving and evaluating code generation, documentation, debugging, and review capabilities for the following high priority programming languages: Python, Java, Javascript, C/C++, Typescript, Rust, PHP, HTML/CSS, SQL, bash/shell. Here, we present our work on improving these coding capabilities via training a code expert, generating synthetic data for SFT, improving formatting with system prompt steering, and creating quality filters to remove bad samples from our training data.

Expert training. We train a code expert which we use to collect high quality human annotations for code throughout subsequent rounds of post-training. This is accomplished by branching the main pre-training run and continuing pre-training on a 1T token mix of mostly (>85\%) code data. Continued pre-training on domainspecific data has been shown to be effective for improving performance in a specific domain (Gururangan et al., 2020). We follow a recipe similar to that of CodeLlama (Rozière et al., 2023). For the last several thousand steps of training we perform long-context finetuning (LCFT) to extend the expert's context length to 16K tokens on a high quality mix of repo-level code data. Finally, we follow the similar post-training modeling recipes described in Section 4.1 to align this model, except with SFT and DPO data mixes primarily targeting code. This model is also used for rejection sampling (Section 4.2.2) for coding prompts.

Synthetic data generation. During development, we identified key issues in code generation, including difficulty in following instructions, code syntax errors, incorrect code generation, and difficulty in fixing bugs. While intensive human annotation could theoretically resolve these issues, synthetic data generation offers a complementary approach at a lower cost and higher scale, unconstrained by the expertise level of annotators. As such, we use Llama 3 and the code expert to generate a large quantity of synthetic SFT dialogs.

We describe three high-level approaches for generating synthetic code data. In total, we generate over 2.7M synthetic examples which were used during SFT.
\begin{itemize}
\item[1.] Synthetic data generation: execution feedback. The 8B and 70B models show significant performance improvements when trained on data generated by a larger, more competent model. However, our initial experiments revealed that training Llama 3 405B on its own generated data is not helpful (and can even degrade performance). To address this limitation, we introduced execution feedback as a source of truth, enabling the model to learn from its mistakes and stay on track. In particular, we generate large dataset of approximately one million synthetic coding dialogues using the following process:
\begin{itemize}
\item[-] Problem description generation: First, we generate a large collection of programming problem descriptions that span a diverse range of topics, including those in the long tail distribution. To achieve this diversity, we sample random code snippets from various sources and prompt the model to generate programming problems inspired by these examples. This allowed us to tap into a wide range of topics and create a comprehensive set of problem descriptions (Wei et al., 2024).
\item[-] Solution generation: Then, we prompt Llama 3 to solve each problem in a given programming language. We observe that adding general rules of good programming to the prompt improves the generated solution quality. Also, we find it is helpful to require the model to explain its thought process in comments.
\item[-] Correctness analysis: After generating a solution, it is crucial to recognize that its correctness is not guaranteed, and including incorrect solutions in the finetuning dataset could harm the model's quality. While we do not ensure complete correctness, we develop methods to approximate it. To achieve this, we extract the source code from the generated solution and applied a combination of static and dynamic analysis techniques to test its correctness, including:
\begin{itemize}
\item[-] Static analysis: We run all generated code through a parser and a linter to ensure syntactic correctness, catching errors such as syntax errors, use of uninitialized variables or non-imported functions, code style issues, typing errors, and others.
\item[-] Unit test generation and execution: For each problem and solution, we prompt the model to generate unit tests, executed in a containerized environment together with the solution, catching run-time execution errors and some semantic errors.
\end{itemize}
\item[-] Error feedback and iterative self-correction: When a solution fails at any step, we prompt the model to revise it. The prompt included the original problem description, the faulty solution, and feedback from the parser/linter/tester (stdout, stderr/ and return code). After a unit test execution failure, the model could either fix the code to pass the existing tests or modify its unit tests to accommodate the generated code. Only dialogs that pass all checks are included in the final dataset, used for supervised finetuning (SFT). Notably, we observed that about 20\% of solutions were initially incorrect but self-corrected, indicating that the model learned from the execution feedback and improved its performance.
\item[-] Fine-tuning and iterative improvement: The finetuning process is conducted over multiple rounds, with each round building on the previous one. After each round, the model is improved, generating higher-quality synthetic data for the next round. This iterative process allows for progressive refinement and enhancement of the model's performance.
\end{itemize}
\item[2.] Synthetic data generation: programming language translation. We observe a performance gap between major programming languages (e.g., Python/C++) and less common ones (e.g., Typescript/PHP). This is not surprising as we have less training data for less common programming languages. To mitigate this, we supplement our existing data by translating data from common programming languages to less common languages (similar to Chen et al. (2023) in the context of reasoning). This is achieved by prompting Llama 3 and ensuring quality via syntax parsing, compilation, and execution. Figure 8 demonstrates an example of synthetic PHP code translated from Python. This improves performance significantly for less common languages as measured by the MultiPL-E (Cassano et al., 2023) benchmark.
\item[3.] Synthetic data generation: backtranslation. To improve certain coding capabilities (e.g., documentation, explanations) where execution feedback is less informative for determining quality, we employ an alternative multi-step approach. Using this procedure, we generated approximately 1.2M synthetic
\end{itemize}

\begin{table}
\begin{tabular}[t]{|l|l|}
\hline \begin{lstlisting}[mathescape=true]
def gushti_cdi():
n = int(input())
arr = list(map(int, input().split()))
points = 0
for i in range(n):
if $\operatorname{arr}[i]==0$ and $i!=0$ :
idx = arr.index(max(arr[:i]))
points += arr[idx]
arr[idx] = 0
return points
for _ in range(int(input())):
print (gushti_cdi())
\end{lstlisting} & \begin{lstlisting}[mathescape=true]
<?php
function gushti_cdi() {
        \$n = (int) fgets(STDIN);
        Sarr = explode(" ", fgets(STDIN));
        \$points = 0;
        for (\$i = 0; \$i < \$n; \$i++) {
                if (\$arr[\$i] == 0 && \$i != 0) (
                    \$maxVal = max(array_slice(\$arr, 0, \$i));
                    \$idx = array_search(\$maxVal, \$arr);
                    \$points += \$arr[\$idx];
                    Sarr [\$idx] = 0;
                }
            }
            return \$points;
    }
\$t = (int) fgets(STDIN);
for (\$i = 0; \$i < \$t; \$i++) {
        echo gushti_cdi() . "\n";
}
?>
\end{lstlisting} \\
\hline
\end{tabular}

**Figure 8 Code translation example. We display an example of using Llama 3 to translate Python code (left) to PHP code (right) to augment our SFT dataset with a wider range of programming languages.**
\end{table}

\begin{table}
\begin{tabular}[t]{|l|l|}
\hline \begin{lstlisting}[mathescape=true]
public static int ClimbStairs (int n)
{
if (n == 1)
{
return 1;
}
if (n == 2)
{
return 2;
}
int[] dp = new int[n + 1];
dp[1] = 1;
dp[2] = 2;
for (int $\mathrm{i}=3$; \(\mathrm{i}<=\mathrm{n} ; \mathrm{i}++)\)
{
$\mathrm{dp}[i]=\mathrm{dp}[i-1]+\mathrm{dp}[i-2] ;$
}
return dp[n];
return dp[n];
}
\end{lstlisting} & \begin{lstlisting}[mathescape=true]
public static int ClimbStairs (int n)
{
// Base cases
if ( $\mathrm{n}==1$ ) return 1;
if ( $\mathrm{n}=-2$ ) return 2;
// Initialize variables to store the number
of ways to climb
int prev = 1;
int curr = 2;
// Calculate the number of ways to climb for
n > 2
for (int $\mathrm{i}=3$; $\mathrm{i}<=\mathrm{n}$; $\mathrm{i}++$ )
{
int temp = curr;
curr = prev + curr;
prev = temp;
}
return curr;
}
\end{lstlisting} \\
\hline
\end{tabular}

**Figure 9 Improving generated code quality with system prompts. Left: without system prompt Right: with system prompt.**
\end{table}
dialogs related to code explanation, generation, documentation, and debugging. Beginning with code snippets from a variety of languages in our pre-training data:
\begin{itemize}
\item[-] Generate: We prompt Llama 3 to generate data that represents our target capability (e.g., we add comments and docstrings for the code snippet, or we ask the model to explain a piece of code).
\item[-] Backtranslate: We then prompt the model to "backtranslate" the synthetically generated data to the original code (e.g., we prompt the model to generate code only from its documentation, or we ask the model to generate code only from its explanation).
\item[-] Filter: Using the original code as a reference, we prompt the Llama 3 to determine the quality of the output (e.g., we ask the model how faithful the backtranslated code is to the original). We then use the generated examples that have the highest self-verification scores in SFT.
\end{itemize}

System prompt steering during rejection sampling. During the rejection sampling process, we used code specific system prompts to improve code readability, documentation, thoroughness, and specificity. Recall, from Section 7 this data is used to finetune the language model. Figure 9 shows an example of how the system prompt helps improve the generated code quality - it adds necessary comments, uses more informative variable names, saves memory, etc.

Filtering training data with execution and model-as-judge signals. As described in Section 4.2.3, we occasionally encounter quality issues in our rejection-sampled data, such as code blocks containing bugs. Detecting these issues in our rejection-sampled data is not as straightforward as it is for our synthetic code data, as the rejection-sampled responses typically contain a mix of natural language and code for which the code may not
always be expected to be executable. (For example, user prompts may explicitly ask for pseudo-code or edits to only a very small snippet of an executable program.) To address this, we utilize the "model-as-judge" approach, where earlier versions of Llama 3 assess and assign a binary (0/1) score based on two criteria: code correctness and code style. We retain only those samples that achieve a perfect score of 2. Initially, this stringent filtering led to a regression in downstream benchmark performance, primarily because it disproportionately removed examples with challenging prompts. To counteract this, we strategically revise the responses of some coding data categorized as most challenging until they met the Llama-based "model-as-judge" criteria. By refining these challenging problems, the coding data achieves a balance between quality and difficulty, resulting in optimal downstream performance.

## 4.3.2 Multilinguality

We describe how we improve Llama 3's multilingual capabilities, including training an expert specialized on substantially more multilingual data, sourcing and generating high quality multilingual instruction tuning data for German, French, Italian, Portuguese, Hindi, Spanish, and Thai, and tackling specific challenges of multilingual language steering to enhance the overall performance of our model.

Expert training. Our Llama 3 pre-training data mix contains significantly more English tokens than non-English tokens. To collect higher quality human annotations in non-English languages, we train a multilingual expert by branching off the pre-training run and continuing to pre-train on a data mix that consists of 90\% multilingual tokens. We then perform post-training on this expert following Section 4.1. This expert model is then used to collect higher quality annotations in non-English languages until pre-training was fully complete.

Multilingual data collection. Our multilingual SFT data is derived primarily from sources described below. The overall distribution is 2.4\% human annotations, 44.2\% data from other NLP tasks, 18.8\% rejection sampled data, and 34.6\% translated reasoning data.
\begin{itemize}
\item[-] Human annotations: We collect high-quality, manually annotated data from linguists and native speakers. These annotations mostly consist of open-ended prompts that represent real world use cases.
\item[-] Data from other NLP tasks: To further augment, we use multilingual training data from other tasks and rewrite into dialog format. For example, we use data from exams-qa (Hardalov et al., 2020) and Conic10k (Wu et al., 2023). To improve language alignment, we also use parallel texts from GlobalVoices (Prokopidis et al., 2016) and Wikimedia (Tiedemann, 2012). We use LID based filtering and Blaser2.0 (Seamless Communication et al., 2023) to remove low quality data. For parallel text data, instead of using the bitext pairs directly, we apply a multilingual template inspired by Wei et al. (2022a) to better simulate real-life conversations in translation and language learning scenarios.
\item[-] Rejection sampled data: We apply rejection sampling on our human annotated prompts to generate high-quality samples for finetuning, with few modifications compared to the process for English data:
\begin{itemize}
\item[-] Generation: We explored randomly choosing the temperature hyperparameter from the range 0.2-1 for diverse generations in early rounds of post-training. With high temperature, responses for multilingual prompts can get creative and inspiring, but are also susceptible to unnecessary or unnatural code-switching. In the final round of post-training, we use a constant value of 0.6 to balance the trade-off. Additionally, we used specialized system prompts to improve response format, structure and general readability.
\item[-] Selection: Prior to reward model based selection, we implement multilingual-specific checks to ensure high language-match rate between the prompt and response (e.g., a romanized Hindi prompt should not expect a response in Hindi Devanagari script).
\end{itemize}
\item[-] Translated data: We try to avoid using machine-translated data to finetune the model in order to prevent translationese (Bizzoni et al., 2020; Muennighoff et al., 2023) or possible name bias (Wang et al., 2022a), gender bias (Savoldi et al., 2021), or cultural bias (Ji et al., 2023). Moreover, we aim to prevent the model from being exposed only to tasks that are rooted in English cultural context, which may not be representative of the linguistic and cultural diversity we aim to capture. We made one exception to this and translated our synthetic quantitative reasoning data (see Section 4.3.3 for details) to improve performance in quantitative reasoning in non-English languages. Due to the simple nature of
\end{itemize}
the language in these math problems, the translated samples were found to have little to no quality issues. We observed strong gains on MGSM (Shi et al., 2022) from adding this translated data.

## 4.3.3 Math and Reasoning

We define reasoning as the ability to perform multi-step computations and arrive at the correct final answer. Several challenges guide our approach to training models that excel in mathematical reasoning:
\begin{itemize}
\item[-] Lack of prompts: As the complexity of questions increases, the number of valid prompts or questions for Supervised Fine-Tuning (SFT) decreases. This scarcity makes it difficult to create diverse and representative training datasets for teaching models various mathematical skills (Yu et al., 2023; Yue et al., 2023; Luo et al., 2023; Mitra et al., 2024; Shao et al., 2024; Yue et al., 2024b).
\item[-] Lack of ground truth chain of thought: Effective reasoning requires a step-by-step solution to facilitate the reasoning process (Wei et al., 2022c). However, there is often a shortage of ground truth chains of thought, which are essential for guiding the model how to break down the problem step-by-step and reach the final answer (Zelikman et al., 2022).
\item[-] Incorrect intermediate steps: When using model-generated chains of thought, the intermediate steps may not always be correct (Cobbe et al., 2021; Uesato et al., 2022; Lightman et al., 2023; Wang et al., 2023a). This inaccuracy can lead to incorrect final answers and needs to be addressed.
\item[-] Teaching models to use external tools: Enhancing models to utilize external tools, such as code interpreters, allows them to reason by interleaving code and text (Gao et al., 2023; Chen et al., 2022; Gou et al., 2023). This capability can significantly improve their problem-solving abilities.
\item[-] Discrepancy between training and inference: There is often a discrepancy between how the model is finetuned during training and how it is used during inference. During inference, the finetuned model may interact with humans or other models, requiring it to improve its reasoning using feedback. Ensuring consistency between training and real-world usage is crucial for maintaining reasoning performance.
\end{itemize}

To address these challenges, we apply the following methodologies:
\begin{itemize}
\item[-] Addressing the lack of prompts: We source relevant pre-training data from mathematical contexts and converted it into a question-answer format which can then be used for supervised finetuning. Additionally, we identify mathematical skills where the model under-performs and actively sourced prompts from humans to teach models such skills. To facilitate this process, we create a taxonomy of mathematical skills (Didolkar et al., 2024) and ask humans to provide relevant prompts/questions accordingly.
\item[-] Augmenting training data with step-wise reasoning traces: We use Llama 3 to generate step-by-step solutions for a set of prompts. For each prompt, the model produces a variable number of generations. These generations are then filtered based on the correct answer (Li et al., 2024a). We also do selfverification where Llama 3 is used to verify whether a particular step-by-step solution is valid for a given question. This process improves the quality of the finetuning data by eliminating instances where the model does not produce valid reasoning traces.
\item[-] Filtering incorrect reasoning traces: We train outcome and stepwise reward models (Lightman et al., 2023; Wang et al., 2023a) to filter training data where the intermediate reasoning steps were incorrect. These reward models are used to eliminate data with invalid step-by-step reasoning, ensuring high-quality data for finetuning. For more challenging prompts, we use Monte Carlo Tree Search (MCTS) with learned step-wise reward models to generate valid reasoning traces, further enhancing the collection of high-quality reasoning data (Xie et al., 2024).
\item[-] Interleaving code and text reasoning: We prompt Llama 3 to solve reasoning problems through a combination of textual reasoning and associated Python code (Gou et al., 2023). Code execution is used as a feedback signal to eliminate cases where the reasoning chain was not valid, ensuring the correctness of the reasoning process.
\item[-] Learning from feedback and mistakes: To simulate human feedback, we utilize incorrect generations (i.e., generations leading to incorrect reasoning traces) and perform error correction by prompting Llama 3 to
\end{itemize}
yield correct generations (An et al., 2023b; Welleck et al., 2022; Madaan et al., 2024a). The iterative process of using feedback from incorrect attempts and correcting them helps improve the model's ability to reason accurately and learn from its mistakes.

## 4.3.4 Long Context

During the final pre-training stage, we extend the context length of Llama 3 from 8K tokens to 128K tokens (see Section 3.4 for more details). Similar to pre-training, we find that during finetuning we must carefully tune the recipe to balance short and long-context capabilities.

SFT and synthetic data generation. Naively applying our existing SFT recipe with only short-context data resulted in significant regressions in long-context capabilities from pre-training, highlighting the need to incorporate long-context data in our SFT data mix. In practice, however, it is largely impractical to get humans to annotate such examples due to the tedious and time-consuming nature of reading lengthy contexts, so we predominantly rely on synthetic data to fill this gap. We use earlier versions of Llama 3 to generate synthetic data based on the key long-context use-cases: (possibly multi-turn) question-answering, summarization for long documents, and reasoning over code repositories, and describe them in greater detail below.
\begin{itemize}
\item[-] Question answering: We carefully curate a set of long documents from our pre-training mix. We split these documents into chunks of 8K tokens, and prompted an earlier version of the Llama 3 model to generate QA pairs conditional on randomly selected chunks. During training, the whole document is used as context.
\item[-] Summarization: We applied hierarchical summarization of long-context documents by first summarizing the chunks of 8K input length using our strongest Llama 3 8K context model and then summarizing the summaries. During training we provide the full document and prompt the model to summarize the document while preserving all the important details. We also generate QA pairs based on the summaries of the documents and prompt the model with questions that require global understanding of the whole long document.
\item[-] Long context code reasoning: We parse Python files to identify import statements and determine their dependencies. From here, we select the most commonly depended-upon files, specifically those referenced by at least five other files. We remove one of these key files from a repository and prompt the model to identify which files depended on the missing file and to generate the necessary missing code.
\end{itemize}

We further categorize these synthetically generated samples based on the sequence length $(16 \mathrm{~K}, 32 \mathrm{~K}, 64 \mathrm{~K}$ and 128K) to enable more fine-grained targeting of input lengths.

Through careful ablations, we observe that mixing 0.1\% of synthetically generated long-context data with the original short-context data optimizes the performance across both short-context and long-context benchmarks.

DPO. We observe that using only short context training data in DPO did not negatively impact long-context performance as long as the SFT model is high quality in long context tasks. We suspect this is due to the fact that our DPO recipe has fewer optimizer steps than SFT. Given this finding, we keep the standard short-context recipe for DPO on top of our long-context SFT checkpoints.

## 4.3.5 Tool Use

Teaching LLMs to use tools such as search engines or code interpreters hugely expands the range of tasks they can solve, transforming them from pure chat models into more general assistants (Nakano et al., 2021; Thoppilan et al., 2022; Parisi et al., 2022; Gao et al., 2023; Mialon et al., 2023a; Schick et al., 2024). We train Llama 3 to interact with the following tools:
\begin{itemize}
\item[-] Search engine. Llama 3 is trained to use Brave Search ${ }^{7}$ to answer questions about recent events that go beyond its knowledge cutoff or that require retrieving a particular piece of information from the web.
\item[-] Python interpreter. Llama 3 can generate and execute code to perform complex computations, read files uploaded by the user and solve tasks based on them such as question answering, summarization, data analysis or visualization.
\end{itemize}

\footnotetext{
${ }^{7}$ https://brave.com/search/api/
}
\begin{itemize}
\item[-] Mathematical computational engine. Llama 3 can use the Wolfram Alpha API ${ }^{8}$ to more accurately solve math, science problems, or retrieve accurate information from Wolfram's database.
\end{itemize}

The resulting model is able to use these tools in a chat setup to solve the user's queries, including in multi-turn dialogs. If a query requires multiple tool calls, the model can write a step-by-step plan, call the tools in sequence, and do reasoning after each tool call.

We also improve Llama 3's zero-shot tool use capabilities - given in-context, potentially unseen tool definitions and a user query, we train the model to generate the correct tool call.

Implementation. We implement our core tools as Python objects with different methods. Zero-shot tools can be implemented as Python functions with descriptions, documentation (i.e., examples for how to use them), and the model only needs the function's signature and docstring as context to generate the appropriate call. We also convert function definitions and calls to JSON format, e.g., for web API calls. All tool calls are executed by the Python interpreter, that must be enabled in the Llama 3 system prompt. Core tools can be individually enabled or disabled in the system prompt.

Data collection. Different from Schick et al. (2024), we rely on human annotations and preferences to teach Llama 3 to use tools. There are two main differences with the post-training pipeline generally used in Llama 3:
\begin{itemize}
\item[-] For tools, dialogs often contain more than a single assistant message (e.g., calling the tool and reasoning about the tool output). Thus, we annotate at the message level to collect granular feedback: annotators provide a preference between two assistant messages with the same context or, if both contain major problems, edit one of the messages. The chosen or edited message is then added to the context and the dialog continues. This provides human feedback for both the assistant's ability of calling the tools and reasoning about the tool outputs. Annotators cannot rank or edit the tool outputs.
\item[-] We do not perform rejection sampling, as we did not observe gains in our tool benchmarks.
\end{itemize}

To accelerate the annotation process, we start by bootstrapping basic tool use capabilities by finetuning on synthetically generated data from previous Llama 3 checkpoints. Thus, annotators have fewer edits to perform. In a similar spirit, as Llama 3 gradually improves through its development, we progressively complexify our human annotation protocols: we start by single-turn tool use annotations, before moving to tool use in dialogs, and finally annotating for multi-step tool use and data analysis.

Tool datasets. To create data for tool usage applications, we leverage the following procedure:
\begin{itemize}
\item[-] Single-step tool use: We start by few-shot generation of synthetic user prompts which, by construction, require a call to one of our core tools (for example, questions that exceed our knowledge cutoff date). Then, still relying on few-shot generation, we generate appropriate tool calls for these prompts, execute them, and add the output to the model's context. Finally, we prompt the model again to generate a final answer to the user's query based on the tool output. We end up with trajectories of the following form: system prompt, user prompt, tool call, tool output, final answer. We also filter around 30\% this dataset to remove tool calls that cannot be executed or other formatting issues.
\item[-] Multi-step tool use: We follow a similar protocol and first generate synthetic data to teach the model basic multi-step tool use capabilities. To do this, we first prompt Llama 3 to generate user prompts that require at least two tool calls, that can be the same or different tools from our core set. Then, conditioned on these prompts, we few-shot prompt Llama 3 to generate a solution consisting of interleaved reasoning steps and tool calls, similar to ReAct (Yao et al., 2022). See Figure 10 for an example of Llama 3 performing a task involving multi-step tool usage.
\item[-] File uploads: We annotate for the following filetypes: .txt, .docx, .pdf, .pptx, .xlsx, .csv, .tsv, .PY, .JSON, .JSONL, .HTML, .XML. Our prompts are based on a provided file, and ask to summarize the contents of the file, find and fix bugs, optimize a piece of code, perform data analysis or visualization. See Figure 11 for an example of Llama 3 performing a task involving a file upload.
\end{itemize}

After finetuning on this synthetic data, we gather human annotations in diverse and challenging scenarios including multi-turn interactions, more than three step tool use, and instances where a tool call does not yield

\footnotetext{
${ }^{8}$ https://products.wolframalpha.com/llm-api/documentation
}

![Figure](./grattafioriLlama3Herd2024_images/964281cf-3b75-4775-9ae9-065d3b6d211b-26_980_1635_219_245.jpg)

**Figure 10 Multi-step tool usage. Example of Llama 3 performing multi-step planning, reasoning, and tool calling to solve a task.**

a satisfying answer. We augment our synthetic data with different system prompts to teach the model to use tools only when activated. To train the model to avoid calling tools for simple queries, we also add queries from easy math or question answering datasets (Berant et al., 2013; Koncel-Kedziorski et al., 2016; Joshi et al., 2017; Amini et al., 2019) and their responses without tools, but with tools activated in system prompt.

Zero-shot tool use data. We improve Llama 3 zero-shot tool use abilities (also referred to as function calling) by finetuning on a large and diverse set of partly synthetic (functions definitions, user query, corresponding call) tuples. We evaluate our model on a set of unseen tools.
\begin{itemize}
\item[-] Single, nested, and parallel function calling: Calls can be simple, nested, i.e. we pass a function call as an argument of another function, or parallel, i.e. the model returns a list of independent function calls. Generating a diverse set of functions, queries and ground truths can be challenging (Mekala et al., 2024), and we resort to mining the Stack (Kocetkov et al., 2022) to ground our synthetic user queries in real functions. More precisely, we extract function calls and their definitions, clean and filter them, e.g. for missing docstrings or non-executable functions, and use Llama 3 to generate a natural language query corresponding to the function call.
\item[-] Multi-turn function calling: We also generate synthetic data for multi-turn dialogs with function calls, following a protocol similar to the one proposed in Li et al. (2023b). We use multiple agents that generate domains, APIs, user queries, API calls, and responses, while also ensuring that the generated data covers a set of diverse domains and realistic APIs. All agents are variants of Llama 3 prompted in different ways depending on their roles and collaborate in a step-by-step manner.
\end{itemize}

## 4.3.6 Factuality

Hallucinations remain a major challenge for large language models. Models tend to be overconfident, even in domains where they have little knowledge. Despite these shortcomings, they are often used as knowledge bases, which can lead to risky outcomes such as the spread of misinformation. While we recognize that factuality can go beyond hallucinations, we took a hallucination-first approach here.

![Figure](./grattafioriLlama3Herd2024_images/964281cf-3b75-4775-9ae9-065d3b6d211b-27_1240_1636_217_244.jpg)

**Figure 11 Processing file uploads. Example of Llama 3 performing analysis and visualization of an uploaded file.**

We follow the principle that post-training should align the model to "know what it knows" rather than add knowledge (Gekhman et al., 2024; Mielke et al., 2020). Our primary approach involves generating data that aligns model generations with subsets of factual data present in the pre-training data. To achieve this, we develop a knowledge probing technique that takes advantage of Llama 3's in-context abilities. This data generation process involves the following procedure:
\begin{itemize}
\item[1.] Extract a data snippet from the pre-training data.
\item[2.] Generate a factual question about these snippets (context) by prompting Llama 3.
\item[3.] Sample responses from Llama 3 to the question.
\item[4.] Score the correctness of the generations using the original context as a reference and Llama 3 as a judge.
\item[5.] Score the informativeness of the generations using Llama 3 as a judge.
\item[6.] Generate a refusal for responses which are consistently informative and incorrect across the generations, using Llama 3.
\end{itemize}

We use data generated from the knowledge probe to encourage the model to only answer questions which it has knowledge about, and refuse answering those questions that it is unsure about. Further, pre-training data is not always factually consistent or correct. We therefore also collect a limited set of labeled factuality data that deals with sensitive topics where factually contradictory or incorrect statements are prevalent.

## 4.3.7 Steerability

Steerability is the ability to direct the model's actions and outcomes to meet developer and user specifications. As Llama 3 is a generic foundational model, it should be maximally steerable to different downstream use cases easily. For Llama 3, we focus on enhancing its steerability through system prompt with natural language instructions, especially around response length, format, tone and character/persona.

Data collection. We collect steerability preference samples within the general English category by asking annotators to design different system prompts for Llama 3. Annotators then engage in conversations with the models to evaluate their consistency in following instructions defined in system prompts over the course of the conversation. We show an example customized system prompt used for enhancing steerability below:

> You are a helpful and cheerful AI Chatbot that acts as a meal plan assistant for busy families. The family consists of 2 adults, 3 teenagers, and 2 preschoolers. Plan two or three days at a time and use leftovers or extra ingredients for the second day's plan. The user will let you know if they want two or three days. If they don't, assume three days. Each plan should include breakfast, lunch, snack, and dinner. Ask the user if they approve of the plan or need adjustments. After they approve provide a grocery list with family size in mind. Always keep family preferences in mind and if there's something that they don't like provide a substitution. If the user is not feeling inspired then ask them what's the one place they wish they could visit on vacation this week and then suggest meals based on that location's culture. Weekend meals can be more complex. Weekday meals should be quick and easy. For breakfast and lunch, easy food like cereal, English muffins with pre-cooked bacon, and other quick easy foods are preferred. The family is busy. Be sure to ask if they have essentials and favorites on hand like coffee or energy drinks so they don't forget to buy it. Remember to be budget-conscious unless it's a special occasion.

Modeling. After we collect the preference data, we leverage this data in reward modeling, rejection sampling, SFT, and DPO to enhance Llama 3's steerability.

