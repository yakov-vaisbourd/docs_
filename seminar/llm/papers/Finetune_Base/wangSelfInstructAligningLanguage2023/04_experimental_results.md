# 4 Experimental Results

We conduct experiments to measure and compare the performance of models under various instruction tuning setups. We first describe our models and other baselines, followed by our experiments.

## 4.1 GPT3 ${ _{\text {Self-Inst }}$ : finetuning GPT3 on its own instruction data}

Given the instruction-generated instruction data, we conduct instruction tuning with the GPT3 model itself ("davinci" engine). As described in §2.3, we use various templates to concatenate the instruction

![Figure](./wangSelfInstructAligningLanguage2023_images/c0364e9b-aeb3-41c0-b14a-38c8a8857c92-05_206_630_210_1179.jpg)

**Figure 4: Distribution of the ROUGE-L scores between generated instructions and their most similar seed instructions.**

![Figure](./wangSelfInstructAligningLanguage2023_images/c0364e9b-aeb3-41c0-b14a-38c8a8857c92-05_660_637_587_1174.jpg)

**Figure 5: Length distribution of the generated instructions, non-empty inputs, and outputs.**

and input, and train the model to generate the output. This finetuning is done through the OpenAI finetuning API. ${ }^{8}$ We use the default hyper-parameters, except that we set the prompt loss weight to 0, and we train the model for 2 epochs. We refer the reader to Appendix A. 3 for additional finetuning details. The resulting model is denoted by GPT3 ${ }_{\text {SELF-Inst }}$.

## 4.2 Baselines

Off-the-shelf LMs. We evaluate T5-LM (Lester et al., 2021; Raffel et al., 2020) and GPT3 (Brown et al., 2020) as the vanilla LM baselines (only pretraining, no additional finetuning). These baselines will indicate the extent to which off-the-shelf LMs are capable of following instructions naturally immediately after pretraining.

Publicly available instruction-tuned models. T0 and T $k$-Instruct are two instruction-tuned models proposed in Sanh et al. (2022) and Wang et al. (2022), respectively, and are demonstrated to be able to follow instructions for many NLP tasks. Both of these models are finetuned from the T5 (Raffel et al., 2020) checkpoints and are publicly available. ${ }^{9}$ For both of these models, we use

\footnotetext{
${ }^{8}$ See OpenAI's documentation on finetuning.
${ }^{9} \mathrm{~T} 0$ is available at here and T $k$-Instruct is here.
}
their largest version with 11B parameters.
Instruction-tuned GPT3 models. We evaluate InstructGPT (Ouyang et al., 2022), which is developed by OpenAI based on GPT3 to follow human instructions better and has been found by the community to have impressive zero-shot abilities. There are various generations of these models, where newer ones use more expansive data or algorithmic novelties. ${ }^{10}$ For our SuperNI experiments in §4.3, we only compare with their text-davinci-001 engine, because their newer engines are trained with the latest user data and are likely to have already seen the SuperNI test set. For our human evaluation on newly written instructions, we include their 001, 002 and 003 engines for completeness.

Additionally, to compare Self-Instruct training with other publicly available instruction tuning data, we further finetune GPT3 model with data from PromptSource and SuperNI, which are used to train the T0 and T $k$-Instruct models. We call them T0 training and SuperNI training for short, respectively. To save the training budget, we sampled 50K instances (but covering all their instructions) for each dataset, which has a comparable size to the instruction data we generated. Based on the findings from Wang et al. (2022) and our early experiments, reducing the number of instances per training task does not degrade the model's generalization performance to unseen tasks.

## 4.3 Experiment 1: Zero-Shot Generalization on SuperNI benchmark

We first evaluate the models' ability to follow instructions on typical NLP tasks in a zero-shot fashion. We use the evaluation set of SuperNI (Wang et al., 2022), which consists of 119 tasks with 100 instances in each task. In this work, we mainly focus on the zero-shot setup, i.e., the model is prompted with the definition of the tasks only, without incontext demonstration examples. For all our requests to the GPT3 variants, we use the deterministic generation mode (temperature as 0 and no nucleus sampling) without specific stop sequences.

Results. We make the following observations from the results in Table 3. Self-Instruct boosts the instruction-following ability of GPT3 by a large margin. The vanilla GPT3 model basically cannot follow human instructions at all. Upon manual analysis, we find that it usually generates irrele-

\footnotetext{
${ }^{10}$ See OpenAI's documentation on their models.
}

\begin{table}
\begin{tabular}[t]{|l|l|l|l|}
\hline & Model & \# Params & ROUGE-L \\
\hline \multicolumn{4}{|c|}{Vanilla LMs} \\
\hline & T5-LM & 11B & 25.7 \\
\hline & GPT3 & 175B & 6.8 \\
\hline \multicolumn{4}{|c|}{} \\
\hline (1) & T0 & 11B & 33.1 \\
\hline & GPT3 + T0 Training & 175B & 37.9 \\
\hline ② & GPT3 ${ }_{\text {Self-Inst }}$ (Ours) & 175B & 39.9 \\
\hline & InstructGPT ${ }_{001}$ & 175B & 40.8 \\
\hline \multicolumn{4}{|c|}{Instruction-tuned w/ SuperNI} \\
\hline & T $k$-Instruct & 11B & 46.0 \\
\hline ③ & → GPT3 + SuperNI Training & 175B & 49.5 \\
\hline & & & \\
\hline
\end{tabular}

**Table 3: Evaluation results on unseen tasks from SuperNI (§4.3). From the results, we see that 1 Self-Instruct can boost GPT3 performance by a large margin (+33.1\%) and (2) nearly matches the performance of InstructGPT ${ **_{001}$. Additionally, ③ it can further improve the performance even when a large amount of labeled instruction data is present.}
\end{table}
vant and repetitive text, and does not know when to stop generation. Compared with other models that are not specifically trained for SuperNI, $\mathrm{GPT3}_{\text {SELF-INST }}$ achieves better performance than T0 or the GPT3 finetuned on the T0 training set, which takes tremendous human labeling efforts. Notably, $\mathrm{GPT}_{\text {Self-Inst }}$ also nearly matches the performance of InstructGPT ${ }_{001}$, which is trained with private user data and human-annotated labels.

Models trained on the SuperNI training set still achieve better performance on its evaluation set, which we attribute to the similar instruction style and formatting. However, we show that Self-Instruct still brings in additional gains when combined with the SuperNI training set, proving its value as complementary data.

## 4.4 Experiment 2: Generalization to User-oriented Instructions on Novel Tasks

Despite the comprehensiveness of SuperNI in collecting existing NLP tasks, most of these NLP tasks were proposed for research purposes and skewed toward classification. To better access the practical value of instruction-following models, a subset of the authors curate a new set of instructions motivated by user-oriented applications. We first brainstorm various domains where large LMs may be useful (e.g., email writing, social media, productivity tools, entertainment, programming), then craft instructions related to each domain along with an input-output instance (again, input is optional). We aim to diversify the styles and formats of these tasks (e.g., instructions may be long or short; in-

![Figure](./wangSelfInstructAligningLanguage2023_images/c0364e9b-aeb3-41c0-b14a-38c8a8857c92-07_698_1508_202_281.jpg)

**Figure 6: Performance of GPT3 model and its instruction-tuned variants, evaluated by human experts on our 252 user-oriented instructions (§4.4). Human evaluators are instructed to rate the models' responses into four levels. The results indicate that GPT3 ${ **_{\text {Self-Inst }}$ outperforms all the other GPT3 variants trained on publicly available instruction datasets. Additionally, GPT3 ${ }_{\text {SELF-Inst }}$ scores nearly as good as InstructGPT ${ }_{001}$ (cf. footnote 1).}

put/output may take the form of bullet points, tables, codes, equations, etc.). In total, we create 252 instructions with 1 instance per instruction. We believe it can serve as a testbed for evaluating how instruction-based models handle diverse and unfamiliar instructions. Table 9 presents a small portion of them. The entire set is available in our GitHub repository. We analyze the overlap between this set set and the seed instructions in §A.1.

Human evaluation setup. Evaluating models' performance on this evaluation set of diverse tasks is extremely challenging because different tasks require different expertise. Indeed, many of these tasks cannot be measured by automatic metrics or even be judged by normal crowdworkers (e.g., writing a program, or converting first-order logic into natural language). To get a more faithful evaluation, we asked the authors of the instructions to judge model predictions. Details on how we set up this human evaluation are described in Appendix B. The evaluators were asked to rate the output based on whether it accurately and effectively completes the task. We implemented a four-level rating system for categorizing the quality of the models' outputs:
\begin{itemize}
\item[-] RATING-A: The response is valid and satisfying.
\item[-] Rating-B: The response is acceptable but has minor errors or imperfections.
\item[-] Rating-C: The response is relevant and responds to the instruction, but it has significant errors in the content. For example, GPT3 might generate a valid output first, but continue to gen
\end{itemize}erate other irrelevant things.
\begin{itemize}
\item[-] RATING-D: The response is irrelevant or completely invalid.
\end{itemize}

Results. Figure 6 shows the performance of GPT3 model and its instruction-tuned counterparts on this newly written instruction set (w. inter-rater agreement $\kappa=0.57$ on the 4-class categorical scale, see Appendix B for details). As anticipated, the vanilla GPT3 LM is largely unable to respond to instructions, and all instruction-tuned models demonstrate comparatively higher performance. Nonetheless, $\mathrm{GPT3}_{\text {Self-Inst }}$ (i.e., GPT3 model finetuned with Self-Instruct) outperforms those counterparts trained on T0 or SuperNI data by a large margin, demonstrating the value of the generated data despite the noise. Compared with InstructGPT ${ }_{001}$, $\mathrm{GPT3}_{\text {SELF-InST }}$ is quite close in performance-if we count acceptable response with minor imperfections (Rating-B) as valid, GPT3 ${ }_{\text {SELF-INST }}$ is only 5\% behind InstructGPT ${ }_{001}$. Lastly, our evaluation confirms the impressive instruction-following ability of InstructGPT ${ }_{002}$ and InstructGPT ${ }_{003}$. Although there are many factors behind this success, we conjecture that future work can largely benefit from improving the quality of our generated data by using human annotators or training a reward model to select better generations, similar to the algorithm used by Ouyang et al. (2022).

## 4.5 Effect of Data Size and Quality

Data size. Self-Instruct provides a way to grow instruction data at a low cost with almost
no human labeling; could more of this generated data lead to better instruction-following ability? We conduct an analysis of the size of generated data by subsampling different numbers of instructions from the generated dataset, finetuning GPT3 on the sampled subsets, and evaluating how the resulting models perform on the 252 user-oriented instruction set. We conduct the same human evaluation as in §4.4. Figure 7 presents the performance of GPT3 ${ }_{\text {Self-Inst }}$ models finetuned with different sizes of generated data. Overall, we see consistent improvement as we grow the data size. However, this improvement almost plateaus after 16K. This is inline with the data scaling experiments in Wang et al. (2022, Fig. 5). Interestingly, when evaluating on SuperNI we found the model's performance gain plateaus earlier at around hundreds of instructions. This may be due to the fact that the new generated data is distinct from typical NLP tasks in SuperNI, indicating that future research may benefit from using a combination of different instruction data for better performance on various types of tasks.

Data quality. Another direction to improve the model's performance is to take our generated data and get better supervision (with less noise). We explore this idea by using InstructGPT ${ }_{003}$ (the best available general-purpose model) to regenerate the output field of all our instances given the instruction and input. We then use this improved version of our data to finetune GPT3. This can be regarded as a distillation of InstructGPT ${ }_{003}$ with our data. As is shown in Figure 7, the resulting model outperforms the counterpart trained with the original data by 10\%, which suggests big room for future work on using our generation pipeline to get initial data and then improving the data quality with human experts or distillation from better models.

