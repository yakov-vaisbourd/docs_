# 1 Introduction

Foundation models are general models of language, vision, speech, and/or other modalities that are designed to support a large variety of AI tasks. They form the basis of many modern AI systems.

The development of modern foundation models consists of two main stages: (1) a pre-training stage in which the model is trained at massive scale using straightforward tasks such as next-word prediction or captioning and (2) a post-training stage in which the model is tuned to follow instructions, align with human preferences, and improve specific capabilities (for example, coding and reasoning).

In this paper, we present a new set of foundation models for language, called Llama 3. The Llama 3 Herd of models natively supports multilinguality, coding, reasoning, and tool usage. Our largest model is dense Transformer with 405B parameters, processing information in a context window of up to 128K tokens. Each member of the herd is listed in Table 1. All the results presented in this paper are for the Llama 3.1 models, which we will refer to as Llama 3 throughout for brevity.

We believe there are three key levers in the development of high-quality foundation models: data, scale, and managing complexity. We seek to optimize for these three levers in our development process:
\begin{itemize}
\item[-] Data. Compared to prior versions of Llama (Touvron et al., 2023a,b), we improved both the quantity and quality of the data we use for pre-training and post-training. These improvements include the development of more careful pre-processing and curation pipelines for pre-training data and the development of more rigorous quality assurance and filtering approaches for post-training data. We pre-train Llama 3 on a corpus of about 15T multilingual tokens, compared to 1.8T tokens for Llama 2.
\item[-] Scale. We train a model at far larger scale than previous Llama models: our flagship language model was pre-trained using $3.8 \times 10^{25}$ FLOPs, almost $50 \times$ more than the largest version of Llama 2. Specifically, we pre-trained a flagship model with 405B trainable parameters on 15.6T text tokens. As expected per
\end{itemize}

\begin{table}
\begin{tabular}[t]{|l|l|l|l|l|l|}
\hline & Finetuned & Multilingual & Long context & Tool use & Release \\
\hline Llama 3 8B & × & $\chi^{1}$ & × & × & April 2024 \\
\hline Llama 3 8B Instruct & ✓ & × & × & × & April 2024 \\
\hline Llama 3 70B & × & $x^{1}$ & × & × & April 2024 \\
\hline Llama 3 70B Instruct & ✓ & × & × & × & April 2024 \\
\hline Llama 3.1 8B & × & ✓ & ✓ & × & July 2024 \\
\hline Llama 3.1 8B Instruct & ✓ & ✓ & ✓ & ✓ & July 2024 \\
\hline Llama 3.1 70B & × & ✓ & ✓ & × & July 2024 \\
\hline Llama 3.1 70B Instruct & ✓ & ✓ & ✓ & ✓ & July 2024 \\
\hline Llama 3.1 405B & × & ✓ & ✓ & × & July 2024 \\
\hline Llama 3.1 405B Instruct & ✓ & ✓ & ✓ & ✓ & July 2024 \\
\hline
\end{tabular}

**Table 1 Overview of the Llama 3 Herd of models. All results in this paper are for the Llama 3.1 models.**
\end{table}
scaling laws for foundation models, our flagship model outperforms smaller models trained using the same procedure. While our scaling laws suggest our flagship model is an approximately compute-optimal size for our training budget, we also train our smaller models for much longer than is compute-optimal. The resulting models perform better than compute-optimal models at the same inference budget. We use the flagship model to further improve the quality of those smaller models during post-training.
\begin{itemize}
\item[-] Managing complexity. We make design choices that seek to maximize our ability to scale the model development process. For example, we opt for a standard dense Transformer model architecture (Vaswani et al., 2017) with minor adaptations, rather than for a mixture-of-experts model (Shazeer et al., 2017) to maximize training stability. Similarly, we adopt a relatively simple post-training procedure based on supervised finetuning (SFT), rejection sampling (RS), and direct preference optimization (DPO; Rafailov et al. (2023)) as opposed to more complex reinforcement learning algorithms (Ouyang et al., 2022; Schulman et al., 2017) that tend to be less stable and harder to scale.
\end{itemize}

The result of our work is Llama 3: a herd of three multilingual ${ }^{1}$ language models with 8B, 70B, and 405B parameters. We evaluate the performance of Llama 3 on a plethora of benchmark datasets that span a wide range of language understanding tasks. In addition, we perform extensive human evaluations that compare Llama 3 with competing models. An overview of the performance of the flagship Llama 3 model on key benchmarks is presented in Table 2. Our experimental evaluation suggests that our flagship model performs on par with leading language models such as GPT-4 (OpenAI, 2023a) across a variety of tasks, and is close to matching the state-of-the-art. Our smaller models are best-in-class, outperforming alternative models with similar numbers of parameters (Bai et al., 2023; Jiang et al., 2023). Llama 3 also delivers a much better balance between helpfulness and harmlessness than its predecessor (Touvron et al., 2023b). We present a detailed analysis of the safety of Llama 3 in Section 5.4.

We are publicly releasing all three Llama 3 models under an updated version of the Llama 3 Community License; see https://llama.meta.com. This includes pre-trained and post-trained versions of our 405B parameter language model and a new version of our Llama Guard model (Inan et al., 2023) for input and output safety. We hope that the open release of a flagship model will spur a wave of innovation in the research community, and accelerate a responsible path towards the development of artificial general intelligence (AGI).

As part of the Llama 3 development process we also develop multimodal extensions to the models, enabling image recognition, video recognition, and speech understanding capabilities. These models are still under active development and not yet ready for release. In addition to our language modeling results, the paper presents results of our initial experiments with those multimodal models.

\footnotetext{
${ }^{1}$ The Llama 3 8B and 70B were pre-trained on multilingual data but were intended for use in English at the time.
}

\begin{table}
\begin{tabular}[t]{|l|l|l|l|l|l|l|l|l|l|l|l|l|}
\hline Category & Benchmark & Llama 3 8B & Gemma 2 9B & Mistral 7B & Llama 3 70B & Mixtral 8x22B & GPT 3.5 Turbo & Llama 3 405B & Nemotron 4 340B & GPT-4 (0125) & GPT-4o & Claude 3.5 Sonnet \\
\hline \multirow{4}{*}{General} & MMLU (5-shot) & 69.4 & 72.3 & 61.1 & 83.6 & 76.9 & 70.7 & 87.3 & 82.6 & 85.1 & 89.1 & 89.9 \\
\hline & MMLU (o-shot, cot) & 73.0 & $72.3^{\triangle}$ & 60.5 & 86.0 & 79.9 & 69.8 & 88.6 & $78.7{ }^{\triangleleft}$ & 85.4 & 88.7 & 88.3 \\
\hline & MMLU-Pro (5-shot, cot) & 48.3 & - & 36.9 & 66.4 & 56.3 & 49.2 & 73.3 & 62.7 & 64.8 & 74.0 & 77.0 \\
\hline & IFEval & 80.4 & 73.6 & 57.6 & 87.5 & 72.7 & 69.9 & 88.6 & 85.1 & 84.3 & 85.6 & 88.0 \\
\hline \multirow{2}{*}{Code} & HumanEval (0-shot) & 72.6 & 54.3 & 40.2 & 80.5 & 75.6 & 68.0 & 89.0 & 73.2 & 86.6 & 90.2 & 92.0 \\
\hline & MBPP EvalPlus (0-shot) & 72.8 & 71.7 & 49.5 & 86.0 & 78.6 & 82.0 & 88.6 & 72.8 & 83.6 & 87.8 & 90.5 \\
\hline \multirow{2}{*}{Math} & GSM8K (8-shot, cot) & 84.5 & 76.7 & 53.2 & 95.1 & 88.2 & 81.6 & 96.8 & $92.3^{\text {✓ }}$ & 94.2 & 96.1 & $96.4^{\text {◇ }}$ \\
\hline & MATH (o-shot, cot) & 51.9 & 44.3 & 13.0 & 68.0 & 54.1 & 43.1 & 73.8 & 41.1 & 64.5 & 76.6 & 71.1 \\
\hline \multirow{2}{*}{Reasoning} & ARC Challenge (0-shot) & 83.4 & 87.6 & 74.2 & 94.8 & 88.7 & 83.7 & 96.9 & 94.6 & 96.4 & 96.7 & 96.7 \\
\hline & GPQA (o-shot, сот) & 32.8 & - & 28.8 & 46.7 & 33.3 & 30.8 & 51.1 & - & 41.4 & 53.6 & 59.4 \\
\hline \multirow{2}{*}{Tool use} & BFCL & 76.1 & - & 60.4 & 84.8 & - & 85.9 & 88.5 & 86.5 & 88.3 & 80.5 & 90.2 \\
\hline & Nexus & 38.5 & 30.0 & 24.7 & 56.7 & 48.5 & 37.2 & 58.7 & - & 50.3 & 56.1 & 45.7 \\
\hline \multirow{3}{*}{Long context} & ZeroSCROLLS/QuALITY & 81.0 & - & - & 90.5 & - & - & 95.2 & - & 95.2 & 90.5 & 90.5 \\
\hline & InfiniteBench/En.MC & 65.1 & - & - & 78.2 & - & - & 83.4 & - & 72.1 & 82.5 & - \\
\hline & NIH/Multi-needle & 98.8 & - & - & 97.5 & - & - & 98.1 & - & 100.0 & 100.0 & 90.8 \\
\hline Multilingual & MGSM (0-shot, сот) & 68.9 & 53.2 & 29.9 & 86.9 & 71.1 & 51.4 & 91.6 & - & 85.9 & 90.5 & 91.6 \\
\hline
\end{tabular}

**Table 2 Performance of finetuned Llama 3 models on key benchmark evaluations. The table compares the performance of the 8B, 70B, and 405B versions of Llama 3 with that of competing models. We boldface the best-performing model in each of three model-size equivalence classes. ${ **^{\triangle}$ Results obtained using 5-shot prompting (no CoT). ${ }^{4}$ Results obtained without CoT. ${ }^{\diamond}$ Results obtained using zero-shot prompting.}
\end{table}

