# 4. Pre-Training

## 4.1. Data Construction

Compared with DeepSeek-V2, we optimize the pre-training corpus by enhancing the ratio of mathematical and programming samples, while expanding multilingual coverage beyond

English and Chinese. Also, our data processing pipeline is refined to minimize redundancy while maintaining corpus diversity. Inspired by Ding et al. (2024), we implement the document packing method for data integrity but do not incorporate cross-sample attention masking during training. Finally, the training corpus for DeepSeek-V3 consists of 14.8T high-quality and diverse tokens in our tokenizer.

In the training process of DeepSeekCoder-V2 (DeepSeek-AI, 2024a), we observe that the Fill-in-Middle (FIM) strategy does not compromise the next-token prediction capability while enabling the model to accurately predict middle text based on contextual cues. In alignment with DeepSeekCoder-V2, we also incorporate the FIM strategy in the pre-training of DeepSeek-V3. To be specific, we employ the Prefix-Suffix-Middle (PSM) framework to structure data as follows:
$$
\left.\langle | \text { fim_begin } \mid>f_{\text {pre }}\langle | \text { fim_hole }\left\rangle f_{\text {suf }}\langle | \text { fim_end }\right|\right\rangle f_{\text {middle }}\langle | \text { eos_token }\rangle \text {. }
$$
This structure is applied at the document level as a part of the pre-packing process. The FIM strategy is applied at a rate of 0.1 , consistent with the PSM framework.

The tokenizer for DeepSeek-V3 employs Byte-level BPE (Shibata et al., 1999) with an extended vocabulary of 128K tokens. The pretokenizer and training data for our tokenizer are modified to optimize multilingual compression efficiency. In addition, compared with DeepSeek-V2, the new pretokenizer introduces tokens that combine punctuations and line breaks. However, this trick may introduce the token boundary bias (Lundberg, 2023) when the model processes multi-line prompts without terminal line breaks, particularly for few-shot evaluation prompts. To address this issue, we randomly split a certain proportion of such combined tokens during training, which exposes the model to a wider array of special cases and mitigates this bias.

## 4.2. Hyper-Parameters

Model Hyper-Parameters. We set the number of Transformer layers to 61 and the hidden dimension to 7168. All learnable parameters are randomly initialized with a standard deviation of 0.006. In MLA, we set the number of attention heads $n_{h}$ to 128 and the per-head dimension $d_{h}$ to 128. The KV compression dimension $d_{c}$ is set to 512, and the query compression dimension $d_{c}^{\prime}$ is set to 1536. For the decoupled queries and key, we set the per-head dimension $d_{h}^{R}$ to 64. We substitute all FFNs except for the first three layers with MoE layers. Each MoE layer consists of 1 shared expert and 256 routed experts, where the intermediate hidden dimension of each expert is 2048. Among the routed experts, 8 experts will be activated for each token, and each token will be ensured to be sent to at most 4 nodes. The multi-token prediction depth $D$ is set to 1 , i.e., besides the exact next token, each token will predict one additional token. As DeepSeek-V2, DeepSeek-V3 also employs additional RMSNorm layers after the compressed latent vectors, and multiplies additional scaling factors at the width bottlenecks. Under this configuration, DeepSeek-V3 comprises 671B total parameters, of which 37B are activated for each token.

Training Hyper-Parameters. We employ the AdamW optimizer (Loshchilov and Hutter, 2017) with hyper-parameters set to $\beta_{1}=0.9, \beta_{2}=0.95$, and weight_decay $=0.1$. We set the maximum sequence length to 4 K during pre-training, and pre-train DeepSeek-V3 on 14.8T tokens. As for the learning rate scheduling, we first linearly increase it from 0 to $2.2 \times 10^{-4}$ during the first 2 K steps. Then, we keep a constant learning rate of $2.2 \times 10^{-4}$ until the model consumes 10T training tokens. Subsequently, we gradually decay the learning rate to $2.2 \times 10^{-5}$ in 4.3T tokens, following a cosine decay curve. During the training of the final 500B tokens, we keep a constant learning rate of $2.2 \times 10^{-5}$ in the first 333B tokens, and switch to another constant learning rate
of $7.3 \times 10^{-6}$ in the remaining 167B tokens. The gradient clipping norm is set to 1.0. We employ a batch size scheduling strategy, where the batch size is gradually increased from 3072 to 15360 in the training of the first 469B tokens, and then keeps 15360 in the remaining training. We leverage pipeline parallelism to deploy different layers of a model on different GPUs, and for each layer, the routed experts will be uniformly deployed on 64 GPUs belonging to 8 nodes. As for the node-limited routing, each token will be sent to at most 4 nodes (i.e., $M=4$ ). For auxiliary-loss-free load balancing, we set the bias update speed $\gamma$ to 0.001 for the first 14.3T tokens, and to 0.0 for the remaining 500B tokens. For the balance loss, we set $\alpha$ to 0.0001, just to avoid extreme imbalance within any single sequence. The MTP loss weight $\lambda$ is set to 0.3 for the first 10T tokens, and to 0.1 for the remaining 4.8T tokens.

**Pressure Testing DeepSeek-V3 128K Context via "Needle In A HayStack"**
![Figure](./deepseek-aiDeepSeekV3TechnicalReport2025_images/7bf953a3-e8a0-4bc9-9115-d38b75ce977a-23_752_1535_861_263.jpg)

Figure 8 | Evaluation results on the "Needle In A Haystack" (NIAH) tests. DeepSeek-V3 performs well across all context window lengths up to 128K.

## 4.3. Long Context Extension

We adopt a similar approach to DeepSeek-V2 (DeepSeek-AI, 2024c) to enable long context capabilities in DeepSeek-V3. After the pre-training stage, we apply YaRN (Peng et al., 2023a) for context extension and perform two additional training phases, each comprising 1000 steps, to progressively expand the context window from 4 K to 32 K and then to 128 K . The YaRN configuration is consistent with that used in DeepSeek-V2, being applied exclusively to the decoupled shared key $\mathbf{k}_{t}^{R}$. The hyper-parameters remain identical across both phases, with the scale $s=40, \alpha=1, \beta=32$, and the scaling factor $\sqrt{t}=0.1 \ln s+1$. In the first phase, the sequence length is set to 32K, and the batch size is 1920. During the second phase, the sequence length is increased to 128K, and the batch size is reduced to 480. The learning rate for both phases is set to $7.3 \times 10^{-6}$, matching the final learning rate from the pre-training stage.

Through this two-phase extension training, DeepSeek-V3 is capable of handling inputs up to 128K in length while maintaining strong performance. Figure 8illustrates that DeepSeek-V3, following supervised fine-tuning, achieves notable performance on the "Needle In A Haystack" (NIAH) test, demonstrating consistent robustness across context window lengths up to 128K.

## 4.4. Evaluations

## 4.4.1. Evaluation Benchmarks

The base model of DeepSeek-V3 is pretrained on a multilingual corpus with English and Chinese constituting the majority, so we evaluate its performance on a series of benchmarks primarily in English and Chinese, as well as on a multilingual benchmark. Our evaluation is based on our internal evaluation framework integrated in our HAI-LLM framework. Considered benchmarks are categorized and listed as follows, where underlined benchmarks are in Chinese and double-underlined benchmarks are multilingual ones:

Multi-subject multiple-choice datasets include MMLU (Hendrycks et al., 2020), MMLU-Redux (Gema et al., 2024), MMLU-Pro (Wang et al., 2024b), MMMLU (OpenAI, 2024b), C-Eval (Huang et al., 2023), and CMMLU (Li et al., 2023).

Language understanding and reasoning datasets include HellaSwag (Zellers et al., 2019), PIQA (Bisk et al., 2020), ARC (Clark et al., 2018), and BigBench Hard (BBH) (Suzgun et al., 2022).

Closed-book question answering datasets include TriviaQA (Joshi et al., 2017) and NaturalQuestions (Kwiatkowski et al., 2019).

Reading comprehension datasets include RACE Lai et al. (2017), DROP (Dua et al., 2019), C3 (Sun et al., 2019a), and CMRC (Cui et al., 2019).

Reference disambiguation datasets include CLUEWSC (Xu et al., 2020) and WinoGrande Sakaguchi et al. (2019).

Language modeling datasets include Pile (Gao et al., 2020).
Chinese understanding and culture datasets include CCPM (Li et al., 2021).
Math datasets include GSM8K (Cobbe et al., 2021), MATH (Hendrycks et al., 2021), MGSM (Shi et al., 2023), and CMath (Wei et al., 2023).

Code datasets include HumanEval (Chen et al., 2021), LiveCodeBench-Base (0801-1101) (Jain et al., 2024), MBPP (Austin et al., 2021), and CRUXEval (Gu et al., 2024).

Standardized exams include AGIEval (Zhong et al., 2023). Note that AGIEval includes both English and Chinese subsets.

Following our previous work (DeepSeek-AI, 2024bs), we adopt perplexity-based evaluation for datasets including HellaSwag, PIQA, WinoGrande, RACE-Middle, RACE-High, MMLU, MMLU-Redux, MMLU-Pro, MMMLU, ARC-Easy, ARC-Challenge, C-Eval, CMMLU, C3, and CCPM, and adopt generation-based evaluation for TriviaQA, NaturalQuestions, DROP, MATH, GSM8K, MGSM, HumanEval, MBPP, LiveCodeBench-Base, CRUXEval, BBH, AGIEval, CLUEWSC, CMRC, and CMath. In addition, we perform language-modeling-based evaluation for Pile-test and use Bits-Per-Byte (BPB) as the metric to guarantee fair comparison among models using different tokenizers.

## 4.4.2. Evaluation Results

In Table 3, we compare the base model of DeepSeek-V3 with the state-of-the-art open-source base models, including DeepSeek-V2-Base (DeepSeek-AI, 2024c) (our previous release), Qwen2.5 72B Base (Qwen, 2024b), and LLaMA-3.1 405B Base (AI@Meta, 2024b). We evaluate all these models with our internal evaluation framework, and ensure that they share the same evaluation setting. Note that due to the changes in our evaluation framework over the past months, the performance

\begin{table}
\begin{tabular}[t]{|l|l|l|l|l|l|l|}
\hline & Benchmark (Metric) & \# Shots & DeepSeek-V2 Base & Qwen2.5 72B Base & LLaMA-3.1 405B Base & DeepSeek-V3 Base \\
\hline \multirow{3}{*}{} & Architecture & - & MoE & Dense & Dense & MoE \\
\hline & \# Activated Params & - & 21B & 72B & 405B & 37B \\
\hline & \# Total Params & - & 236B & 72B & 405B & 671B \\
\hline \multirow{16}{*}{\begin{tabular}[t]{l}
English \\
English
\end{tabular}} & Pile-test (BPB) & - & 0.606 & 0.638 & 0.542 & 0.548 \\
\hline & BBH (em) & 3-shot & 78.8 & 79.8 & 82.9 & 87.5 \\
\hline & MMLU (em) & 5-shot & 78.4 & 85.0 & 84.4 & 87.1 \\
\hline & MMLU-Redux (em) & 5-shot & 75.6 & 83.2 & 81.3 & 86.2 \\
\hline & MMLU-Pro (em) & 5-shot & 51.4 & 58.3 & 52.8 & 64.4 \\
\hline & DROP (F1) & 3-shot & 80.4 & 80.6 & 86.0 & 89.0 \\
\hline & ARC-Easy (em) & 25-shot & 97.6 & 98.4 & 98.4 & 98.9 \\
\hline & ARC-Challenge (em) & 25-shot & 92.2 & 94.5 & 95.3 & 95.3 \\
\hline & HellaSwag (em) & 10-shot & 87.1 & 84.8 & 89.2 & 88.9 \\
\hline & PIQA (em) & 0-shot & 83.9 & 82.6 & 85.9 & 84.7 \\
\hline & WinoGrande (em) & 5-shot & 86.3 & 82.3 & 85.2 & 84.9 \\
\hline & RACE-Middle (em) & 5-shot & 73.1 & 68.1 & 74.2 & 67.1 \\
\hline & RACE-High (em) & 5-shot & 52.6 & 50.3 & 56.8 & 51.3 \\
\hline & TriviaQA (em) & 5-shot & 80.0 & 71.9 & 82.7 & 82.9 \\
\hline & NaturalQuestions (em) & 5-shot & 38.6 & 33.2 & 41.5 & 40.0 \\
\hline & AGIEval (em) & 0-shot & 57.5 & 75.8 & 60.6 & 79.6 \\
\hline \multirow{5}{*}{Code} & HumanEval (Pass@1) & 0-shot & 43.3 & 53.0 & 54.9 & 65.2 \\
\hline & MBPP (Pass@1) & 3-shot & 65.0 & 72.6 & 68.4 & 75.4 \\
\hline & LiveCodeBench-Base (Pass@1) & 3-shot & 11.6 & 12.9 & 15.5 & 19.4 \\
\hline & CRUXEval-I (em) & 2-shot & 52.5 & 59.1 & 58.5 & 67.3 \\
\hline & CRUXEval-O (em) & 2-shot & 49.8 & 59.9 & 59.9 & 69.8 \\
\hline \multirow{4}{*}{Math} & GSM8K (em) & 8-shot & 81.6 & 88.3 & 83.5 & 89.3 \\
\hline & MATH (em) & 4-shot & 43.4 & 54.4 & 49.0 & 61.6 \\
\hline & MGSM (em) & 8-shot & 63.6 & 76.2 & 69.9 & 79.8 \\
\hline & CMath (em) & 3-shot & 78.7 & 84.5 & 77.3 & 90.7 \\
\hline \multirow{6}{*}{Chinese} & CLUEWSC (em) & 5-shot & 82.0 & 82.5 & 83.0 & 82.7 \\
\hline & C-Eval (em) & 5-shot & 81.4 & 89.2 & 72.5 & 90.1 \\
\hline & CMMLU (em) & 5-shot & 84.0 & 89.5 & 73.7 & 88.8 \\
\hline & CMRC (em) & 1-shot & 77.4 & 75.8 & 76.0 & 76.3 \\
\hline & C3 (em) & 0-shot & 77.4 & 76.7 & 79.7 & 78.6 \\
\hline & CCPM (em) & 0-shot & 93.0 & 88.5 & 78.6 & 92.0 \\
\hline Multilingual & MMMLU-non-English (em) & 5-shot & 64.0 & 74.8 & 73.8 & 79.4 \\
\hline
\end{tabular}

**Table 3 | Comparison among DeepSeek-V3-Base and other representative open-source base models. All models are evaluated in our internal framework and share the same evaluation setting. Scores with a gap not exceeding 0.3 are considered to be at the same level. DeepSeek-V3-Base achieves the best performance on most benchmarks, especially on math and code tasks.**
\end{table}
of DeepSeek-V2-Base exhibits a slight difference from our previously reported results. Overall, DeepSeek-V3-Base comprehensively outperforms DeepSeek-V2-Base and Qwen2.5 72B Base, and surpasses LLaMA-3.1 405B Base in the majority of benchmarks, essentially becoming the strongest open-source model.

From a more detailed perspective, we compare DeepSeek-V3-Base with the other open-source base models individually. (1) Compared with DeepSeek-V2-Base, due to the improvements in our model architecture, the scale-up of the model size and training tokens, and the enhancement of data quality, DeepSeek-V3-Base achieves significantly better performance as expected. (2) Compared with Qwen2.5 72B Base, the state-of-the-art Chinese open-source model, with only half of the activated parameters, DeepSeek-V3-Base also demonstrates remarkable advantages,
especially on English, multilingual, code, and math benchmarks. As for Chinese benchmarks, except for CMMLU, a Chinese multi-subject multiple-choice task, DeepSeek-V3-Base also shows better performance than Qwen2.5 72B. (3) Compared with LLaMA-3.1 405B Base, the largest open-source model with 11 times the activated parameters, DeepSeek-V3-Base also exhibits much better performance on multilingual, code, and math benchmarks. As for English and Chinese language benchmarks, DeepSeek-V3-Base shows competitive or better performance, and is especially good on BBH, MMLU-series, DROP, C-Eval, CMMLU, and CCPM.

Due to our efficient architectures and comprehensive engineering optimizations, DeepSeekV3 achieves extremely high training efficiency. Under our training framework and infrastructures, training DeepSeek-V3 on each trillion tokens requires only 180K H800 GPU hours, which is much cheaper than training 72B or 405B dense models.

\begin{table}
\begin{tabular}[t]{|l|l|l|l|l|l|}
\hline Benchmark (Metric) & \# Shots & Small MoE Baseline & Small MoE w/ MTP & Large MoE Baseline & Large MoE w/ MTP \\
\hline \# Activated Params (Inference) & - & 2.4B & 2.4B & 20.9B & 20.9B \\
\hline \# Total Params (Inference) & - & 15.7B & 15.7B & 228.7B & 228.7B \\
\hline \# Training Tokens & - & 1.33T & 1.33T & 540B & 540B \\
\hline Pile-test (BPB) & - & 0.729 & 0.729 & 0.658 & 0.657 \\
\hline BBH (em) & 3-shot & 39.0 & 41.4 & 70.0 & 70.7 \\
\hline MMLU (em) & 5-shot & 50.0 & 53.3 & 67.5 & 66.6 \\
\hline DROP (F1) & 1-shot & 39.2 & 41.3 & 68.5 & 70.6 \\
\hline TriviaQA (em) & 5-shot & 56.9 & 57.7 & 67.0 & 67.3 \\
\hline NaturalQuestions (em) & 5-shot & 22.7 & 22.3 & 27.2 & 28.5 \\
\hline HumanEval (Pass@1) & 0-shot & 20.7 & 26.8 & 44.5 & 53.7 \\
\hline MBPP (Pass@1) & 3-shot & 35.8 & 36.8 & 61.6 & 62.2 \\
\hline GSM8K (em) & 8-shot & 25.4 & 31.4 & 72.3 & 74.0 \\
\hline MATH (em) & 4-shot & 10.7 & 12.6 & 38.6 & 39.8 \\
\hline
\end{tabular}

**Table 4 | Ablation results for the MTP strategy. The MTP strategy consistently enhances the model performance on most of the evaluation benchmarks.**
\end{table}

## 4.5. Discussion

## 4.5.1. Ablation Studies for Multi-Token Prediction

In Table 4. we show the ablation results for the MTP strategy. To be specific, we validate the MTP strategy on top of two baseline models across different scales. At the small scale, we train a baseline MoE model comprising 15.7B total parameters on 1.33T tokens. At the large scale, we train a baseline MoE model comprising 228.7B total parameters on 540B tokens. On top of them, keeping the training data and the other architectures the same, we append a 1-depth MTP module onto them and train two models with the MTP strategy for comparison. Note that during inference, we directly discard the MTP module, so the inference costs of the compared models are exactly the same. From the table, we can observe that the MTP strategy consistently enhances the model performance on most of the evaluation benchmarks.

## 4.5.2. Ablation Studies for the Auxiliary-Loss-Free Balancing Strategy

In Table 5, we show the ablation results for the auxiliary-loss-free balancing strategy. We validate this strategy on top of two baseline models across different scales. At the small scale, we train a baseline MoE model comprising 15.7B total parameters on 1.33T tokens. At the large scale, we train a baseline MoE model comprising 228.7B total parameters on 578B tokens.

\begin{table}
\begin{tabular}[t]{|l|l|l|l|l|l|}
\hline Benchmark (Metric) & \# Shots & Small MoE Aux-Loss-Based & Small MoE Aux-Loss-Free & Large MoE Aux-Loss-Based & Large MoE Aux-Loss-Free \\
\hline \# Activated Params & - & 2.4B & 2.4B & 20.9B & 20.9B \\
\hline \# Total Params & - & 15.7B & 15.7B & 228.7B & 228.7B \\
\hline \# Training Tokens & - & 1.33T & 1.33T & 578B & 578B \\
\hline Pile-test (BPB) & - & 0.727 & 0.724 & 0.656 & 0.652 \\
\hline BBH (em) & 3-shot & 37.3 & 39.3 & 66.7 & 67.9 \\
\hline MMLU (em) & 5-shot & 51.0 & 51.8 & 68.3 & 67.2 \\
\hline DROP (F1) & 1-shot & 38.1 & 39.0 & 67.1 & 67.1 \\
\hline TriviaQA (em) & 5-shot & 58.3 & 58.5 & 66.7 & 67.7 \\
\hline NaturalQuestions (em) & 5-shot & 23.2 & 23.4 & 27.1 & 28.1 \\
\hline HumanEval (Pass@1) & 0-shot & 22.0 & 22.6 & 40.2 & 46.3 \\
\hline MBPP (Pass@1) & 3-shot & 36.6 & 35.8 & 59.2 & 61.2 \\
\hline GSM8K (em) & 8-shot & 27.1 & 29.6 & 70.7 & 74.5 \\
\hline MATH (em) & 4-shot & 10.9 & 11.1 & 37.2 & 39.6 \\
\hline
\end{tabular}

**Table 5 | Ablation results for the auxiliary-loss-free balancing strategy. Compared with the purely auxiliary-loss-based method, the auxiliary-loss-free strategy consistently achieves better model performance on most of the evaluation benchmarks.**
\end{table}

Both of the baseline models purely use auxiliary losses to encourage load balance, and use the sigmoid gating function with top-K affinity normalization. Their hyper-parameters to control the strength of auxiliary losses are the same as DeepSeek-V2-Lite and DeepSeek-V2, respectively. On top of these two baseline models, keeping the training data and the other architectures the same, we remove all auxiliary losses and introduce the auxiliary-loss-free balancing strategy for comparison. From the table, we can observe that the auxiliary-loss-free strategy consistently achieves better model performance on most of the evaluation benchmarks.

## 4.5.3. Batch-Wise Load Balance VS. Sequence-Wise Load Balance

The key distinction between auxiliary-loss-free balancing and sequence-wise auxiliary loss lies in their balancing scope: batch-wise versus sequence-wise. Compared with the sequence-wise auxiliary loss, batch-wise balancing imposes a more flexible constraint, as it does not enforce in-domain balance on each sequence. This flexibility allows experts to better specialize in different domains. To validate this, we record and analyze the expert load of a 16B auxiliaryloss-based baseline and a 16B auxiliary-loss-free model on different domains in the Pile test set. As illustrated in Figure 9, we observe that the auxiliary-loss-free model demonstrates greater expert specialization patterns as expected.

To further investigate the correlation between this flexibility and the advantage in model performance, we additionally design and validate a batch-wise auxiliary loss that encourages load balance on each training batch instead of on each sequence. The experimental results show that, when achieving a similar level of batch-wise load balance, the batch-wise auxiliary loss can also achieve similar model performance to the auxiliary-loss-free method. To be specific, in our experiments with 1B MoE models, the validation losses are: 2.258 (using a sequencewise auxiliary loss), 2.253 (using the auxiliary-loss-free method), and 2.253 (using a batch-wise auxiliary loss). We also observe similar results on 3B MoE models: the model using a sequencewise auxiliary loss achieves a validation loss of 2.085, and the models using the auxiliary-loss-free method or a batch-wise auxiliary loss achieve the same validation loss of 2.080.

In addition, although the batch-wise load balancing methods show consistent performance advantages, they also face two potential challenges in efficiency: (1) load imbalance within

![Figure](./deepseek-aiDeepSeekV3TechnicalReport2025_images/7bf953a3-e8a0-4bc9-9115-d38b75ce977a-28_760_1554_293_255.jpg)

**Figure 9 | Expert load of auxiliary-loss-free and auxiliary-loss-based models on three domains in the Pile test set. The auxiliary-loss-free model shows greater expert specialization patterns than the auxiliary-loss-based one. The relative expert load denotes the ratio between the actual expert load and the theoretically balanced expert load. Due to space constraints, we only present the results of two layers as an example, with the results of all layers provided in Appendix C.**

certain sequences or small batches, and (2) domain-shift-induced load imbalance during inference. The first challenge is naturally addressed by our training framework that uses large-scale expert parallelism and data parallelism, which guarantees a large size of each micro-batch. For the second challenge, we also design and implement an efficient inference framework with redundant expert deployment, as described in Section 3.4, to overcome it.

