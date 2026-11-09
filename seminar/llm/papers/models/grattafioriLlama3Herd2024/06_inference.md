# 6 Inference

We investigate two main techniques to make inference with the Llama 3 405B model efficient: (1) pipeline parallelism and (2) FP8 quantization. We have publicly released our implementation of FP8 quantization.

## 6.1 Pipeline Parallelism

When using a BF16 number representation for the model parameters, Llama 3 405B does not fit in the GPU memory of a single machine with 8 Nvidia H100 GPUs. To address this issue, we parallelize model inference using BF16 precision across 16 GPUs on two machines. Within each machine, the high NVLink bandwidth

\begin{table}
\begin{tabular}[t]{|l|l|l|l|l|l|}
\hline Metric & Jailbreaks & Injections & Out-of-Distribution Jailbreaks & Multilingual Jailbreaks & Indirect Injections \\
\hline TPR & 99.9\% & 99.5\% & 97.5\% & 91.5\% & 71.4\% \\
\hline FPR & 0.4\% & 0.8\% & 3.9\% & 5.3\% & 1.0\% \\
\hline AUC & 0.997 & 1.000 & 0.975 & 0.959 & 0.996 \\
\hline
\end{tabular}

**Table 28 Performance of Prompt Guard. We include in- and out-of-distribution evaluations, a multilingual jailbreak built using machine translation, and a dataset of indirect injections from CyberSecEval.**
\end{table}

![Figure](./grattafioriLlama3Herd2024_images/964281cf-3b75-4775-9ae9-065d3b6d211b-52_509_1609_617_238.jpg)

**Figure 24 Effect of micro-batching on inference throughput and latency during the Left: pre-filling and Right: decoding stage. The numbers in the plot correspond to the (micro-)batch size.**

enables the use of tensor parallelism (Shoeybi et al., 2019). Across nodes, however, connectivity has lower bandwidth and higher latency, so we use pipeline parallelism (Huang et al., 2019) instead.

During training with pipeline parallelism, bubbles are a major efficiency concern (see Section 3.3). However, they are not an issue during inference, since inference does not involve a backward pass that requires a pipeline flush. Therefore, we use micro-batching to improve inference throughput with pipeline parallelism.

We evaluate the effect of using two micro-batches in inference workloads of 4,096 input tokens and 256 output tokens both during the key-value cache pre-fill stage of inference and during the decoding stage. We find that micro-batching improves throughput of inference with the same local batch size; see Figure 24. These improvements result from micro-batching enabling concurrent execution of micro batches in both these stages. The additional synchronization points due to micro-batching also increase latency but, overall, micro-batching still leads to a better throughput-latency trade-off.

## 6.2 FP8 Quantization

We perform experiments leveraging the native FP8 support of H100 GPUs to perform low-precision inference. To enable low-precision inference, we apply FP8 quantization to most matrix multiplications inside the model. In particular, we quantize most parameters and activations in the feedforward network layers in the model, which account for roughly 50\% of the inference compute time. We do not quantize parameters in the self-attention layers of the model. We leverage dynamic scaling factors for better accuracy (Xiao et al., 2024b), optimizing our CUDA kernels ${ }^{15}$ to reduce the overhead of calculating the scales. We find that the quality of Llama 3 405B is sensitive to certain types of quantization, and make a few additional changes to increase the model output quality:
\begin{itemize}
\item[1.] Akin to Zhang et al. (2021), we do not perform quantization in the first and last Transformer layers.
\item[2.] High-perplexity tokens such as dates can lead to large activation values. In turn, these can lead to high dynamic scaling factors in FP8 and a non-negligible number of underflows, leading to errors in decoding.
\end{itemize}

\footnotetext{
${ }^{15}$ Our FP8 kernels are available at https://github.com/pytorch/FBGEMM/tree/main/fbgemm_gpu/experimental/gen_ai. We provide usage examples at https://github.com/meta-llama/llama-agentic-system.
}

![Figure](./grattafioriLlama3Herd2024_images/964281cf-3b75-4775-9ae9-065d3b6d211b-53_651_1644_258_254.jpg)

**Figure 25 Illustration of tensor-wise and row-wise FP8 quantization. Right: Row-wise quantization enables the use of more granular activation factors than Left: tensor-wise quantization.**

![Figure](./grattafioriLlama3Herd2024_images/964281cf-3b75-4775-9ae9-065d3b6d211b-53_320_1055_1152_503.jpg)

**Figure 26 Reward score distribution for Llama 3 405B using BF16 and FP8 inference. Our FP8 quantization approach has negligible impact on the model's responses.**

To address this issue, we upper bound the dynamic scaling factors to 1200.
\begin{itemize}
\item[3.] We use row-wise quantization, computing scaling factors across rows for parameter and activation matrices (see Figure 25). We find this works better than a tensor-wise quantization approach.
\end{itemize}

Effect of quantization errors. Evaluations on standard benchmarks often suggest that FP8 inference performs on par with BF16 inference even without these mitigations. However, we find that such benchmarks do not adequately reflect the effects of FP8 quantization. When scaling factors are not upper bounded, the model occasionally produces corrupted responses even though the benchmark performance is strong. Instead of relying on benchmarks to measure distribution changes due to quantization, we find it is better to analyze the distribution of reward-model scores for 100,000 responses produced using both FP8 and BF16. Figure 26 shows the resulting reward distribution for our quantization approach. The results in the figure show that our approach to FP8 quantization has very limited impact on the model's response.

Experimental evaluation of efficiency. Figure 27 depicts the throughput-latency trade-off of performing FP8 inference with Llama 3 405B in the pre-fill and decoding stages, using 4,096 input tokens and 256 output tokens. The figure compares the efficiency of FP8 inference with that of the two-machine BF16 inference approach described in Section 6.1. The results show that use of FP8 inference leads to throughput improvements of up to 50\% during the pre-fill stage, and a substantially better throughput-latency trade-off during decoding.

![Figure](./grattafioriLlama3Herd2024_images/964281cf-3b75-4775-9ae9-065d3b6d211b-54_516_1597_258_238.jpg)

**Figure 27 Throughput-latency trade-off in FP8 inference with Llama 3 405B compared with BF16 inference using different pipeline parallelization setups. Left: Results for pre-filling. Right: Results for decoding.**

