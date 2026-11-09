# 3 Pre-Training

Language model pre-training involves: (1) the curation and filtering of a large-scale training corpus, (2) the development of a model architecture and corresponding scaling laws for determining model size, (3) the development of techniques for efficient pre-training at large scale, and (4) the development of a pre-training recipe. We present each of these components separately below.

## 3.1 Pre-Training Data

We create our dataset for language model pre-training from a variety of data sources containing knowledge until the end of 2023. We apply several de-duplication methods and data cleaning mechanisms on each data source to obtain high-quality tokens. We remove domains that contain large amounts of personally identifiable information (PII), and domains with known adult content.

## 3.1.1 Web Data Curation

Much of the data we utilize is obtained from the web and we describe our cleaning process below.
PII and safety filtering. Among other mitigations, we implement filters designed to remove data from websites are likely to contain unsafe content or high volumes of PII, domains that have been ranked as harmful according to a variety of Meta safety standards, and domains that are known to contain adult content.

Text extraction and cleaning. We process the raw HTML content for non-truncated web documents to extract high-quality diverse text. To do so, we build a custom parser that extracts the HTML content and optimizes for precision in boilerplate removal and content recall. We evaluate our parser's quality in human evaluations, comparing it with popular third-party HTML parsers that optimize for article-like content, and found it to perform favorably. We carefully process HTML pages with mathematics and code content to preserve the structure of that content. We maintain the image alt attribute text since mathematical content is often represented as pre-rendered images where the math is also provided in the alt attribute. We experimentally evaluate different cleaning configurations. We find markdown is harmful to the performance of a model that is primarily trained on web data compared to plain text, so we remove all markdown markers.

De-duplication. We apply several rounds of de-duplication at the URL, document, and line level:
\begin{itemize}
\item[-] URL-level de-duplication. We perform URL-level de-duplication across the entire dataset. We keep the most recent version for pages corresponding to each URL.
\item[-] Document-level de-duplication. We perform global MinHash (Broder, 1997) de-duplication across the entire dataset to remove near duplicate documents.
\item[-] Line-level de-duplication. We perform aggressive line-level de-duplication similar to ccNet (Wenzek et al., 2019). We remove lines that appeared more than 6 times in each bucket of 30M documents. Although our manual qualitative analysis showed that the line-level de-duplication removes not only leftover boilerplate from various websites such as navigation menus, cookie warnings, but also frequent high-quality text, our empirical evaluations showed strong improvements.
\end{itemize}

Heuristic filtering. We develop heuristics to remove additional low-quality documents, outliers, and documents with excessive repetitions. Some examples of heuristics include:
\begin{itemize}
\item[-] We use duplicated n-gram coverage ratio (Rae et al., 2021) to remove lines that consist of repeated content such as logging or error messages. Those lines could be very long and unique, hence cannot be filtered by line-dedup.
\item[-] We use "dirty word" counting (Raffel et al., 2020) to filter out adult websites that are not covered by domain block lists.
\item[-] We use a token-distribution Kullback-Leibler divergence to filter out documents containing excessive numbers of outlier tokens compared to the training corpus distribution.
\end{itemize}

Model-based quality filtering. Further, we experiment with applying various model-based quality classifiers to sub-select high-quality tokens. These include using fast classifiers such as fasttext (Joulin et al., 2017) trained to recognize if a given text would be referenced by Wikipedia (Touvron et al., 2023a), as well as more compute-intensive Roberta-based classifiers (Liu et al., 2019a) trained on Llama 2 predictions. To train a quality classifier based on Llama 2, we create a training set of cleaned web documents, describe the quality requirements, and instruct Llama 2's chat model to determine if the documents meets these requirements. We use DistilRoberta (Sanh et al., 2019) to generate quality scores for each document for efficiency reasons. We experimentally evaluate the efficacy of various quality filtering configurations.

Code and reasoning data. Similar to DeepSeek-AI et al. (2024), we build domain-specific pipelines that extract code and math-relevant web pages. Specifically, both the code and reasoning classifiers are DistilRoberta models trained on web data annotated by Llama 2. Unlike the general quality classifier mentioned above, we conduct prompt tuning to target web pages containing math deduction, reasoning in STEM areas and code interleaved with natural language. Since the token distribution of code and math is substantially different than that of natural language, these pipelines implement domain-specific HTML extraction, customized text features and heuristics for filtering.

Multilingual data. Similar to our processing pipelines for English described above, we implement filters to remove data from websites that are likely to contain PII or unsafe content. Our multilingual text processing pipeline has several unique features:
\begin{itemize}
\item[-] We use a fasttext-based language identification model to categorize documents into 176 languages.
\item[-] We perform document-level and line-level de-duplication within data for each language.
\end{itemize}
\begin{itemize}
\item[-] We apply language-specific heuristics and model-based filters to remove low-quality documents.
\end{itemize}

In addition, we perform quality ranking of multilingual documents using a multilingual Llama 2-based classifier to ensure that high-quality content is prioritized. We determine the amount of multilingual tokens used in pre-training experimentally, balancing model performance on English and multilingual benchmarks.

## 3.1.2 Determining the Data Mix

To obtain a high-quality language model, it is essential to carefully determine the proportion of different data sources in the pre-training data mix. Our main tools in determining this data mix are knowledge classification and scaling law experiments.

Knowledge classification. We develop a classifier to categorize the types of information contained in our web data to more effectively determine a data mix. We use this classifier to downsample data categories that are over-represented on the web, for example, arts and entertainment.

Scaling laws for data mix. To determine the best data mix, we perform scaling law experiments in which we train several small models on a data mix and use that to predict the performance of a large model on that mix (see Section 3.2.1). We repeat this process multiple times for different data mixes to select a new data mix candidate. Subsequently, we train a larger model on this candidate data mix and evaluate the performance of that model on several key benchmarks.

Data mix summary. Our final data mix contains roughly 50\% of tokens corresponding to general knowledge, 25\% of mathematical and reasoning tokens, 17\% code tokens, and 8\% multilingual tokens.

## 3.1.3 Annealing Data

Empirically, we find that annealing (see Section 3.4.3) on small amounts of high-quality code and mathematical data can boost the performance of pre-trained models on key benchmarks. Akin to Li et al. (2024b), we perform annealing with a data mix that upsamples high-quality data in select domains. We do not include any training sets from commonly used benchmarks in our annealing data. This enables us to assess the true few-shot learning capabilities and out-of-domain generalization of Llama 3.

Following OpenAI (2023a), we evaluate the efficacy of annealing on the GSM8k (Cobbe et al., 2021) and MATH (Hendrycks et al., 2021b) training sets in annealing. We find that annealing improved the performance of a pre-trained Llama 3 8B model on the GSM8k and MATH validation sets by 24.0\% and 6.4\%, respectively. However, the improvements on the 405B model are negligible, suggesting that our flagship model has strong in-context learning and reasoning capabilities and does not require specific in-domain training samples to obtain strong performance.

Using annealing to assess data quality. Similar to Blakeney et al. (2024), we find that annealing enables us to judge the value of small domain-specific datasets. We measure the value of such datasets by annealing the learning rate of a 50\% trained Llama 3 8B model linearly to 0 on 40B tokens. In those experiments, we assign 30\% weight to the new dataset and the remaining 70\% weight to the default data mix. Using annealing to evaluate new data sources is more efficient than performing scaling law experiments for every small dataset.

## 3.2 Model Architecture

Llama 3 uses a standard, dense Transformer architecture (Vaswani et al., 2017). It does not deviate significantly from Llama and Llama 2 (Touvron et al., 2023a,b) in terms of model architecture; our performance gains are primarily driven by improvements in data quality and diversity as well as by increased training scale.

We make a few small modifications compared to Llama 2:
\begin{itemize}
\item[-] We use grouped query attention (GQA; Ainslie et al. (2023)) with 8 key-value heads to improve inference speed and to reduce the size of key-value caches during decoding.
\item[-] We use an attention mask that prevents self-attention between different documents within the same sequence. We find that this change had limited impact during in standard pre-training, but find it to be important in continued pre-training on very long sequences.
\end{itemize}

\begin{table}
\begin{tabular}[t]{|l|l|l|l|}
\hline & 8B & 70B & 405B \\
\hline Layers & 32 & 80 & 126 \\
\hline Model Dimension & 4,096 & 8192 & 16,384 \\
\hline FFN Dimension & 14,336 & 28,672 & 53,248 \\
\hline Attention Heads & 32 & 64 & 128 \\
\hline Key/Value Heads & 8 & 8 & 8 \\
\hline Peak Learning Rate & $3 \times 10^{-4}$ & $1.5 \times 10^{-4}$ & $8 \times 10^{-5}$ \\
\hline Activation Function & & SwiGLU & \\
\hline Vocabulary Size & \multicolumn{3}{|c|}{128,000} \\
\hline Positional Embeddings & \multicolumn{3}{|c|}{RoPE ( $\theta=500,000$ )} \\
\hline
\end{tabular}

**Table 3 Overview of the key hyperparameters of Llama 3. We display settings for 8B, 70B, and 405B language models.**
\end{table}
\begin{itemize}
\item[-] We use a vocabulary with 128K tokens. Our token vocabulary combines 100K tokens from the tiktoken ${ }^{3}$ tokenizer with 28K additional tokens to better support non-English languages. Compared to the Llama 2 tokenizer, our new tokenizer improves compression rates on a sample of English data from 3.17 to 3.94 characters per token. This enables the model to "read" more text for the same amount of training compute. We also found that adding 28K tokens from select non-English languages improved both compression ratios and downstream performance, with no impact on English tokenization.
\item[-] We increase the RoPE base frequency hyperparameter to 500,000. This enables us to better support longer contexts; Xiong et al. (2023) showed this value to be effective for context lengths up to 32,768.
\end{itemize}

Llama 3 405B uses an architecture with 126 layers, a token representation dimension of 16,384, and 128 attention heads; see Table 3 for details. This leads to a model size that is approximately compute-optimal according to scaling laws on our data for our training budget of $3.8 \times 10^{25}$ FLOPs.

## 3.2.1 Scaling Laws

We develop scaling laws (Hoffmann et al., 2022; Kaplan et al., 2020) to determine the optimal model size for our flagship model given our pre-training compute budget. In addition to determining the optimal model size, a major challenge is to forecast the flagship model's performance on downstream benchmark tasks, due to a couple of issues: (1) Existing scaling laws typically predict only next-token prediction loss rather than specific benchmark performance. (2) Scaling laws can be noisy and unreliable because they are developed based on pre-training runs conducted with small compute budgets (Wei et al., 2022b).

To address these challenges, we implement a two-stage methodology to develop scaling laws that accurately predict downstream benchmark performance:
\begin{itemize}
\item[1.] We first establish a correlation between the compute-optimal model's negative log-likelihood on downstream tasks and the training FLOPs.
\item[2.] Next, we correlate the negative log-likelihood on downstream tasks with task accuracy, utilizing both the scaling law models and older models trained with higher compute FLOPs. In this step, we specifically leverage the Llama 2 family of models.
\end{itemize}

This approach enables us to predict downstream task performance given a specific number of training FLOPs for compute-optimal models. We use a similar method to select our pre-training data mix (see Section 3.4).

Scaling law experiments. Concretely, we construct our scaling laws by pre-training models using compute budgets between $6 \times 10^{18} \mathrm{FLOPs}$ and $10^{22}$ FLOPs. At each compute budget, we pre-train models ranging in size between 40M and 16B parameters, using a subset of model sizes at each compute budget. In these training runs, we use a cosine learning rate schedule with a linear warmup for 2,000 training steps. The peak learning rate is set between $2 \times 10^{-4}$ and $4 \times 10^{-4}$ depending on the size of the model. We set the cosine decay to 0.1 of the peak value. The weight decay at each step is set to 0.1 times the learning rate at that step. We use a fixed batch size for each compute scale, ranging between 250K and 4M.

\footnotetext{
${ }^{3}$ https://github.com/openai/tiktoken/tree/main
}

![Figure](./grattafioriLlama3Herd2024_images/964281cf-3b75-4775-9ae9-065d3b6d211b-08_481_653_228_284.jpg)

**Figure 2 Scaling law IsoFLOPs curves between $6 \times 10^{18**$ and $10^{22}$ FLOPs. The loss is the negative loglikelihood on a held-out validation set. We approximate measurements at each compute scale using a second degree polynomial.}

![Figure](./grattafioriLlama3Herd2024_images/964281cf-3b75-4775-9ae9-065d3b6d211b-08_479_655_230_1182.jpg)

**Figure 3 Number of training tokens in identified computeoptimal models as a function of pre-training compute budget. We include the fitted scaling-law prediction as well. The compute-optimal models correspond to the parabola minimums in Figure 2.**

These experiments give rise to the IsoFLOPs curves in Figure 2. The loss in these curves is measured on a separate validation set. We fit the measured loss values using a second-degree polynomial and identify the minimums of each parabola. We refer to minimum of a parabola as the compute-optimal model at the corresponding pre-training compute budget.

We use the compute-optimal models we identified this way to predict the optimal number of training tokens for a specific compute budget. To do so, we assume a power-law relation between compute budget, $C$, and the optimal number of training tokens, \(N^{\star}(C)\) :
$$
N^{\star}(C)=A C^{\alpha} .
$$
We fit $A$ and $\alpha$ using the data from Figure 2. We find that \((\alpha, A)=(0.53,0.29)\); the corresponding fit is shown in Figure 3. Extrapolation of the resulting scaling law to $3.8 \times 10^{25}$ FLOPs suggests training a 402B parameter model on 16.55T tokens.

An important observation is that IsoFLOPs curves become flatter around the minimum as the compute budget increases. This implies that performance of the flagship model is relatively robust to small changes in the trade-off between model size and training tokens. Based on this observation, we ultimately decided to train a flagship model with 405B parameters.

Predicting performance on downstream tasks. We use the resulting compute-optimal models to forecast the performance of the flagship Llama 3 model on benchmark data sets. First, we linearly correlate the (normalized) negative log-likelihood of correct answer in the benchmark and the training FLOPs. In this analysis, we use only the scaling law models trained up to $10^{22}$ FLOPs on the data mix described above. Next, we establish a sigmoidal relation between the log-likelihood and accuracy using both the scaling law models and Llama 2 models, which were trained using the Llama 2 data mix and tokenizer. We show the results of this experiment on the ARC Challenge benchmark in Figure 4). We find this two-step scaling law prediction, which extrapolates over four orders of magnitude, to be quite accurate: it only slightly underestimates the final performance of the flagship Llama 3 model.

## 3.3 Infrastructure, Scaling, and Efficiency

We describe our hardware and infrastructure that powered Llama 3 405B pre-training at scale and discuss several optimizations that leads to improvements in training efficiency.

## 3.3.1 Training Infrastructure

The Llama 1 and 2 models were trained on Meta's AI Research SuperCluster (Lee and Sengupta, 2022). As we scaled further, the training for Llama 3 was migrated to Meta's production clusters (Lee et al., 2024).This

![Figure](./grattafioriLlama3Herd2024_images/964281cf-3b75-4775-9ae9-065d3b6d211b-09_509_1143_221_490.jpg)

**Figure 4 Scaling law forecast for ARC Challenge. Left: Normalized negative log-likelihood of the correct answer on the ARC Challenge benchmark as a function of pre-training FLOPs. Right: ARC Challenge benchmark accuracy as a function of the normalized negative log-likelihood of the correct answer. This analysis enables us to predict model performance on the ARC Challenge benchmark before pre-training commences. See text for details.**

setup optimizes for production-grade reliability, which is essential as we scale up training.
Compute. Llama 3 405B is trained on up to 16K H100 GPUs, each running at 700W TDP with 80GB HBM3, using Meta's Grand Teton AI server platform (Matt Bowman, 2022). Each server is equipped with eight GPUs and two CPUs. Within a server, the eight GPUs are connected via NVLink. Training jobs are scheduled using MAST (Choudhury et al., 2024), Meta's global-scale training scheduler.

Storage. Tectonic (Pan et al., 2021), Meta's general-purpose distributed file system, is used to build a storage fabric (Battey and Gupta, 2024) for Llama 3 pre-training. It offers 240 PB of storage out of 7,500 servers equipped with SSDs, and supports a sustainable throughput of 2 TB/s and a peak throughput of 7 TB/s. A major challenge is supporting the highly bursty checkpoint writes that saturate the storage fabric for short durations. Checkpointing saves each GPU's model state, ranging from 1 MB to 4 GB per GPU, for recovery and debugging. We aim to minimize GPU pause time during checkpointing and increase checkpoint frequency to reduce the amount of lost work after a recovery.

Network. Llama 3 405B used RDMA over Converged Ethernet (RoCE) fabric based on the Arista 7800 and Minipack2 Open Compute Project ${ }^{4}$ OCP rack switches. Smaller models in the Llama 3 family were trained using Nvidia Quantum2 Infiniband fabric. Both RoCE and Infiniband clusters leverage 400 Gbps interconnects between GPUs. Despite the underlying network technology differences between these clusters, we tune both of them to provide equivalent performance for these large training workloads. We elaborate further on our RoCE network since we fully own its design.
\begin{itemize}
\item[-] Network topology. Our RoCE-based AI cluster comprises $24 \mathrm{~K} \mathrm{GPUs}^{5}$ connected by a three-layer Clos network (Lee et al., 2024). At the bottom layer, each rack hosts 16 GPUs split between two servers and connected by a single Minipack2 top-of-the-rack (ToR) switch. In the middle layer, 192 such racks are connected by Cluster Switches to form a pod of 3,072 GPUs with full bisection bandwidth, ensuring no oversubscription. At the top layer, eight such pods within the same datacenter building are connected via Aggregation Switches to form a cluster of 24K GPUs. However, network connectivity at the aggregation layer does not maintain full bisection bandwidth and instead has an oversubscription ratio of 1:7. Our model parallelism methods (see Section 3.3.2) and training job scheduler (Choudhury et al., 2024) are all optimized to be aware of network topology, aiming to minimize network communication across pods.
\item[-] Load balancing. LLM training produces fat network flows that are hard to load balance across all available network paths using traditional methods such as Equal-Cost Multi-Path (ECMP) routing. To address this challenge, we employ two techniques. First, our collective library creates 16 network flows between two GPUs, instead of just one, thereby reducing the traffic per flow and providing more flows
\end{itemize}

\footnotetext{
${ }^{4}$ Open Compute Project: https://www.opencompute.org/
${ }^{5}$ Note that we use only up to 16K of these 24K GPUs for Llama 3 pre-training.
}

\begin{table}
\begin{tabular}[t]{|l|l|l|l|l|l|l|l|l|l|}
\hline GPUs & TP & CP & PP & DP & Seq. Len. & Batch size/DP & Tokens/Batch & TFLOPs/GPU & BF16 MFU \\
\hline 8,192 & 8 & 1 & 16 & 64 & 8,192 & 32 & 16M & 430 & 43\% \\
\hline 16,384 & 8 & 1 & 16 & 128 & 8,192 & 16 & 16M & 400 & 41\% \\
\hline 16,384 & 8 & 16 & 16 & 8 & 131,072 & 16 & 16M & 380 & 38\% \\
\hline
\end{tabular}

**Table 4 Scaling configurations and MFU for each stage of Llama 3 405B pre-training. See text and Figure 5 for descriptions of each type of parallelism.**
\end{table}
for load balancing. Second, our Enhanced-ECMP (E-ECMP) protocol effectively balances these 16 flows across different network paths by hashing on additional fields in the RoCE header of packets.
\begin{itemize}
\item[-] Congestion control. We use deep-buffer switches in the spine (Gangidi et al., 2024) to accommodate transient congestion and buffering caused by collective communication patterns. This setup helps limit the impact of persistent congestion and network back pressure caused by slow servers, which is common in training. Finally, better load balancing through E-ECMP significantly reduces the chance of congestion. With these optimizations, we successfully run a 24K GPU cluster without traditional congestion control methods such as Data Center Quantized Congestion Notification (DCQCN).
\end{itemize}

## 3.3.2 Parallelism for Model Scaling

To scale training for our largest models, we use 4D parallelism-a combination of four different types of parallelism methods-to shard the model. This approach efficiently distributes computation across many GPUs and ensures each GPU's model parameters, optimizer states, gradients, and activations fit in its HBM. Our implementation of 4D parallelism is illustrated in Figure 5. It combines tensor parallelism (TP; Krizhevsky et al. (2012); Shoeybi et al. (2019); Korthikanti et al. (2023)), pipeline parallelism (PP; Huang et al. (2019); Narayanan et al. (2021); Lamy-Poirier (2023)), context parallelism (CP; Liu et al. (2023a)), and data parallelism (DP; Rajbhandari et al. (2020); Ren et al. (2021); Zhao et al. (2023b)).

Tensor parallelism splits individual weight tensors into multiple chunks on different devices. Pipeline parallelism partitions the model vertically into stages by layers, so that different devices can process in parallel different stages of the full model pipeline. Context parallelism divides the input context into segments, reducing memory bottleneck for very long sequence length inputs. We use fully sharded data parallelism (FSDP; Rajbhandari et al., 2020; Ren et al., 2021; Zhao et al., 2023b), which shards the model, optimizer, and gradients while implementing data parallelism which processes data in parallel on multiple GPUs and synchronizes after each training step. Our use of FSDP for Llama 3 shards optimizer states and gradients, but for model shards we do not reshard after forward computation to avoid an extra all-gather communication during backward passes.

GPU utilization. Through careful tuning of the parallelism configuration, hardware, and software, we achieve an overall BF16 Model FLOPs Utilization (MFU; Chowdhery et al. (2023)) of 38-43\% for the configurations shown in Table 4. The slight drop in MFU to 41\% on 16K GPUs with $\mathrm{DP}=128$ compared to 43\% on 8K GPUs with $\mathrm{DP}=64$ is due to the lower batch size per DP group needed to keep the global tokens per batch constant during training.

Pipeline parallelism improvements. We encountered several challenges with existing implementations:
\begin{itemize}
\item[-] Batch size constraint. Current implementations have constraints on supported batch size per GPU, requiring it to be divisible by the number of pipeline stages. For the example in Figure 6, the depth-first schedule (DFS) of pipeline parallelism (Narayanan et al., 2021) requires $N=\mathrm{PP}=4$, while the breadth-first schedule (BFS; Lamy-Poirier (2023)) requires $N=M$, where $M$ is the total number of micro-batches and $N$ is the number of contiguous micro-batches for the same stage's forward or backward. However, pre-training often needs flexibility to adjust batch size.
\item[-] Memory imbalance. Existing pipeline parallelism implementations lead to imbalanced resource consumption. The first stage consumes more memory due to the embedding and the warm-up micro-batches.
\item[-] Computation imbalance. After the last layer of the model, we need to calculate output and loss, making this stage the execution latency bottleneck.
\end{itemize}

![Figure](./grattafioriLlama3Herd2024_images/964281cf-3b75-4775-9ae9-065d3b6d211b-11_679_1502_287_294.jpg)

**Figure 5 Illustration of 4D parallelism. GPUs are divided into parallelism groups in the order of [TP, CP, PP, DP], where DP stands for FSDP. In this example, 16 GPUs are configured with a group size of $|\mathrm{TP**|=2,|\mathrm{CP}|=2,|\mathrm{PP}|=2$, and $|\mathrm{DP}|=2$. A GPU's position in 4D parallelism is represented as a vector, $\left[D_{1}, D_{2}, D_{3}, D_{4}\right]$, where $D_{i}$ is the index on the $i$-th parallelism dimension. In this example, GPU0[TP0, CP0, PP0, DP0] and GPU1[TP1, CP0, PP0, DP0] are in the same TP group, GPU0 and GPU2 are in the same CP group, GPU0 and GPU4 are in the same PP group, and GPU0 and GPU8 are in the same DP group.}

To address these issues, we modify our pipeline schedule as shown in Figure 6, which allows setting $N$ flexibly-in this case $N=5$, which can run a arbitrary number of micro-batches in each batch. This allows us to run: (1) fewer micro-batches than the number of stages when we have batch size limit at large scale; or (2) more micro-batches to hide point-to-point communication, finding a sweet spot between DFS and breadth first schedule (BFS) for the best communication and memory efficiency. To balance the pipeline, we reduce one Transformer layer each from the first and the last stages, respectively. This means that the first model chunk on the first stage has only the embedding, and the last model chunk on the last stage has only output projection and loss calculation. To reduce pipeline bubbles, we use an interleaved schedule (Narayanan et al., 2021) with $V$ pipeline stages on one pipeline rank. Overall pipeline bubble ratio is $\frac{\mathrm{PP}-1}{V * M}$. Further, we adopt asynchronous point-to-point communication in PP, which considerably speeds up training, especially in cases when the document mask introduces extra computation imbalance. We enable TORCH_NCCL_AVOID_RECORD_STREAMS to reduce memory usage from asynchronous point-to-point communication. Finally, to reduce memory cost, based on detailed memory allocation profiling, we proactively deallocate tensors that will not be used for future computation, including the input and output tensors of each pipeline stage, that will not be used for future computation. With these optimizations, we could pre-train Llama 3 on sequences of 8K tokens without activation checkpointing.

Context parallelism for long sequences. We utilize context parallelism (CP) to improve memory efficiency when scaling the context length of Llama 3 and enable training on extremely long sequences up to 128K in length. In CP, we partition across the sequence dimension, and specifically we partition the input sequence into $2 \times \mathrm{CP}$ chunks so each CP rank receives two chunks for better load balancing. The $i$-th CP rank received both the $i$-th and the \((2 \times \mathrm{CP}-1-i)\)-th chunks.

Different from existing CP implementations that overlap communication and computation in a ring-like structure (Liu et al., 2023a), our CP implementation adopts an all-gather based method where we first all-gather the key (K) and value (V) tensors, and then compute attention output for the local query (Q) tensor chunk. Although the all-gather communication latency is exposed in the critical path, we still adopt this approach for two main reasons: (1) it is easier and more flexible to support different types of attention masks in all-gather based CP attention, such as the document mask; and (2) the exposed all-gather latency

![Figure](./grattafioriLlama3Herd2024_images/964281cf-3b75-4775-9ae9-065d3b6d211b-12_271_1557_258_285.jpg)

**Figure 6 Illustration of pipeline parallelism in Llama 3. Pipeline parallelism partitions eight pipeline stages (0 to 7) across four pipeline ranks (PP ranks 0 to 3), where the GPUs with rank 0 run stages 0 and 4, the GPUs with P rank 1 run stages 1 and 5, etc. The colored blocks (0 to 9) represent a sequence of micro-batches, where $M$ is the total number of micro-batches and $N$ is the number of continuous micro-batches for the same stage's forward or backward. Our key insight is to make $N$ tunable.**

is small as the communicated K and V tensors are much smaller than Q tensor due to the use of GQA (Ainslie et al., 2023). Hence, the time complexity of attention computation is an order of magnitude larger than all-gather ( \(O\left(S^{2}\right)\) versus \(O(S)\), where $S$ represents the sequence length in the full causal mask), making the all-gather overhead negligible.

Network-aware parallelism configuration. The order of parallelism dimensions, [TP, CP, PP, DP], is optimized for network communication. The innermost parallelism requires the highest network bandwidth and lowest latency, and hence is usually constrained to within the same server. The outermost parallelism may spread across a multi-hop network and should tolerate higher network latency. Therefore, based on the requirements for network bandwidth and latency, we place parallelism dimensions in the order of [TP, CP, PP, DP]. DP (i.e., FSDP) is the outermost parallelism because it can tolerate longer network latency by asynchronously prefetching sharded model weights and reducing gradients. Identifying the optimal parallelism configuration with minimal communication overhead while avoiding GPU memory overflow is challenging. We develop a memory consumption estimator and a performance-projection tool which helped us explore various parallelism configurations and project overall training performance and identify memory gaps effectively.

Numerical stability. By comparing training loss between different parallelism setups, we fixed several numerical issues that impact training stability. To ensure training convergence, we use FP32 gradient accumulation during backward computation over multiple micro-batches and also reduce-scatter gradients in FP32 across data parallel workers in FSDP. For intermediate tensors, e.g., vision encoder outputs, that are used multiple times in the forward computation, the backward gradients are also accumulated in FP32.

## 3.3.3 Collective Communication

Our collective communication library for Llama 3 is based on a fork of Nvidia's NCCL library, called NCCLX. NCCLX significantly improves the performance of NCCL, especially for higher latency networks. Recall that the order of parallelism dimensions is [TP, CP, PP, DP], where DP corresponds to FSDP. The outermost parallelism dimensions, PP and DP, may communicate through a multi-hop network, with latency up to tens of microseconds. The original NCCL collectives-all-gather and reduce-scatter in FSDP, and point-to-point in PP-require data chunking and staged data copy. This approach incurs several inefficiencies, including (1) requiring a large number of small control messages to be exchanged over the network to facilitate data transfer, (2) extra memory-copy operations, and (3) using extra GPU cycles for communication. For Llama 3 training, we address a subset of these inefficiencies by tuning chunking and data transfer to fit our network latencies, which can be as high as tens of microseconds for a large cluster. We also allow small control messages to traverse our network at a higher priority, especially avoiding being head-of-line blocked in deep-buffer core switches. Our ongoing work for future Llama versions involves making deeper changes in NCCLX to holistically address all the aforementioned problems.

\begin{table}
\begin{tabular}[t]{|l|l|l|l|}
\hline Component & Category & Interruption Count & \% of Interruptions \\
\hline Faulty GPU & GPU & 148 & 30.1\% \\
\hline GPU HBM3 Memory & GPU & 72 & 17.2\% \\
\hline Software Bug & Dependency & 54 & 12.9\% \\
\hline Network Switch/Cable & Network & 35 & 8.4\% \\
\hline Host Maintenance & Unplanned Maintenance & 32 & 7.6\% \\
\hline GPU SRAM Memory & GPU & 19 & 4.5\% \\
\hline GPU System Processor & GPU & 17 & 4.1\% \\
\hline NIC & Host & 7 & 1.7\% \\
\hline NCCL Watchdog Timeouts & Unknown & 7 & 1.7\% \\
\hline Silent Data Corruption & GPU & 6 & 1.4\% \\
\hline GPU Thermal Interface + Sensor & GPU & 6 & 1.4\% \\
\hline SSD & Host & 3 & 0.7\% \\
\hline Power Supply & Host & 3 & 0.7\% \\
\hline Server Chassis & Host & 2 & 0.5\% \\
\hline IO Expansion Board & Host & 2 & 0.5\% \\
\hline Dependency & Dependency & 2 & 0.5\% \\
\hline CPU & Host & 2 & 0.5\% \\
\hline System Memory & Host & 2 & 0.5\% \\
\hline
\end{tabular}

**Table 5 Root-cause categorization of unexpected interruptions during a 54-day period of Llama 3 405B pre-training. About 78\% of unexpected interruptions were attributed to confirmed or suspected hardware issues.**
\end{table}

## 3.3.4 Reliability and Operational Challenges

The complexity and potential failure scenarios of 16K GPU training surpass those of much larger CPU clusters that we have operated. Moreover, the synchronous nature of training makes it less fault-tolerant-a single GPU failure may require a restart of the entire job. Despite these challenges, for Llama 3, we achieved higher than 90\% effective training time while supporting automated cluster maintenance, such as firmware and Linux kernel upgrades (Vigraham and Leonhardi, 2024), which resulted in at least one training interruption daily. The effective training time measures the time spent on useful training over the elapsed time.

During a 54-day snapshot period of pre-training, we experienced a total of 466 job interruptions. Of these, 47 were planned interruptions due to automated maintenance operations such as firmware upgrades or operatorinitiated operations like configuration or dataset updates. The remaining 419 were unexpected interruptions, which are classified in Table 5. Approximately 78\% of the unexpected interruptions are attributed to confirmed hardware issues, such as GPU or host component failures, or suspected hardware-related issues like silent data corruption and unplanned individual host maintenance events. GPU issues are the largest category, accounting for 58.7\% of all unexpected issues. Despite the large number of failures, significant manual intervention was required only three times during this period, with the rest of issues handled by automation.

To increase the effective training time, we reduced job startup and checkpointing time, and developed tools for fast diagnosis and problem resolution. We extensively use PyTorch's built-in NCCL flight recorder (Ansel et al., 2024), a feature that captures collective metadata and stack traces into a ring buffer, and hence allowing us to diagnose hangs and performance issues quickly at scale, particularly with regard to NCCLX. Using this, we efficiently record every communication event and the duration of each collective operation, and also automatically dump tracing data on NCCLX watchdog or heartbeat timeout. We enable more computationally intensive tracing operations and metadata collection selectively as needed live in production through online configuration changes (Tang et al., 2015) without needing a code release or job restart.

Debugging issues in large-scale training is complicated by the mixed use of NVLink and RoCE in our network. Data transfer over NVLink typically occurs through load/store operations issued by CUDA kernels, and failures in either the remote GPU or NVLink connectivity often manifest as stalled load/store operations within CUDA kernels without returning a clear error code. NCCLX enhances the speed and accuracy of failure
detection and localization through a tight co-design with PyTorch, allowing PyTorch to access NCCLX's internal state and track relevant information. While stalls due to NVLink failures cannot be completely prevented, our system monitors the state of the communication library and automatically times out when such a stall is detected. Additionally, NCCLX traces the kernel and network activities of each NCCLX communication and provides a snapshot of the failing NCCLX collective's internal state, including finished and pending data transfers between all ranks. We analyze this data to debug NCCLX scaling issues.

Sometimes, hardware issues may cause still-functioning but slow stragglers that are hard to detect. Even a single straggler can slow down thousands of other GPUs, often appearing as functioning but slow communications. We developed tools to prioritize potentially problematic communications from selected process groups. By investigating just a few top suspects, we were usually able to effectively identify the stragglers.

One interesting observation is the impact of environmental factors on training performance at scale. For Llama 3 405B , we noted a diurnal 1-2\% throughput variation based on time-of-day. This fluctuation is the result of higher mid-day temperatures impacting GPU dynamic voltage and frequency scaling.

During training, tens of thousands of GPUs may increase or decrease power consumption at the same time, for example, due to all GPUs waiting for checkpointing or collective communications to finish, or the startup or shutdown of the entire training job. When this happens, it can result in instant fluctuations of power consumption across the data center on the order of tens of megawatts, stretching the limits of the power grid. This is an ongoing challenge for us as we scale training for future, even larger Llama models.

## 3.4 Training Recipe

The recipe used to pre-train Llama 3 405B consists of three main stages: (1) initial pre-training, (2) long-context pre-training, and (3) annealing. The three stages are described separately below. We use similar recipes to pre-train the 8B and 70B models.

## 3.4.1 Initial Pre-Training

We pre-train Llama 3 405B using AdamW with a peak learning rate of $8 \times 10^{-5}$, a linear warm up of 8,000 steps, and a cosine learning rate schedule decaying to $8 \times 10^{-7}$ over 1,200,000 steps. We use a lower batch size early in training to improve training stability, and increase it subsequently to improve efficiency. Specifically, we use an initial batch size of 4M tokens and sequences of length 4,096, and double these values to a batch size of 8M sequences of 8,192 tokens after pre-training 252M tokens. We double the batch size again to 16M after pre-training on 2.87T tokens. We found this training recipe to be very stable: we observed few loss spikes and did not require interventions to correct for model training divergence.

Adjusting the data mix. We made a several adjustments to the pre-training data mix during training to improve model performance on particular downstream tasks. In particular, we increased the percentage of non-English data during pre-training to improve the multilingual performance of Llama 3. We also upsample mathematical data to improve the model's mathematical reasoning performance, we added more recent web data in the later stages of pre-training to advance the model's knowledge cut-off, and we downsampled subsets of the pre-training data that were later identified as being lower quality.

## 3.4.2 Long Context Pre-Training

In the final stages of pre-training, we train on long sequences to support context windows of up to 128K tokens. We do not train on long sequences earlier because the compute in self-attention layers grows quadratically in the sequence length. We increase the supported context length in increments, pre-training until the model has successfully adapted to the increased context length. We assess successful adaptation by measuring whether (1) model performance on short-context evaluations has recovered completely and (2) the model perfectly solves "needle in a haystack" tasks up to that length. In Llama 3 405B pre-training, we increased context length gradually in six stages, starting from the original 8K context window and ending in the final 128K context window. This long-context pre-training stage was performed using approximately 800B training tokens.

![Figure](./grattafioriLlama3Herd2024_images/964281cf-3b75-4775-9ae9-065d3b6d211b-15_576_1543_299_275.jpg)

**Figure 7 Illustration of the overall post-training approach for Llama 3. Our post-training strategy involves rejection sampling, supervised finetuning, and direct preference optimization. See text for details.**

## 3.4.3 Annealing

During pre-training on the final 40M tokens, we linearly annealed the learning rate to 0, maintaining a context length of 128K tokens. During this annealing phase, we also adjusted the data mix to upsample data sources of very high quality; see Section 3.1.3. Finally, we compute the average of model checkpoints (Polyak (1991) averaging) during annealing to produce the final pre-trained model.

