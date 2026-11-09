\title{
Why Can GPT Learn In-Context? Language Models Implicitly Perform Gradient Descent as Meta-Optimizers
}

\author{
Damai Dai ${ }^{\dagger}$, Yutao Sun ${ }^{\|}$, Li Dong ${ }^{\ddagger}$, Yaru Hao ${ }^{\ddagger}$, Shuming Ma ${ }^{\ddagger}$, Zhifang Sui ${ }^{\dagger}$, Furu Wei ${ }^{\ddagger}$ \\ ${ }^{\dagger}$ MOE Key Lab of Computational Linguistics, Peking University \\ ${ }^{| |}$Tsinghua University ${ }^{\ddagger}$ Microsoft Research \\ \{daidamai,szf\}@pku.edu.cn \\ \{lidong1,fuwei\}@microsoft.com
}

\begin{abstract}
Large pretrained language models have shown surprising in-context learning (ICL) ability. With a few demonstration input-label pairs, they can predict the label for an unseen input without parameter updates. Despite the great success in performance, its working mechanism still remains an open question. In this paper, we explain language models as metaoptimizers and understand in-context learning as implicit finetuning. Theoretically, we figure out that Transformer attention has a dual form of gradient descent. On top of it, we understand ICL as follows: GPT first produces meta-gradients according to the demonstration examples, and then these meta-gradients are applied to the original GPT to build an ICL model. We comprehensively compare the behaviors of in-context learning and explicit finetuning on real tasks to provide empirical evidence that supports our understanding. Experimental results show that in-context learning behaves similarly to explicit finetuning from multiple perspectives. Inspired by the dual form between Transformer attention and gradient descent, we design a momentum-based attention by analogy with gradient descent with momentum. The improved performance over vanilla attention further supports our understanding from another perspective, and more importantly, shows the potential to utilize our understanding for future model design. The code is available at https://aka.ms/icl.
\end{abstract}

# 1 Introduction

In recent years, large pretrained language models, especially in Transformer-based architectures (e.g., GPT; Brown et al. 2020), have shown strong emergent in-context learning (ICL) ability (Wei et al., 2022; Dong et al., 2023). Different from finetuning which needs additional parameter updates, ICL just needs several demonstration examples prepended

\footnotetext{
*Contribution during internship at Microsoft Research.
}

![Figure](./daiWhyCanGPT_images/c0a6693c-d31e-4d1f-975c-196084ab4f6d-01_829_749_733_1069.jpg)

**Figure 1: According to the demonstration examples, GPT produces meta-gradients for in-context learning (ICL) through forward computation. ICL works by applying these meta-gradients to the model through attention. The meta-optimization process of ICL shares a dual view with finetuning that explicitly updates the model parameters with back-propagated gradients.**

before the query input, and then the model can predict labels for unseen inputs. On numerous downstream tasks, large GPT models can achieve surprising performance, which even exceeds smaller models with supervised finetuning. However, although ICL has achieved great performance, its working mechanism is still an open question to be investigated.

In this paper, we explain in-context learning as a process of meta-optimization and analyze connections between GPT-based in-context learning and finetuning. Concentrating on the attention modules, we figure out that the Transformer attention has a dual form of gradient descent. On top of it, we propose a novel perspective to explain in-
context learning: (1) a pretrained GPT serves as a meta-optimizer; (2) it produces meta-gradients according to the demonstration examples through forward computation; (3) the meta-gradients are applied to the original language model through attention to build an ICL model. As illustrated in Figure 1, in-context learning and explicit finetuning share a dual view of gradient descent, where ICL produces meta-gradients through forward computation, while finetuning computes gradients by back-propagation. Therefore, it is reasonable to understand in-context learning as implicit finetuning.

In order to provide empirical evidence to support our understanding, we conduct comprehensive experiments based on real tasks. On six classification tasks, we compare the model predictions, attention outputs, attention weights to query tokens, and attention weights to training tokens between in-context learning and finetuning. Experimental results validate that the behavior of in-context learning is similar to explicit finetuning from multiple perspectives. These results are strong evidence to prove the reasonability of our understanding of in-context learning as implicit finetuning.

Further, inspired by the dual form between Transformer attention and gradient descent, we design a momentum-based attention, which regards the attention values as meta-gradients and applies the momentum mechanism (Polyak, 1964; Sutskever et al., 2013) to them. Experiments on both language modeling and in-context learning show that our momentum-based attention consistently outperforms vanilla attention, which supports our understanding of meta-optimization again from another perspective. We note that beyond this preliminary attempt, our understanding may have more potential to enlighten model design, which is worth investigating in the future.

Our contributions are summarized as follows:
\begin{itemize}
\item[-] We figure out a dual form between Transformer attention and gradient descent, and explain ICL as a process of meta-optimization.
\item[-] We analyze connections between in-context learning and explicit finetuning and propose to understand ICL as implicit finetuning.
\item[-] We provide several lines of empirical evidence to prove that ICL and explicit finetuning behave similarly from multiple perspectives.
\item[-] We design a momentum-based attention and validate its effectiveness, which supports our
\end{itemize} understanding of meta-optimization again and shows the potential of our understanding to enlighten future model design.

# 2 Background

## 2.1 In-Context Learning with GPT

In this paper, we focus on ICL for classification tasks using GPT (Brown et al., 2020). A GPT model is stacked with $L$ identical Transformer (Vaswani et al., 2017) decoder layers where each layer consists of an attention module and a feed-forward network. For a classification task, given a query input text $x$ and a candidate answer set $Y=\left\{y_{1}, y_{2}, \ldots, y_{m}\right\}$, we need to predict a label $\hat{y}$ conditional on $n$ demonstration examples \(C=\left\{\left(x_{1}^{\prime}, y_{1}^{\prime}\right),\left(x_{2}^{\prime}, y_{2}^{\prime}\right), \ldots,\left(x_{n}^{\prime}, y_{n}^{\prime}\right)\right\}\), where \(\left(x_{i}^{\prime}, y_{i}^{\prime}\right)\) is an input-label pair different from the query one. Formally, given a GPT model $\mathcal{M}$, we first compute the probability of each answer $y_{j}$ :
$$
P_{\mathcal{M}}\left(y_{j} \mid C, x\right) .
$$
Since the label space is restricted for classification, we predict the final answer $\hat{y}$ by selecting the answer with the highest probability from the candidate answer set $Y$ :
$$
\hat{y}=\arg \max _{y_{j}} P_{\mathcal{M}}\left(y_{j} \mid C, x\right) .
$$
In practice, we usually use a pre-defined template to format the demonstrations and prepend them before the query input. Let \(\mathcal{T}(\cdot)\) be the function that formats an example, e.g.:
$$
\mathcal{T}(x, y)=\text { Sentence: } x . \text { Sentiment: } y .
$$
The contextual model input $I$ is organized like
$$
\mathcal{T}\left(x_{1}^{\prime}, y_{1}^{\prime}\right) \mathcal{T}\left(x_{2}^{\prime}, y_{2}^{\prime}\right) \ldots \mathcal{T}\left(x_{n}^{\prime}, y_{n}^{\prime}\right) \mathcal{T}\left(x, \_\right) .
$$
Feeding this contextual input into $\mathcal{M}$, the probability of an answer $y_{j}$ is computed as
$$
\begin{gathered}
l_{j}=\mathcal{M}(I) \cdot \mathbf{e}_{y_{j}} \\
P_{\mathcal{M}}\left(y_{j} \mid C, x\right)=\operatorname{softmax}\left(l_{j}\right)
\end{gathered}
$$
where \(\mathcal{M}(I)\) denotes the output hidden state at the last token position; $\mathbf{e}_{y_{j}}$ denotes the output word embedding of $y_{j}$; and $l_{j}$ is the logit corresponding to the $j$-th answer.

## 2.2 Dual Form Between Attention and Linear Layers Optimized by Gradient Descent

The idea in this paper to explain language models as meta-optimizers is inspired by Aizerman et al. (1964); Irie et al. (2022). They present that linear layers optimized by gradient descent have a dual form of linear attention. Let $W_{0}, \Delta W \in \mathbb{R}^{d_{\text {out }} \times d_{\text {in }}}$ be the initialized parameter matrix and the update matrix, respectively, and $\mathbf{x} \in \mathbb{R}^{d_{\text {in }}}$ be the input representation. A linear layer optimized by gradient descent can be formulated as
$$
\mathcal{F}(\mathbf{x})=\left(W_{0}+\Delta W\right) \mathbf{x} .
$$
In the back-propagation algorithm, $\Delta W$ is computed by accumulating the outer products of historic input representations $\mathbf{x}_{i}^{\prime T} \in \mathbb{R}^{d_{\text {in }}}$ and the error signals $\mathbf{e}_{i} \in \mathbb{R}^{d_{\text {out }}}$ of their corresponding outputs:
$$
\Delta W=\sum_{i} \mathbf{e}_{i} \otimes \mathbf{x}_{i}^{\prime},
$$
where $\mathbf{e}_{i}$ is derived from the historic output gradients by multiplying $-\gamma$, the negative learning rate. Combing Equation (7) and Equation (8), we can derive the dual form of linear layers optimized by gradient descent:
$$
\begin{aligned}
\mathcal{F}(\mathbf{x}) & =\left(W_{0}+\Delta W\right) \mathbf{x} \\
& =W_{0} \mathbf{x}+\Delta W \mathbf{x} \\
& =W_{0} \mathbf{x}+\sum_{i}\left(\mathbf{e}_{i} \otimes \mathbf{x}_{i}^{\prime}\right) \mathbf{x} \\
& =W_{0} \mathbf{x}+\sum_{i} \mathbf{e}_{i}\left(\mathbf{x}_{i}^{\prime T} \mathbf{x}\right) \\
& =W_{0} \mathbf{x}+\operatorname{LinearAttn}\left(E, X^{\prime}, \mathbf{x}\right),
\end{aligned}
$$
where \(\operatorname{LinearAttn}(V, K, \mathbf{q})\) denotes the linear attention operation, in which we regard the historic output error signals $E$ as values, the historic inputs $X^{\prime}$ as keys, and the current input $\mathbf{x}$ as the query.

# 3 Understanding In-Context Learning (ICL) as Implicit Finetuning

We first qualitatively analyze the Transformer attention under a relaxed linear attention form to figure out a dual form between it and gradient descent. Then, we compare in-context learning with explicit finetuning to analyze connections between these two optimization forms. Based on these theoretical findings, we propose to understand in-context learning as implicit finetuning.

## 3.1 Understanding Transformer Attention as Meta-Optimization

Let $\mathbf{x} \in \mathbb{R}^{d}$ be the input representation of a query token $t$, and $\mathbf{q}=W_{Q} \mathbf{x} \in \mathbb{R}^{d^{\prime}}$ be the attention query vector. In the ICL setting, the attention result of a head is formulated as
$$
\begin{aligned}
\mathcal{F}_{\mathrm{ICL}}(\mathbf{q}) & =\operatorname{Attn}(V, K, \mathbf{q}) \\
= & W_{V}\left[X^{\prime} ; X\right] \operatorname{softmax}\left(\frac{\left(W_{K}\left[X^{\prime} ; X\right]\right)^{T} \mathbf{q}}{\sqrt{d}}\right)
\end{aligned}
$$
where $W_{Q}, W_{K}, W_{V} \in \mathbb{R}^{d^{\prime} \times d}$ are the projection matrices for computing the attention queries, keys, and values, respectively; $\sqrt{d}$ denotes the scaling factor; $X$ denotes the input representations of query tokens before $t ; X^{\prime}$ denotes the input representations of the demonstration tokens; and $\left[X^{\prime} ; X\right]$ denotes the matrix concatenation. For ease of qualitative analysis, we approximate the standard attention to relaxed linear attention by removing the softmax operation and the scaling factor:
$$
\begin{aligned}
\mathcal{F}_{\mathrm{ICL}}(\mathbf{q}) & \approx W_{V}\left[X^{\prime} ; X\right]\left(W_{K}\left[X^{\prime} ; X\right]\right)^{T} \mathbf{q} \\
& =W_{V} X\left(W_{K} X\right)^{T} \mathbf{q}+W_{V} X^{\prime}\left(W_{K} X^{\prime}\right)^{T} \mathbf{q} \\
& =\widetilde{\mathcal{F}}_{\mathrm{ICL}}(\mathbf{q})
\end{aligned}
$$

We define \(W_{\mathrm{ZSL}}=W_{V} X\left(W_{K} X\right)^{T}\) as the initialized parameters to be updated since $W_{\text {ZSL } \mathbf{q}}$ is the attention result in the zero-shot learning (ZSL) setting, where no demonstrations are given. Following the reverse direction of Equation (9), we derive a dual form of the Transformer attention:
$$
\begin{aligned}
\widetilde{\mathcal{F}}_{\mathrm{ICL}} & (\mathbf{q})=W_{\mathrm{ZSL}} \mathbf{q}+W_{V} X^{\prime}\left(W_{K} X^{\prime}\right)^{T} \mathbf{q} \\
& =W_{\mathrm{ZSL}} \mathbf{q}+\operatorname{LinearAttn}\left(W_{V} X^{\prime}, W_{K} X^{\prime}, \mathbf{q}\right) \\
& =W_{\mathrm{ZSL}} \mathbf{q}+\sum_{i} W_{V} \mathbf{x}_{i}^{\prime}\left(\left(W_{K} \mathbf{x}_{i}^{\prime}\right)^{T} \mathbf{q}\right) \\
& =W_{\mathrm{ZSL}} \mathbf{q}+\sum_{i}\left(\left(W_{V} \mathbf{x}_{i}^{\prime}\right) \otimes\left(W_{K} \mathbf{x}_{i}^{\prime}\right)\right) \mathbf{q} \\
& =W_{\mathrm{ZSL}} \mathbf{q}+\Delta W_{\mathrm{ICL}} \mathbf{q} \\
& =\left(W_{\mathrm{ZSL}}+\Delta W_{\mathrm{ICL}}\right) \mathbf{q} .
\end{aligned}
$$
As shown in the above equations, the attention to the demonstration tokens is equivalent to parameter updates $\Delta W_{\text {ICL }}$ that take effect on $W_{\text {ZSL }}$. In addition, by analogy with $E$ in Equation (9), we regard $W_{V} X^{\prime}$ as meta-gradients, which are used to compute the update matrix $\Delta W_{\text {ICL }}$.

In summary, we explain in-context learning as a process of meta-optimization: (1) a pretrained GPT model serves as a meta-optimizer; (2) it produces meta-gradients according to the demonstration examples through forward computation; (3) through attention, the meta-gradients are applied to the original language model to build an ICL model.

## 3.2 Comparing ICL with Finetuning

Based on the above understanding of in-context learning, we further compare the meta-optimization of in-context learning with the explicit optimization of finetuning to analyze connections between them. Considering that ICL directly takes effect on only the attention keys and values, we design a specific finetuning setting as the compared baseline, which also updates only the parameters for the key and value projection. Also in the relaxed linear attention form, the attention result of a finetuned head is formulated as
$$
\begin{aligned}
\widetilde{\mathcal{F}}_{\mathrm{FT}}(\mathbf{q}) & =\left(W_{V}+\Delta W_{V}\right) X X^{T}\left(W_{K}+\Delta W_{K}\right)^{T} \mathbf{q} \\
& =\left(W_{\mathrm{ZSL}}+\Delta W_{\mathrm{FT}}\right) \mathbf{q},
\end{aligned}
$$
where $\Delta W_{K}$ and $\Delta W_{V}$ denote the parameter updates to $W_{K}$ and $W_{V}$, respectively, which are acquired by back-propagation from task-specific training objectives; and $\Delta W_{\mathrm{FT}}$ is the updates to $W_{\text {ZSL }}$ introduced by finetuning.

For a more fair comparison with in-context learning, we further restrict the finetuning setting as follows: (1) we specify the training examples as the demonstration examples for in-context learning; (2) we train each example for only one step in the same order as demonstrated for in-context learning; (3) we format each training example with the same template used for ICL \(\mathcal{T}\left(x_{i}^{\prime}, y_{i}^{\prime}\right)\) and use the causal language modeling objective for finetuning.

Comparing in-context learning and this finetuning setting, we find that ICL has many properties in common with finetuning. We organize these common properties into the following four aspects.

Both Perform Gradient Descent Comparing Equation (12) and Equation (13), we find that both in-context learning and finetuning introduce updates $\left(\Delta W_{\text {ICL }}\right.$ v.s. \(\left.\Delta W_{\text {FT }}\right)\) to $W_{\text {ZSL }}$, which drive from implicit and explicit gradient descent, respectively. The main difference is that ICL produces meta-gradients by forward computation while finetuning acquires real gradients by back-propagation.

Same Training Information The metagradients of ICL are produced according to the demonstration examples. The gradients of finetuning are also derived from the same training examples. That is to say, in-context learning and finetuning share the same source of training information.

Same Causal Order of Training Examples Incontext learning and our finetuning setting share the same causal order of training examples. ICL uses decoder-only Transformers so the subsequent tokens in the demonstrations will not affect the preceding ones. For our finetuning setting, we use the same order of training examples and train only one epoch, so we can also guarantee that the subsequent examples have no effect on the preceding ones.

Both Aim at Attention Compared with zeroshot learning, the direct effect of in-context learning and our finetuning are both restricted to the computation of attention keys and values. For ICL, the model parameters are unchanged and it encodes demonstration information into additional keys and values to change the attention behavior. For finetuning, due to our restriction, the training information can be introduced to only the projection matrices for attention keys and values as well.

Considering the above common properties between in-context learning and finetuning, we show that it is reasonable to understand in-context learning as implicit finetuning. In the rest of this paper, we compare ICL and explicit finetuning empirically from multiple perspectives to provide quantitative results to support this understanding.

# 4 Experiments

## 4.1 Experimental Settings

We analyze two off-the-shelf pretrained GPT models with 1.3 billion and 2.7 billion model parameters, respectively, which are released by fairseq ${ }^{1}$. In the rest of this paper, we call them GPT 1.3B and GPT 2.7B for short. All experiments are conducted on NVIDIA V100 GPUs with 32 GB memory.

For each task, we use the same template to format examples for zero-shot learning (ZSL), finetuning (FT), and in-context learning (ICL). Details of the templates used for each task are provided in Appendix A. The answer prediction processes for ZSL and finetuning are the same with ICL as described in Section 2.1, except that they do not have demonstration examples.

For in-context learning, we fix the max number of demonstration examples to 32 and tune the random seed for each task to find a set of demonstration examples that achieves the best validation performance. For explicit finetuning, we use the same demonstration examples for in-context learning as the training examples and use SGD as the optimizer. For a fair comparison, we fine-tune the

\footnotetext{
${ }^{1}$ https://github.com/facebookresearch/fairseq
}

\begin{table}
\begin{tabular}[t]{|l|l|l|l|l|l|l|}
\hline & SST2 & SST5 & MR & Subj & AGNews & CB \\
\hline \# Validation Examples & 872 & 1101 & 1066 & 2000 & 7600 & 56 \\
\hline \# Label Types & 2 & 5 & 2 & 2 & 4 & 3 \\
\hline ZSL Accuracy (GPT 1.3B) & 70.5 & 39.3 & 65.9 & 72.6 & 46.3 & 37.5 \\
\hline FT Accuracy (GPT 1.3B) & 73.9 & 39.5 & 73.0 & 77.8 & 65.3 & 55.4 \\
\hline ICL Accuracy (GPT 1.3B) & 92.7 & 45.0 & 89.0 & 90.0 & 79.2 & 57.1 \\
\hline ZSL Accuracy (GPT 2.7B) & 71.4 & 35.9 & 60.9 & 75.2 & 39.8 & 42.9 \\
\hline FT Accuracy (GPT 2.7B) & 76.9 & 39.1 & 80.0 & 86.1 & 65.7 & 57.1 \\
\hline ICL Accuracy (GPT 2.7B) & 95.0 & 46.5 & 91.3 & 90.3 & 80.3 & 55.4 \\
\hline
\end{tabular}

**Table 1: Statistics of six classification datasets (rows 1-2) and validation accuracy in the zero-shot learning (ZSL), finetuning (FT), and in-context learning (ICL) settings on these datasets (rows 3-8).**
\end{table}

\begin{table}
\begin{tabular}[t]{|l|l|l|l|l|l|l|l|}
\hline Model & SST2 & SST5 & MR & Subj & AGNews & CB & Average \\
\hline GPT 1.3B & 91.84 & 66.67 & 97.08 & 87.17 & 83.08 & 87.50 & 85.56 \\
\hline GPT 2.7B & 96.83 & 71.60 & 95.83 & 87.63 & 84.44 & 100.00 & 89.39 \\
\hline
\end{tabular}

**Table 2: Rec2FTP for two GPT models on six datasets. From the perspective of model prediction, ICL can cover most of the correct behavior of finetuning.**
\end{table}
model for only one epoch and the training examples are provided in the same order as demonstrated for in-context learning. We tune the learning rate for finetuning and select the one that achieves the best validation performance. Details of the search range and selected value for the random seeds and learning rates are shown in Appendix B.

## 4.2 Evaluation Datasets

We compare in-context learning and finetuning based on six datasets spanning three sorts of classification tasks. SST2 (Socher et al., 2013), SST5 (Socher et al., 2013), MR (Pang and Lee, 2005) and Subj (Pang and Lee, 2004) are four datasets for sentiment classification; AG-News (Zhang et al., 2015) is a topic classification dataset; and CB (De Marneffe et al., 2019) is used for natural language inference. Statistics of the number of validation examples and label types are summarized in Table 1.

For reference, we present the validation accuracy in the ZSL, finetuning, and ICL settings on six classification datasets in Table 1. Compared with ZSL, ICL and finetuning both achieve considerable improvements, which means the optimizations they make are both helpful to these downstream tasks.

## 4.3 ICL Covers Most of Correct Predictions of Finetuning

We compute a recall to finetuning prediction (Rec2FTP) to measure ICL can cover how much behavior of finetuning from the perspective of the model prediction. We first count $N_{\text {FT>ZSL }}$, the number of query examples that finetuning can predict correctly but ZSL cannot. Then, among these examples, we count \(N_{\text {(FT>ZSL) } \wedge(\text { ICL }>\text { ZSL })}\), the number that ICL can also predict correctly. Finally, we compute the Rec2FTP score as \(\frac{N_{(\text {FT> ZSL }) \wedge(\text { ICL }>\text { ZSL })}}{N_{\text {FT }>\text { ZSL }}}\). A higher Rec2FTP score suggests that ICL covers more correct behavior of finetuning from the perspective of the model prediction.

We show the Rec2FTP scores for two GPT models on six datasets in Table 2. As shown in the table, on average, ICL can correctly predict more than 85\% of the examples that finetuning can correct from ZSL. These results indicate that from the perspective of model prediction, ICL can cover most of the correct behavior of finetuning.

## 4.4 ICL Tends to Change Attention Outputs in the Same Direction as Finetuning

From the perspective of representation, we compute a similarity of the attention output updates (SimAOU) to measure the similarity between the updates that ICL and finetuning make. For a query example, let \(\mathbf{h}_{\mathrm{X}}^{(l)}\) denote the normalized output rep-

\begin{table}
\begin{tabular}[t]{|l|l|l|l|l|l|l|l|l|}
\hline Model & Metric & SST2 & SST5 & MR & Subj & AGNews & CB & Average \\
\hline \multirow{2}{*}{GPT 1.3B} & SimAOU (Random $\Delta$ ) & 0.002 & 0.003 & 0.001 & 0.002 & 0.002 & 0.003 & 0.002 \\
\hline & SimAOU ( $\Delta \mathrm{FT}$ ) & 0.110 & 0.080 & 0.222 & 0.191 & 0.281 & 0.234 & 0.186 \\
\hline \multirow{2}{*}{GPT 2.7B} & SimAOU (Random $\Delta$ ) & 0.000 & -0.002 & 0.000 & 0.001 & -0.002 & 0.000 & -0.001 \\
\hline & SimAOU ( $\Delta \mathrm{FT}$ ) & 0.195 & 0.323 & 0.157 & 0.212 & 0.333 & 0.130 & 0.225 \\
\hline
\end{tabular}

**Table 3: SimAOU for two GPT models on six datasets. ICL updates are much more similar to finetuning updates than to random updates. From the perspective of representation, ICL tends to change attention output representations in the same direction as finetuning changes.**
\end{table}

\begin{table}
\begin{tabular}[t]{|l|l|l|l|l|l|l|l|l|}
\hline Model & Metric & SST2 & SST5 & MR & Subj & AGNews & CB & Average \\
\hline \multirow{2}{*}{GPT 1.3B} & SimAM (Before Finetuning) & 0.555 & 0.391 & 0.398 & 0.378 & 0.152 & 0.152 & 0.338 \\
\hline & SimAM (After Finetuning) & 0.585 & 0.404 & 0.498 & 0.490 & 0.496 & 0.177 & 0.442 \\
\hline \multirow{2}{*}{GPT 2.7B} & SimAM (Before Finetuning) & 0.687 & 0.380 & 0.314 & 0.346 & 0.172 & 0.228 & 0.355 \\
\hline & SimAM (After Finetuning) & 0.687 & 0.492 & 0.347 & 0.374 & 0.485 & 0.217 & 0.434 \\
\hline
\end{tabular}

**Table 4: SimAM for two models on six datasets. From the perspective of attention behavior, compared with attention weights before finetuning, ICL is more inclined to generate similar attention weights to those after finetuning.**
\end{table}
resentation of the last token at the $l$-th attention layer in setting X. The updates of ICL and finetuning compared with ZSL are \(\mathbf{h}_{\text {ICL }}^{(l)}-\mathbf{h}_{\text {ZSL }}^{(l)}\) and \(\mathbf{h}_{\mathrm{FT}}^{(l)}-\mathbf{h}_{\mathrm{ZSL}}^{(l)}\), respectively. We compute the cosine between these two updates to get SimAOU ( $\Delta \mathbf{F T}$ ) at the $l$-th layer. A higher SimAOU \((\Delta \mathrm{FT})\) means ICL is more inclined to update the attention output in the same direction as finetuning. For comparison, we also compute a baseline metric called SimAOU (Random $\Delta$ ) that computes the similarity between ICL updates and randomly generated updates.

We present the SimAOU scores averaged across examples and layers for two GPT models on six datasets in Table 3. From the table, we find that SimAOU (Random $\Delta$ ) is always around zero, while SimAOU ( $\Delta \mathrm{FT}$ ) remains much more positive. These results indicate that ICL updates are much more similar to finetuning updates than to random updates. From the perspective of representation, we prove that ICL tends to change the attention outputs in the same direction as finetuning.

## 4.5 ICL Is Inclined to Generate Similar Attention Weights to Finetuning

From the perspective of attention behavior, we compute a similarity of the attention map (SimAM) to measure the similarity of the attention map to query tokens for ICL and finetuning. For a query example, let \(\mathbf{m}_{\mathrm{X}}^{(l, h)}\) denote the attention weights before softmax of the last token at the $h$-th attention head in the $l$-th attention layer in setting X. For ICL, we omit the attention to the demonstration tokens and only monitor the attention weights to the query tokens. First, before finetuning, we compute the cosine between \(\mathbf{m}_{\text {ICL }}^{(l, h)}\) and \(\mathbf{m}_{\text {ZSL }}^{(l, h)}\) and then average the similarity across attention heads to get SimAM (Before Finetuning) at each layer. Similarly, after finetuning, we compute the cosine between \(\mathbf{m}_{\text {ICL }}^{(l, h)}\) and \(\mathbf{m}_{\mathrm{FT}}^{(l, h)}\) to get SimAM (After Finetuning). A higher SimAM (After Finetuning) over SimAM (Before Finetuning) indicates that the attention behavior of ICL is more similar to a finetuned model than a non-finetuned one.

Table 4 demonstrates the SimAM scores averaged across examples and layers for two GPT models on six datasets. We observe that compared with attention weights before finetuning, ICL is more inclined to generate similar attention weights to attention weights after finetuning. Again, from the perspective of attention behavior, we prove that ICL behaves similarly to finetuning.

## 4.6 ICL and Finetuning Tend to Pay Similar Attention to Training Tokens

Since we understand ICL as a process of metaoptimization, we also compare the attention to training tokens for ICL and finetuning with the Kendall rank correlation coefficient (Kendall, 1948). For a query example, let \(\mathbf{m}_{\text {ICL }}^{(l)}\) denote the ICL attention weights to the demonstration tokens

\begin{table}
\begin{tabular}[t]{|l|l|l|l|l|l|l|l|l|}
\hline Model & Metric & SST2 & SST5 & MR & Subj & AGNews & CB & Average \\
\hline \multirow{2}{*}{GPT 1.3B} & Kendall (ICL, Random) & 0.000 & -0.001 & 0.000 & 0.001 & -0.001 & 0.000 & 0.000 \\
\hline & Kendall (ICL, FT) & 0.192 & 0.151 & 0.173 & 0.181 & 0.190 & 0.274 & 0.193 \\
\hline \multirow{2}{*}{GPT 2.7B} & Kendall (ICL, Random) & -0.001 & 0.000 & 0.000 & 0.000 & 0.000 & -0.001 & 0.000 \\
\hline & Kendall (ICL, FT) & 0.213 & 0.177 & 0.264 & 0.203 & 0.201 & 0.225 & 0.214 \\
\hline
\end{tabular}

**Table 5: Kendall rank correlation coefficients for two GPT models on six datasets. Compared with random attention weights, ICL attention weights to training tokens are much more similar to finetuning attention weights.**
\end{table}
of the last query token in the $l$-th attention layer, which is summed across attention heads. For finetuning, we first record all the attention queries \(Q^{\prime(l, h)} \in \mathbb{R}^{d^{\prime} \times N}\) of the training tokens, and then use the inner product between them and the attention query \(\mathbf{q}^{(l, h)} \in \mathbb{R}^{d^{\prime}}\) of the last token in the query example as the finetuning attention weights to the training tokens: \(\mathbf{m}_{\mathrm{FT}}^{(l)}=\sum_{h} Q^{\prime(l, h)^{T}} \mathbf{q}^{(l, h)}\), which is also summed across attention heads. The Kendall coefficient between \(\mathbf{m}_{\text {ICL }}^{(l)}\) and \(\mathbf{m}_{\text {FT }}^{(l)}\) is computed as Kendall (ICL, FT) \(=\frac{P_{c}-P_{d}}{N(N-1) / 2}\), where $N$ denotes the number of training tokens, $P_{c}$ denotes the number of concordant pairs, and $P_{d}$ denotes the number of discordant pairs. A higher Kendall coefficient means that the orders of attention weights to training tokens of ICL and finetuning are more similar. For comparison, we also compute the Kendall coefficient between \(\mathbf{m}_{\text {ICL }}^{(l)}\) and randomly generated attention weights \(\mathbf{m}_{\text {Random }}^{(l)}\), which we call Kendall (ICL, Random).

Table 5 shows the Kendall correlation coefficients averaged across examples and layers for two GPT models on six datasets. We find that Kendall (ICL, Random) is always near zero, while Kendall (ICL, FT) always maintains a distinctly positive value. These results suggest that ICL and finetuning tend to pay similar attention to training tokens.

# 5 Momentum-Based Attention Inspired by Dual Form of Transformer Attention

We have figured out the dual form between Transformer attention and gradient descent. As illustrated in Figure 2, inspired by this dual view, we investigate whether we can utilize momentum (Polyak, 1964; Sutskever et al., 2013), a widely used technique for optimization algorithms, to improve Transformer attention.

Gradient descent with momentum averages gra-

![Figure](./daiWhyCanGPT_images/c0a6693c-d31e-4d1f-975c-196084ab4f6d-07_269_755_726_1063.jpg)

**Figure 2: Inspired by the dual form between attention and gradient descent, we introduce the momentum mechanism into Transformer attention by analogy with gradient descent with momentum.**

dients among timestamps:
$$
\Theta_{t}=\Theta_{t-1}-\gamma \sum_{i=1}^{t-1} \eta^{t-i} \nabla f_{\Theta_{i}},
$$
where $\gamma$ is the learning rate and $\eta$ is a scalar between 0 and 1. As stated in Section 3.1, the attention values serve as meta-gradients. By analogy with gradient descent with momentum, we try to use Exponential Moving Average (EMA; Hunter 1986) to average the attention values to build the momentum-based attention:
$$
\begin{aligned}
\operatorname{MoAttn}\left(V, K, \mathbf{q}_{t}\right) & =\operatorname{Attn}\left(V, K, \mathbf{q}_{t}\right)+\operatorname{EMA}(V) \\
& =V \operatorname{softmax}\left(\frac{K^{T} \mathbf{q}_{t}}{\sqrt{d}}\right)+\sum_{i=1}^{t-1} \eta^{t-i} \mathbf{v}_{i}
\end{aligned}
$$
where $\mathbf{v}_{i}$ is the $i$-th attention value vector. The momentum of attention value vectors explicitly strengthens the recency bias of attention, which has been shown helpful for language modeling (Press et al., 2022). Therefore, we assume that introducing momentum into attention will contribute to faster convergence and better performance.

Experiments on Language Modeling First, we evaluate the effect of momentum-based attention on language modeling. We train two GPT models with 350 M parameters from scratch, where one is the vanilla Transformer, and another applies momentum to attention. More training details are provided in Appendix C. We evaluate the perplexity

\begin{table}
\begin{tabular}[t]{|l|l|l|l|l|}
\hline Model & Train ${ }_{1024}$ & Valid ${ }_{256}$ & Valid ${ }_{512}$ & Valid ${ }_{1024}$ \\
\hline Transformer & 17.61 & 19.50 & 16.87 & 15.14 \\
\hline Transformer ${ }_{\text {MoAttn }}$ & 17.55 & 19.37 & 16.73 & 15.02 \\
\hline
\end{tabular}

**Table 6: Perplexity on the training set and validation sets with different input lengths for language modeling. Momentum-based attention achieves a consistent perplexity improvement compared with the vanilla Transformer.**
\end{table}

\begin{table}
\begin{tabular}[t]{|l|l|l|l|l|l|l|l|}
\hline Model & SST5 & IMDB & MR & CB & ARC-E & PIQA & Average \\
\hline Transformer & 25.3 & 64.0 & 61.2 & 43.9 & 48.2 & 68.7 & 51.9 \\
\hline Transformer ${ }_{\text {MoAttn }}$ & 27.4 & 70.3 & 64.8 & 46.8 & 50.0 & 69.0 & 54.7 \\
\hline
\end{tabular}

**Table 7: Accuracy on six in-context learning datasets. Introducing momentum into attention improves the accuracy of the vanilla Transformer by 2.8 on average.**
\end{table}
of these two models on the training set and three validation sets with input lengths of 256, 512, and 1024, respectively. The results are shown in Table 6. On all of the validation sets, applying momentum to attention introduces a consistent perplexity improvement compared with the vanilla Transformer.

Experiments on In-Context Learning We also evaluate the in-context learning ability of the above language models to verify the effectiveness of momentum-based attention on downstream tasks. We consider six datasets for sentiment analysis (SST5 (Socher et al., 2013), IMDB (Maas et al., 2011), and MR (Pang and Lee, 2005)), natural language inference (CB (De Marneffe et al., 2019)), and multi-choice selection (ARC-E (Clark et al., 2018) and PIQA (Bisk et al., 2020)). For all of these datasets, we use up to 32 examples as demonstrations. As shown in Table 7, compared with vanilla Transformer, using momentum-based attention achieves consistently higher accuracy on all of these datasets.

The performance improvements on both language modeling and in-context learning prove our deduction that introducing momentum will improve Transformer attention. From another perspective, these results further support our understanding of Transformer attention as meta-optimization.

# 6 Related Work

Recently, some pieces of work have attempted to understand the inference mechanism of in-context learning. Xie et al. (2022) explain in-context learning as implicit Bayesian inference. They state that in-context learning emerges when language models can infer the shared latent concept among the demonstration examples, which is learned during pretraining. On another aspect, Olsson et al. (2022) focus on specific modules in Transformers. They find some induction heads in Transformers that refer to abstract patterns in previous sequences to help predict the next token. They indicate that the induction heads drive the ability of in-context learning. Different from them, we concentrate on the learning algorithm of ICL and explain it as a process of meta-optimization.

Some other work also studies the learning algorithm of ICL. As a case study, Garg et al. (2022) show that Transformers can be trained to in-context learn a class of linear functions and the performance is comparable to the least squares estimator. Based on linear regression, Akyürek et al. (2022) prove that they can construct parameters of Transformers to implement gradient-descent-based learning algorithms. Further, they show that models trained with an in-context learning objective tend to match the behavior of models computed by explicit learning algorithms. Also based on regression tasks, von Oswald et al. (2022) show that linear attention-only Transformers with constructed parameters that implement gradient descent and models learned by an in-context learning objective are highly related. Compared with them, we are the first ones to explain in-context learning in real scenarios. To be specific, (1) we analyze in-context learning for off-the-shelf GPT models, instead of models trained from scratch by an ICL objective; (2) our experiments are based on real NLP tasks, instead of toy ones like linear regression.

# 7 Conclusion

In this paper, we aim to explain the working mechanism of GPT-based ICL. Theoretically, we figure
out a dual form between Transformer attention and gradient descent, and propose to understand ICL as a process of meta-optimization. Further, we analyze connections between ICL and explicit finetuning and show the reasonability to regard ICL as implicit finetuning. Empirically, we comprehensively compare ICL and finetuning based on six real NLP tasks. The results prove that ICL behaves similarly to explicit finetuning from multiple perspectives. Further, inspired by our understanding of meta-optimization, we design a momentum-based attention that achieves consistent performance improvements over vanilla attention. We believe our understanding will have more potential to enlighten ICL applications and model design in the future.

# Limitations

Although the ability of in-context learning has been found for different architectures (e.g., Transformer and LSTM), we consider only Transformer-based in-context learning in this paper because Transformer is the current mainstream architecture of NLP. However, as for in-context learning itself, figuring out how it works for other architectures is also a meaningful problem, which we encourage to study in the future.

As for the dual form we point out between Transformer attention and gradient descent, we consider a relaxed form of linear attention for qualitative analysis. Although the experimental results support our understanding well, the mechanism of standard Transformer attention without approximation may be more complex and should be studied more clearly in the future.

As for empirical experiments, our analysis needs to record a large number of intermediate results (e.g., attention output representations, and attention weights to query tokens and demonstration tokens) for thousands of validation examples. Considering the storage space and computational cost of analysis, we only analyze GPT models with up to 2.7B parameters and leave larger models such as GPT 13B for future work. In addition, for the clarity of the problem definition and the convenience of experiments, our analysis is based on only classification tasks. Although classification is a representative application of in-context learning, other tasks like multiple choice and open-ended generation are not considered in this paper and could be investigated in the future.

# Acknowledgement

Damai Dai and Zhifang Sui are supported by the National Key Research and Development Program of China 2020AAA0106700 and NSFC project U19A2065.

# References

Mark A Aizerman, Emmanuil M Braverman, and Lev I Rozonoer. 1964. Theoretical foundation of potential functions method in pattern recognition. Avtomatika i Telemekhanika, 25(6):917-936.

Ekin Akyürek, Dale Schuurmans, Jacob Andreas, Tengyu Ma, and Denny Zhou. 2022. What learning algorithm is in-context learning? investigations with linear models. CoRR, abs/2211.15661.

Yonatan Bisk, Rowan Zellers, Ronan Le Bras, Jianfeng Gao, and Yejin Choi. 2020. PIQA: reasoning about physical commonsense in natural language. In The Thirty-Fourth AAAI Conference on Artificial Intelligence, AAAI 2020, pages 7432-7439. AAAI Press.

Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. 2020. Language models are few-shot learners. In Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020.

Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. 2018. Think you have solved question answering? try arc, the AI2 reasoning challenge. CoRR, abs/1803.05457.

Marie-Catherine De Marneffe, Mandy Simons, and Judith Tonhauser. 2019. The commitmentbank: Investigating projection in naturally occurring discourse. In proceedings of Sinn und Bedeutung, volume 23, pages 107-124.

Qingxiu Dong, Lei Li, Damai Dai, Ce Zheng, Zhiyong Wu, Baobao Chang, Xu Sun, Jingjing Xu, Lei Li, and Zhifang Sui. 2023. A survey for in-context learning. CoRR, abs/2301.00234.

Shivam Garg, Dimitris Tsipras, Percy Liang, and Gregory Valiant. 2022. What can transformers learn incontext? A case study of simple function classes. CoRR, abs/2208.01066.

J Stuart Hunter. 1986. The exponentially weighted moving average. Journal of quality technology, 18(4):203-210.

Kazuki Irie, Róbert Csordás, and Jürgen Schmidhuber. 2022. The dual form of neural networks revisited: Connecting test time predictions to training patterns via spotlights of attention. In International Conference on Machine Learning, ICML 2022, volume 162 of Proceedings of Machine Learning Research, pages 9639-9659. PMLR.

Maurice George Kendall. 1948. Rank correlation methods.

Louis Kirsch, James Harrison, Jascha Sohl-Dickstein, and Luke Metz. 2022. General-purpose in-context learning by meta-learning transformers. CoRR, abs/2212.04458.

Louis Kirsch and Jürgen Schmidhuber. 2021. Meta learning backpropagation and improving it. In Advances in Neural Information Processing Systems 34: Annual Conference on Neural Information Processing Systems 2021, NeurIPS 2021, pages 14122-14134.

Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts. 2011. Learning word vectors for sentiment analysis. In Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies, pages 142-150, Portland, Oregon, USA. Association for Computational Linguistics.

Catherine Olsson, Nelson Elhage, Neel Nanda, Nicholas Joseph, Nova DasSarma, Tom Henighan, Ben Mann, Amanda Askell, Yuntao Bai, Anna Chen, Tom Conerly, Dawn Drain, Deep Ganguli, Zac Hatfield-Dodds, Danny Hernandez, Scott Johnston, Andy Jones, Jackson Kernion, Liane Lovitt, Kamal Ndousse, Dario Amodei, Tom Brown, Jack Clark, Jared Kaplan, Sam McCandlish, and Chris Olah. 2022. In-context learning and induction heads. CoRR, abs/2209.11895.

Bo Pang and Lillian Lee. 2004. A sentimental education: Sentiment analysis using subjectivity summarization based on minimum cuts. In Proceedings of the 42nd Annual Meeting of the Association for Computational Linguistics (ACL-04), pages 271-278, Barcelona, Spain.

Bo Pang and Lillian Lee. 2005. Seeing stars: Exploiting class relationships for sentiment categorization with respect to rating scales. In ACL 2005, 43rd Annual Meeting of the Association for Computational Linguistics, Proceedings of the Conference, pages 115-124. The Association for Computer Linguistics.

Boris T Polyak. 1964. Some methods of speeding up the convergence of iteration methods. Ussr computational mathematics and mathematical physics, 4(5):1-17.

Ofir Press, Noah A. Smith, and Mike Lewis. 2022. Train short, test long: Attention with linear biases enables input length extrapolation. In The Tenth International Conference on Learning Representations, ICLR 2022. OpenReview.net. Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, Christopher D. Manning, Andrew Ng, and Christopher Potts. 2013. Recursive deep models for semantic compositionality over a sentiment treebank. In Proceedings of the 2013 Conference on Empirical Methods in Natural Language Processing, pages 1631-1642, Seattle, Washington, USA. Association for Computational Linguistics.

Ilya Sutskever, James Martens, George E. Dahl, and Geoffrey E. Hinton. 2013. On the importance of initialization and momentum in deep learning. In Proceedings of the 30th International Conference on Machine Learning, ICML 2013, volume 28 of JMLR Workshop and Conference Proceedings, pages 1139-1147. JMLR.org.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in Neural Information Processing Systems, pages 5998-6008. Curran Associates, Inc.

Johannes von Oswald, Eyvind Niklasson, Ettore Randazzo, João Sacramento, Alexander Mordvintsev, Andrey Zhmoginov, and Max Vladymyrov. 2022. Transformers learn in-context by gradient descent. ArXiv preprint, abs/2212.07677.

Jason Wei, Yi Tay, Rishi Bommasani, Colin Raffel, Barret Zoph, Sebastian Borgeaud, Dani Yogatama, Maarten Bosma, Denny Zhou, Donald Metzler, Ed H. Chi, Tatsunori Hashimoto, Oriol Vinyals, Percy Liang, Jeff Dean, and William Fedus. 2022. Emergent abilities of large language models. CoRR, abs/2206.07682.

Sang Michael Xie, Aditi Raghunathan, Percy Liang, and Tengyu Ma. 2022. An explanation of in-context learning as implicit bayesian inference. In The Tenth International Conference on Learning Representations, ICLR 2022. OpenReview.net.

Xiang Zhang, Junbo Jake Zhao, and Yann LeCun. 2015. Character-level convolutional networks for text classification. In Advances in Neural Information Processing Systems 28: Annual Conference on Neural Information Processing Systems 2015, pages 649-657.

# Appendix

# A Templates for In-Context Learning

We demonstrate the templates used to format examples and the candidate answer sets for six classification datasets used in our experiments in Table 8.

# B Hyper-Parameters for In-Context Learning and Finetuning

We perform grid search to find the best random seed for ICL and the best learning rate for finetuning. The search range for all the datasets is the same. For random seeds, we search in $\{1,2,3,4,5,6,7\}$. For learning rates, the search base values are $\{1,2,3,4,5,6,7,8,9\}$ and we scale them to 0.1 , $0.01,0.001$, and 0.0001 times, i.e., we have $9 \times 4=$ 36 values to search. As an exception, for GPT 1.3B finetuned on SST5, we perform a more fine-grained search and finally set its learning rate to 0.00016 since the finetuned model cannot outperform the zero-shot learning with the above 36 learning rates.

In Table 9, we present the details of the selected random seeds and learning rates for two GPT models on six classification datasets.

# C Hyper-Parameters for Training Language Models from Scratch

The hyper-parameters for training two language models from scratch are summarized in Table 10.

\begin{table}
\begin{tabular}[t]{|l|l|l|}
\hline Dataset & Template & Candidate Answer Set \\
\hline SST2 & Sentence: \{Sentence\} Label: \{Label\} & \{ Negative, Positive \} \\
\hline SST5 & Sentence: \{Sentence\} Label: \{Label\} & \{ terrible, bad, neutral, good, great \} \\
\hline MR & Review: \{Sentence\} Sentiment: \{Label\} & \{ Negative, Positive \} \\
\hline Subj & Input: \{Sentence\} Type: \{Label\} & \{ objective, subjective \} \\
\hline AGNews & Classify the news articles into the categories of World, Sports, Business, and Technology. News: \{Sentence\} Type: \{Label\} & \{ World, Sports, Business, Technology \} \\
\hline CB & \{Premise\} Question: \{Hypothesis\} True, False, or Neither? Answer: \{Label\} & \{ True, False, Neither \} \\
\hline
\end{tabular}

**Table 8: Formatting templates and candidate answer sets for six classification datasets.**
\end{table}

\begin{table}
\begin{tabular}[t]{|l|l|l|l|}
\hline Hyper-Parameter & Dataset & GPT 1.3B & GPT 2.7B \\
\hline \multirow{6}{*}{Random Seed} & SST2 & 2 & 7 \\
\hline & SST5 & 5 & 5 \\
\hline & MR & 5 & 1 \\
\hline & Subj & 4 & 4 \\
\hline & AGNews & 3 & 3 \\
\hline & CB & 3 & 3 \\
\hline \multirow{6}{*}{Learning Rate} & SST2 & 0.0005 & 0.007 \\
\hline & SST5 & 0.00016 & 0.04 \\
\hline & MR & 0.003 & 0.001 \\
\hline & Subj & 0.003 & 0.002 \\
\hline & AGNews & 0.2 & 0.2 \\
\hline & CB & 0.08 & 0.01 \\
\hline
\end{tabular}

**Table 9: Selected random seeds and learning rates for two GPT models on six classification datasets.**
\end{table}

\begin{table}
\begin{tabular}[t]{|l|l|}
\hline Hyper-parameter & Value \\
\hline Embedding \& Hidden Dimension & 1024 \\
\hline FFN Inner Hidden Dimension & 4096 \\
\hline Number of Attention Heads & 16 \\
\hline Number of Transformer Layers & 24 \\
\hline Number of Parameters & 350M \\
\hline Sequence Length & 1024 \\
\hline Batch Size & 512K Tokens \\
\hline Optimizer & Adam \\
\hline Adam Betas & (0.9, 0.98) \\
\hline Adam Epsilon & $1 \mathrm{e}-6$ \\
\hline Maximum Learning Rate & 3e-4 \\
\hline Learning Rate Scheduler & Polynomial Decay \\
\hline Total Training Steps & 500 K \\
\hline Warm-up Steps & 20 K \\
\hline Gradient Clip Norm & 2.0 \\
\hline
\end{tabular}

**Table 10: Hyper-parameters for training two language models from scratch.**
\end{table}
