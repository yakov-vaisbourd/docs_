# 1 Introduction

Many applications in natural language processing rely on adapting one large-scale, pre-trained language model to multiple downstream applications. Such adaptation is usually done via fine-tuning, which updates all the parameters of the pre-trained model. The major downside of fine-tuning is that the new model contains as many parameters as in the original model. As larger models are trained every few months, this changes from a mere "inconvenience" for GPT-2 (Radford et al., b) or RoBERTa large (Liu et al., 2019) to a critical deployment challenge for GPT-3 (Brown et al., 2020) with 175 billion trainable parameters ${ }^{1}$

Many sought to mitigate this by adapting only some parameters or learning external modules for new tasks. This way, we only need to store and load a small number of task-specific parameters in ad-

![Figure](./huLoRALowRankAdaptation2021_images/2f8c4be1-43fa-4600-99ef-0945e0605540-01_359_400_1824_1351.jpg)

**Figure 1: Our reparametrization. We only train $A$ and $B$.**

dition to the pre-trained model for each task, greatly boosting the operational efficiency when deployed. However, existing techniques

\footnotetext{
${ }^{*}$ Equal contribution.
${ }^{0}$ Compared to V1, this draft includes better baselines, experiments on GLUE, and more on adapter latency.
${ }^{1}$ While GPT-3 175B achieves non-trivial performance with few-shot learning, fine-tuning boosts its performance significantly as shown in Appendix A
}
often introduce inference latency (Houlsby et al., 2019; Rebuffi et al., 2017) by extending model depth or reduce the model's usable sequence length (Li \& Liang, 2021; Lester et al., 2021; Hambardzumyan et al., 2020; Liu et al., 2021) (Section 3). More importantly, these method often fail to match the fine-tuning baselines, posing a trade-off between efficiency and model quality.

We take inspiration from Li et al. (2018a); Aghajanyan et al., (2020) which show that the learned over-parametrized models in fact reside on a low intrinsic dimension. We hypothesize that the change in weights during model adaptation also has a low "intrinsic rank", leading to our proposed Low-Rank Adaptation (LoRA) approach. LoRA allows us to train some dense layers in a neural network indirectly by optimizing rank decomposition matrices of the dense layers' change during adaptation instead, while keeping the pre-trained weights frozen, as shown in Figure 1. Using GPT-3 175 B as an example, we show that a very low rank (i.e., $r$ in Figure 1 can be one or two) suffices even when the full rank (i.e., $d$ ) is as high as 12,288, making LoRA both storage- and compute-efficient.

LoRA possesses several key advantages.
\begin{itemize}
\item[-] A pre-trained model can be shared and used to build many small LoRA modules for different tasks. We can freeze the shared model and efficiently switch tasks by replacing the matrices $A$ and $B$ in Figure 1, reducing the storage requirement and task-switching overhead significantly.
\item[-] LoRA makes training more efficient and lowers the hardware barrier to entry by up to 3 times when using adaptive optimizers since we do not need to calculate the gradients or maintain the optimizer states for most parameters. Instead, we only optimize the injected, much smaller low-rank matrices.
\item[-] Our simple linear design allows us to merge the trainable matrices with the frozen weights when deployed, introducing no inference latency compared to a fully fine-tuned model, by construction.
\item[-] LoRA is orthogonal to many prior methods and can be combined with many of them, such as prefix-tuning. We provide an example in Appendix E.
\end{itemize}

Terminologies and Conventions We make frequent references to the Transformer architecture and use the conventional terminologies for its dimensions. We call the input and output dimension size of a Transformer layer $d_{\text {model }}$. We use $W_{q}, W_{k}, W_{v}$, and $W_{o}$ to refer to the query/key/value/output projection matrices in the self-attention module. $W$ or $W_{0}$ refers to a pretrained weight matrix and $\Delta W$ its accumulated gradient update during adaptation. We use $r$ to denote the rank of a LoRA module. We follow the conventions set out by (Vaswani et al., 2017; Brown et al., 2020) and use Adam (Loshchilov \& Hutter, 2019; Kingma \& Ba, 2017) for model optimization and use a Transformer MLP feedforward dimension $d_{f f n}=4 \times d_{\text {model }}$.

