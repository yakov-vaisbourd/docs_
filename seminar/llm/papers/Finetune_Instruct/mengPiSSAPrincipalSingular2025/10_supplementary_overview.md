# The Supplementary Material for The Paper "PiSSA: Principal Singular Values and Singular Vectors Adaptation of Large Language Models."
\begin{itemize}
\item[-] In Section A, we combined PiSSA with two improved LoRA methods, and the experimental results show that these improvements can further enhance the effectiveness of PiSSA.
\item[-] In Section B, we use fast singular value decomposition to initialize PiSSA. The results indicate that the performance of fast singular value decomposition approaches that of SVD decomposition in just several seconds. This ensures that the cost of converting from LoRA/QLoRA to PiSSA/QPiSSA is negligible.
\item[-] In Section C, we demonstrate that the trained PiSSA adapter can be losslessly converted to LoRA, allowing for integration with the original model, facilitating sharing, and enabling the use of multiple PiSSA adapters.
\item[-] In Section D, we explore the experimental effects of using different precisions.
\item[-] In Section E, we discuss the effects of QPiSSA during multiple rounds of SVD decomposition, which can significantly reduce quantization errors without increasing training or inference costs.
\item[-] In Section F, we compare the use of high, medium, and low singular values and vectors to initialize adapters. The experimental results show that initializing adapters with principal singular values and vectors yields the best fine-tuning performance.
\item[-] In Section G, we used a normal distribution function to fit all linear layers of multiple models and calculated their mu and sigma. The experimental results show that after using PiSSA for initialization, the distribution of the remaining models, as described in Section 3 of the main text, is indeed narrower than that of the original models.
\item[-] In Section H, we provide a comprehensive comparison of quantization errors among QLoRA, LoftQ, and QPiSSA, theoretically explaining why QPiSSA reduce quantization errors.
\item[-] In Section I, we combine QPiSSA with various quantization methods beyond Normal Float 4bit, including INT8 and GPTQ. QPiSSA effectively reduces quantization error in these formats, enhancing fine-tuning performance.
\item[-] In Section J, we trained Mistral-7B and Gemma-7B for a sufficient number of steps. The results indicate that PiSSA and LoRA are less prone to overfitting compared to full parameter fine-tuning.
\item[-] In Section K, we offer a more detailed comparison of PiSSA and LoRA at different ranks. It is evident that PiSSA consistently outperforms LoRA in terms of loss convergence, quantization error reduction, and final performance across different ranks.
\item[-] In Section L, we describe the detail setting for NLU task.
\end{itemize}

