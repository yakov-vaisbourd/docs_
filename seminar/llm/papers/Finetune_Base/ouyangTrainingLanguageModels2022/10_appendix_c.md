# C Additional model details

All model architectures use the GPT-3 architecture (Brown et al., 2020). For the reward models and value functions, the unembedding layer of the original model is replaced with a projection layer to output a scalar value. All models use fp16 weights and activations, with fp32 master copies of weights. The same byte pair encodings as in Brown et al. (2020) are used for all models. All our language models and RL policies have a context length of 2 k tokens. We filter out prompts that are longer than 1k tokens and limit the maximum response length to 1k tokens.

All models are trained with the Adam optimizer, with $\beta_{1}=0.9$ and $\beta_{2}=0.95$.

# C. 1 Details of SFT training

We train our SFT models for 16 epochs with residual dropout of 0.2 . We use a cosine LR schedule down to 10\% of the original learning rate, with no learning rate warmup. For our 1.3B and 6B models, we use an LR of 9.65e-6 and a batch size of 32. For 175B, we use a LR of 5.03e-6 and a batch size of 8. To select learning rates, we did a geometric search over 7 LRs for 1.3B and 6B, and 5 LRs for 175B. We also tuned the number of epochs using geometric search. Our final models

\begin{table}

**Table 13: Labeler satisfaction survey**
\begin{tabular}[t]{|l|l|}
\hline \multicolumn{2}{|c|}{It was clear from the instructions what I was supposed to do.} \\
\hline Strongly agree & 57.9\% \\
\hline Agree & 42.1\% \\
\hline Neither agree nor disagree & 0\% \\
\hline Disagree & 0\% \\
\hline Strongly disagree & 0\% \\
\hline \multicolumn{2}{|c|}{I found the task enjoyable and engaging.} \\
\hline Strongly agree & 57.9\% \\
\hline Agree & 36.8\% \\
\hline Neither agree nor disagree & 5.3\% \\
\hline Disagree & 0\% \\
\hline Strongly disagree & 0\% \\
\hline \multicolumn{2}{|c|}{I found the task repetitive.} \\
\hline Strongly agree & 0\% \\
\hline Agree & 31.6\% \\
\hline Neither agree nor disagree & 31.6\% \\
\hline Disagree & 36.8\% \\
\hline Strongly disagree & 0\% \\
\hline \multicolumn{2}{|c|}{I was paid fairly for doing the task.} \\
\hline Strongly agree & 47.4\% \\
\hline Agree & 42.1\% \\
\hline Neither agree nor disagree & 10.5\% \\
\hline Disagree & 0\% \\
\hline Strongly disagree & 0\% \\
\hline \multicolumn{2}{|c|}{Overall, I'm glad I did this task.} \\
\hline Strongly agree & 78.9\% \\
\hline Agree & 21.1\% \\
\hline Neither agree nor disagree & 0\% \\
\hline Disagree & 0\% \\
\hline Strongly disagree & 0\% \\
\hline
\end{tabular}
\end{table}
were selected based on the RM score, which we've found to be more predictive of human preference results compared to validation loss.

# C. 2 Details of RM training

We trained a single 6B reward model which we used for all PPO models of all sizes. Larger 175B RMs had the potential to achieve lower validation loss, but (1) their training was more unstable which made them less suitable for use as initializations for the PPO value functions, and (2) using a 175B RM and value function greatly increase the compute requirements of PPO. In preliminary experiments, we found that 6B RMs were stable across a wide range of learning rates, and led to equally strong PPO models.

The final reward model was initialized from a 6B GPT-3 model that was fine-tuned on a variety of public NLP datasets (ARC, BoolQ, CoQA, DROP, MultiNLI, OpenBookQA, QuAC, RACE, and Winogrande). This was mostly for historical reasons; we find similar results when initializing the RM from the GPT-3 or SFT models. We trained for a single epoch over the full reward model training set (see Table 6) at a learning rate of $\operatorname{lr}=9 \mathrm{e}-6$, a cosine learning rate schedule (dropping to $10 \%$ of its initial value by the end of training), and a batch size of 64. Training did not appear to be very sensitive to the learning rate or schedule; changes of up to 50\% in the learning rate resulted in similar performance. Training was quite sensitive to the number of epochs: multiple epochs quickly overfit the model to the training data with obvious deterioration in the validation loss. The batch size here represents the distinct number of prompts per batch. Each prompt had between $K=4$ and $K=9$
labeled completions, from which there were up to $\binom{K}{2}$ possible comparisons. Ties were dropped. Therefore, a single batch could contain up to $64 \times\binom{ K}{2} \leq 2,304$ comparisons.

# C. 3 Details of the initialization models for RLHF

We initialize the RLHF models from a pretrained GPT-3 model and apply supervised fine-tuning for 2 epochs on the demonstration dataset. We also mix in 10\% pretraining data during fine-tuning, since we find it helpful for PPO training (see Appendix E. 11 for details). Cosine learning rate schedule is used and the learning rate eventually decays to 10\% of the peak learning rate. We use a batch size of 32 for 1.3 B and 6 B models and 8 for the 175 B model. We compare a few different peak learning rates for each model and pick the one with low losses on both the demonstration and the pretraining validation datasets. A log linear sweep of 5 values of the LR's are compared for 1.3B and 6 B models and 3 values are compared for the 175 B model. The resultant LR's for the $1.3 \mathrm{~B}, 6 \mathrm{~B}$, and 175B models are 5e-6, 1.04e-5 and 2.45e-6, respectively.

# C. 4 Details of RLHF training

We then initialize the RL policies from the above supervised fine-tuned models with pretraining mix. These models are also used to compute the KL reward, in the same way as Stiennon et al. (2020), with $\beta=0.02$ (see Equation 2). We train all the RL models for 256k episodes. These episodes include about 31 k unique prompts, after filtering out prompts with PII and deduplication based on common prefixes. The batch size for each iteration is 512, with a minibatch size of 64. In other words, each batch is randomly split into 8 minibatches and is trained on for only a single inner epoch (Schulman et al., 2017). A constant learning rate is applied with a warmup over the first 10 iterations, starting with one tenth of the peak learning rate. Exponential moving averages of the weights are applied, with a decay rate of 0.992 . No discount is applied when estimating the generalized advantage (Schulman et al., 2016). The PPO clip ratio is set to 0.2, and the sampling temperature is 1 for rollouts.
As previously mentioned, for all PPO models we use a 6B RM and a 6B value function, and the latter is initialized from the former. By using the same 6 B reward model and value function on policies of all model sizes, it's easier to compare the effect of policy model size on policy performance. A fixed learning rate of 9e-6 for the value function is used for 1.3B and the 6B policies and 5e-6 for the 175B policy.

Our initial RLHF experiments showed regressions on public NLP datasets, such as SQuADv2 and DROP, and we mitigate the regressions by mixing in pretraining gradients during PPO training. We use 8 times more pretraining examples than the number of the RL training episodes. The pretraining data is randomly drawn from the dataset used to train the GPT-3 models. For each minibatch, we compute the PPO gradients and pretraining gradients in consecutive steps and accumulate them both into the gradient buffers. We multiply the pretraining gradients by a coefficient, $\gamma=27.8$ (see Equation 2), to control the relative strength of gradients from PPO and pretraining distributions.

# C. 5 FLAN and T0 models

We obtain our FLAN and T0 baselines by fine-tuning a 175B GPT-3 model on the FLAN and T0 datasets. For T0, note that we trained on the T0++ version of the dataset. Because T0 contains much more data (96M datapoints) than FLAN (1.2M datapoints), we subsampled T0 to 1 million datapoints to make the amount of training data comparable for each model. Note that the original models train on epochs where datapoints can be repeated, but in our epochs we go through every datapoint without repeats (to better match the way we trained our SFT baselines). We applied a cosine learning rate schedule, and try initial learning rates of 4e-6 and 6e-6 for each dataset. The learning rate decays to 10\% of its peak at the end of training, and we use a batch size of 64 for both experiments.

To choose the best FLAN checkpoint, we use our 6B reward model to score the completions on the validation set of prompts. As shown in Figure 13, the reward saturates after the initial 400k examples of training. This indicates that training for even longer will unlikely improve the human eval performance. We picked the checkpoint with the highest RM score for our human evaluation, which is the one trained with learning rate of 4e-6 and for 896k examples.
We perform two similar experiments to find the best T0 checkpoint. In one experiment, we used a batch size of 128, a learning rate of 4e-6 and 1.28 million examples. The other experiment used a

![Figure](./ouyangTrainingLanguageModels2022_images/ba26513b-feb1-4cca-a1ca-3c06344fb35e-43_799_1324_258_382.jpg)

**Figure 13: Tuning FLAN and T0 based on reward model scores**

batch size of 64, a learning rate of 6e-6 and 1 million examples. Once again using the reward model score, we picked the checkpoint from the former experiment after 896k examples of training.

