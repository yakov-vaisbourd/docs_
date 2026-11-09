# D. Additional Experimental Results

In this section, we provide additional experimental results that are not covered in the main body.

# D.1. Ablation Studies on GP-VL

As mentioned in Section 6, we observe an abnormal phenomenon that SFT fails to achieve comparable indistribution performance with RL (see Figure 5 subplot row 1 column 3). To further explore this, we conduct ablation studies over different hyperparameter choices.

SFT. We ablate the hyperparameter choices under the same task setting of GP-VL in Section 5.1. For experiments fine-tuning all parameters, we search learning rates from $\left\{1 \times 10^{-4}, 1 \times 10^{-4}, 1 \times 10^{-5}, 1 \times 10^{-6}, 5 \times 10^{-7}, 1 \times\right.$ $\left.10^{-7}\right\}$. Freezing the vision encoder, we search learning rates $\left\{1 \times 10^{-6}, 1 \times 10^{-7}\right\}$. Freezing vision encoder and adapter, we search learning rates $\left\{1 \times 10^{-6}, 5 \times 10^{-7}, 1 \times\right.$ $\left.10^{-7}\right\}$. We provide the in-distribution success rate curve in Figure 16.

RL. Finding suitable hyperparameters for RL experiments requires minimal effort. We conduct a search over learning rates $2 \times 10^{-6}, 1 \times 10^{-6}$, with the in-distribution success rate curves shown in Figure 17. All parameters are tunable in our RL experiments.

# D.2. More results on V-IRL-VL

Echoing per-step accuracy results in Figure 5, we report the overall success rate of V-IRL-VL in Figure 18. Due to the task's complexity, both training methods achieve overall success rates no higher than 1\%. For V-IRL, the overall

![Figure](./chuSFTMemorizesRL2025_images/c29435c9-0689-4d6d-8ce6-4fa35255cad0-19_412_690_878_1129.jpg)

**Figure 17: Ablation studies on GeneralPoints-VL RL. Echoing Figure 16, we ablate the learning rate and rreport the in-distribution episode success rate (\%) of the two experiments. All components are tunable here.**

success rate is a significantly more demanding metric since it aggregates per-step errors. For example, a random policy achieving 10\% per-step accuracy would achieve achieve only approximately $10^{-8} \%$ success rate on enough routes averaging 10 steps in length.

# D.3. Failure Cases

In this section, we present 2 failure cases in our experiments as mentioned in Sections 5.4 and 6.

Without SFT, RL fails. In Figure 9, we present the training dynamics of failed RL experiments without SFT initialization. We additionally provide output examples of these experiments in Figure 20, where the model tends to generate unstructured response and fail.

RL cannot save overfitted checkpoints. As shown in Figure 19, RL cannot recover the out-of-distribution performance when initialized from a extremely overfitted checkpoint that has an initial per-step accuracy of less than 1\%. We additionally provide an output example in Figure 19, where the model fails to adjust to the new rule.

![Figure](./chuSFTMemorizesRL2025_images/c29435c9-0689-4d6d-8ce6-4fa35255cad0-20_451_756_284_193.jpg)

**Figure 18: Overall success rate (\%) - GFLOPs for V-IRL-VL under rule variants. Due to the nature of the task requiring aggregating a trajectory of correct actions, neither training method achieves reasonable outof-distribution performance.**

![Figure](./chuSFTMemorizesRL2025_images/c29435c9-0689-4d6d-8ce6-4fa35255cad0-20_458_756_281_1121.jpg)

**Figure 19: Out-of-distribution per-step accuracy (\%) - GFLOPs for V-IRL-VL under rule variants with overfitted initial checkpoint. Evaluation metric details can be found in Appendix C.3.**

\begin{lstlisting}[mathescape=true]
System Prompt ( $\boldsymbol{v}_{0}^{\text {in }}$ )
[Task Description]
You are an expert 24 points card game player. You are observing these four cards in the image. Note that 'J', 'Q',
and 'K' count as '11', '12', and '13' respectively, and each card must be used once. Your goal is to output a formula
that evaluates to 24 using numbers from the cards and operators such as '+', '-', '*', '/', '(', ')', and '='.
[Input]
Cards: ['2', 'J', '3', 'K']
[Output]
Your response should be a valid json file in the following format:
{
"cards": [x, y, z, w], where 'J', 'Q', and 'K' count as '10',
"number": [a, b, c, d], where a, b, c, and d are the numbers on the cards,
"formula": "an equation that equals 24",
}
Appending model and verifier outputs to obtain $\boldsymbol{v}_{t}^{\text {in }}$
\(\boldsymbol{v}_{t}^{\text {in }}=\left[\boldsymbol{v}_{0}^{\text {out }}, \boldsymbol{v}_{0}^{\text {ver }}, \boldsymbol{v}_{1}^{\text {out }}, \boldsymbol{v}_{1}^{\text {ver }}, \ldots, \boldsymbol{v}_{t-1}^{\text {out }}, \boldsymbol{v}_{t-1}^{\text {ver }}\right] \quad \quad \triangleright \boldsymbol{v}_{t}^{\text {in }}=\operatorname{concat}\left(\boldsymbol{v}_{0}^{\text {in }},\left[\boldsymbol{v}_{k}^{\text {out }}, \boldsymbol{v}_{k}^{\text {ver }}\right]_{k=0}^{t-1}\right)\)
Failed model output ( $\boldsymbol{v}_{t}^{\text {out }}$ )
To solve this problem, we can use a brute force approach by generating all possible combinations of the cards and
checking if any of them can be used to form an equation that equals 24 .
Here is a Python solution for this problem:
"spython
def calculate_formula(cards):
# Define the mapping of cards to their corresponding numbers
⋯
Verifier Output ( $\boldsymbol{v}_{t}^{\text {ver }}$ )
You failed this trial because your formula is incorrect. \(\quad \square \boldsymbol{v}_{t+1}^{\text {in }}=\operatorname{concat}\left(\boldsymbol{v}_{t}^{\text {in }}, \boldsymbol{v}_{t}^{\text {out }}, \boldsymbol{v}_{t}^{\text {ver }}\right)\)
\end{lstlisting}

Figure 20: Example model outputs without SFT initialization. We record model responses using prompts similar to the one shown in Figure 11. The results demonstrate that Llama-3.2-Vision-11B fails to follow instructions properly. We omit the long response which tries to solve the puzzle via code but fails to finish within finite context length.

System Prompt ( $\boldsymbol{v}_{0}^{\text {in }}$ )
[Task Description]
You are an expert in navigation. You will receive a sequence of instructions to follow while observing your surrounding street views. You are also provided with your observation and action history in text. Your goal is to first analyze the instruction and identify the next sentence to be executed. Then, you need to provide the action to be taken based on the current observation and instruction.
[Instruction]
\begin{itemize}
\item[1.] First, turn right to face northwest.
\item[2.] Move forward until you reach next intersection where Korean War Memorial is on your left.
\end{itemize}
![](./chuSFTMemorizesRL2025_images/c29435c9-0689-4d6d-8ce6-4fa35255cad0-21_344_344_406_1427.jpg)
\begin{itemize}
\item[3.] Turn left to face southwest.
\item[4.] Move forward until you reach next intersection where Korean War Memorial is on your left behind.
\item[5.] Turn right to face north.
\item[6.] Move forward until you reach next intersection.
\item[7.] Turn left to face east.
\item[8.] Move forward until you reach next intersection.
\item[9.] Turn left to face north.
\item[10.] Move forward until you reach next intersection.
\item[11.] Turn right to face east.
\item[12.] Move forward until you reach next intersection.
\item[13.] Turn left to face northeast.
\item[14.] Move forward until you reach next intersection where 9/11 Memorial \& Museum is on your left.
\item[15.] Turn right to face northwest.
\item[16.] Move forward until you reach destination where The destination 9/11 Memorial \& Museum is on your right front.
\end{itemize}
[Action space]
"forward()": indicates moving forward one step
"turn_direction(x)": indicates adjust the ego agent direction towards x direction. x could be any following ['left', 'right', 'slightly left', 'slightly right']
"stop()": indicates the navigation is finished.
[Observations and actions sequence]
O_1: You observe an image of 4 views;
A_1:
[Output]
\{
"current observation": latest observation from the streetview grid,
"current instruction": analyze the full instruction and identify the sentence to be executed,
"action": the action to be taken chosen from the action space,
\}
Appending model and verifier outputs to obtain $\boldsymbol{v}_{t}^{\text {in }}$
$\boldsymbol{v}_{t}^{\text {in }}=\left[\boldsymbol{v}_{0}^{\text {out }}, \boldsymbol{v}_{0}^{\text {ver }}, \boldsymbol{v}_{1}^{\text {out }}, \boldsymbol{v}_{1}^{\text {ver }}, \ldots, \boldsymbol{v}_{t-1}^{\text {out }}, \boldsymbol{v}_{t-1}^{\text {ver }}\right]$ \(\triangleright \boldsymbol{v}_{t}^{\text {in }}=\operatorname{concat}\left(\boldsymbol{v}_{0}^{\text {in }},\left[\boldsymbol{v}_{k}^{\text {out }}, \boldsymbol{v}_{k}^{\text {ver }}\right]_{k=0}^{t-1}\right)\)

Failed model output \(\left(\boldsymbol{v}_{t}^{\text {out }}\right)\)
\{
"current observation": "No landmarks nearby",
"current instruction": "Turn right to face northwest.",
"action": "turn_direction(northwest)",
\}
Verifier Output ( $\boldsymbol{v}_{t}^{\text {ver }}$ )
Incorrect action. - Fail this sample.

Figure 21: Failed example of V-IRL transition due to overfitting. This phenomenon happens more frequently during scaling up supervised fine-tuning.
