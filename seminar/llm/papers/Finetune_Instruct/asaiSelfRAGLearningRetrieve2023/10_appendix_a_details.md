# A Self-Rag Details

# A. 1 Reflection Tokens.

Definitions of reflection tokens. Below, we provide a detailed definition of reflection type and output tokens. The first three aspects will be provided at each segment level, while the final aspect is only given at each output level.
\begin{itemize}
\item[-] Retrieval-on-demand (Retrieve): Given an input and previous-step generation (if applicable), an LM determines whether the continuation requires factual grounding. No indicates retrieval is unnecessary as the sequence does not require factual grounding or may not be enhanced by knowledge retrieval, Yes indicates retrieval is necessary. We additionally have continue to use evidence, which indicates that a model can continue to use the evidence retrieved previously. For instance, a passage may contain rich factual information, and thus Self-Rag generates multiple segments based on the passage.
\item[-] Relevant ( IsREL ): Retrieved knowledge may not be always relevant to the input. This aspect indicates whether the evidence provides useful information (Relevant) or not (Irrelevant).
\item[-] Supported ( Issup ): Attribution is the concept of whether the output is fully supported by certain evidence (Menick et al., 2022; Bohnet et al., 2022). This aspect judges how much information in the output is entailed by the evidence. We evaluate attributions in three scale, Fully supported, Partially supported, and No support / Contradictory, following Yue et al. (2023); Nakano et al. (2021).
\item[-] Useful ( IsUSE ): Following the definitions from Liu et al. (2023a), we define the perceived utility as whether the response is a helpful and informative answer to the query, independently from whether it is in fact factual or not. This can be also viewed as plausibility in Menick et al. (2022). For usefulness, we use a five-scale evaluation (1 is the lowest and 5 is the highest).
\end{itemize}

Details of GPT-4-based data collections. We use the instruction and demonstration pairs to prompt GPT-4, listed in Section D. Following an official recommendation, we separate instructions and outputs with "\#\#". We use the temperature 1 and set the maximum output token counts to be 200. We discard instances where GPT-4 does not follow the designated output formats or output sequences that do not match our expected category names. As a result, we collected 1,2594 for Retrieve, 11,181 for Issup , 19,317 for relevance, 3,831 for utility.

Manual analysis of the GPT-4 predictions. The authors of this paper manually assess randomly sampled 20 instances for each aspect and check if GPT-4 predictions match their assessments given the same instruction, demonstrations, and test instances. We found our assessments show high agreement with GPT-4 predictions, especially for relevance (95\%), retrieval necessity (95\%), and the degree of support (90\%). Agreement was slightly lower in usefulness (80\%), mostly due to the disagreement between 1 and 2 or 4 and 5 .

# A. 2 Self-Rag Training

Overview of training. Algorithm 2 provides a high-level overview of our training.
Full list of seed datasets. To sample diverse input-output pairs, we sample instances of the OpenInstruct (Wang et al., 2023) dataset. In particular, we use their ShareGPT, GPT-4 Alpaca, Alpaca, OpenAssistant, and FLAN subsets subsets. We also sample instances from a couple of knowledgeintensive datasets, Natural Questions (Kwiatkowski et al., 2019), Wizard of Wikipedia (Dinan et al., 2019) and FEVER (Thorne et al., 2018) from the KILT benchmark (Petroni et al., 2021), ASQA (Stelmakh et al., 2022) and multiple QA datasets including ARC-Easy and OpenBookQA (Mihaylov et al., 2018). Table 3 shows the full list of training instances, and in total, we use 145,619 instances.

Performance of the Critic $\mathcal{C}$. We evaluate the accuracy of reward predictions by splitting GPT-4 generated feedback into training, development, and test sets. The accuracy of the reward model is as follows. Table 5 shows the model performance of predicting GPT-4 judgments. As you can see, overall our fine-tuned reward model shows high prediction matching with GPT-4 predicted feedback.
\begin{lstlisting}[mathescape=true]
Algorithm 2 Self-Rag Training
Input input-output data $\mathcal{D}=\{X, Y\}$, generator $\mathcal{M}, \mathcal{C} \theta$
Initialize $\mathcal{C}$ with a pre-trained LM
Sample data $\left\{X^{\text {sample }}, Y^{\text {sample }}\right\} \sim\{X, Y\} \quad \triangleright$ Training Critic LM (Section 3.2.1)
for \((x, y) \in\left(X^{\text {sample }}, Y^{\text {sample }}\right)\) do $\quad \square$ Data collections for $\mathcal{C}$
Prompt GPT-4 to collect a reflection token $r$ for \((x, y)\)
Add \(\{(x, y, r)\}\) to $\mathcal{D}_{\text {critic }}$
Update $\mathcal{C}$ with next token prediction loss □ Critic learning; Eq. 1
Initialize $\mathcal{M}$ with a pre-trained LM $\quad>$ Training Generator LM (Section 3.2.2)
for \((x, y) \in(X, Y) \mathbf{d o} \quad \triangleright\) Data collection for $\mathcal{M}$ with $\mathcal{D}_{\text {critic }}$
Run $\mathcal{C}$ to predict $r$ given \((x, y)\)
Add \((x, y, r)\) to $\mathcal{D}_{g e n}$
Update $\mathcal{M}$ on $\mathcal{D}_{\text {gen }}$ with next token prediction loss $\quad \triangleright$ Generator LM learning; Eq. 2
\end{lstlisting}

\begin{table}
\begin{tabular}[t]{|l|l|l|l|}
\hline Dataset name & category & Data source & the number of instances \\
\hline GPT-4 Alpaca & Instruction-following & Open-Instruct & 26,168 \\
\hline Stanford Alpaca & Instruction-following & Open-Instruct & 25,153 \\
\hline FLAN-V2 & Instruction-following & Open-Instruct & 17,817 \\
\hline ShareGPT & Instruction-following & Open-Instruct & 13,406 \\
\hline Open Assistant 1 & Instruction-following & Open-Instruct & 9,464 \\
\hline Wizard of Wikipedia & Knowledge-intensive & KILT & 17,367 \\
\hline Natural Questions & Knowledge-intensive & KILT & 15,535 \\
\hline FEVER & Knowledge-intensive & KILT & 9,966 \\
\hline OpenBoookQA & Knowledge-intensive & HF Dataset & 4,699 \\
\hline Arc-Easy & Knowledge-intensive & HF Dataset & 2,147 \\
\hline ASQA & Knowledge-intensive & ASQA & 3,897 \\
\hline
\end{tabular}

**Table 3: The generator LM $\mathcal{M**$ training data statistics.}
\end{table}

\begin{table}
\begin{tabular}[t]{|l|l|l|l|l|}
\hline base LM & Retrieve & IsSup & IsRel & IsUse \\
\hline Llama2-7B & 93.8 & 93.5 & 80.2 & 73.5 \\
\hline FLAN-3B & 85.6 & 73.1 & 82.0 & 72.1 \\
\hline
\end{tabular}

**Figure 5: Reward prediction accuracy using GPT-4 predictions as ground-truth predictions.**
\end{table}
While our final model uses Llama2-7B as a base LM, we also train and compare FLAN-3B (Wei et al., 2022) model on the same data, to investigate the effectiveness of different data sizes affect final reward predictions. In most aspects, our reward model shows higher than 80\% accuracy, indicating the powerful ability of fine-tuned specialized LMs to evaluate text. While both models show relatively lower performance on IsUSE , this is because both models often confuse between the two highest cases (5 and 4), where human annotators can also disagree.

Details of $\mathcal{M}$ data creation. Here, we provide detailed data creation procedures. Algorithm 3 summarizes the process. Here we set $y_{t}$ to $y$ for simplification. Once we train the critic model, we first run it on input data from the aforementioned datasets, to predict whether retrieval is needed or not. For the instances where the critic predicts Retrieve $=\mathrm{NO}$, we only predict the IsUSE given input and output. For the instances where the critic predicts Retrieve = Yes, we first retrieve passages using the input and the entire output as queries, to find passages that are relevant to the entire output. We then split output sentences using Spacy ${ }^{7}$ For each sentence, we run $\mathcal{C}$ to predict whether the retrieval is necessary or not, given the input, preceding segments, and the initial retrieved passage. If $\mathcal{C}$ predicts Retrieve $=\mathrm{N} \circ$, then do not insert any paragraph at the $t$ th segment. If $\mathcal{C}$ predicts Retrieve $=\mathrm{Yes}$, then we use the original input and the $t$ th segment as a retrieval query to find relevant passages for the $t$-th segment. For each retrieved passage, we predict ISREL and ISSUP . If there is any passage and continuation with Isrel =Relevant and Issup =Fully Supported / IsSup =Partially

\footnotetext{
https://spacy.io/
}

Supported, then we sample it as the continuation. If there is more than one passage satisfying this criterion, we use the one with the highest retrieval score. If there are only IsREL =Irrelevant or IsSUP =No Support passages, we randomly sample one passage.
\begin{lstlisting}[mathescape=true]
Algorithm $3 \mathcal{M}_{\text {gen }}$ Data creation
Input Input-output data $\mathcal{D}=X, Y$
for \((x, y) \in\{X, Y\}\) do
Given \((x, y) \mathcal{C}\) predicts Retrieve
if Retrieve is predicted then
Retrieve relevant passages $\mathbf{D}$ using $\mathcal{R}$ given \((x, y) \quad \triangleright\) Retrieve passages
for $d \in \mathbf{D}$ do
$\mathcal{C}$ predicts IsREL for each $d$ △ Predict relevance of passages
$\mathcal{C}$ predicts Issup for each \((y, d) \quad\) ▷ Predict supports of outputs
$\mathcal{C}$ predicts IsUSE for each $d$ > Predict overall utility ( $t=\mathcal{T}$ only)
Sample $d$
else if Retrieve is not predicted then
$\mathcal{C}$ predicts IsUSE given $x, y$
Add augmented \((x, y, d, r)\) to $\mathcal{D}_{g e n}$
\end{lstlisting}

Training examples. Table 4 show several training examples used for $\mathcal{M}$ training.

# A. 3 Self-Rag Inference

Details of beam-search score calculations. We first compute scores for each critique type by taking the normalized probabilities of desirable tokens. For Isrel, we compute the score as follows:
$$
s(\text { ISREL })=\frac{p(\text { ISREL }=\text { RELEVANT })}{p(\text { ISREL }=\text { RELEVANT })+p(\text { ISREL }=\text { IRRELEVANT })} .
$$
For Issup , we compute the score as follows:
$$
s(\boxed{\text { ISREL }})=\frac{p(\text { ISSUP }=\text { FULLY })}{S}+0.5 \times \frac{p(\text { ISSUP }=\text { PARTIALLY })}{S},
$$
where $S=\sum_{t \in\{\text { Fully, Partially, No }\}} p($ IsSup \(=t)\). For IsUSE where we have a five-scale score, we compute the weighted sum of the scores. We assigns weighted scores of $w=\{-1,-0.5,0,0.5,1\}$ to the tokens IsUSE $=\{1,2,3,4,5\}$, and compute the final scores as follows:
$$
s(\text { ISUSE })=\sum_{i}^{5} w_{i} \frac{p(\text { ISUSE }=i)}{S},
$$
where $S=\sum_{t \in\{1,2,3,4,5\}} p($ ISUSE \(=t)\).
Details of adaptive retrieval. For retrieval based on soft constraints, we trigger retrieval if the following condition is satisfied:
$$
\frac{p(\text { Retrieve }=\mathrm{YES})}{p(\text { Retrieve }=\mathrm{YES})+p(p(\text { Retrieve }=\mathrm{NO})}>\delta .
$$

