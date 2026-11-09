# 5 Experimental Setup

We evaluate the effectiveness of FLARE on 4 diverse knowledge-intensive tasks using few-shot incontext learning (Radford et al., 2019; Brown et al., 2020; Liu et al., 2023). We follow previous works (Trivedi et al., 2022) to sub-sample at most 500 examples from each dataset due to the cost of running experiments. Datasets, metrics, and settings are summarized in Table 7 of Appendix B. The hyperparameters of FLARE are selected based on the development set and listed in Table 9. FLARE refers to FLARE $_{\text {direct }}$ if not specifically stated.

Multihop QA The goal of multihop QA is to answer complex questions through information retrieval and reasoning. We use 2WikiMultihopQA (Ho et al., 2020) which contains 2-hop complex

\footnotetext{
${ }^{4}$ Since KNN-LM uses the contextualized representation corresponding to the current decoding position to retrieve relevant information which encodes all previous tokens. Strictly speaking, $\boldsymbol{q}_{t}$ should be $\boldsymbol{y}_{<t}$.
}
questions sourced from Wikipedia articles that require composition, comparison, or inference, e.g., "Why did the founder of Versus die?" We follow Wang et al. (2022) to generate both the chain-ofthought and the final answer. Experimental setting details are included in Appendix B.

We use regular expressions to extract the final answer from the output and compare it with the reference answer using exact match (EM), and tokenlevel $\mathrm{F}_{1}$, precision, and recall.

Commonsense reasoning Commonsense reasoning requires world and commonsense knowledge to generate answers. We use StrategyQA (Geva et al., 2021) which is a collection of crowdsourced yes/no questions, e.g., "Would a pear sink in water?" We follow Wei et al. (2022) to generate both the chain-of-thought and the final yes/no answer. Details are included in Appendix B.

We extract the final answer and match it against the gold answer using exact match.

Long-form QA Long-form QA aims to generate comprehensive answers to questions seeking complex information (Fan et al., 2019; Stelmakh et al., 2022). We use ASQA (Stelmakh et al., 2022) as our testbed where inputs are ambiguous questions with multiple interpretations, and outputs should cover all of them. For example, "Where do the Philadelphia Eagles play their home games?" could be asking about the city, sports complex, or stadium. We found in many cases it is challenging even for humans to identify which aspect of the question is ambiguous. Therefore, we created another setting (ASQA-hint) where we provide a brief hint to guide LMs to stay on track when generating answers. The hint for the above case is "This question is ambiguous in terms of which specific location or venue is being referred to." Experimental setting details are included in Appendix B.

We use metrics from Stelmakh et al. (2022), including EM, RoBERTa-based QA score (Disambig$\mathrm{F}_{1}$ ), ROUGE (Lin, 2004), and an overall score combining Disambig- $\mathrm{F}_{1}$ and ROUGE (DR).

Open-domain summarization The goal of opendomain summarization is to generate a comprehensive summary about a topic by gathering information from open web (Giorgi et al., 2022). We use WikiAsp (Hayashi et al., 2021) which aims to generate aspect-based summaries about entities from 20 domains in Wikipedia, e.g., "Generate a summary about Echo School (Oregon) including the

![Figure](./jiangActiveRetrievalAugmented2023_images/db4f0e65-0f0e-44fb-a560-94088cacb1ee-07_379_1551_263_258.jpg)

**Figure 4: Comparision between FLARE and baselines across all tasks/datasets. We report the primary metric for each dataset: EM for 2WikiMultihopQA, StrategyQA, and ASQA, and UniEval for WikiAsp.**

following aspects: academics, history." Experimental setting details are included in Appendix B.

Metrics include ROUGE, named entity-based $\mathrm{F}_{1}$, and UniEval (Zhong et al., 2022) which measures factual consistency.

