# 6 Case Study: Math Word Problems

We evaluate on the popular GSM8K dataset with grade school math questions (Cobbe et al., 2021). We sample 200 and 300 question-answer pairs from the official training set for training and development, respectively. Our final evaluations use the 1.3 k official test set examples. We report extensive comparisons on the development set to avoid overfitting on test. Following prior work on GSM8K, we evaluate the accuracy of the final numerical value that appears in the LM output.

Programs Considered For this task, we consider three simple DSPy programs: a one-step Predict module (vanilla), a two-step ChainOfThought module (CoT), and finally a multi-stage ComparerOfThoughts module (ThoughtReflection). These are fully defined by the following code:
```
vanilla = dspy.Predict("question -> answer") # GSM8K Program 'vanilla'
CoT = dspy.ChainOfThought("question -> answer") # GSM8K Program 'CoT'
class ThoughtReflection(dspy.Module):
    def __init__(self, num_attempts):
        self.predict = dspy.ChainOfThought("question -> answer", n=num_attempts)
        self.compare = dspy.MultiChainComparison('question -> answer', M=num_attempts)
    def forward(self, question):
        completions = self.predict(question=question).completions
        return self.compare(question=question, completions=completions)
reflection = ThoughtReflection(num_attempts=5) # GSM8K Program 'reflection'
```

In reflection, five reasoning chains are sampled from the LM (alongside their answers) and they are compared in parallel by a built-in MultiChainComparison module, which generalizes Yoran et al. (2023). This generates a new answer taking into account the patterns from the five attempts. Critically, the modules used are all generic, none is specific math problems or particular LM.

Compiling As we discussed in Section 4, DSPy programs can be compiled into new, optimized programs. In our experiments, we evaluate the programs zero-shot (no compiling) as well as a number of strategies for compiling. Our simplest compiler is LabeledFewShot:
```
fewshot = dspy.LabeledFewShot(k=8).compile(program, trainset=trainset)
```

Here, program can be any DSPy module. This simply samples $\mathrm{k}=8$ random demonstrations from the trainset for the fields common to the training examples and the signature(s), in this case, question and answer, but not the reasoning for instance. We report the average of 3-5 runs (depending on the setting) when applying such random sampling.

Next, we also consider bootstrapping few-shot examples with random search:
```
tp = BootstrapFewShotWithRandomSearch(metric=gsm8k_accuracy)
bootstrap = tp.compile(program, trainset=trainset, valset=devset)
```

This will generate demonstration chains for examples in the training set and optimize the selection of demonstrations (from this set) to self-improve the program's modules. As the name indicates, this is done with random search, treating the selection of demonstrations as a parameter to optimize.

Next, if desired, this bootstrapping process can be nested in DSPy. In particular, we can use the optimized bootstrap program itself to further bootstrap another program. This is relevant, for example, whenever the original zero-shot program performs relatively poorly.
```
bootstrap2 = tp.compile(program, teacher=bootstrap, trainset=trainset, valset=devset)
```

And lastly, we consider ensembling these bootstraps:
```
l # A program that ensembles the top-7 candidate programs from a bootstrapping compiler run
    (in particular 'bootstrap' or, when applicable, 'bootstrap2') with majority voting.
2 ensemble = Ensemble(reduce_fn=dspy.majority).compile(bootstrap.programs[:7])
```

GSM8K includes human reasoning chains. Above, trainset does not include these reasoning chains. We also evaluate with trainset_human_CoT, which extends the examples in trainset with the human reasoning string. These two datasets can be used interchangeably as the value for the trainset parameter above. We note here that compiling generally runs on the order of minutes (or tens of minutes) as even the more expensive settings only require running the program a few thousand times (e.g., 10-20 trials over 150-300 validation examples) and they can occur in parallel.

Results Our results are summarized in Table 1, which includes dev results as well as our evaluation of promising representatives of each approach on the test set. First, the vanilla program results show that GPT-3.5 and llama2-13b-chat struggle with math word problems when they have to predict the answers directly, that is, without using a reasoning chain first. This is most pronounced in the absence of good demonstrations, which can be seen in the none compilation setting (i.e., zero-shot instruction) and the fewshot setting (i.e., sampling random question-answer pairs). Interestingly, however, vanilla is helped substantially by compiling with bootstrap and by iterating this process into bootstrap×2. On inspecting the prompts bootstrapped (Appendix F), we see that the prompt allows the LM to leverage the answer field for reasoning first, which is permitted as the metric extracts the final numerical value for evaluation.

Next, we consider the CoT program. While the expert human reasoning chains (+human_CoT) provide a large boost when available, we can match or surpass this using bootstrap, substantiating our hypothesis that DSPy can cut the need for hand-crafted prompts. Beyond this, we see that the reflection program, while only a few lines longer than the others, is a clear winner, though CoT is quite effective with ensemble. Overall, the bootstrap compilation procedure leads to large gains for every program, across both LMs. Indeed, all programs in this table are expressed by composing two to four DSPy modules and teleprompters, and they reveal overall that-in the new paradigm prescribed by DSPy-it's composing the right generic modules, rather than manipulating string prompts, that improves different LMs from 4-20\% accuracy to 49-88\% accuracy.

We can informally compare with the following. Zhang et al. (2022) reports $48 \%$ for text-davinci-002, which aligns closely with our llama2-13b-chat results, and reports $59.4 \%$ with codex when employing a manual CoT approach and $62.8 \%$ with an automatic CoT method. Wang et al. (2022b) report 57\% for CoT prompting with PaLM 540-B, which becomes 74\% upon adding self-consistency. The Llama2 authors (Touvron et al., 2023) presents 28.7\% for 1lama2-13b, 42.2\% for 1lama2-34b, and 56.8\% for llama2-70b. Intriguingly, our program with the 13b variant of the model is competitive with their 34b-based results even though we don't use human reasoning chains in our program. Zhao et al. (2023b) reports 80.8\% for CoT with gpt-3.5-turbo from April 2023. The GPT-4 authors (OpenAI, 2023) reports that GPT-3.5 scores 57.1\% and GPT-4 elevates this to 92\% but they note that GPT-4 was in fact pre-trained on a subset of GSM8K's training set.

