# 3 The DSPy Programming Model

We present DSPy, which treats LMs as abstract devices for text generation, ${ }^{3}$ and optimizes their usage in arbitrary computational graphs. DSPy programs are expressed in Python: each program takes the task input (e.g., a question to answer or a paper to summarize) and returns the output (e.g., an answer or a summary) after a series of steps. DSPy contributes three abstractions toward automatic optimization: signatures, modules, and teleprompters. Signatures abstract the input/output behavior of a module; modules replace existing hand-prompting techniques and can be composed in arbitrary pipelines; and teleprompters optimize all modules in the pipeline to maximize a metric.

\footnotetext{
${ }^{3}$ We assume access to one or more LMs, which consume a prompt string and return text completions. This may be a promptable LM capable of in-context learning (e.g., GPT-3.5 or Llama2-7b) or a smaller finetuneable LM (e.g., T5-base). An LM may be selected as the default; operations will use it unless configured otherwise.
}

## 3.1 Natural Language Signatures can abstract prompting \& finetuning

Instead of free-form string prompts, DSPy programs use natural language signatures to assign work to the LM. A DSPy signature is natural-language typed declaration of a function: a short declarative spec that tells DSPy what a text transformation needs to do (e.g., "consume questions and return answers"), rather than how a specific LM should be prompted to implement that behavior. More formally, a DSPy signature is a tuple of input fields and output fields (and an optional instruction). A field consists of field name and optional metadata. ${ }^{4}$ In typical usage, the roles of fields are inferred by DSPy as a function of field names. For instance, the DSPy compiler will use in-context learning to interpret question differently from answer and will iteratively refine its usage of these fields.

Signatures offer two benefits over prompts: they can be compiled into self-improving and pipelineadaptive prompts or finetunes. This is primarily done by bootstrapping (Sec 4) useful demonstrating examples for each signature. Additionally, they handle structured formatting and parsing logic to reduce (or, ideally, avoid) brittle string manipulation in user programs.

In practice, DSPy signatures can be expressed with a shorthand notation like question -> answer, so that line 1 in the following is a complete DSPy program for a basic question-answering system (with line 2 illustrating usage and line 3 the response when GPT-3.5 is the LM):
```
qa = dspy.Predict("question -> answer")
qa(question="Where is Guaraní spoken?")
# Out: Prediction(answer='Guaraní is spoken mainly in South America.')
```

In the shorthand notation, each field's name indicates the semantic role that the input (or output) field plays in the transformation. DSPy will parse this notation and expand the field names into meaningful instructions for the LM, so that english_document -> french_translation would prompt for English to French translation. When needed, DSPy offers more advanced programming interfaces for expressing more explicit constraints on signatures (Appendix A).

## 3.2 Parameterized \& templated Modules can abstract prompting techniques

Akin to type signatures in programming languages, DSPy signatures simply define an interface and provide type-like hints on the expected behavior. To use a signature, we must declare a module with that signature, like we instantiated a Predict module above. A module declaration like this returns a function having that signature.

The Predict Module The core module for working with signatures in DSPy is Predict (simplified pseudocode in Appendix D.1). Internally, Predict stores the supplied signature, an optional LM to use (initially None, but otherwise overrides the default LM for this module), and a list of demonstrations for prompting (initially empty). Like layers in PyTorch, the instantiated module behaves as a callable function: it takes in keyword arguments corresponding to the signature input fields (e.g., question), formats a prompt to implement the signature and includes the appropriate demonstrations, calls the LM, and parses the output fields. When Predict detects it's being used in compile mode, it will also internally track input/output traces to assist the teleprompter at bootstrapping the demonstrations.

Other Built-in Modules DSPy modules translate prompting techniques into modular functions that support any signature, contrasting with the standard approach of prompting LMs with task-specific details (e.g., hand-written few-shot examples). To this end, DSPy includes a number of more sophisticated modules like ChainOfThought, ProgramOfThought, MultiChainComparison, and ReAct. ${ }^{5}$ These can all be used interchangeably to implement a DSPy signature. For instance, simply chang-

\footnotetext{
${ }^{4}$ String descriptions of the task and the fields are also optional and usually omitted. Fields can carry optional field prefix and description. By default, fields are assumed to hold free-form strings; we are actively exploring optional data type as a way to specify constraints on valid values (e.g., bool or int) and more gracefully handle formatting and parsing logic, though this feature is not core to DSPy at the time of writing.
${ }^{5}$ These modules generalize prompting techniques from the literature, respectively, by Wei et al. (2022), Chen et al. (2022), Yoran et al. (2023), and Yao et al. (2022) and, in doing so, generalize the ideas on zero-shot prompting and rationale self-generation from Kojima et al. (2022), Zelikman et al. (2022), Zhang et al. (2022), and Huang et al. (2022) to parameterized modules that can bootstrap arbitrary multi-stage pipelines.
}
ing Predict to ChainOfThought in the above program leads to a system that thinks step by step before committing to its output field.

Importantly, all of these modules are implemented in a few lines of code by expanding the userdefined signature and calling Predict one or more times on new signatures as appropriate. For instance, we show a simplified implementation of the built-in ChainOfThought below.
```
class ChainOfThought(dspy.Module):
    def __init__(self, signature):
        # Modify signature from '*inputs -> *outputs' to '*inputs -> rationale, *outputs'.
        rationale_field = dspy.OutputField(prefix="Reasoning: Let’s think step by step.")
        signature = dspy. Signature(signature).prepend_output_field(rationale_field)
        # Declare a sub-module with the modified signature.
        self.predict = dspy.Predict(signature)
    def forward(self, **kwargs):
        # Just forward the inputs to the sub-module.
        return self.predict(**kwargs)
```

This is a fully-fledged module capable of learning effective few-shot prompting for any LM or task. We contrast that with Appendix C, which copies long reasoning prompts hand-written by sources ranging from recent research to popular prompting libraries.

Parameterization Uniquely, DSPy parameterizes these prompting techniques. To understand this parameterization, observe that any LM call seeking to implement a particular signature needs to specify parameters that include: (1) the specific LM to call (Chen et al., 2023), (2) the prompt instructions (Yang et al., 2023) and the string prefix of each signature field and, most importantly, (3) the demonstrations used as few-shot prompts (for frozen LMs) or as training data (for finetuning). We focus primarily on automatically generating and selecting useful demonstrations. In our case studies, we find that bootstrapping good demonstrations gives us a powerful way to teach sophisticated pipelines of LMs new behaviors systematically.

Tools DSPy programs may use tools, which are modules that execute computation. We support retrieval models through a dspy. Retrieve module. At the time of writing, DSPy has built-in support for ColBERTv2, Pyserini, and Pinecone retrievers, and we have explored experimental dspy. SQL for executing SQL queries and dspy. PythonInterpreter for executing Python code in a sandbox.

Programs DSPy modules can be composed in arbitrary pipelines in a define-by-run interface. Inspired directly by PyTorch and Chainer, one first declares the modules needed at initialization, allowing DSPy to keep track of them for optimization, and then one expresses the pipeline with arbitrary code that calls the modules in a forward method. As a simple illustration, we offer the following simple but complete retrieval-augmented generation (RAG) system.
```
class RAG(dspy.Module):
    def __init__(self, num_passages=3):
        # 'Retrieve' will use the user's default retrieval settings unless overriden.
        self.retrieve = dspy.Retrieve(k=num_passages)
        # 'ChainOfThought' with signature that generates answers given retrieval & question.
        self.generate_answer = dspy.ChainOfThought("context, question -> answer")
    def forward(self, question):
        context = self.retrieve(question).passages
        return self.generate_answer(context=context, question=question)
```

To highlight modularity, we use ChainOfThought as a drop-in replacement of the basic Predict. One can now simply write RAG()("Where is Guaraní spoken?") to use it. Notice that, if we use a signature "context, question -> search_query", we get a system that generates search queries rather than answers.

## 3.3 Teleprompters can automate prompting for arbitrary pipelines

When compiling a DSPy program, we generally invoke a teleprompter, which is an optimizer that takes the program, a training set, and a metric-and returns a new optimized program. Different teleprompters (Sec 4) apply different strategies for optimization.

In DSPy, training sets may be small, potentially a handful of examples, though larger data enables more powerful optimization. Training examples may be incomplete, i.e., only input values are necessary. Labels for the pipeline steps are not required, unless they need to be used in the metric. In practice, we typically assume labels only for (at most) the program's final output, not the intermediate steps. This label-efficiency is critical for modularity: building a new pipeline in DSPy requires simply recompiling the new pipeline's code, not annotating data specific to the new pipeline.

Metrics can be simple notions like exact match (EM) or F1, but they can be entire DSPy programs that balance multiple concerns. For example, we may compile the RAG module above against a dataset of question-answer pairs qa_trainset and the metric EM. The goal of optimization here is to effectively bootstrap few-shot demonstrations. The following code achieves this:
```
# Small training set with only questions and final answers.
qa_trainset = [dspy.Example(question="What is the capital of France?", answer="Paris")]
# The teleprompter will bootstrap missing labels: reasoning chains and retrieval contexts.
teleprompter = dspy.BootstrapFewShot(metric=dspy.evaluate.answer_exact_match)
compiled_rag = teleprompter.compile(RAG(), trainset=qa_trainset)
```

In this example, the BootstrapFewShot teleprompter (Sec 4, Appendix E.1) simulates RAG on the training example(s). It will collect demonstrations of each module (i.e., examples of its input-output behavior) that collectively lead to valid output (i.e., respecting the signatures and the metric).

If one wanted to push the compiled program to be extractive given its retrieved contexts, one could define a custom metric to use in place of dspy.evaluate.answer_exact_match:
```
def answer_and_context_match(example, pred, trace=None):
    answer_match = dspy.evaluate.answer_exact_match(example, pred)
    # Is the prediction a substring of some passage?
    context_match = any((pred.answer.lower() in c) for c in pred.context)
    return answer_match and context_match
```

Notice that behavior like this might be more accurately checked by another DSPy program that checks for faithful grounding of answers. Such metrics are fully supported and encouraged in DSPy.

Teleprompters can be composed by specifying a teacher program. DSPy will sample demonstrations from this program for prompt optimization. This composition can enable very rich pipelines, where expensive programs (e.g., complex expensive ensembles using large LMs) supervise cheap programs (e.g., simple pipelines using smaller LMs). One may start with compiled_rag from above (say, compiled to use a large Llama2-13b-chat LM) but now fine-tune Flan-T5-large to create an efficient program:
```
# Larger set of questions with *no labels*. Labels for all steps will be bootstrapped.
unlabeled_questions = [dspy.Example(question="What is the capital of Germany?"), ...]
# As we assumes no answer, we use 'answer_passage_match' to filter ungrounded answers.
finetuning_teleprompter = BootstrapFinetune(metric=dspy.evaluate.answer_passage_match)
# We set 'teacher=compiled_rag' to compose. Bootstrapping will now use 'compiled_rag'.
compiled_rag_via_finetune = finetuning_teleprompter.compile(RAG(), teacher=compiled_rag,
    trainset=unlabeled_questions, target=’google/flan-t5-large’)
```

