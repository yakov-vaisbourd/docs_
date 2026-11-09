# 2. Background and Experimental Setup

Neural autoregressive language models (LMs) take as input a sequence of tokens and output a probability distribution over the next token. Large neural LMs can perform tasks in a zero- or few-shot manner using in-context learning (Radford et al., 2019; Brown et al., 2020). To do so, a natural language prompt is fed into the model. This prompt contains three components: a format, a set of training examples, and a permutation (ordering) of the training examples.

Prompt Format The prompt format is a template which consists of placeholders for the training and test example(s) and possibly a natural language description of the task. For example, the format of the prompt in Section 1 is a template with the style: "Input:" input "Sentiment:" label. Many alternate formats exist, e.g., one could frame the task as question answering.

Prompt Training Examples The prompt's training examples are used to teach the LM how to solve the task at hand. The prompt from Section 1 consists of two training examples; we refer to this as "two-shot" learning. We also consider "zero-shot" learning, where no training examples are present.

Training Example Permutation When training examples are used, they have a particular permutation, e.g., the "Subpar acting" example comes first in the prompt from Section 1. The permutation matters because neural language models update their hidden states in a left-to-right-fashion.

To make predictions on an input, we slot it into the test placeholder and generate from the LM. For example, see the "Amazing." test example in the prompt from Section 1. For generation tasks, we generate greedily from the LM until it produces a newline character. For classification tasks, the probability for each class is given by the probability assigned to its associated label name, e.g., the words "Negative" and "Positive" for sentiment classification.

## 2.1. Datasets and Prompt Formats

We use datasets for three tasks: text classification, fact retrieval, and information extraction. We use a fixed prompt format for each dataset unless otherwise specified. We show the format and examples from each dataset in Appendix B.

Text Classification We study text classification using six datasets: sentiment analysis using SST-2 (Socher et al., 2013), 6-way question classification using TREC (Voorhees \& Tice, 2000), textual entailment using 3-way CB (de Marneffe et al., 2019) and binary RTE (Dagan et al., 2005) from SuperGLUE (Wang et al., 2019), and topic classification

![Figure](./zhaoCalibrateUseImproving2021_images/bfa78706-72f5-4d26-85c6-dd246cd5745f-03_574_786_247_206.jpg)

**Figure 2. There is high variance in GPT-3's accuracy as we change the prompt's training examples, as well as the permutation of the examples. Here, we select ten different sets of four SST-2 training examples. For each set of examples, we vary their permutation and plot GPT-3 2.7B's accuracy for each permutation (and its quartiles).**

using the 4-way AGNews (Zhang et al., 2015) and 14-way DBPedia (Zhang et al., 2015) datasets. The prompt in Section 1 shows an example of the sentiment analysis task.

Fact Retrieval We evaluate fact retrieval with LAMA (Petroni et al., 2019). The dataset consists of knowledge base triples that are placed into templates with missing objects, e.g. "Obama was born in". We use these templates as our prompts, and remove the relations where the missing answer is not at the end of the template (left-to-right LMs cannot solve these). The answers are always single tokens, and we report average accuracy across all triples.

Information Extraction We consider information extraction using two slot filling datasets, ATIS (Hemphill et al., 1990) and MIT Movies trivia10k13 (Liu et al., 2012). We use two random slots for each dataset, airline and departure date for ATIS, and director name and movie genre for MIT Movies. The answer for both datasets is a span of text from the input, e.g., the ATIS airline task is to predict "american airlines" when given the sentence "list a flight on american airlines from toronto to san diego". We use Exact Match between the model's generated output and the ground-truth span as our evaluation metric.

## 2.2. Model Details

We run our experiments on three sizes of GPT-3 $(2.7 \mathrm{~B}, 13 \mathrm{~B}$, and 175 B parameters) as well as GPT-2 (1.5B parameters). We access GPT-3 using the OpenAI API. We release code to replicate our experiments. ${ }^{1}$

\footnotetext{
${ }^{1}$ https://www.github.com/tonyzhaozh/few-shot-learning
}

![Figure](./zhaoCalibrateUseImproving2021_images/bfa78706-72f5-4d26-85c6-dd246cd5745f-03_584_801_241_1063.jpg)

**Figure 3. There is high variance in GPT-3's accuracy as we change the prompt format. In this figure, we use ten different prompt formats for SST-2. For each format, we plot GPT-3 2.7B's accuracy for different sets of four training examples, along with the quartiles.**

