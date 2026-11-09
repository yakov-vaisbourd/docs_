# 2 Method

Annotating large-scale instruction data can be challenging for humans because it requires 1) creativity to come up with novel tasks and 2) expertise for writing the solutions to each task. Here, we detail our process for Self-Instruct, which refers to the pipeline of generating tasks with a vanilla pretrained language model itself, filtering the generated data, and then conducting instruction tuning with this generated data in order to align the LM to follow instructions better. This pipeline is depicted in Figure 2.

## 2.1 Defining Instruction Data

The instruction data we want to generate contains a set of instructions $\left\{I_{t}\right\}$, each of which defines a task $t$ in natural language. Task $t$ has $n_{t} \geq 1$ input-output instances \(\left\{\left(X_{t, i}, Y_{t, i}\right)\right\}_{i=1}^{n_{t}}\). A model $M$ is expected to produce the output, given the task instruction and the corresponding input: \(M\left(I_{t}, X_{t, i}\right)=Y_{t, i}\), for $i \in\left\{1, \ldots, n_{t}\right\}$. Note that the instruction and instance input does not have a strict boundary in many cases. For example, "write an essay about school safety" can be a valid instruction that we expect models to respond to directly, while it can also be formulated as "write an essay about the following topic" as the instruction, and "school safety" as an instance input. To encourage the diversity of the data format, we allow such instructions that do not require additional input (i.e., $X$ is empty).

## 2.2 Automatic Instruction Data Generation

Our pipeline for data generation consists of four steps: 1) generating task instructions, 2) determining if the instruction represents a classification task, 3) instance generation with either an input-first or output-first approach, and 4) filtering low-quality data.

Instruction Generation. At the first step, SelfInstruct generates new instructions from a small set of seed human-written instructions in a bootstrapping fashion. We initiate the task pool with 175 tasks ( 1 instruction and 1 instance for each task). ${ }^{3}$ For every step, we sample 8 task instructions from this pool as in-context examples. Of the 8 instructions, 6 are from the human-written

\footnotetext{
${ }^{3}$ These tasks were newly written by the authors and their labmates at UW, without reference to existing datasets or the test set used in this work. We provide more details about these tasks and analyze their similarity to the test tasks in Appendix §A.1.
}
tasks, and 2 are from the model-generated tasks in previous steps to promote diversity. The prompting template is shown in Table 5.

Classification Task Identification. Because we need two different approaches for classification and non-classification tasks, we next identify whether the generated instruction represents a classification task or not. ${ }^{4}$ We prompt the LM in a few-shot way to determine this, using 12 classification instructions and 19 non-classification instructions from the seed tasks. The prompting template is shown in Table 6.

Instance Generation. Given the instructions and their task type, we generate instances for each instruction independently. This is challenging because it requires the model to understand what the target task is, based on the instruction, figure out what additional input fields are needed and generate them, and finally complete the task by producing the output. We found that pretrained LMs can achieve this to a large extent when prompted with instruction-input-output in-context examples from other tasks. A natural way to do this is the Inputfirst Approach, where we can ask an LM to come up with the input fields first based on the instruction, and then produce the corresponding output. This generation order is similar to how models are used to respond to instruction and input, but here with in-context examples from other tasks. The prompting template is shown in Table 7.

However, we found that this approach can generate inputs biased toward one label, especially for classification tasks (e.g., for grammar error detection, it usually generates grammatical input). Therefore, we additionally propose an Output-first Approach for classification tasks, where we first generate the possible class labels, and then condition the input generation on each class label. The prompting template is shown in Table 8. ${ }^{5}$ We apply the outputfirst approach to the classification tasks identified in the former step, and the input-first approach to the remaining non-classification tasks.

Filtering and Postprocessing. To encourage diversity, a new instruction is added to the task pool only when its ROUGE-L similarity with any exist-

\footnotetext{
${ }^{4}$ More concretely, we regard tasks that have a small limited output label space as classification tasks.
${ }^{5}$ In this work, we use a fixed set of seed tasks for prompting the instance generation, and thus only generate a small number of instances per task in one round. Future work can use randomly sampled tasks to prompt the model to generate a larger number of instances in multiple rounds.
}
ing instruction is less than 0.7. We also exclude instructions that contain some specific keywords (e.g., image, picture, graph) that usually can not be processed by LMs. When generating new instances for each instruction, we filter out instances that are exactly the same or those with the same input but different outputs. Invalid generations are identified and filtered out based on heuristics (e.g., instruction is too long or too short, instance output is a repetition of the input).

## 2.3 Finetuning the LM to Follow Instructions

After creating large-scale instruction data, we use it to finetune the original LM (i.e., Self-Instruct). To do this, we concatenate the instruction and instance input as a prompt and train the model to generate the instance output in a standard supervised way. To make the model robust to different formats, we use multiple templates to encode the instruction and instance input together. For example, the instruction can be prefixed with "Task:" or not, the input can be prefixed with "Input:" or not, "Output:" can be appended at the end of the prompt or not, and different numbers of break lines can be put in the middle, etc.

