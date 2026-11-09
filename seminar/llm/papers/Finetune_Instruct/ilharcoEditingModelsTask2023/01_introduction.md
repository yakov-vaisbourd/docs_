# 1 Introduction

Pre-trained models are commonly used as backbones of machine learning systems. In practice, we often want to edit models after pre-training, ${ }^{1}$ to improve performance on downstream tasks [105; 100; 63; 39], mitigate biases or unwanted behavior [85; 59; 82; 71], align models with human preferences [4; 74; 44; 32], or update models with new information [104; 15; 69; 70].

In this work, we present a new paradigm for editing neural networks based on task vectors, which encode the information necessary to do well on a given task. Inspired by recent work on weight interpolation [27; 100; 63; 99; 39; 55; 2; 20], we obtain such vectors by taking the weights of a model fine-tuned on a task and subtracting the corresponding pre-trained weights (Figure 1a).

We show that we can edit a variety of models with task arithmetic-performing simple arithmetic operations on task vectors (Figure 1b-d). For example, negating a vector can be used to remove undesirable behaviors or unlearn tasks, while adding task vectors leads to better multi-task models, or even improves performance on a single task. Finally, when tasks form an analogy relationship, task vectors can be combined to improve performance on tasks where data is scarce.

Forgetting via negation. Users can negate task vectors to mitigate undesirable behaviors (e.g., toxic generations), or even to forget specific tasks altogether, like OCR. In Section 3, we negate a task vector from a language model fine-tuned on toxic data [77; 8], reducing the proportion of generations classified as toxic, with little change in fluency. We also negate task vectors for image classification tasks, resulting in substantially lower accuracy on the task we wish to forget with little loss on ImageNet accuracy [16].

\footnotetext{
*Correspondence to gamaga@cs.washington.edu.
${ }^{1}$ We use the term editing to refer to any intervention done to a model done after the pre-training stage.
}

![Figure](./ilharcoEditingModelsTask2023_images/a9c92ee0-c150-48d6-ad01-59723be068c6-02_415_1377_275_380.jpg)

**Figure 1: An illustration of task vectors and the arithmetic operations we study for editing models. (a) A task vector is obtained by subtracting the weights of a pre-trained model from the weights of the same model after fine-tuning (Section 2). (b) Negating a task vector degrades performance on the task, without substantial changes in control tasks (Section 3). (c) Adding task vectors together improves the performance of the pre-trained model on the tasks under consideration (Section 4). (d) When tasks form an analogy relationship such as supervised and unsupervised learning on two different data sources, it is possible to improve performance on a supervised target task using only vectors from the remaining three combinations of objectives and datasets (Section 5).**

Learning via addition. Adding task vectors results in better multi-task models, or improved performance on a single task. In Section 4, we add task vectors from various image models (CLIP, Radford et al. [78]) and compare the performance of the resulting model with using multiple specialized fine-tuned models. We find that the single resulting model can be competitive with using multiple specialized models. Adding two task vectors maintains 98.9\% of the accuracy, and the average performance on the entire set of tasks increases as more task vectors are added. Moreover, adding a task vector from a different task can improve performance on a target task using text models (T5, Raffel et al. [79]).

Task analogies. When we can form task analogies of the form " $A$ is to $B$ as $C$ is to $D$ ", combining task vectors from the first three tasks improves performance on the fourth, even when little or no training data is available. In Section 5, we show that we can improve domain generalization to a new target task without using labeled data from that task. More specifically, accuracy on a sentiment analysis task improves by combining a task vector from a second sentiment analysis dataset and task vectors produced using unlabeled data from both domains. We also use analogies between classifying pictures and sketches of objects to improve accuracy on subgroups where little or no data is available.

Overall, editing models with task arithmetic is simple, fast and effective. There is no extra cost at inference time in terms of memory or compute, since we only do element-wise operations on model weights. Moreover, vector operations are cheap, allowing users to experiment quickly with multiple task vectors. With task arithmetic, practitioners can reuse or transfer knowledge from models they create, or from the multitude of publicly available models all without requiring access to data or additional training. ${ }^{2}$

