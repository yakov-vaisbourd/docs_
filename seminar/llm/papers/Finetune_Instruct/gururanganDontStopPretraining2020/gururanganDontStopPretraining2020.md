\title{
Don't Stop Pretraining: Adapt Language Models to Domains and Tasks
}

\author{
Suchin Gururangan ${ }^{\dagger}$ Ana Marasović ${ }^{\dagger ◇}$ Swabha Swayamdipta ${ }^{\dagger}$ Kyle Lo ${ }^{\dagger}$ Iz Beltagy ${ }^{\dagger}$ Doug Downey ${ }^{\dagger}$ Noah A. Smith ${ }^{\dagger ◇}$ \\ ${ }^{\dagger}$ Allen Institute for Artificial Intelligence, Seattle, WA, USA \\ ${ }^{\diamond}$ Paul G. Allen School of Computer Science \& Engineering, University of Washington, Seattle, WA, USA \\ \{suching, anam, swabhas, kylel, beltagy, dougd, noah\}@allenai.org
}

\begin{abstract}
Language models pretrained on text from a wide variety of sources form the foundation of today's NLP. In light of the success of these broad-coverage models, we investigate whether it is still helpful to tailor a pretrained model to the domain of a target task. We present a study across four domains (biomedical and computer science publications, news, and reviews) and eight classification tasks, showing that a second phase of pretraining indomain (domain-adaptive pretraining) leads to performance gains, under both high- and low-resource settings. Moreover, adapting to the task's unlabeled data (task-adaptive pretraining) improves performance even after domain-adaptive pretraining. Finally, we show that adapting to a task corpus augmented using simple data selection strategies is an effective alternative, especially when resources for domain-adaptive pretraining might be unavailable. Overall, we consistently find that multiphase adaptive pretraining offers large gains in task performance.
\end{abstract}

