# 3 Forgetting via Negation

In this section, we show that negating a task vector is an effective way to reduce its performance on a target task, without substantially hurting performance elsewhere. Forgetting or "unlearning" can help mitigate undesired biases learned when pre-training; forgetting tasks altogether may be desirable to comply with regulations or for ethical reasons like preventing an image classifier to recognize faces, or to "read" personal information via OCR.

These interventions should not have a substantial effect on how models behave when processing data outside the scope of the edit [69; 39]. Accordingly, we measure accuracy on control tasks, in addition to evaluating on the target tasks from which the task vector originated. Our experiments showcase the effectiveness of negating task vectors for editing image classification and text generation models.

## 3.1 Image classification

For image classification, we use CLIP models [78] and task vectors from eight tasks studied by Ilharco et al. [39]; Radford et al. [78], ranging from satellite imagery recognition to classifying traffic signs: Cars [47], DTD [12], EuroSAT [36], GTSRB [87], MNIST [51], RESISC45 [10], SUN397 [101], and SVHN [72]. We explore additional tasks including OCR and person identification in Appendix B. For the control task, we use ImageNet [16]. We generate task vectors by fine-tuning on each of the target tasks, as detailed in Appendix B.1.

We compare against two additional baselines, fine-tuning by moving in the direction of increasing loss (i.e., with gradient ascent), as in Golatkar et al. [34]; Tarun et al. [90], and against using a random vector where each layer has the same magnitude as the corresponding layer of task vector. Additional details are in Appendix B.2.

As shown in Table 1, negating the task vectors is the most effective editing strategy for decreasing accuracy on the target task with little impact on the control task. For example, negative task vectors decrease the average target accuracy of ViT-L/14 by 45.8 percentage points with little change in accuracy on the control task. In contrast, using a random vector does not have much impact on target accuracy, while fine-tuning with gradient ascent severely deteriorates performance on control tasks. We present additional results in Appendix B.

## 3.2 Text generation

We study whether we can mitigate a particular model behavior by negating a task vector trained to do that behavior. In particular, we aim to reduce the amount of toxic generations produced by GPT-2 models of various sizes [77]. We generate task vectors by fine-tuning on data from Civil

\begin{table}

**Table 1: Forgetting image classification tasks via negation. Results are shown for CLIP models, reporting average accuracy (\%) on the eight target tasks we wish to forget (Cars, DTD, EuroSAT, GTSRB, MNIST, RESISC45, SUN397 and SVHN), and the control task (ImageNet). Negating task vectors reduce the accuracy of a pre-trained ViT-L/14 by 45.8 percentage points on the target tasks, with little loss on the control task. Additional details and results are shown in Appendix B.**
\begin{tabular}[t]{|l|l|l|l|l|l|l|}
\hline \multirow{2}{*}{Method} & \multicolumn{2}{|c|}{ViT-B/32} & \multicolumn{2}{|c|}{ViT-B/16} & \multicolumn{2}{|c|}{ViT-L/14} \\
\hline & Target (↓) & Control (↑) & Target (↓) & Control (↑) & Target (↓) & Control (↑) \\
\hline Pre-trained & 48.3 & 63.4 & 55.2 & 68.3 & 64.8 & 75.5 \\
\hline Fine-tuned & 90.2 & 48.2 & 92.5 & 58.3 & 94.0 & 72.6 \\
\hline Gradient ascent & 2.73 & 0.25 & 1.93 & 0.68 & 3.93 & 16.3 \\
\hline Random vector & 45.7 & 61.5 & 53.1 & 66.0 & 60.9 & 72.9 \\
\hline Negative task vector & 24.0 & 60.9 & 21.3 & 65.4 & 19.0 & 72.9 \\
\hline
\end{tabular}
\end{table}

\begin{table}

**Table 2: Making language models less toxic with negative task vectors. Results are shown for the GPT-2 Large model. Negative task vectors decrease the amount of toxic generations by 6×, while resulting in a model with comparable perplexity on a control task (WikiText-103). Additional details and results are shown in Appendix C.**
\begin{tabular}[t]{|l|l|l|l|}
\hline Method & \% toxic generations (↓) & Avg. toxicity score (↓) & WikiText-103 perplexity (↓) \\
\hline Pre-trained & 4.8 & 0.06 & 16.4 \\
\hline Fine-tuned & 57 & 0.56 & 16.6 \\
\hline Gradient ascent & 0.0 & 0.45 & $>10^{10}$ \\
\hline Fine-tuned on non-toxic & 1.8 & 0.03 & 17.2 \\
\hline Random vector & 4.8 & 0.06 & 16.4 \\
\hline Negative task vector & 0.8 & 0.01 & 16.9 \\
\hline
\end{tabular}
\end{table}

Comments [8] where the toxicity score is higher than 0.8, and then negating such task vectors. As in Section 3.1, we also compare against baselines that use gradient ascent when fine-tuning [34; 90], and using a random task vector of the same magnitude. Additionally, we compare against fine-tuning on non-toxic samples from Civil Comments (toxicity scores smaller than 0.2), similar to Liu et al. [57]. We measure the toxicity of one thousand model generations with Detoxify [35]. For the control task, we measure the perplexity of the language models on WikiText-103 [66].

As shown in Table 2, editing with negative task vectors is effective, reducing the amount of generations classified as toxic from $4.8 \%$ to $0.8 \%$, while maintaining perplexity on the control task within 0.5 points of the pre-trained model. In contrast, fine-tuning with gradient ascent lowers toxic generations by degrading performance on the control task to an unacceptable level, while fine-tuning on non-toxic data is worse than task vectors both in reducing task generations and on the control task. As an experimental control, adding a random vector has little impact either on toxic generations or perplexity on WikiText-103. We present additional experimental details and results in Appendix C.

