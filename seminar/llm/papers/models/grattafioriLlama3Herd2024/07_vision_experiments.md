# 7 Vision Experiments

We perform a series of experiments in which we incorporate visual-recognition capabilities into Llama 3 via a compositional approach that consists of two main stages. First, we compose a pre-trained image encoder (Xu et al., 2023) and the pre-trained language model by introducing and training a set of cross-attention layers between the two models (Alayrac et al., 2022) on a large number of image-text pairs. This leads to the model illustrated in Figure 28. Second, we introduce temporal aggregator layers and additional video cross-attention layers that operate on a large collection of video-text pairs to learn the model to recognize and process temporal information from videos.

A compositional approach to foundation model development has several advantages: (1) it enables us to parallelize the development of the vision and language modeling capabilities; (2) it circumvents complexities of joint pre-training on visual and language data that stem from tokenization of visual data, differences in background perplexities of tokens originating from different modalities, and contention between modalities; (3) it guarantees that model performance on text-only tasks is not affected by the introduction of visual-recognition capabilities, and (4) the cross-attention architecture ensures that we do not have to expend compute passing full-resolution images through the increasingly LLM backbones (specifically, the feed-forward networks in each transformer layer), making it more efficient during inference. We note that our multimodal models are still under development and not yet ready for release.

Before presenting the results of our experiments in Section 7.6 and 7.7, we describe the data we used to train visual recognition capabilities, the model architecture of the vision components, how we scale training of those components, and our pre-training and post-training recipes.

## 7.1 Data

We describe our image and video data separately below.

## 7.1.1 Image Data

Our image encoder and adapter are trained on image-text pairs. We construct this dataset via a complex data processing pipeline that consists of four main stages: (1) quality filtering, (2) perceptual de-duplication, (3) resampling, and (4) optical character recognition. We also apply a series of safety mitigations.
\begin{itemize}
\item[-] Quality filtering. We implement quality filters that remove non-English captions and low-quality captions via heuristics such as low alignment scores produced by (Radford et al., 2021). Specifically, we remove all image-text pairs below a certain CLIP score.
\item[-] De-duplication. De-duplicating large-scale training datasets benefits model performance because it reduces training compute spent on redundant data (Esser et al., 2024; Lee et al., 2021; Abbas et al.,
\end{itemize}

![Figure](./grattafioriLlama3Herd2024_images/964281cf-3b75-4775-9ae9-065d3b6d211b-55_769_1532_285_296.jpg)

**Figure 28 Illustration of the compositional approach to adding multimodal capabilities to Llama 3 that we study in this paper. This approach leads to a multimodal model that is trained in five stages: (1) language model pre-training, (2) multi-modal encoder pre-training, (3) vision adapter training, (4) model finetuning, and (5) speech adapter training.**

2023) and memorization (Carlini et al., 2023; Somepalli et al., 2023). Hence, we de-duplicate our training data for both efficiency and privacy reasons. To do so, we use an internal version of the state-of-the-art SSCD copy-detection model (Pizzi et al., 2022) to de-duplicate images at scale. For all images, we first compute a 512-dimensional representation using the SSCD model. We use those embeddings to perform a nearest neighbor (NN) search for each image across all images in our data set, using a cosine similarity measure. We define examples above a certain similarity threshold as duplicates. We group these duplicates using a connected-components algorithm, and maintain only one image-text pair per connected component. We increase the efficiency of our de-duplication pipeline by: (1) pre-clustering the data using k-means clusters and (2) using FAISS (Johnson et al., 2019) for NN searches and clustering.
\begin{itemize}
\item[-] Resampling. We ensure diversity of the image-text pairs via resampling akin to Xu et al. (2023); Mahajan et al. (2018); Mikolov et al. (2013). First, we construct a vocabulary of n-grams by parsing high-quality text sources. Next, we compute the frequency of each vocabulary n-gram in our dataset. We then resample the data as follows: If any of the n-grams in a caption occurs less than $T$ times in the vocabulary, we keep the corresponding image-text pair. Otherwise, we independently sample each of the n-grams $n_{i}$ in the caption with probability $\sqrt{T / f_{i}}$ where $f_{i}$ indicates the frequency of n-gram $n_{i}$; we keep the image-text pair if any of the n-grams was sampled. This resampling aids performance on low-frequency categories and fine-grained recognition tasks.
\item[-] Optical character recognition. We further improve our image-text data by extracting text written in the image and concatenating it with the caption. The written text is extracted using a proprietary optical character recognition (OCR) pipeline. We observe that adding OCR data into the training data greatly improves tasks that require OCR capabilities, such as document understanding.
\end{itemize}

Transcribing documents. To improve the performance of our models on document understanding tasks, we render pages from documents as images and paired the images with their respective text. The document text is obtained either directly from the source or via a document parsing pipeline.

Safety. We focus primarily on ensuring that the pre-training dataset for image recognition does not contain
unsafe content, such as sexual abuse material (CSAM) (Thiel, 2023). We scan all our training images for CSAM using perceptual hashing approaches such as PhotoDNA (Farid, 2021) as well as internal, proprietary classifiers. We also use a proprietary media-risk retrieval pipeline to identify and remove image-text pairs that we consider to be NSFW, for example, because they contain sexual or violent content. We believe that minimizing the prevalence of such material in the training dataset improves the safety of the final model without impacting its helpfulness. Finally, we perform face blurring on all images in our training set. We test the model against human generated prompts that refer to an attached image.

Annealing data. We create an annealing dataset by resampling the image-caption pairs to a smaller volume of ~350M examples using n-grams. Since the n-grams resampling favor richer text descriptions, this selects a higher-quality data subset. We augment the resulting data with ~150M examples from five additional sources:
\begin{itemize}
\item[-] Visual grounding. We link noun phrases in the text to bounding boxes or masks in the image. The grounding information (bounding boxes and masks) are specified in the image-text pair in two ways. (1) We overlay boxes or masks with marks on the image and use marks in the text as reference, akin to set-of-marks (Yang et al., 2023a). (2) We insert normalized \(\left(x_{\text {min }}, y_{\text {min }}, x_{\text {max }}, y_{\text {max }}\right)\) coordinates directly into the text, demarcated by special tokens.
\item[-] Screenshot parsing. We render screenshots from HTML code and task the model with predicting the code that produced a specific element in the screenshot, akin to Lee et al. (2023). The element of interest is indicated in the screenshot via a bounding box.
\item[-] Question-answer pairs. We include question-answer pairs, enabling us to use volumes of questionanswering data that are too large to be used in model finetuning.
\item[-] Synthetic captions. We include images with synthetic captions that were generated by an early version of the model. Compared to original captions, we find that synthetic captions provide a more comprehensive description of images than the original captions.
\item[-] Synthetically-generated structured images. We also include synthetically generated images for a variety of domains such as charts, tables, flowcharts, math equations and textual data. These images are accompanied by a structured representation such as the corresponding markdown or LaTeX notation. Besides improving recognition capabilities of the model for these domains, we find this data useful to generate question-answer pairs via the text model for finetuning.
\end{itemize}

## 7.1.2 Video Data

For video pre-training, we use a large dataset of video-text pairs. Our dataset is curated through a multi-stage process. We filter and clean the associated texts using rule-based heuristics, such as ensuring a minimum length and fixing capitalization. Then, we run language identification models to filter out non-English texts. We run OCR detection models to filter out videos with excessive overlaid text. To ensure reasonable alignment between the video-text pairs, we use CLIP (Radford et al., 2021) style image-text and video-text contrastive models. We first compute image-text similarity using a single frame in the videos and filtered out low similarity pairs, and then subsequently filter out pairs with low video-text alignment. Some of our data contains static or low-motion videos; we filter out such data using motion-score based filtering (Girdhar et al., 2023). We do not apply any filters on the visual quality of the videos such as aesthetic scores or resolution filtering.

Our dataset contains videos with an average duration of 21 seconds and a median duration of 16 seconds, with over 99\% videos being under a minute. The spatial resolution varies significantly between 320p and 4K videos, with over 70\% of the videos having a short side greater than 720 pixels. The videos have varying aspect ratios with almost all videos having between aspect ratio between 1:2 and 2:1, with a 1:1 median.

## 7.2 Model Architecture

Our visual-recognition model consists of three main components: (1) an image encoder, (2) an image adapter, and (3) a video adapter.

Image encoder. Our image encoder is a standard vision transformer (ViT; Dosovitskiy et al. (2020)) that is trained to align images and text (Xu et al., 2023). We use the ViT-H/14 variant of the image encoder,
which has 630 M parameters that were trained on 2.5B image-text pairs for five epochs. The image encoder is pre-trained on images with resolution 224 × 224; images were split up into 16 × 16 patches of equal size (i.e., a patch size of $14 x 14$ pixels). As also demonstrated by prior work such as ViP-Llava (Cai et al., 2024), we observe that image encoders trained via a contrastive text alignment objective are unable to preserve fine-grained localization information. To alleviate this, we employ a multi-layer feature extraction, where features from the $4^{\text {th }}, 8^{\text {th }}, 16^{\text {th }}, 24^{\text {th }}$ and $31^{\text {st }}$ layers are also provided in addition to the final layer features. In addition, we further insert 8 gated self-attention layers (making a total of 40 transformer blocks) prior to pre-training of the cross-attention layers to learn alignment-specific features. The image encoder therefore eventually has a total 850M parameters with the additional layers. With the multi-layer features, the image encoder produces a 7680-dimensional representation for each of the resulting 16 × 16 = 256 patches. The parameters of the image encoder are not frozen during subsequent training stages as we found it to improve performance, especially in domains such as text recognition.

Image adapter. We introduce cross-attention layers between the visual token representations produced by the image encoder and the token representations produced by the language model (Alayrac et al., 2022). The cross-attention layers are applied after every fourth self-attention layer in the core language model. Like the language model itself, the cross-attention layers use generalized query attention (GQA) for increased efficiency. The cross-attention layers introduce substantial numbers of additional trainable parameters into the model: for Llama 3 405B, the cross-attention layers have $\approx 100 \mathrm{~B}$ parameters. We pre-train our image adapter in two stages: (1) initial pre-training followed by (2) annealing:
\begin{itemize}
\item[-] Initial pre-training. We pre-train our image adapter on our dataset of ~6B image-text pairs described above. For compute efficiency reasons, we resize all images to fit within at most four tiles of $336 \times 336$ pixels each, where we arrange the tiles to support different aspect ratios, e.g., $672 \times 672,672 \times 336$, and 1344 × 336.
\item[-] Annealing. We continue training the image adapter on ~500M images from the annealing dataset described above. During annealing, we increase the per-tile image resolution to improve performance on tasks that require higher-resolution images, for example, infographics understanding.
\end{itemize}

Video adapter. Our model takes as input up to 64 frames (uniformly sampled from a full video), each of which is processed by the image encoder. We model temporal structure in videos through two components: (i) encoded video frames are aggregated by a temporal aggregator which merges 32 consecutive frames into one, (ii) additional video cross attention layers are added before every fourth image cross attention layer. The temporal aggregator is implemented as a perceiver resampler (Jaegle et al., 2021; Alayrac et al., 2022). We pre-train using 16 frames per video (aggregated to 1 frame), but increase the number of input frames to 64 during supervised finetuning. The video aggregator and cross attention layers have 0.6B and 4.6B parameters for Llama 3 7B and 70B, respectively.

## 7.3 Model Scaling

After the visual-recognition components are added to Llama 3, the model contains self-attention layers, crossattention layers, and a ViT image encoder. To train adapters for the smaller 8B and 70B parameter models, we found a combination of data and tensor parallelization is the most efficient. Model or pipeline parallelism does not increase efficiency at these scales because the gathering of model parameters would dominate the computation. We do, however, use pipeline parallelism (in addition to data and tensor parallelism) when training the adapter for the 405B parameter model. Training at this scale introduces three new challenges in addition to those outlined in Section 3.3: model heterogeneity, data heterogeneity, and numerical instabilities.

Model heterogeneity. The model computation is heterogeneous because more computation is performed on some tokens than on others. In particular, image tokens are processed by the image encoder and the crossattention layers, whereas text tokens are only processed by the language backbone. This heterogeneity leads to bottlenecks in the scheduling of pipeline parallelism. We address this problem by ensuring each pipeline stage contains five layers: namely, four self-attention layers in the language backbone and a cross-attention layer. (Recall that we introduce a cross-attention layer after every fourth self-attention layer.) In addition, we replicate the image encoder on all pipeline stages. Because we train on paired image-text data, this enables us to perform load balancing between the image and text parts of the computation.

Data heterogeneity. The data is heterogeneous because, on average, images have more tokens than the associated text: an image has 2,308 tokens, whereas the associated text contains an average of only 192 tokens. As a result, the computation of cross-attention layers requires more time and memory than the computation of self-attention layers. We address this problem by introducing sequence parallelization in the image encoder, so that each GPU processes roughly the same number of tokens. Because the average text size is relatively short, we also use a substantially larger micro-batch size ( 8 instead of 1 ).

Numerical instabilities. After the image encoder is added to the model, we find that performing gradient accumulation in bf16 led to numerical instabilities. The most likely explanation for this is that image tokens are introduced into the language backbone via all cross-attention layers. This implies that numerical deviations in the representation of an image token have an outsized impact on the overall computation because the errors are compounded. We address this by performing gradient accumulation in FP32.

## 7.4 Pre-training

Image. We initialize from the pre-trained text model and vision encoder weights. The vision encoder is unfrozen, while the text model weights are kept frozen as explained above. First, we train the model using 6B image-text pairs where each image is resized to fit within four tiles of 336 × 336 pixels. We use a global batch size of 16,384 and a cosine learning rate schedule with initial learning rate $10 \times 10^{-4}$ and a weight decay of 0.01. The initial learning rate was determined based on small-scale experiments. However, these findings did not generalize well to very long training schedules and dropped the learning rate a few times during training when the loss values became stagnant. After the base pre-training, we increase the image resolution further and continue training the same weights on the annealing dataset. The optimizer is re-initialized via warm-up to learning rate $2 \times 10^{-5}$ and again follows a cosine schedule.

Video. For video pre-training, we start from the image pre-trained and annealed weights as described above. We add the video aggregator and cross-attention layers as described in the architecture, initialized randomly. We freeze all the parameters in the model except the video-specific ones (the aggregator and video cross-attention), and train them on the video pre-training data. We use the same training hyperparameters as the image annealing stage, with small differences in the learning rate. We uniformly sample 16 frames from the full video, and represent each frame using four chunks, each of size of 448 × 448 pixels. We use an aggregation factor of 16 in the video aggregator, hence obtaining one effective frame, which the text tokens cross-attend to. We use a global batch size of 4,096, a sequence length of 190 tokens, and a learning rate of $10^{-4}$ during training.

## 7.5 Post-Training

In this section, we describe the post-training recipe for our vision adapters. After pre-training, we fine-tune the model on highly curated multi-modal conversational data to enable chat capabilities. We further implement direct preference optimization (DPO) to boost human evaluation performance and rejection sampling to improve multi-modal reasoning capabilities. Finally, we add a quality-tuning stage where we continue finetuning the model on a very small set of high-quality conversational data which further boosts human evaluation while retaining performance across benchmarks. More details on each of these steps are provided below.

## 7.5.1 Supervised Finetuning Data

We describe our supervised finetuning (SFT) data for image and video capabilities separately below.
Image. We utilize a mix of different datasets for supervised finetuning.
\begin{itemize}
\item[-] Academic datasets. We convert a highly filtered collection of existing academic datasets to questionanswer pairs using templates or via LLM rewriting. The LLM rewriting's purpose is to augment the data with different instructions and to improve the language quality of answers.
\item[-] Human annotations. We collect multi-modal conversation data via human annotators for a wide range of tasks (open-ended question-answering, captioning, practical use cases, etc.) and domains (e.g., natural images and structured images). Annotators are provided with images and asked to write conversations. To ensure diversity, we cluster large-scale datasets and sampled images uniformly across different clusters. Further, we acquire additional images for a few specific domains by expanding a seed via k-nearest
\end{itemize}
\begin{itemize}
\item[] neighbors. Annotators are also provided with intermediate checkpoints of existing models to facilitate model-in-the-loop style annotations, so that model generations can be utilized as a starting point by the annotators to then provide additional human edits. This is an iterative process, in which model checkpoints would be regularly updated with better performing versions trained on the latest data. This increases the volume and efficiency of human annotations, while also improving their quality.
\item[-] Synthetic data. We explore different ways to generate synthetic multi-modal data by using textrepresentations of images and a text-input LLM. The high-level idea is to utilize the reasoning capabilities of text-input LLMs to generate question-answer pairs in the text domain, and replace the text representation with its corresponding images to produce synthetic multi-modal data. Examples include rendering texts from question-answer datasets as images or rendering table data into synthetic images of tables and charts. Additionally, we use captions and OCR extractions from existing images to generate additional conversational or question-answer data related to the images.
\end{itemize}

Video. Similar to the image adapter, we use academic datasets with pre-existing annotations and convert them into appropriate textual instructions and target responses. The targets are converted to open-ended responses or multiple-choice options, whichever is more appropriate. We ask humans to annotate videos with questions and corresponding answers. The annotators are asked to focus on questions that could not be answered based on a single frame, to steer the annotators towards questions that require temporal understanding.

## 7.5.2 Supervised Finetuning Recipe

We describe our supervised finetuning (SFT) recipe for image and video capabilities separately below.
Image. We initialize from the pre-trained image adapter, but hot-swap the pre-trained language model's weights with the instruction tuned language model's weights. The language model weights are kept frozen to maintain text-only performance, i.e., we only update the vision encoder and image adapter weights.

Our approach to finetune the model is similar to Wortsman et al. (2022). First, we run a hyperparameter sweep using multiple random subsets of data, learning rates and weight decay values. Next, we rank the models based on their performance. Finally, we average the weights of the top- $K$ models to obtain the final model. The value of $K$ is determined by evaluating the averaged models and selecting the instance with highest performance. We observe that the averaged models consistently yield better results compared to the best individual model found via grid search. Further, this strategy reduces sensitivity to hyperparameters.

Video. For video SFT, we initialize the video aggregator and cross-attention layers using the pre-trained weights. The rest of the parameters in the model, the image weights and the LLM, are initialized from corresponding models following their finetuning stages. Similar to video pre-training, we then finetune only the video parameters on the video SFT data. For this stage, we increase the video length to 64 frames, and use an aggregation factor of 32 to get two effective frames. The resolution of the chunks is also increased to be consistent with the corresponding image hyperparameters.

## 7.5.3 Preference Data

We built multimodal pair-wise preference datasets for reward modeling and direct preference optimization.
\begin{itemize}
\item[-] Human annotations. The human-annotated preference data consists of comparisons between two different model outputs, labeled as "chosen" and "rejected", with 7-scale ratings. The models used to generate responses are sampled on-the-fly from a pool of the best recent models, each with different characteristics. We update the model pool weekly. Besides preference labels, we also request annotators to provide optional human edits to correct inaccuracies in "chosen" responses because vision tasks have a low tolerance for inaccuracies. Note that human editing is an optional step because there is a trade-off between volume and quality in practice.
\item[-] Synthetic data. Synthetic preference pairs could also be generated by using text-only LLMs to edit and deliberately introduce errors in the supervised finetuning dataset. We took the conversational data as input, and use an LLM to introduce subtle but meaningful errors (e.g., change objects, change attributes, add mistakes in calculations, etc.). These edited responses are used as negative "rejected" samples and paired with the "chosen" original supervised finetuning data.
\end{itemize}
\begin{itemize}
\item[-] Rejection sampling. Furthermore, to create more on-policy negative samples, we leveraged the iterative process of rejection sampling to collect additional preference data. We discuss our usage of rejection sampling in more detail in the following sections. At a high-level, rejection sampling is used to iteratively sample high-quality generations from a model. Therefore, as a by-product, all generations that are not selected can be used as negative rejected samples and used as additional preference data pairs.
\end{itemize}

## 7.5.4 Reward Modeling

We train a vision reward model (RM) on top of the vision SFT model and the language RM. The vision encoder and the cross-attention layers are initialized from the vision SFT model and unfrozen during training, while the self-attention layers are initialized from the language RM and kept frozen. We observe that freezing the language RM part generally leads to better accuracy, especially on tasks that require the RM to judge based on its knowledge or the language quality. We adopt the same training objective as the language RM, but adding a weighted regularization term on the square of the reward logits averaged over the batch, which prevents the reward scores from drifting.

The human preference annotations in Section 7.5.3 are used to train the vision RM. We follow the same practice as language preference data (Section 4.2.1) to create two or three pairs with clear ranking (edited > chosen > rejected). In addition, we also synthetically augment the negative responses by perturbing the words or phrases related to the information in the image (such as numbers or visual texts). This encourages the vision RM to ground its judgement based on the actual image content.

## 7.5.5 Direct Preference Optimization

Similar to the language model (Section 4.1.4), we further train the vision adapters with Direct Preference Optimization (DPO; Rafailov et al. (2023)) using the preference data described in Section 7.5.3. To combat the distribution shift during post-training rounds, we only keep recent batches of human preference annotations while dropping batches that are sufficiently off-policy (e.g., if the base pre-trained model is changed). We find that instead of always freezing the reference model, updating it in an exponential moving average (EMA) fashion every k-steps helps the model learn more from the data, resulting in better performance in human evaluations. Overall, we observed that the vision DPO model consistently performs better than its SFT starting point in human evaluations for every finetuning iteration.

## 7.5.6 Rejection Sampling

Most available question-answer pairs only contain the final answer and lack the chain-of-thought explanation that is required to train a model that generalizes well for reasoning tasks. We use rejection sampling to generate the missing explanations for such examples and boost the model's reasoning capabilities.

Given a question-answer pair, we generate multiple answers by sampling the finetuned model with different system prompts or temperature. Next, we compare the generated answers to the ground-truth via heuristics or an LLM judge. Finally, we retrain the model by adding the correct answers back into the finetuning data mix. We find it useful to keep multiple correct answers per question.

To ensure we only add high-quality examples back into training, we implemented the following two guardrails. First, we find that some examples contain incorrect explanations, despite the final answer being correct. We observed that this pattern occurs more frequently for questions where only a small fraction of the generated answers is correct. Therefore, we drop answers for questions where the probability of the answer being correct is below a certain threshold. Second, raters prefer some answers over others due to differences in language or style. We use the reward model to select top- $K$ highest-quality answers and add them back into training.

## 7.5.7 Quality Tuning

We curate a small but highly selective SFT dataset where all samples have been rewritten and verified either by humans or our best models to meet our highest standards. We train DPO models with this data to improve response quality, calling the process Quality-Tuning (QT). We find that QT significantly improves human evaluations without affecting generalization verified by benchmarks when the QT dataset covers a wide range

\begin{table}
\begin{tabular}[t]{|l|l|l|l|l|l|l|l|}
\hline & Llama 3-V 8B & Llama 3-V 70B & Llama 3-V 405B & GPT-4V & GPT-4o & Gemini 1.5 Pro & Claude 3.5 \\
\hline MMMU (val, cot) & 49.6 & 60.6 & 64.5 & 56.4 & 69.1 & 62.2 & 68.3 \\
\hline VQAv2 (test-dev) & 78.0 & 79.1 & 80.2 & 77.2 & - & 80.2 & - \\
\hline AI2 Diagram (test) & 84.4 & 93.0 & 94.1 & 78.2 & 94.2 & 94.4 & 94.7 \\
\hline ChartQA (test, CoT) & 78.7 & 83.2 & 85.8 & 78.4 & 85.7 & 87.2 & 90.8 \\
\hline TextVQA (val) & 78.2 & 83.4 & 84.8 & 78.0 & - & 78.7 & - \\
\hline DocVQA (test) & 84.4 & 92.2 & 92.6 & 88.4 & 92.8 & $93.1^{\triangle}$ & 95.2 \\
\hline
\end{tabular}

**Table 29 Image understanding performance of our vision module attached to Llama 3. We compare model performance to GPT-4V, GPT-4o, Gemini 1.5 Pro, and Claude 3.5 Sonnet. ${ **^{\triangle}$ Results obtained using external OCR tools.}
\end{table}
of tasks and proper early stopping is applied. We select checkpoints at this stage purely based on benchmarks to ensure capabilities are retained or improved.

## 7.6 Image Recognition Results

We evaluate the performance of the image understanding capabilities of Llama 3 on a range of tasks spanning natural image understanding, text understanding, charts understanding and multimodal reasoning:
\begin{itemize}
\item[-] MMMU (Yue et al., 2024a) is a challenging dataset for mulitmodal reasoning where model is expected to understand images and solve college-level problems spanning 30 different disciplines. This includes both multiple-choice and open ended questions. We evaluate our model on the validation set with 900 images, in line with other works.
\item[-] VQAv2 (Antol et al., 2015) tests the ability of a model to combine image understanding, language understanding and commonsense knowlege to answer generic questions about natural images
\item[-] AI2 Diagram (Kembhavi et al., 2016) evaluates models capability to parse scientific diagrams and answer questions about the same. We use the same evaluation protocol as Gemini and x.ai, and report scores using a transparent bounding box.
\item[-] ChartQA (Masry et al., 2022) is a challenging benchmark for charts understanding. This requires model to visually understand different kinds of charts and answer logical questions about the charts.
\item[-] TextVQA (Singh et al., 2019) is a popular benchmark dataset that requires models to read and reason about text in images to answer questions about them. This tests the OCR understanding ability of the model on natural images.
\item[-] DocVQA (Mathew et al., 2020) is a benchmark dataset focused on document analysis and recognition. It contains images of a wide range of documents which evaluates a model's ability to perform OCR understanding and reason about the contents of a document to answer questions about them.
\end{itemize}
Table 29 presents the results of our experiments. The results in the table show that our vision module attached to Llama 3 performs competitively across a wide range of image-recognition benchmarks at varying model capacities. Using the resulting Llama 3-V 405B model, we outperform GPT-4V on all benchmarks, while being slightly behind Gemini 1.5 Pro and Claude 3.5 Sonnet. Llama 3 405B appears particularly competitive on document understanding tasks.

## 7.7 Video Recognition Results

We evaluate our video adapter for Llama 3 on three benchmarks:
\begin{itemize}
\item[-] PerceptionTest (Pătrăucean et al., 2023) evaluates the model's ability to answer temporal reasoning questions focusing on skills (memory, abstraction, physics, semantics) and different types of reasoning (descriptive, explanatory, predictive, counterfactual). It consists of $11.6 K$ test QA pairs, each with an on-average $23 s$ long video, filmed by 100 participants worldwide to show perceptually interesting tasks. We focus on the multiple-choice question answering task, where each question is paired with
\end{itemize}

\begin{table}
\begin{tabular}[t]{|l|l|l|l|l|l|l|l|}
\hline & Llama 3-V 8B & Llama 3-V 70B & Gemini 1.0 Pro & Gemini 1.0 Ultra & Gemini 1.5 Pro & GPT-4V & GPT-4o \\
\hline PerceptionTest (test) & 53.8 & 60.8 & 51.1 & 54.7 & - & - & - \\
\hline TVQA (val) & 82.5 & 87.9 & - & - & - & 87.3 & - \\
\hline NExT-QA (test) & 27.3 & 30.3 & 28.0 & 29.9 & - & - & - \\
\hline ActivityNet-QA (test) & 52.7 & 56.3 & 49.8 & 52.2 & 57.5 & - & 61.9 \\
\hline
\end{tabular}

**Table 30 Video understanding performance of our vision module attached to Llama 3. We find that across range of tasks covering long-form and temporal video understanding, our vision adapters for Llama3 8B and 70B parameters are competitive and sometimes even outperform alternative models.**
\end{table}
\begin{itemize}
\item[] three possible options. We report performance on the held-out test split which is accessed by submitting our predictions to an online challenge server. ${ }^{16}$
\item[-] NExT-QA (Xiao et al., 2021) is another temporal and causal reasoning benchmark, with a focus on open-ended question answering. It consists of $1 K$ test videos each on-average $44 s$ in length, paired with $9 K$ questions. The evaluation is performed by comparing the model's responses with the ground truth answer using Wu-Palmer Similarity (WUPS) (Wu and Palmer, 1994). ${ }^{17}$
\item[-] TVQA (Lei et al., 2018) evaluates the model's ability to perform compositional reasoning, requiring spatiotemporal localization of relevant moments, recognition of visual concepts, and joint reasoning with subtitle-based dialogue. This dataset, being derived from popular TV shows, additionally tests for the model's ability to leverage its outside-knowledge of those TV shows in answering the questions. It consists of over $15 K$ validation QA pairs, with each corresponding video clip being on-average $76 s$ in length. It also follows a multiple-choice format with five options for each question, and we report performance on the validation set following prior work (OpenAI, 2023b).
\item[-] ActivityNet-QA (Yu et al., 2019) evaluates the model's ability to reason over long video clips to understand actions, spatial relations, temporal relations, counting, etc. It consists of $8 K$ test QA pairs from 800 videos, each on-average 3 minutes long. For evaluation, we follow the protocol from prior work (Google, 2023; Lin et al., 2023; Maaz et al., 2024), where the model generates short one-word or one-phrase answers, and the correctness of the output is evaluated using the GPT-3.5 API which compares it to the ground truth answer. We report the average accuracy as evaluated by the API.
\end{itemize}

When performing inference, we uniformly sample frames from the full video clip and pass those frames into the model with a short text prompt. Since most of our benchmarks involve answering multiple-choice questions, we use the following prompt: Select the correct answer from the following options: \{question\}. Answer with the correct option letter and nothing else. For benchmarks that require producing a short answer (e.g., ActivityNet-QA and NExT-QA), we use the following prompt: Answer the question using a single word or phrase. \{question\}. For NExT-QA, since the evaluation metric (WUPS) is sensitive to the length and the specific words used, we additionally prompt the model to be specific and respond with the most salient answer, for instance specifying "living room" instead of simply responding with "house" when asked a location question. For benchmarks that contain subtitles (i.e., TVQA), we include the subtitles corresponding to the clip in the prompt during inference.

We present the performance of Llama 3 8B and 70B in Table 30. We compare Llama 3's performance with that of two Gemini and two GPT-4 models. Note that all our results are zero-shot, as we do not include any part of these benchmarks in our training or finetuning data. We find that our Llama 3 models that train a small video adapter during post-training are very competitive, and in some cases even better, than other models that potentially leverage native multimodal processing all the way from pre-training. Llama 3 performs particularly well on video recognition given that we only evaluate the 8B and 70B parameter models. Llama 3 achieves its best performance on PerceptionTest, suggesting the model has a strong ability to perform complex temporal reasoning. On long-form activity understanding tasks like ActivityNet-QA, Llama 3 is able to obtain strong results even though it is processing only up to 64 frames, which means that for a 3-minute long video the model only processes one frame every 3 seconds.

\footnotetext{
${ }^{16}$ See https://eval.ai/web/challenges/challenge-page/2091/overview.
${ }^{17}$ See https://github.com/doc-doc/NExT-OE.
}

![Figure](./grattafioriLlama3Herd2024_images/964281cf-3b75-4775-9ae9-065d3b6d211b-63_309_1543_266_288.jpg)

**Figure 29 Architecture of our speech interface for Llama 3.**

