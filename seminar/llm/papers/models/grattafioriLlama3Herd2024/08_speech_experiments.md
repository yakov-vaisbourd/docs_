# 8 Speech Experiments

We perform experiments to study a compositional approach of integrating speech capabilities into Llama 3, resembling the method we used for visual recognition. On the input side, an encoder, together with an adapter, is incorporated to process speech signals. We leverage a system prompt (in text) to enable different modes of operation for speech understanding in Llama 3. If no system prompt is provided, the model acts as a general-purpose spoken dialogue model which can effectively respond to the user speech in a manner that is consistent with the text-only version of Llama 3. The dialogue history is introduced as the prompt prefix to improve the multi-round dialogue experience. We also experiment with system prompts that enable the use of Llama 3 for automatic speech recognition (ASR) and automatic speech translation (AST). The speech interface of Llama 3 supports up to 34 languages. ${ }^{18}$ It also allows for the interleaved input of text and speech, enabling the model to solve advanced audio-comprehension tasks.

We also experiment with a speech generation approach in which we implement a streaming text-to-speech (TTS) system that generates speech waveforms on-the-fly during language model decoding. We design the speech generator for Llama 3 based on a proprietary TTS system and do not fine-tune the language model for speech generation. Instead, we focus on improving speech synthesis latency, accuracy, and naturalness by leveraging Llama 3 embeddings at inference time. The speech interface is illustrated in Figure 28 and 29.

## 8.1 Data

## 8.1.1 Speech Understanding

The training data can be categorized into two types. The pre-training data includes a large amount of unlabeled speech, which is used to initialize the speech encoder in a self-supervised manner. The supervised finetuning data includes speech recognition, speech translation, and spoken dialogue data; this data is used to unlock specific abilities when integrated with the large language model.

Pre-training data. To pre-train the speech encoder, we curate a dataset of approximately 15M hours of speech recordings encompassing a large number of languages. We filter our audio data using a voice activity detection (VAD) model and select audio samples with a VAD threshold above 0.7 for pre-training. In speech pre-training data, we also focus on ensuring the absence of PII. We use the Presidio Analyzer to identify such PII.

Speech recognition and translation data. Our ASR training data contains 230K hours of manually transcribed speech recordings that span 34 languages. Our AST training data contains 90K hours of translations in two directions: from 33 languages to English and from English to 33 languages. This data contains both supervised and synthetic data generated using the NLLB toolkit (NLLB Team et al., 2022). The use of synthetic AST data enables us to increase model quality for low-resource languages. The speech segments in our data have a maximum length of 60 seconds.

Spoken dialogue data. To finetune the speech adapter for spoken dialogue, we synthetically generate responses

\footnotetext{
${ }^{18}$ The speech interface supports the following 34 languages: Arabic, Bengali, Chinese, Czech, Dutch, English, Finnish, French, German, Greek, Gujarati, Hindi, Hungarian, Indonesian, Italian, Japanese, Kannada, Korean, Malayalam, Marathi, Persian, Polish, Portuguese, Romanian, Russian, Spanish, Swahili, Swedish, Tamil, Telugu, Thai, Turkish, Urdu, Vietnamese.
}
for speech prompts by asking the language model to respond to transcriptions of those prompts (Fathullah et al., 2024). We generate synthetic data this way using a subset of the ASR dataset with 60K hours of speech. In addition, we generate 25K hours of synthetic data by running the Voicebox TTS system (Le et al., 2024) on subsets of the data used to finetune Llama 3. We used several heuristics to select a subset of finetuning data that matches the distribution of speech. These heuristics include focusing on relatively short prompts with a simple structure and without non-text symbols.

## 8.1.2 Speech Generation

The speech generation datasets mainly consist of those for training the text normalization (TN) model and the prosody model (PM). Both training data are augmented with an additional input feature of the Llama 3 embeddings to provide contextual information.

Text normalization data. Our TN training dataset includes 55K samples that cover a wide range of semiotic classes (e.g., number, date, time) that require non-trivial normalization. Each sample is a pair of written-form text and the corresponding normalized spoken-form text, with an inferred sequence of handcrafted TN rules that carry out the normalization.

Prosody model data. The PM training data includes linguistic and prosodic features extracted from a 50K-hour TTS dataset, which are paired transcripts and audios recorded by professional voice actors in studio settings.

Llama 3 embedding. The Llama 3 embeddings are taken as the output of the 16th decoder layer. We work exclusively with the Llama 3 8B model and extract the embeddings for a given text (i.e. written-form input text for TN or the audio transcript for PM) as if they are generated by the Llama 3 model with an empty user prompt. In a given sample, each chunk in the Llama 3 token sequence is explicitly aligned with the corresponding chunks in native input sequence for TN or PM, i.e., TN-specific text tokens (demarcated by unicode category) or phone-rate features respectively. This allows for training the TN and PM modules with streaming input of Llama 3 tokens and embeddings.

## 8.2 Model Architecture

## 8.2.1 Speech Understanding

On the input side, the speech module consists of two successive modules: a speech encoder and an adapter. The output of the speech module is directly fed into the language model as token representation, enabling direct interaction between speech and text tokens. Furthermore, we incorporate two new special tokens to enclose the sequence of speech representations. The speech module differs substantially from the vision module (see Section 7), which feeds multi-modal information into the language model via cross-attention layers. By contrast, the speech module generates embeddings that can be seamlessly integrated with text tokens, enabling the speech interface to leverage all the capabilities of the Llama 3 language model.

Speech encoder. Our speech encoder is a Conformer (Gulati et al., 2020) model with 1B parameters. The input to the model consists of 80 -dimensional mel-spectrogram features, which are first processed by a stride- 4 stacking layer followed by a linear projection to reduce the frame length to 40 ms. The resulting features are processed by an encoder with 24 Conformer layers. Each Conformer layer has a latent dimension of 1536, and consists of two Macron-net style feed-forward networks with dimension 4096, a convolution module with kernel size 7, and a rotary attention module (Su et al., 2024) with 24 attention heads.

Speech adapter. The speech adapter contains about 100M parameters. It is composed of a convolution layer, a rotary Transformer layer, and a linear layer. The convolution layer has a kernel size of 3 and a stride of 2, which is designed to reduce the speech frame length to 80ms. This allows the model to provide more coarse-grained features to the language model. The Transformer layer has a latent dimension of 3072 and a feed-forward network with a dimension of 4096 which further processes the information from speech with context after the convolutional downsampling. Finally, the linear layer maps the output dimension to match that of the language-model embedding layer.

## 8.2.2 Speech Generation

We use Llama 3 8B embeddings in two key components for speech generation: Text Normalization and Prosody Modeling. The TN module ensures semantic correctness by contextually transforming written text into spoken form. The PM module enhances naturalness and expressiveness by predicting prosodic features using these embeddings. Together, they enable accurate and natural speech generation.

Text normalization. As a determinant of the semantic correctness of generated speech, the text normalization (TN) module carries out context-aware transformation from written-form text into the respective spoken form which is eventually verbalized by the downstream components. For example, the written-form text 123 is read as a cardinal number (one hundred twenty three) or spelled digit-by-digit (one two three) depending on the semantic context. The TN system consists of a streaming LSTM-based sequence-tagging model that predicts the sequence of handcrafted TN rules used to transform the input text (Kang et al., 2024). The neural model also takes in Llama 3 embeddings via cross attention to leverage the contextual information encoded therein, enabling minimal text token lookahead and streaming input/output.

Prosody modeling. To enhance the naturalness and expressiveness of synthesized speech, we integrate a decoder-only Transformer-based Prosody model (PM) (Radford et al., 2021) that takes the Llama 3 embeddings as an additional input. This integration leverages the linguistic capabilities of Llama 3, utilizing both its textual output and intermediate embeddings at the token rate (Devlin et al., 2018; Dong et al., 2019; Raffel et al., 2020; Guo et al., 2023) to enhance the prediction of prosody features, thus reducing the lookahead required by the model.

The PM integrates several input components to generate comprehensive prosody predictions: linguistic features derived from the text normalization front-end detailed above, tokens, and embeddings. The PM predicts three key prosodic features: log duration of each phone, log F0 (fundamental frequency) average, and log power average across the phone duration. The model comprises a uni-directional Transformer and six attention heads. Each block includes cross-attention layers and dual fully connected layers with a hidden dimension of 864. A distinctive feature of the PM is its dual cross-attention mechanism, with one layer dedicated to linguistic inputs and the other to Llama embeddings. This setup efficiently manages varying input rates without requiring explicit alignment.

## 8.3 Training Recipe

## 8.3.1 Speech Understanding

Training of the speech module is done in two stages. The first stage, speech pre-training, leverages unlabeled data to train a speech encoder that exhibits strong generalization capabilities across languages and acoustic conditions. In the second stage, supervised fine-tuning, the adapter and pre-trained encoder are integrated with the language model, and trained jointly with it while the LLM stays frozen. This enables the model to respond to speech input. This stage uses labeled data corresponding to speech understanding abilities.

Multilingual ASR and AST modeling often results in language confusion/interference, which leads to degraded performance. A popular way to mitigate this is to incorporate language identification (LID) information, both on the source and target side. This can lead to improved performance in the predetermined set of directions, but it does come with potential loss of generality. For instance, if a translation system expects LID on both source and target side, then the model will not likely to show good zero-shot performance in directions that were not seen in training. So our challenge is to design a system that allows LID information to some extent, but keeps the model general enough such that we can have the model do speech translation in unseen directions. To address this, we design system prompts which only contain LID for the text to be emitted (target side). There is no LID information for the speech input (source side) in these prompts, which also potentially allows it to work with code-switched speech. For ASR, we use the following system prompt: Repeat after me in \{language\}:, where \{language\} comes from one of the 34 languages (English, French, etc.) For speech translation, the system prompt is: Translate the following sentence into \{language\}:. This design has been shown to be effective in prompting the language model to respond in the desired language. We used the same system prompts during training and inference.

Speech pre-training. We use the self-supervised BEST-RQ algorithm (Chiu et al., 2022) to pre-train the speech
encoder. We apply a mask of 32-frame length with a probability of 2.5\% to the input mel-spectrogram. If the speech utterances are longer than 60 seconds, we perform a random crop of 6K frames, corresponding to 60 seconds of speech. We quantize mel-spectrogram features by stacking 4 consecutive frames, projecting the 320-dimensional vectors to a 16-dimensional space, and performing a nearest-neighbor search with respect to cosine similarity metric within a codebook of 8,192 vectors. To stabilize pre-training, we employ 16 different codebooks. The projection matrix and codebooks are randomly initialized and are not updated throughout the model training. The multi-softmax loss is used only on masked frames for efficiency reasons. The encoder is trained for 500K steps with a global batch size of 2,048 utterances.

Supervised finetuning. Both the pre-trained speech encoder and the randomly initialized adapter are further jointly optimized with Llama 3 in the supervised finetuning stage. The language model remains unchanged during this process. The training data is a mixture of ASR, AST, and spoken dialogue data. The speech model for Llama 3 8B is trained for 650K updates, using a global batch size of 512 utterances and an initial learning rate of $10^{-4}$. The speech model for Llama 3 70B is trained for 600 K updates, using a global batch size of 768 utterances and an initial learning rate of $4 \times 10^{-5}$.

## 8.3.2 Speech Generation

To support real-time processing, the prosody model employs a lookahead mechanism that considers a fixed number of future phones and a variable number of future tokens. This ensures consistent lookahead while processing incoming text, which is crucial for low-latency speech synthesis applications.

Training. We develop a dynamic alignment strategy utilizing causal masking to facilitate streamability in speech synthesis. This strategy incorporates a lookahead mechanism for a fixed number of future phones and a variable number of future tokens, aligning with the chunking process during text normalization (Section 8.1.2). For each phone, the token lookahead includes the maximum number of tokens defined by the chunk size, resulting in variable lookahead for Llama embeddings but fixed lookahead for phonemes.

The Llama 3 embeddings are sourced from the Llama 3 8B model, which remains frozen during the training of the Prosody Model. The input phone-rate features include both linguistic and speaker/style controllability elements. The model training is conducted with a batch size of 1,024 utterances, each with a maximum length of 500 phones. We employ a learning rate of $9 \times 10^{-4}$ using the AdamW optimizer, training over 1 million updates with a learning rate warmup for the first 3,000 updates, following a cosine schedule.

Inference. During inference, the same lookahead mechanism and causal masking strategy are employed to ensure consistency between training and real-time processing. The PM handles incoming text in a streaming manner, updating the input phone by phone for phone-rate features and chunk by chunk for token-rate features. The new chunk input is updated only when the first phone for that chunk is current, maintaining the alignment and lookahead as during training.

For prosody target prediction, we employ a delayed pattern approach (Kharitonov et al., 2021), which enhances the model's ability to capture and reproduce long-range prosodic dependencies. This approach contributes to the naturalness and expressiveness of the synthesized speech, ensuring low-latency and high-quality output.

## 8.4 Speech Understanding Results

We evaluate the speech understanding capabilities of our speech interface for Llama 3 on three tasks: (1) automatic speech recognition, (2) speech translation, and (3) spoken question answering. We compare the performance of our speech interface for Llama 3 with three state-of-the-art models for speech understanding: Whisper (Radford et al., 2023), SeamlessM4T (Barrault et al., 2023), and Gemini. ${ }^{19}$ In all the evaluations, we used greedy search for Llama 3 token prediction.

Speech recognition. We evaluate the ASR performance on the English datasets of Multilingual LibriSpeech (MLS; Pratap et al. (2020)), LibriSpeech (Panayotov et al., 2015), VoxPopuli (Wang et al., 2021a), and a subset of the multilingual FLEURS dataset (Conneau et al., 2023). In evaluation, the decoding results are post-processed using the Whisper text normalizer to ensure consistency in comparing with the reported results of other models. On all benchmarks, we measure the word error rate of our speech interface for Llama 3

\footnotetext{
${ }^{19}$ Due to technical limitations, we compare with the performance of Gemini on MLS reported in the original paper.
}

\begin{table}
\begin{tabular}[t]{|l|l|l|l|l|l|l|}
\hline & Llama 3 8B & Llama 3 70B & Whisper & SeamlessM4T v2 & Gemini 1.0 Ultra & Gemini 1.5 Pro \\
\hline MLS (English) & 4.9 & 4.4 & 6.2 (v2) & 6.5 & 4.4 & 4.2 \\
\hline LibriSpeech (test-other) & 3.4 & 3.1 & 4.9 (v2) & 6.2 & - & - \\
\hline VoxPopuli (English) & 6.2 & 5.7 & 7.0 (v2) & 7.0 & - & - \\
\hline FLEURS (34 languages) & 9.6 & 8.2 & 14.4 (v3) & 11.7 & - & - \\
\hline
\end{tabular}

**Table 31 Word error rate of our speech interface for Llama 3 on speech recognition tasks. We report the performance of Whisper, SeamlessM4T, and Gemini for reference.**
\end{table}

\begin{table}
\begin{tabular}[t]{|l|l|l|l|l|}
\hline & Llama 3 8B & Llama 3 70B & Whisper v2 & SeamlessM4T v2 \\
\hline FLEURS (33 lang. → English) & 29.5 & 33.7 & 21.9 & 28.6 \\
\hline Covost 2 (15 lang. → English) & 34.4 & 38.8 & 33.8 & 37.9 \\
\hline
\end{tabular}

**Table 32 BLEU score of our speech interface for Llama 3 on speech translation tasks. We report the performance of Whisper and SeamlessM4T for reference.**
\end{table}
on the standard test set of those benchmarks, except for Chinese, Japanese, Korean and Thai, where the character error rate is reported.

Table 31 shows the results of ASR evaluations. It demonstrates the strong performance of Llama 3 (and multi-modal foundation models more generally) on speech recognition tasks: our model outperforms models that are tailored to speech like Whisper ${ }^{20}$ and SeamlessM4T on all benchmarks. On MLS English, Llama 3 performs similarly to Gemini.

Speech translation. We also evaluate our models on speech translation tasks in which the model is asked to translate non-English speech into English text. We use the FLEURS and Covost 2 (Wang et al., 2021b) datasets in these evaluations, measuring BLEU scores of the translated English. Table 32 presents the results of these experiments. ${ }^{21}$ The performance of our models in speech translation highlights the advantages of multimodal foundation models for tasks such as speech translation.

Spoken question answering. The speech interface of Llama 3 demonstrates remarkable question answering capabilities. The model can effortlessly comprehend code-switched speech without any prior exposure to such data. Notably, although the model was trained only on single-turn dialogue, it is capable of engaging in extended, coherent multi-turn dialogue sessions. Figure 30 presents a few examples that highlight these multilingual and multi-turn capabilities.

Safety. We evaluate the safety of our speech model on MuTox (Costa-jussà et al., 2023), a multilingual audio-based dataset of 20,000 utterances for English and Spanish and 4,000 for 19 other languages, each with toxicity labels attached. The audio is passed as input to the model and the output is evaluated for toxicity, after cleaning some special characters. We apply the MuTox classifier (Costa-jussà et al., 2023) and compare the results with Gemini 1.5 Pro. We evaluate the percentage of added toxicity (AT), when the input prompt is safe and the output is toxic, and the percentage of lost toxicity (LT), when the input prompt is toxic and the answer is safe. Table 33 shows the results for English and an average across all 21 languages that we evaluated on. ${ }^{22}$ The percentage of added toxicity is very low: our speech models have the lowest percentage of added toxicity for English, with less than 1\%. It removes significantly more toxicity than it adds.

## 8.5 Speech Generation Results

For speech generation, we focus on evaluating the quality of token-wise input streaming models with the Llama 3 embeddings for the text normalization and prosody modeling tasks. The evaluation focuses on

\footnotetext{
${ }^{20}$ On FLEURS ASR, Malayalam is not officially reported for Whisper v3, so we use the average of 33 languages.
${ }^{21}$ On Covost 2, we evaluate only on 15 (out of 21) languages.
${ }^{22}$ Note that for Gemini, we encountered that a significant number of responses were empty, which could be due to safety filters on their side (though some empty responses were for non-toxic input) or to rate limits. To conduct the analysis, we assumed that all the empty responses are safe. This is the most conservative approach for results and the upper bound of what Gemini results would look like.
}

![Figure](./grattafioriLlama3Herd2024_images/964281cf-3b75-4775-9ae9-065d3b6d211b-68_733_1598_262_264.jpg)

**Figure 30 Transcribed dialogue examples using the speech interface for Llama 3. The examples illustrate zero-shot multi-turn and code-switching capabilities.**

\begin{table}
\begin{tabular}[t]{|l|l|l|l|l|l|l|}
\hline \multirow[b]{2}{*}{Language} & \multicolumn{2}{|c|}{Llama 3 8B} & \multicolumn{2}{|c|}{Llama 3 70B} & \multicolumn{2}{|c|}{Gemini 1.5 Pro} \\
\hline & AT (↓) & LT (↑) & AT (↓) & LT (↑) & AT (↓) & LT (↑) \\
\hline English & 0.84 & 15.09 & 0.68 & 15.46 & 1.44 & 13.42 \\
\hline Overall & 2.31 & 9.89 & 2.00 & 10.29 & 2.06 & 10.94 \\
\hline
\end{tabular}

**Table 33 Speech toxicity of our speech interface to Llama 3 on the MuTox dataset. AT refers to added toxicity (\%) and LT refers to lost toxicity (\%).**
\end{table}
comparisons with models that do not take the Llama 3 embeddings as an additional input.
Text normalization. To measure the effect of Llama 3 embeddings, we experimented with changing the amount of right context the model uses. We trained the model using a right context of 3 TN tokens (demarcated by unicode category). This model is compared to models that do not use the Llama 3 embeddings, using a 3-token right context or a full bi-directional context. As expected, Table 34 shows using the full right context improves performance for the model without Llama 3 embeddings. However, the model that incorporates the Llama 3 embeddings outperforms all other models, hence enabling token-rate input/output streaming without relying on long context in the input.

Prosody modeling. To evaluate the performance of the our prosody model (PM) with Llama 3 8B, we conducted two sets of human evaluation comparing models with and without Llama 3 embeddings. Raters listened to samples from different models and indicated their preferences. To generate the final speech waveform, we use an inhouse transformer based acoustic model (Wu et al., 2021) that predicts spectral features and a WaveRNN neural vocoder (Kalchbrenner et al., 2018) to generate the final speech waveform.

\begin{table}
\begin{tabular}[t]{|l|l|l|}
\hline Model & Context & Accuracy \\
\hline Without Llama 3 8B & 3 & 73.6\% \\
\hline Without Llama 3 8B & $\infty$ & 88.0\% \\
\hline With Llama 3 8B & 3 & 90.7\% \\
\hline
\end{tabular}

**Table 34 Sample-wise text normalization (TN) accuracy. We compare models with or without Llama 3 8B embeddings, and using different right-context values.**
\end{table}

First, we compare directly to a streaming baseline model without Llama 3 embeddings. In the second test, the Llama 3 8B PM is compared to a non-streaming baseline model without Llama 3 embeddings. As shown in Table 35, the Llama 3 8B PM is preferred 60\% of the time compared to the streaming baseline, and

\begin{table}
\begin{tabular}[t]{|l|l|l|l|}
\hline Model & Preference & Model & Preference \\
\hline PM for Llama 3 8B & 60.0\% & PM for Llama 3 8B & 63.6\% \\
\hline Streaming phone-only baseline & 40.0\% & Non-streaming phone-only baseline & 36.4\% \\
\hline
\end{tabular}

**Table 35 Prosody Modeling (PM) evaluation. Left: Rater preferences of PM for Llama 3 8B vs. streaming phone-only baseline. Right: Rater preferences of PM for Llama 3 8B vs. non-streaming phone-only baseline.**
\end{table}
63.6\% of the time compared to the non-streaming baseline, indicating a significant improvement in perceived quality. The key advantage of the Llama 3 8B PM is its token-wise streaming capability (Section 8.2.2), which maintains low latency during inference. This reduces the model's lookahead requirements, enabling more responsive and real-time speech synthesis compared to non-streaming baselines. Overall, the Llama 3 8B prosody model consistently outperforms the baseline models, demonstrating its effectiveness in enhancing the naturalness and expressiveness of synthesized speech.

