# 5 Experimental Setup

## 5.1 Datasets and Metrics

We evaluate on three standard neural generation datasets for the table-to-text task: E2E (Novikova et al., 2017), WebNLG (Gardent et al., 2017), and DART (Radev et al., 2020). The datasets are ordered by increasing complexity and size. E2E only has 1 domain (i.e. restaurant reviews); WebNLG has 14 domains, and DART is open-domain, using open-domain tables from Wikipedia.

The E2E dataset contains approximately 50K examples with 8 distinct fields; it contains multiple test references for one source table, and the average output length is 22.9. We use the official evaluation script, which reports BLEU (Papineni et al., 2002), NIST (Belz and Reiter, 2006), METEOR (Lavie and Agarwal, 2007), ROUGE-L (Lin, 2004), and CIDEr (Vedantam et al., 2015).

The WebNLG (Gardent et al., 2017) dataset consists of 22 K examples, and the input $x$ is a sequence of (subject, property, object) triples. The average output length is 22.5. In the training and validation splits, the input describes entities from 9 distinct DBpedia categories (e.g., Monument). The test split consists of two parts: the first half contains DB categories seen in training data, and the second half contains 5 unseen categories. These unseen categories are used to evaluate extrapolation. We use the official evaluation script, which reports BLEU, METEOR and TER (Snover et al., 2006).

DART (Radev et al., 2020) is an open domain table-to-text dataset, with similar input format (entity-relation-entity triples) as WebNLG. The average output length is 21.6. It consists of 82K examples from WikiSQL, WikiTableQuestions, E2E, and WebNLG and applies some manual or automated conversion. We use the official evaluation script and report BLEU, METEOR, TER, MoverScore (Zhao et al., 2019), BERTScore (Zhang et al., 2020b) and BLEURT (Sellam et al., 2020).

For the summarization task, we use the XSUM (Narayan et al., 2018) dataset, which is an abstrac-

\footnotetext{
${ }^{4} P_{\theta}$ has a dimension of \(\left|\mathrm{P}_{\mathrm{idx}}\right| \times \operatorname{dim}\left(h_{i}\right)\) while $P_{\theta}$ has a dimension of $\left|\mathrm{P}_{\mathrm{idx}}\right| \times k$, where we choose $k=512$ for table-to-text and 800 for summarization. MLP $_{\theta}$ maps from dimension $k$ to \(\operatorname{dim}\left(h_{i}\right)\)
}
tive summarization dataset on news articles. There are 225 K examples. The average length of the articles is 431 words and the average length of the summaries is 23.3. We report ROUGE-1, ROUGE-2 and ROUGE-L.

## 5.2 Methods

For table-to-text generation, we compare prefixtuning with three other methods: fine-tuning (FineTUNE), fine-tuning only the top 2 layers (FT-top2), and adapter-tuning (Adapter). ${ }^{5}$ We also report the current state-of-the-art results on these datasets: On E2E, Shen et al. (2019) uses a pragmatically informed model without pretraining. On WebNLG, Kale (2020) fine-tunes T5-large. On DART, no official models trained on this dataset version are released. ${ }^{6}$ For summarization, we compare against fine-tuning BART (Lewis et al., 2020).

## 5.3 Architectures and Hyperparameters

For table-to-text, we use GPT-2 medium and GPT-$2_{\text {LARGE }}$; the source tables are linearized. ${ }^{7}$ For summarization, we use BART ${ }_{\text {LARGE }},{ }^{8}$ and the source articles are truncated to 512 BPE tokens.

Our implementation is based on the Hugging Face Transformer models (Wolf et al., 2020). At training time, we use the AdamW optimizer (Loshchilov and Hutter, 2019) and a linear learning rate scheduler, as suggested by the Hugging Face default setup. The hyperparameters we tune include the number of epochs, batch size, learning rate, and prefix length. Hyperparameter details are in the appendix. A default setting trains for 10 epochs, using a batch size of 5, a learning rate of $5 \cdot 10^{-5}$ and a prefix length of 10 . The table-to-text models are trained on TITAN Xp or GeForce GTX TITAN X machines. Prefix-tuning takes 0.2 hours per epochs to train on 22 K examples, whereas finetuning takes around 0.3 hours. The summarization models are trained on Tesla V100 machines, taking 1.25h per epoch on the XSUM dataset.

At decoding time, for the three table-to-text datasets, we use beam search with a beam size of 5. For summarization, we use a beam size of 6

\footnotetext{
${ }^{5}$ Same implementation as Lin et al. (2020).
${ }^{6}$ The official benchmark model is trained on v.1.0.0 while the release dataset is v1.1.1.
${ }^{7}$ In comparison with natural language utterances, the linearized table is in an unnatural format, which might be challenging for pretrained LMs.
${ }^{8}$ We didn't include GPT-2 results for summarization because in our preliminary experiment, fine-tuning GPT-2 significantly underperforms fine-tuning BART on XSUM.
}
and length normalization of 0.8 . Decoding takes 1.2 seconds per sentence (without batching) for table-to-text, and 2.6 seconds per batch (using a batch size of 10) for summarization.

