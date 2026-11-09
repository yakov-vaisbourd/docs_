# A Related Work

Table 11 shows which of the strategies for continued pretraining have already been explored in the prior work from the Related Work (§6). As evident from the table, our work compares various strategies as well as their interplay using a pretrained language model trained on a much more heterogeneous pretraining corpus.

# A. 1 RoBERTA's Pretraining Corpus

ROBERTA was trained on data from BookCorPUS (Zhu et al., 2015), ${ }^{6}$ Wikipedia, ${ }^{7}$ a portion of the CCNews dataset (Nagel, 2016), ${ }^{8}$ OpenWebText corpus of Web content extracted from URLs shared on Reddit (Gokaslan and Cohen, 2019), ${ }^{9}$ and a subset of CommonCrawl that it is said to resemble the "story-like" style of Winograd schemas (STORIES; Trinh and Le, 2018). ${ }^{10}$

# A. 2 State of the Art

In this section, we specify the models achieving state of the art on our tasks. See the caption of

\footnotetext{
${ }^{6}$ https://github.com/soskek/bookcorpus
${ }^{7}$ https://github.com/google-research/ bert
${ }^{8}$ https://github.com/fhamborg/ news-please
${ }^{9}$ https://github.com/jcpeterson/ openwebtext
${ }^{10}$ https://github.com/tensorflow/models/ tree/master/research/lm_commonsense
}
Table 5 for the reported performance of these models. For ACL-ARC, that is SciBERT (Beltagy et al., 2019), a BERT-base model for trained from scratch on scientific text. For ChemProt and SciERC, that is S2ORC-BERT (Lo et al., 2020), a similar model to SciBERT. For AGNews and IMDB, XLNet-large, a much larger model. For RCT, Cohan et al. (2019). For HyperPartisan, Longformer, a modified Transformer language model for long documents (Beltagy et al., 2020). Thongtan and Phienthrakul (2019) report a higher number (97.42) on IMDB, but they train their word vectors on the test set. Our baseline establishes the first benchmark for the Helpfulness dataset.

