# B Datasets and Settings

Datasets, metrics, and experimental settings are summarized in Table 7.

\footnotetext{
${ }^{5}$ https://www.nltk.org/api/nltk.tokenize. PunktSentenceTokenizer.html
}

Multihop QA For "Why did the founder of Versus die?", the output we aim to generate is "The founder of Versus was Gianni Versace. Gianni Versace was shot and killed on the steps of his Miami Beach mansion on July 15, 1997. So the answer is shot." We use 8 exemplars from Trivedi et al. (2022) listed in Prompt D. 4 for in-context learning, BM25 as the retriever, and Wikipedia articles as the retrieval corpus. Similar to the observation in Trivedi et al. (2022), we found incorporating retrieval results for exemplars improves the performance, we use the input $\boldsymbol{x}$ of each exemplar to retrieve several documents and then add them using the format in Prompt D.1. We found increasing the number of retrieval documents often increases performance. Therefore, we use the maximum number of documents that can fit within the input length limit of text-davinci-003, which is 2 for 2WikiMultihopQA.

Commonsense Reasoning For "Would a pear sink in water?", the output we aim to generate is "The density of a pear is about $0.6 \mathrm{~g} / \mathrm{cm}^{3}$, which is less than water. Objects less dense than water float. Thus, a pear would float. So the final answer is no." We use 6 exemplars from Wei et al. (2022) listed in Prompt D.5, BM25 on the Wikipedia corpus, and 3 retrieved documents to run experiments.

Long-form QA For "Where do the Philadelphia Eagles play their home games?", the output we aim to generate is "We need to consider the different possible locations or venues that could be considered the home field of the Philadelphia Eagles. These include the city, the sports complex, or the stadium. Therefore, this question has 3 interpretations and the answers are: (1) The city is Philadelphia. (2) The sports complex is the South Philadelphia Sports Complex. (3) The stadium is the Lincoln Financial Field stadium." For both the original setting (ASQA) and the setting with hints (ASQA-hint), we manually annotate 8 exemplars (Prompt D. 6 and D.8), use BM25 on the Wikipedia corpus, and 3 retrieved documents to run experiments.

Open-domain Summarization The original WikiAsp dataset is designed for multi-document summarization and provides a list of references to systems. We converted it into the open-domain setting by removing the associated references and instead gathering information from the open web. For "Generate a summary about Echo School (Ore-
gon) including the following aspects: academics, history.", the output we aim to generate is "\# Academics. In 2008, 91\% of the school's seniors received their high school diploma... \# History. The class of 2008 was the 100th class in the school's history." where \# is used to indicate aspects. We manually annotate 4 exemplars (Prompt D.10), and use the Bing search engine to retrieve 5 documents from the open web. To avoid leaking, we exclude several Wikipedia-related domains listed in Table 8 from Bing's search results.

