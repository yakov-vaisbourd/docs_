\title{
Active Retrieval Augmented Generation
}

\author{
Zhengbao Jiang ${ }^{1 *}$ Frank F. Xu ${ }^{1 *}$ Luyu Gao ${ }^{1 *}$ Zhiqing Sun ${ }^{1 *}$ Qian Liu ${ }^{2}$ Jane Dwivedi-Yu ${ }^{3}$ Yiming Yang ${ }^{1}$ Jamie Callan ${ }^{1}$ Graham Neubig ${ }^{1}$ \\ ${ }^{1}$ Language Technologies Institute, Carnegie Mellon University \\ ${ }^{2}$ Sea AI Lab ${ }^{3}$ FAIR, Meta \\ \{zhengbaj,fangzhex,luyug,zhiqings,gneubig\}@cs.cmu.edu
}

\begin{abstract}
Despite the remarkable ability of large language models (LMs) to comprehend and generate language, they have a tendency to hallucinate and create factually inaccurate output. Augmenting LMs by retrieving information from external knowledge resources is one promising solution. Most existing retrieval augmented LMs employ a retrieve-and-generate setup that only retrieves information once based on the input. This is limiting, however, in more general scenarios involving generation of long texts, where continually gathering information throughout generation is essential. In this work, we provide a generalized view of $a c$ tive retrieval augmented generation, methods that actively decide when and what to retrieve across the course of the generation. We propose Forward-Looking Active REtrieval augmented generation (FLARE), a generic method which iteratively uses a prediction of the upcoming sentence to anticipate future content, which is then utilized as a query to retrieve relevant documents to regenerate the sentence if it contains low-confidence tokens. We test FLARE along with baselines comprehensively over 4 longform knowledge-intensive generation tasks/datasets. FLARE achieves superior or competitive performance on all tasks, demonstrating the effectiveness of our method. ${ }^{1}$
\end{abstract}

