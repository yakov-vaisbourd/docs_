# 6 Conclusion

This work introduces Self-Rag, a new framework to enhance the quality and factuality of LLMs through retrieval on demand and self-reflection. Self-Rag trains an LM to learn to retrieve, generate, and critique text passages and its own generation by predicting the next tokens from its original vocabulary as well as newly added special tokens, called reflection tokens. Self-Rag further enables the tailoring of LM behaviors at test time by leveraging reflection tokens. Our holistic evaluations on six tasks using multiple metrics demonstrate that Self-Rag significantly outperforms LLMs with more parameters or with conventional retrieval-augmented generation approaches.

