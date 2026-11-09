# A Scalability and Computational Efficiency of the Tree-Building Process

To assess the computational efficiency and cost-effectiveness of RAPTOR's tree-building process, we conducted experiments on a consumer-grade laptop, specifically an Apple M1 Mac with 16GB of RAM. These experiments aimed to demonstrate the scalability and feasibility of RAPTOR on typical hardware. We varied the context length from 12,500 to 78,000 tokens and measured both the token expenditure and the time required to complete the tree-building process, from initial splitting and embedding to the construction of the final root node.

![Figure](./sarthiRAPTORRecursiveAbstractive2024_images/4beeda7d-949d-4d45-a072-4861cfa71615-15_452_1376_1796_373.jpg)

**Figure 5: Token cost as a function of document length for QASPER, NarrativeQA, and QuALITY. RAPTOR tree construction costs scale linearly with document length for each of the datasets.**

Token Expenditure We empirically investigated the relationship between the initial document length and the total number of tokens expended during the tree-building process, which includes both the prompt and completion tokens. The document lengths varied significantly across the three
datasets examined: QuALITY, QASPER, and NarrativeQA. Figure 5 illustrates a clear linear correlation between the initial document length and the total token expenditure, emphasizing that RAP-TOR maintains a linear token scaling regardless of document complexity or length.

![Figure](./sarthiRAPTORRecursiveAbstractive2024_images/4beeda7d-949d-4d45-a072-4861cfa71615-16_537_602_516_752.jpg)

**Figure 6: Build time as a function of document length for documents of up to 80,000 tokens. RAP-TOR tree construction time scales linearly with document length for each of the datasets.**

Build Time We also empirically observed a consistent linear trend between the document length and the build time, as shown in Figure 6. This suggests that RAPTOR scales linearly in terms of time, making it a viable solution for efficiently processing large corpora of varying lengths.

Conclusion Overall, our empirical results indicate that RAPTOR scales both in terms of tokens expended and build time. Even as the complexity and volume of the input text grow, the cost of constructing the tree scales predictably and linearly. This demonstrates that RAPTOR is computationally efficient and well-suited for processing large and diverse corpora.

