# D Cost Analysis

More powerful models are cost-efficient for instruction proposal Despite higher per-token costs, we find larger, human-aligned models (models trained to follow human instructions (Ouyang et al., 2022)) dominate the accuracy-cost frontier of APE (Figure 11). Compared to smaller models not fined-tuned with human instructions, they tend to generate more concise instructions (Figure 12), significantly reducing the cost of APE scoring. Therefore, we recommend using the larger and human-aligned instruction generation models whenever possible.

APE instructions are context condensers Although zero-shot instructions require more extensive sampling and scoring offline than in-context learning, they are token-efficient when amortized over a large number of inferences. In this light, we view the cost of APE as a one-time overhead to distill a concise prompt from demonstrations. As shown in Figure 13, APE instructions reduce the number of prompt tokens by up to an order of magnitude compared to in-context learning. Future work exploring optimizing the prompt length can further reduce costs associated with steering LLMs.

![Figure](./zhouLargeLanguageModels2023_images/64ce2797-9859-4de8-95b4-245b838f0cd8-24_718_1311_885_404.jpg)

**Figure 11: The accuracy-cost frontier of APE across eight OpenAI models. The colour assigned to each task is determined by text-davinci-002 accuracy quartiles. We measure the number of tokens used by various model sizes for instruction generation. We also measure the number of tokens used to score 250 generated instructions on ten validation input-output pairs on InstructGPT (i.e., text-davinci-002). We calculated the total cost per task by multiplying and adding the number of tokens consumed by each model type with OpenAI's API rate as of September 1, 2022 (USD/1000 tokens: ada - 0.0004, babbage - 0.0005, curie - 0.0020, davinci - 0.0200). Counter-intuitively, smaller models are more expensive. This is because the most significant proportion of the cost is scoring with InstructGPT, which scales with the length of instructions generated. Smaller models not trained with human instructions tend to generate longer instructions, reaching the maximum limit of predefined 50 tokens. Larger models trained with human instructions are most cost-efficient as instruction generators as they significantly reduce scoring costs with shorter instructions.**

![Figure](./zhouLargeLanguageModels2023_images/64ce2797-9859-4de8-95b4-245b838f0cd8-25_713_1304_408_406.jpg)

**Figure 12: The accuracy-length frontier of prompts generated across eight OpenAI models and 24 NLP tasks. Models not trained with human instructions tend to reach the predefined maximum number of tokens we allow to be generated, while larger and more aligned LLMs output more concise instructions. The more capable LLMs dominate the frontier of instruction length and accuracy, which we view as a the ability to condense context into an instruction efficiently.**

![Figure](./zhouLargeLanguageModels2023_images/64ce2797-9859-4de8-95b4-245b838f0cd8-25_576_1302_1641_412.jpg)

**Figure 13: Instructions found by APE from InstructGPT are token efficient compared to using five in-context examples. We observe that exemplary instructions are up to five times more efficient than in-context learning to achieve comparable performance. Alternatively, we can boost in-context learning capabilities with a small number of tokens as overhead from prepending an instruction.**

