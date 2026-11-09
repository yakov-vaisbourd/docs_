# 7 Related Work

We refer to subsection 2.2 and section 4 for extensively discussion on single-time and multi-time retrieval augmented LMs, which is the most relevant area to this paper.

Iterative and adaptive retrieval Iterative retrieval and refinement has been studied in both text and code generation tasks (Peng et al., 2023; Zhang et al., 2023; Zemlyanskiy et al., 2022; Yu et al., 2023). FLARE differs from these methods in the granularity of generation and retrieval strategies. Adaptive retrieval has been studied in single-time retrieval scenarios based on either question popularity or generation probabilities (Mallen et al., 2022; Li et al., 2023), while we focus on long-form generation requiring active information access.

Browser-enhanced LMs WebGPT (Nakano et al., 2021) and WebCPM (Qin et al., 2023) train LMs to interact with browser to enhance factuality using reinforcement learning or supervised training where multiple queries can be triggered before generation. FLARE is built on text-based retrievers but can be combined with a browser to potentially improve retrieval quality.

