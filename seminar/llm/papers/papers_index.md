# LLM Literature Review: Fine-Tuning, Instruction Tuning & Prompt Engineering

Index of papers held in the Zotero collection **ML > LLM/VLM > SFT**, converted to Markdown via `/mathpix` and stored under this folder (`4_docs/llm/papers/`). Each entry links to (a) the local vault copy and (b) the Zotero item/PDF, so a Claude session can jump straight to either — e.g. to open the PDF for a highlight, or to attach a note/annotation back onto the Zotero item.

## How to act on these in Zotero (for Claude sessions)
- **Zotero item key** below opens the item in the desktop app: `zotero://select/library/items/<ITEM_KEY>`
- **Zotero MCP tools** (`mcp__zotero-mcp__*`) operate on the item key directly — no need to resolve via search:
  - `get_item_details` / `get_item_abstract` — fetch metadata/abstract
  - `get_annotations` — read existing highlights/annotations on the attached PDF
  - `write_note` — attach a new standalone or child note to the item (e.g. link back to the vault `.md` summary)
  - `write_tag` — add/update tags on the item
  - `get_content` / `search_fulltext` — pull full text or search within the PDF
- The **attachment key** (the PDF child item) is what `get_annotations` and PDF-level operations need — pass that, not the parent item key, when the target is annotations on the reader.
- Local PDF path (for tools that need a filesystem path rather than the API) is under `C:\Users\yakov\Zotero\storage\<ATTACHMENT_KEY>\`.
- To cross-link an Obsidian note back to Zotero, use the `zotero://select/library/items/<ITEM_KEY>` URL directly in the note (Obsidian renders it as a clickable external link) or add it as a `write_note` child note on the Zotero side pointing back to the vault path.
- Vault copies over 500 lines were split into a per-paper subfolder (main file + one `.md` per section), per this vault's `/mathpix` workflow; shorter papers stay as a single `.md` file. Links below point to whichever form applies.

---

## Finetune Instruct (PEFT methods & Retrieval-Augmented Generation)

### LoRA: Low-Rank Adaptation of Large Language Models
- **Authors:** Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen
- **Published:** 2021-10-16 (arXiv)
- **Focus:** Freezes pretrained weights and injects trainable low-rank decomposition matrices into each Transformer layer, cutting trainable parameters and GPU memory by orders of magnitude while matching or beating full fine-tuning quality.
- **Vault copy:** [Finetune_Instruct/huLoRALowRankAdaptation2021/huLoRALowRankAdaptation2021.md](Finetune_Instruct/huLoRALowRankAdaptation2021/huLoRALowRankAdaptation2021.md)
- **Zotero item:** `zotero://select/library/items/NBVZ3T7R` (attachment key `KZJ4YFEW`)

### Parameter-Efficient Transfer Learning for NLP
- **Authors:** Neil Houlsby, Andrei Giurgiu, Stanisław Jastrzębski, Bruna Morrone, et al.
- **Focus:** Introduces adapter modules — small bottleneck layers inserted into a frozen pretrained Transformer — as a parameter-efficient alternative to full fine-tuning across many downstream NLP tasks.
- **Vault copy:** [Finetune_Instruct/houlsbyParameterEfficientTransferLearning.md](Finetune_Instruct/houlsbyParameterEfficientTransferLearning.md)
- **Zotero item:** `zotero://select/library/items/M8EUMRGN` (attachment key `DXLC2W5A`)

### Don't Stop Pretraining: Adapt Language Models to Domains and Tasks
- **Authors:** Suchin Gururangan, Ana Marasović, Swabha Swayamdipta, Kyle Lo, Iz Beltagy, Doug Downey, Noah A. Smith
- **Published:** 2020
- **Focus:** Shows that a second phase of pretraining on domain- and task-specific unlabeled text (domain-/task-adaptive pretraining) reliably improves downstream performance, even after broad-coverage pretraining.
- **Vault copy:** [Finetune_Instruct/gururanganDontStopPretraining2020/gururanganDontStopPretraining2020.md](Finetune_Instruct/gururanganDontStopPretraining2020/gururanganDontStopPretraining2020.md)
- **Zotero item:** `zotero://select/library/items/Q5KITTAM` (attachment key `B4X522G2`)

### Editing Models with Task Arithmetic
- **Authors:** Gabriel Ilharco, Marco Tulio Ribeiro, Mitchell Wortsman, Suchin Gururangan, Ludwig Schmidt, Hannaneh Hajishirzi, Ali Farhadi
- **Published:** 2023-03-31 (arXiv)
- **Focus:** Defines "task vectors" (fine-tuned minus pretrained weights) that can be added, negated, and combined via simple arithmetic to steer model behavior across multiple tasks without further training.
- **Vault copy:** [Finetune_Instruct/ilharcoEditingModelsTask2023/ilharcoEditingModelsTask2023.md](Finetune_Instruct/ilharcoEditingModelsTask2023/ilharcoEditingModelsTask2023.md)
- **Zotero item:** `zotero://select/library/items/2659K7Y8` (attachment key `7EJAQB8N`)

### PiSSA: Principal Singular Values and Singular Vectors Adaptation of Large Language Models
- **Authors:** Fanxu Meng, Zhaohui Wang, Muhan Zhang
- **Published:** 2025-04-09 (arXiv)
- **Focus:** Reworks LoRA's initialization by seeding the low-rank adapters with the principal singular components of the original weight matrix (rather than noise + zeros), giving faster convergence and better accuracy at the same architecture and cost.
- **Vault copy:** [Finetune_Instruct/mengPiSSAPrincipalSingular2025/mengPiSSAPrincipalSingular2025.md](Finetune_Instruct/mengPiSSAPrincipalSingular2025/mengPiSSAPrincipalSingular2025.md)
- **Zotero item:** `zotero://select/library/items/64GGVGPU` (attachment key `NRWIIV6Q`)

### Prefix-Tuning: Optimizing Continuous Prompts for Generation
- **Authors:** Xiang Lisa Li, Percy Liang
- **Published:** 2021-01-01 (arXiv)
- **Focus:** Keeps the LM frozen and instead optimizes a small continuous "prefix" vector prepended to the input, learning only ~0.1% of parameters while matching or beating full fine-tuning on generation tasks.
- **Vault copy:** [Finetune_Instruct/liPrefixTuningOptimizingContinuous2021/liPrefixTuningOptimizingContinuous2021.md](Finetune_Instruct/liPrefixTuningOptimizingContinuous2021/liPrefixTuningOptimizingContinuous2021.md)
- **Zotero item:** `zotero://select/library/items/GZQG7ZKV` (attachment key `4XSUX27P`)

### The Power of Scale for Parameter-Efficient Prompt Tuning
- **Authors:** Brian Lester, Rami Al-Rfou, Noah Constant
- **Published:** 2021-09-02 (arXiv)
- **Focus:** Learns "soft prompts" via backpropagation on a frozen LM; shows the gap to full model tuning closes as model scale grows into the billions of parameters, simplifying prefix-tuning further.
- **Vault copy:** [Finetune_Instruct/lesterPowerScaleParameterEfficient2021/lesterPowerScaleParameterEfficient2021.md](Finetune_Instruct/lesterPowerScaleParameterEfficient2021/lesterPowerScaleParameterEfficient2021.md)
- **Zotero item:** `zotero://select/library/items/LVGT9PDW` (attachment key `J9Z2WJLC`)

### Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks
- **Authors:** Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, et al.
- **Published:** 2021-04-12 (arXiv)
- **Focus:** Foundational RAG paper combining a pretrained seq2seq generator (parametric memory) with a dense Wikipedia retriever (non-parametric memory), fine-tuned end-to-end for open-domain QA and knowledge-intensive generation.
- **Vault copy:** [Finetune_Instruct/lewisRetrievalAugmentedGenerationKnowledgeIntensive2021.md](Finetune_Instruct/lewisRetrievalAugmentedGenerationKnowledgeIntensive2021.md)
- **Zotero item:** `zotero://select/library/items/ED4VIC3C` (attachment key `JH7VMPBJ`)

### Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection
- **Authors:** Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, Hannaneh Hajishirzi
- **Published:** 2023-10-17 (arXiv)
- **Focus:** Trains a single LM to adaptively decide when to retrieve, and to critique its own retrieved passages and generations via special "reflection tokens," improving factuality and citation accuracy over fixed-retrieval RAG.
- **Vault copy:** [Finetune_Instruct/asaiSelfRAGLearningRetrieve2023/asaiSelfRAGLearningRetrieve2023.md](Finetune_Instruct/asaiSelfRAGLearningRetrieve2023/asaiSelfRAGLearningRetrieve2023.md)
- **Zotero item:** `zotero://select/library/items/IZETPSRS` (attachment key `NBYQT7FZ`)

### From Local to Global: A Graph RAG Approach to Query-Focused Summarization
- **Authors:** Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva Mody, et al.
- **Published:** 2025-02-19 (arXiv)
- **Focus:** Builds an LLM-derived entity knowledge graph with pre-generated community summaries, then answers global "sensemaking" questions over an entire corpus by map-reducing over community summaries — where plain RAG fails.
- **Vault copy:** [Finetune_Instruct/edgeLocalGlobalGraph2025/edgeLocalGlobalGraph2025.md](Finetune_Instruct/edgeLocalGlobalGraph2025/edgeLocalGlobalGraph2025.md)
- **Zotero item:** `zotero://select/library/items/ZCL36GNI` (attachment key `XB6KK9C7`)

### RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval
- **Authors:** Parth Sarthi, Salman Abdullah, Aditi Tuli, Shubh Khanna, Anna Goldie, Christopher D. Manning
- **Published:** 2024-01-31 (arXiv)
- **Focus:** Recursively embeds, clusters, and summarizes text chunks into a tree of multi-level summaries, letting retrieval integrate information at different levels of abstraction — strong results on multi-step QA (QuALITY).
- **Vault copy:** [Finetune_Instruct/sarthiRAPTORRecursiveAbstractive2024/sarthiRAPTORRecursiveAbstractive2024.md](Finetune_Instruct/sarthiRAPTORRecursiveAbstractive2024/sarthiRAPTORRecursiveAbstractive2024.md)
- **Zotero item:** `zotero://select/library/items/ALWVW2B3` (attachment key `GT3MLL3V`)

### Active Retrieval Augmented Generation (FLARE)
- **Authors:** Zhengbao Jiang, Frank F. Xu, Luyu Gao, Zhiqing Sun, Qian Liu, Jane Dwivedi-Yu, Yiming Yang, Jamie Callan, Graham Neubig
- **Published:** 2023-10-22 (arXiv)
- **Focus:** Proposes FLARE, which iteratively predicts the upcoming sentence and re-retrieves whenever that prediction contains low-confidence tokens — retrieval decisions made actively throughout generation rather than once up front.
- **Vault copy:** [Finetune_Instruct/jiangActiveRetrievalAugmented2023/jiangActiveRetrievalAugmented2023.md](Finetune_Instruct/jiangActiveRetrievalAugmented2023/jiangActiveRetrievalAugmented2023.md)
- **Zotero item:** `zotero://select/library/items/ZVLWIA9L` (attachment key `C5R3363Z`)

---

## Finetune Base (Instruction Tuning, RLHF & Post-Training)

### Self-Instruct: Aligning Language Models with Self-Generated Instructions
- **Authors:** Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A. Smith, Daniel Khashabi, Hannaneh Hajishirzi
- **Published:** 2023-05-25 (arXiv)
- **Focus:** Bootstraps an instruction-tuning dataset from a pretrained LM's own generations (instructions, inputs, outputs), then filters and fine-tunes on it — a near annotation-free path to instruction-following ability.
- **Vault copy:** [Finetune_Base/wangSelfInstructAligningLanguage2023/wangSelfInstructAligningLanguage2023.md](Finetune_Base/wangSelfInstructAligningLanguage2023/wangSelfInstructAligningLanguage2023.md)
- **Zotero item:** `zotero://select/library/items/MADIMUAS` (attachment key `3F43FNFL`)

### LIMA: Less Is More for Alignment
- **Authors:** Chunting Zhou, Pengfei Liu, Puxin Xu, Srini Iyer, Jiao Sun, Yuning Mao, Xuezhe Ma, et al.
- **Published:** 2023-05-18 (arXiv)
- **Focus:** Fine-tunes a 65B LLaMA on only 1,000 carefully curated prompt/response pairs (no RLHF) and shows strong, competitive performance — evidence that most knowledge comes from pretraining and only limited instruction data is needed to elicit it.
- **Vault copy:** [Finetune_Base/zhouLIMALessMore2023/zhouLIMALessMore2023.md](Finetune_Base/zhouLIMALessMore2023/zhouLIMALessMore2023.md)
- **Zotero item:** `zotero://select/library/items/QTDMX5XY` (attachment key `49P38CFG`)

### Finetuned Language Models Are Zero-Shot Learners (FLAN)
- **Authors:** Jason Wei, Maarten Bosma, Vincent Y. Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M. Dai, Quoc V. Le
- **Published:** 2022-02-08 (arXiv)
- **Focus:** Instruction-tunes a 137B LM across 60+ NLP datasets verbalized as natural-language instructions (FLAN), substantially improving zero-shot generalization to unseen task types and surpassing zero-shot GPT-3 on most benchmarks.
- **Vault copy:** [Finetune_Base/weiFinetunedLanguageModels2022/weiFinetunedLanguageModels2022.md](Finetune_Base/weiFinetunedLanguageModels2022/weiFinetunedLanguageModels2022.md)
- **Zotero item:** `zotero://select/library/items/YRVCEPJF` (attachment key `MV4H3BKG`)

### Training language models to follow instructions with human feedback (InstructGPT)
- **Authors:** Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, et al.
- **Published:** 2022-03-04 (arXiv)
- **Focus:** Introduces the SFT → reward model → RLHF pipeline (InstructGPT); a 1.3B RLHF'd model is preferred by human raters over the 175B base GPT-3, with improved truthfulness and reduced toxicity.
- **Vault copy:** [Finetune_Base/ouyangTrainingLanguageModels2022/ouyangTrainingLanguageModels2022.md](Finetune_Base/ouyangTrainingLanguageModels2022/ouyangTrainingLanguageModels2022.md)
- **Zotero item:** `zotero://select/library/items/SSD6EF3L` (attachment key `T6P7GIXS`)

### SFT Memorizes, RL Generalizes: A Comparative Study of Foundation Model Post-training
- **Authors:** Tianzhe Chu, Yuexiang Zhai, Jihan Yang, Shengbang Tong, Saining Xie, Dale Schuurmans, Quoc V. Le, Sergey Levine, Yi Ma
- **Published:** 2025-05-26 (arXiv)
- **Focus:** Directly compares SFT vs. RL post-training on rule-based text and visual reasoning tasks; finds RL (with outcome-based reward) generalizes to unseen rule/visual variants while SFT tends to memorize — though SFT still helps by stabilizing output format for subsequent RL.
- **Vault copy:** [Finetune_Base/chuSFTMemorizesRL2025/chuSFTMemorizesRL2025.md](Finetune_Base/chuSFTMemorizesRL2025/chuSFTMemorizesRL2025.md)
- **Zotero item:** `zotero://select/library/items/KP2RA66Q` (attachment key `XLGYA46W`)

### Scaling Instruction-Finetuned Language Models (Flan-PaLM / Flan-T5)
- **Authors:** Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Yunxuan Li, Xuezhi Wang, Mostafa Dehghani, et al.
- **Published:** 2022-12-06 (arXiv)
- **Focus:** Scales instruction finetuning along number of tasks, model size, and chain-of-thought data; Flan-PaLM 540B (1.8K tasks) beats PaLM 540B by +9.4% average and sets SOTA on MMLU — also releases the widely-used Flan-T5 checkpoints.
- **Vault copy:** [Finetune_Base/chungScalingInstructionFinetunedLanguage2022/chungScalingInstructionFinetunedLanguage2022.md](Finetune_Base/chungScalingInstructionFinetunedLanguage2022/chungScalingInstructionFinetunedLanguage2022.md)
- **Zotero item:** `zotero://select/library/items/69S9ISTS` (attachment key `M7DCZQU3`)

### TailSFT: Filtered Fine-Tuning Improves Post-Training Performance
- **Authors:** Sadhika Malladi, Samy Jelassi, Dylan Foster, Jordan T. Ash, Akshay Krishnamurthy
- **Published:** 2026-08-26 (arXiv)
- **Focus:** Filters out already-well-fit training sequences during SFT to focus learning on the under-modeled "tail" of the data distribution, improving pass@K coverage and yielding better base checkpoints for subsequent RL (GRPO).
- **Vault copy:** [Finetune_Base/malladiTailSFTFilteredFineTuning2026/malladiTailSFTFilteredFineTuning2026.md](Finetune_Base/malladiTailSFTFilteredFineTuning2026/malladiTailSFTFilteredFineTuning2026.md)
- **Zotero item:** `zotero://select/library/items/WN86QWUN` (attachment key `FMCEKX8L`)

---

## Prompt Engineering (In-Context Learning & Prompt Optimization)

### Large Language Models as Optimizers (OPRO)
- **Authors:** Chengrun Yang, Xuezhi Wang, Yifeng Lu, Hanxiao Liu, Quoc V. Le, Denny Zhou, Xinyun Chen
- **Published:** 2024-04-15 (arXiv)
- **Focus:** Uses an LLM itself as a black-box optimizer (OPRO) — the optimization trajectory is described in natural language in the prompt, and the LLM proposes improved solutions each step; applied to prompt optimization with large gains on GSM8K and Big-Bench Hard.
- **Vault copy:** [Prompt_Engineering/yangLargeLanguageModels2024/yangLargeLanguageModels2024.md](Prompt_Engineering/yangLargeLanguageModels2024/yangLargeLanguageModels2024.md)
- **Zotero item:** `zotero://select/library/items/CPMND2W8` (attachment key `Y239PZC3`)

### DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines
- **Authors:** Omar Khattab, Arnav Singhvi, Paridhi Maheshwari, Zhiyuan Zhang, Keshav Santhanam, et al.
- **Published:** 2023-10-05 (arXiv)
- **Focus:** Replaces hand-written prompt templates with DSPy, a programming model that treats LM pipelines as text-transformation graphs of parameterized, declarative modules, compiled/optimized automatically to maximize a target metric.
- **Vault copy:** [Prompt_Engineering/khattabDSPyCompilingDeclarative2023/khattabDSPyCompilingDeclarative2023.md](Prompt_Engineering/khattabDSPyCompilingDeclarative2023/khattabDSPyCompilingDeclarative2023.md)
- **Zotero item:** `zotero://select/library/items/F5S6CBM8` (attachment key `N8H754GI`)

### Calibrate Before Use: Improving Few-Shot Performance of Language Models
- **Authors:** Tony Z. Zhao, Eric Wallace, Shi Feng, Dan Klein, Sameer Singh
- **Published:** 2021-06-10 (arXiv)
- **Focus:** Shows few-shot prompting accuracy is highly unstable due to model bias toward certain answers (recency/majority effects); introduces "contextual calibration" using a content-free input to correct this bias, substantially improving and stabilizing GPT-3/GPT-2 accuracy.
- **Vault copy:** [Prompt_Engineering/zhaoCalibrateUseImproving2021/zhaoCalibrateUseImproving2021.md](Prompt_Engineering/zhaoCalibrateUseImproving2021/zhaoCalibrateUseImproving2021.md)
- **Zotero item:** `zotero://select/library/items/RQYSVAQJ` (attachment key `W8HJFJBA`)

### Why Can GPT Learn In-Context? Language Models Implicitly Perform Gradient Descent as Meta-Optimizers
- **Authors:** Damai Dai, Yutao Sun, Li Dong, Yaru Hao, Shuming Ma, Zhifang Sui, Furu Wei
- **Focus:** Argues Transformer attention has a dual form to gradient descent, so in-context learning can be understood as implicit fine-tuning — demonstration examples produce "meta-gradients" applied to the model at inference time; motivates a momentum-based attention variant.
- **Vault copy:** [Prompt_Engineering/daiWhyCanGPT.md](Prompt_Engineering/daiWhyCanGPT.md)
- **Zotero item:** `zotero://select/library/items/TNRRFUZC` (attachment key `C6EIAG2N`)

### What Makes Good In-Context Examples for GPT-3?
- **Authors:** Jiachang Liu, Dinghan Shen, Yizhe Zhang, Bill Dolan, Lawrence Carin, Weizhu Chen
- **Published:** 2022
- **Focus:** Finds that retrieving in-context examples semantically similar to the test input (via a retrieval module, ideally task-finetuned) substantially outperforms random example selection for GPT-3 few-shot prompting.
- **Vault copy:** [Prompt_Engineering/liuWhatMakesGood2022.md](Prompt_Engineering/liuWhatMakesGood2022.md)
- **Zotero item:** `zotero://select/library/items/T2W32QVX` (attachment key `4UFYMU25`)

### Large Language Models Are Human-Level Prompt Engineers (APE)
- **Authors:** Yongchao Zhou, Andrei Ioan Muresanu, Ziwen Han, Keiran Paster, Silviu Pitis, Harris Chan, Jimmy Ba
- **Published:** 2023-03-10 (arXiv)
- **Focus:** Treats instructions as "programs" and proposes Automatic Prompt Engineer (APE), which searches over LLM-proposed instruction candidates to maximize a score function — matches or beats human-written prompts on 19/24 NLP tasks.
- **Vault copy:** [Prompt_Engineering/zhouLargeLanguageModels2023/zhouLargeLanguageModels2023.md](Prompt_Engineering/zhouLargeLanguageModels2023/zhouLargeLanguageModels2023.md)
- **Zotero item:** `zotero://select/library/items/WUJS5JCS` (attachment key `Z637C34W`)

### Rethinking the Role of Demonstrations: What Makes In-Context Learning Work?
- **Authors:** Sewon Min, Xinxi Lyu, Ari Holtzman, Mikel Artetxe, Mike Lewis, Hannaneh Hajishirzi, Luke Zettlemoyer
- **Published:** 2022
- **Focus:** Finds that randomly replacing labels in in-context demonstrations barely hurts performance — ground-truth input-label mapping is not the key driver; instead, the label space, input distribution, and overall sequence format matter most.
- **Vault copy:** [Prompt_Engineering/minRethinkingRoleDemonstrations2022/minRethinkingRoleDemonstrations2022.md](Prompt_Engineering/minRethinkingRoleDemonstrations2022/minRethinkingRoleDemonstrations2022.md)
- **Zotero item:** `zotero://select/library/items/TVI25QSV` (attachment key `NFGMPU86`)
