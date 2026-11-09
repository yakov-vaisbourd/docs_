# Deep Learning Seminar 2026: RL, LLMs, and Flow Matching

Welcome to our group's 13-week Deep Learning Seminar. This series is divided into three core blocks: Reinforcement Learning (RL), Large Language Models (LLMs), and Diffusion & Flow Matching.

Each session combines theoretical foundations with a **15-20 minute hands-on coding lab**. All code is designed to run on the Google Colab Free Tier.

## Group Assignments

| Topic | Assigned Members |
| :--- | :--- |
| **Reinforcement Learning** | Israel, Omer, Doron, Eran, Snir |
| **LLM** | Oryan, Dmitry, Itamar, Itay |
| **Diffusion and Flow Matching** | Dor, Tomer, Yakov, Tali |

---

## Block 1: Reinforcement Learning (RL)

**Core Focus:** Training agents to maximize rewards in dynamic environments.
**Primary Resources:** [DeepMind RL Lectures](https://www.youtube.com/playlist?list=PLqYmG7hTraZDVH599EItlEWsUOsJbAodm),  [CleanRL Library](https://docs.cleanrl.dev/)

| Lecture | Title & Goal | Theoretical Concepts | Hands-on Implementation (Code) | Resource Links |
| :--- | :--- | :--- | :--- | :--- |
| **1** | **RL Fundamentals**<br>The language of RL. | Markov Decision Processes (MDPs), Rewards, Returns, the Bellman Equation. | **Gridworld Manual:** Write a script to manually control an agent via arrow keys, logging the $(S, A, R, S')$ tuples. | [Gymnasium Environments](https://gymnasium.farama.org/) |
| **2** | **Value-Based (DQN)**<br>Learning action values. | Q-Learning, Deep Q-Networks, Experience Replay, Target Networks. | **Solving CartPole:** Run CleanRL's `dqn.py` on `CartPole-v1`. Watch the agent learn to balance the pole live. | [CleanRL DQN Code](https://docs.cleanrl.dev/rl-algorithms/dqn/)<br>[Original DQN Paper](https://arxiv.org/abs/1312.5602) |
| **3** | **Policy Gradients**<br>Directly optimizing the policy. | The REINFORCE algorithm, Direct Policy Optimization, On-policy vs. Off-policy. | **Loss Comparison:** Plot and compare the training stability and loss curves of DQN against a basic Policy Gradient. | [PyTorch Official REINFORCE](https://github.com/pytorch/examples/blob/main/reinforcement_learning/reinforce.py) |
| **4** | **Modern RL (PPO)**<br>The algorithm behind RLHF. | Advantage Actor-Critic (A2C), Proximal Policy Optimization (PPO), clipping. | **LunarLander PPO:** Train `ppo.py` on `LunarLander-v2`. Alter the learning rate to demonstrate the brittleness of RL. | [CleanRL PPO Code](https://docs.cleanrl.dev/rl-algorithms/ppo/) |
| **5** | **Maximum Entropy RL (SAC)**<br>Off-policy continuous control. | The exploration-exploitation trade-off, Soft Actor-Critic (SAC), Squashed Gaussian policies. | **BipedalWalker SAC:** Run CleanRL’s `sac.py` to observe how entropy regularization helps in high-dimensional continuous action spaces. | [CleanRL SAC Code](https://docs.cleanrl.dev/rl-algorithms/sac/)<br>[SAC Paper](https://arxiv.org/abs/1801.01290) |

---

## Block 2: Large Language Models (LLMs)

**Core Focus:** Moving from raw text prediction to aligned chat assistants.
**Primary Resource:** [Deep Dive into LLMs like ChatGPT - Andrej Karpathy](https://www.youtube.com/watch?v=7xTGNNLPyMI&t=1814s)

| Lecture | Title & Goal | Theoretical Concepts | Hands-on Implementation (Code) | Resource Links |
| :--- | :--- | :--- | :--- | :--- |
| **6** | **The Foundation**<br>Understand what goes into the model. | Mental models of LLMs, dataset curation (FineWeb), BPE tokenization. | **Tokenizer Gym:** Run raw text through `tiktoken` to observe word splitting and calculate compression ratios. | [Tiktoken GitHub](https://github.com/openai/tiktoken)<br>[FineWeb Explorer](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1) |
| **7** | **The Architecture**<br>Understand the Base Model. | Transformer blocks, self-attention, next-token prediction, pre-training scale. | **Code a Head:** Build a single Self-Attention head from scratch in PyTorch (matrix multiplication + softmax). | [nanoGPT Repository](https://github.com/karpathy/nanoGPT)<br>[Attention is All You Need](https://arxiv.org/abs/1706.03762) |
| **8** | **Refinement (SFT)**<br>From document completer to assistant. | Formatting chat data (User/Assistant), supervised fine-tuning, limitations of SFT. | **LoRA in 5 Mins:** Use Unsloth on Google Colab to fine-tune `Llama-3-8B` on a custom conversational dataset. | [Unsloth AI](https://github.com/unslothai/unsloth)<br>[HuggingFace PEFT](https://huggingface.co/docs/peft/index) |
| **9** | **Alignment & Future**<br>State-of-the-art capabilities. | RLHF, Proximal Policy Optimization (PPO), Reward Models, Thinking/Reasoning models. | **Monologue Inspection:** Analyze the "Thinking" logs of a reasoning model (like DeepSeek-R1) to understand internal intermediate steps. | [DeepSeek-R1 Paper](https://arxiv.org/abs/2501.12948)<br>[InstructGPT Paper](https://arxiv.org/abs/2203.02155) |

---

## Block 3: Diffusion & Flow Matching

**Core Focus:** Generating complex data distributions from pure noise.
**Primary Resource:** [MIT 6.S184: Intro to Generative AI](https://diffusion.csail.mit.edu/2025/index.html)

| Lecture | Title & Goal | Theoretical Concepts | Hands-on Implementation (Code) | Resource Links |
| :--- | :--- | :--- | :--- | :--- |
| **10** | **Generative Intuition**<br>The mechanics of destruction. | Probability paths, generating data from noise, ordinary differential equations (ODEs). | **The Noise Schedule:** Write a loop that adds step-by-step Gaussian noise to 2D points, visualizing the data-to-noise transition. | [MIT 6.S184 Materials](https://diffusion.csail.mit.edu/2025/index.html) |
| **11** | **Diffusion Models**<br>Reversing the process. | Denoising Diffusion Probabilistic Models (DDPM), Forward vs. Reverse processes, Score Matching. | **Unrolling the Swiss Roll:** Train a simple MLP to denoise a 2D "Swiss Roll" dataset, watching the points clump back into a spiral. | [SmallDiffusion Repo](https://github.com/yuanchenyang/smalldiffusion)<br>[DDPM Paper](https://arxiv.org/abs/2006.11239) |
| **12** | **Flow Matching**<br>A more efficient generative path. | Conditional Flow Matching (CFM), vector fields, why Flow Matching scales better. | **Straight vs. Curved:** Plot the learned vector fields. Show that Diffusion creates curved, wiggly paths, while Flow Matching creates straight, optimal lines. | [TorchCFM Library](https://github.com/atong01/conditional-flow-matching)<br>[Medium: Diffusion vs Flow](https://medium.com/@raajanwankhade/how-probability-moves-diffusion-vs-flow-matching-54a39159079c) |
| **13** | **Latent Space & Guidance**<br>Applying concepts to real images. | Latent Diffusion (Stable Diffusion architectures), Classifier-Free Guidance (CFG). | **Latent Surgery:** Use Hugging Face `diffusers` to extract the latent noise of a generated image, tweak it, and regenerate it. | [HF Diffusers Docs](https://huggingface.co/docs/diffusers/index)<br>[Stable Diffusion Paper](https://arxiv.org/abs/2112.10752) |

---

### Speaker Guidelines
* **Assignment Strategy:** Please claim lectures based on your experience level. Foundation sessions for newer members; complex topics for advanced participants.
* **Preparation:** Prepare a Google Colab notebook for the "Hands-on" section.
* **Pre-training:** If code takes >5 mins to train, provide pre-trained weights (`.pt` or `.bin`).
* **Code Repository:** All materials will be stored in a dedicated GitHub repository for each block.
