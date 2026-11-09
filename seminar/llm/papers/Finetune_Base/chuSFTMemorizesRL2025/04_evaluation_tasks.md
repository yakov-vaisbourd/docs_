# 4. Evaluation Tasks

To evaluate the generalization of different post-training methods, we select two tasks that each offer rule and visual variations. The first task, GeneralPoints, is a new environment we have designed that allows assessment of arithmetic reasoning abilities (Section 4.1). The second task, V-IRL (Yang et al., 2024a), is chosen to examine the model's reasoning capabilities in an open-world visual navigation domain (Section 4.2).

## 4.1. The General Points Environment

Our original GeneralPoints environment, instantiated on top of the Points24 environment (Zhai et al., 2024a), is designed to evaluate generalization of arithmetic reasoning. Each state $s$ of the environment contains 4 cards, described as text (in the GP-L variant) or presented as an image (in the GP-VL variant); see Figure 2 left for a visual example of GeneralPoints. The goal is to produce an equation that equals a target number (24 by default) using all 4 numbers from the cards exactly once. Detailed examples of the state-action transitions are provided in Appendix A.2. Note that when input from GeneralPoints is presented in an image (GP-VL), it naturally introduces additional visual challenges requiring the VLM to recognize all cards before solving the equation.

![Figure](./chuSFTMemorizesRL2025_images/c29435c9-0689-4d6d-8ce6-4fa35255cad0-04_288_1553_232_262.jpg)

**Figure 2: An example of the sequential revision formulation with a verifier. The model generate the next answer $\boldsymbol{v**_{t+1}^{\text {out }}$ conditioned on all previous answers and information ( $\boldsymbol{v}_{i}^{\text {out }}, \boldsymbol{v}_{t}^{\text {ver }}, 0 \leq i \leq t$ ) from the verifier.}

System Prompt ( $\boldsymbol{v}_{0}^{\text {in }}$ )
[Task Description] You are an expert in \{task name\}, you are observing \{purely language/vision-language inputs + <image>\}. You are currently at \{state related info\}. Please follow \{tasks rules\}.
[Output] Your response should be a valid json file in the following format: \{task related information and answer\}

Appending previous model and verifier outputs to obtain $\boldsymbol{v}_{\boldsymbol{t}}^{\text {in }}$
$$
\boldsymbol{v}_{t}^{\mathrm{in}}=\left[\boldsymbol{v}_{0}^{\text {out }}, \boldsymbol{v}_{0}^{\text {ver }}, \boldsymbol{v}_{1}^{\text {out }}, \boldsymbol{v}_{1}^{\text {ver }}, \ldots, \boldsymbol{v}_{t-1}^{\text {out }}, \boldsymbol{v}_{t-1}^{\text {ver }}\right] \quad \triangleright \boldsymbol{v}_{t}^{\text {in }}=\operatorname{concat}\left(\boldsymbol{v}_{0}^{\text {in }},\left[\boldsymbol{v}_{k}^{\text {out }}, \boldsymbol{v}_{k}^{\text {ver }}\right]_{k=0}^{t-1}\right)
$$

Model output \(\left(v_{t}^{\text {out }}\right)\) and Verifier Output \(\left(v_{t}^{\text {ver }}\right)\)
$$
\text { \{ Task related json outputs }\},\{\text { You success/fail }\} . \quad \triangleright \boldsymbol{v}_{t+1}^{\text {in }}=\operatorname{concat}\left(\boldsymbol{v}_{t}^{\text {in }}, \boldsymbol{v}_{t}^{\text {out }}, \boldsymbol{v}_{t}^{\text {ver }}\right)
$$

Figure 3: An template of our prompt update for constructing $\boldsymbol{v}_{t+1}^{\text {in }}$. The brown parts marks the task and related information, and the purple parts denote the state \(\left(s_{t}\right)\) specific info. The blue and red describe the output from the model and verifier, respectively.

Rule variations. To study whether the model learns arithmetic operations or simply memorizes the post-training data, we introduce rule variations in GeneralPoints. These variations consist of interpreting the symbols 'J', 'Q', and 'K' either as '11', '12', and '13', respectively, or all as the same number '10'. These variations ensure a rigorous evaluation of the model's ability to generalize arithmetic reasoning across diverse settings. Each rule is specified as text in the input prompt, see the \{tasks rules\} part in Figure 3. For studying ruled based generalization, we post-train the model using one rule, then evaluate using a different rule.

Visual variations. The GeneralPoints environment can also be naturally customized to evaluate generalization across visual variants. Since the major visual challenge is to recognize the number of each card, agnostic to the the color of the cards, we consider the cards with different colors as visual variants of the task. In the visual generalization setting, we train the model using cards of one color, then test OOD performance using the other color.

## 4.2. The V-IRL Environment

While the GeneralPoints environment is designed to assess arithmetic reasoning abilities, we further utilize the V-IRL environment (Yang et al., 2024a) to study spatial reasoning ability in an open-world navigation domain that uses realistic visual input. As in GeneralPoints we consider two versions of the environment, one (V-IRL-L) that consists of pure language descriptions, ${ }^{2}$ and another (V-IRL-VL) that includes vision-language input. The major visual challenge in V-IRL involves recognizing different landmarks from the visual observation ${ }^{3}$ before taking an action. The goal is to navigate to a target location by following a set of instructions that contain spatial information. A detailed example of one environment step is shown in Appendix B.2.

Rule variations. To evaluate whether the model possesses spatial knowledge or simply memorizes posttraining data, we consider two distinct action space configurations. The first variant utilizes an absolute orientation action space, which includes \{'north', 'northeast', 'east', 'southeast', 'south', 'southwest', 'west', 'northwest'\}. The second variant employs a relative orientation action space, containing \{ 'left', 'right', 'slightly left', 'slightly right'\}. This relative configuration adjusts the current orientation by 90 degrees or 45 degrees to the left or right, respectively. An

\footnotetext{
${ }^{2}$ The visual input can be parsed into pure text description, see more details in Yang et al. (2024a) and an illustration of pure text the version in Figure 14.
${ }^{3}$ See Figure 4, the model needs to recognize landmarks like The Dutch, Lola Taverna, and Shuka from the visual observation, and relate these landmarks with the textual instructions for taking the right action.
}

![Figure](./chuSFTMemorizesRL2025_images/c29435c9-0689-4d6d-8ce6-4fa35255cad0-05_1195_1518_226_274.jpg)

**Figure 4: Demonstration of one navigation task in V-IRL. Agent navigates from place to place following the given linguistic navigation instructions in V-IRL. The navigation procedure is shown at the top, with the navigation instructions displayed below. Visual observation-related information is highlighted in green, while action-related information is marked in orange.**

overview of a navigation task in V-IRL is provided in Figure 4, and a detailed state-action transition in V-IRL is provided in Figure 13 (in Appendix B.2).

Visual variations. The key visual challenge in V-IRL is to recognize landmarks from the visual observations (e.g., the green parts in Figure 4). Since the V-IRL environment contains visual observations from different cities, we can assess visual generalization in V-IRL by training the model to navigate in one location and then evaluate its performance in different locations.

