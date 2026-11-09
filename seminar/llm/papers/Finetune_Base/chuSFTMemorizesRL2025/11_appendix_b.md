# B. Details on the V-IRL Environment

Similar to Appendix A, we present the design details for V-IRL discussed in Section 4.2. First, we introduce the database used for this environment (Appendix B.1) and demonstrate transition examples (Appendix B.2). We then describe the environment by explaining its fundamental component-route. Finally, we outline our modifications and reward design choices made to adapt the original V-IRL for reinforcement learning training (Appendix B.3).

# B.1. Data

Leveraging the data collection pipeline of Yang et al. (2024a), we construct a training database with 1000 unique routes from New York City. We evaluate all rule-variant experiments and visual in-distribution experiments using randomly sampled routes from this database. For visual out-of-distribution experiments, we directly adopt the VLN mini benchmark from Yang et al. (2024a). This benchmark consists of 18 distinct routes across nine cities: Milan, New Delhi, Buenos Aires, London, Hong Kong, New York, ${ }^{4}$ Melbourne, Lagos, and San Francisco, with two routes per city.

\footnotetext{
${ }^{4}$ These NYC routes in the VLN mini benchmark do not overlap with our training data.
}
```
System Prompt ( (1i)\
[Task Description]
You are an expert 24 points card game player. You are observing these four cards
in the image. Note that 'J', 'Q', and 'K' count as '10', and each card must be used
once. Your goal is to output a formula that evaluates to 24 using numbers from the
cards and operators such as '+', '-', '*', '/', '(', ')', and '='.
[Output]
Your response should be a valid json file in the following format:
{
"cards": [x, y, z, w], where 'J', 'Q', and 'K' count as '10',
```

![](./chuSFTMemorizesRL2025_images/c29435c9-0689-4d6d-8ce6-4fa35255cad0-14_344_342_314_1427.jpg)
```
"number": [a, b, c, d], where a, b, c, and d are the numbers on the cards,
"formula": "an equation that equals 24",
}
Appending model and verifier outputs to obtain vt \( 2) \)
```

![](./chuSFTMemorizesRL2025_images/c29435c9-0689-4d6d-8ce6-4fa35255cad0-14_43_1519_846_275.jpg)
```
Model output ( (vt
{
"cards": ['A', '3', 'K', '6'],
"number": [1, 3, 13, 6],
"formula": " (1+6)*3+13=24",
}
Verifier Output ((vt
You failed this trial because your formula is incorrect.
    > 岳 in 
```

Figure 11: An example of our prompt update for constructing $\boldsymbol{v}_{t+1}^{\text {in }}$ using $\boldsymbol{v}_{t}^{\text {in }}, \boldsymbol{v}_{t}^{\text {out }}$ and $\boldsymbol{v}_{t}^{\text {ver }}$. This example provides an optional vision input for VLMs, adding a visual recognition challenge. The brown parts marks the task and related information, and the purple parts denote the state \(\left(s_{t}\right)\) specific info. The blue and red describe the output from the model and verifier, respectively.

# B.2. Detailed Examples on the Transition Dynamics

We provide detailed transition examples of the V-IRL environment in Figure 13 (vision and language) and Figure 14 (pure language).

# B.3. Additional Details on the Environmental Design

Concept of route. The route serves as the fundamental navigation object in the V-IRL environment. As illustrated in Figure 4, each route corresponds to a real-world path with associated language instructions and visual signals. Using Figure 4 as an example, a route comprises:
\begin{itemize}
\item[-] Destination: Shuka
\item[-] Starting point: Start
\item[-] Turning points: The Dutch, Lola Taverna
\item[-] Straight road: Roads connecting turning points, starting point, and destination
\item[-] Street views: 360-degree panoramic views at each movable point
\item[-] Oracle information: Expert observation data for each movable point
\item[-] Expert trajectory
\item[-] Instruction
\end{itemize}

Although the instructions in Figures 4, 13 and 14 are presented in different formats, they convey equivalent information, with Figure 4 using natural language.

Simplification and arguments. We simplify the original V-IRL design from Yang et al. (2024a) to better accommodate RL training. The modifications include eliminating the 2-stage navigation pipeline that required a separate visual detector for street view processing, and removing online queries to reduce training time and cost. Our V-IRL environment contains 2 additional configuration arguments compared with the original design:
\begin{itemize}
\item[-] Action space: two options
\begin{itemize}
\item[-] Absolute direction: "turn_direction(x)" where $\mathrm{x} \in\{$ 'north', 'northeast',
\end{itemize}
\end{itemize}
```
System Prompt ( (1i
[Task Description]
You are an expert 24 points card game player. You are observing these four cards in the image. Note that 'J', 'Q',
and 'K' count as '11', '12', and '13' respectively, and each card must be used once. Your goal is to output a formula
that evaluates to 24 using numbers from the cards and operators such as '+', '-', '*', '/', '(', ')', and '='.
[Input]
Cards: ['A', '3', 'K', '6']
[Output]
Your response should be a valid json file in the following format:
{
"cards": [x, y, z, w], where 'J', 'Q', and 'K' count as '10',
"number": [a, b, c, d], where a, b, c, and d are the numbers on the cards,
"formula": "an equation that equals 24",
}
Appending model and verifier outputs to obtain 
```

![](./chuSFTMemorizesRL2025_images/c29435c9-0689-4d6d-8ce6-4fa35255cad0-15_47_1523_892_273.jpg)
```
Model output (vt \( ( 
{
"cards": ['A', '3', 'K', '6'],
"number": [1, 3, 13, 6],
"formula": " (1+6)*3+13=24",
}
Verifier Output ( (vt
```

![](./chuSFTMemorizesRL2025_images/c29435c9-0689-4d6d-8ce6-4fa35255cad0-15_39_1521_1268_275.jpg)

Figure 12: An example of our prompt update for constructing $\boldsymbol{v}_{t+1}^{\text {in }}$ using $\boldsymbol{v}_{t}^{\text {in }}, \boldsymbol{v}_{t}^{\text {out }}$ and $\boldsymbol{v}_{t}^{\text {ver }}$. This example provides an optional vision input for VLMs, adding a visual recognition challenge. The brown parts marks the task and related information, and the purple parts denote the state \(\left(s_{t}\right)\) specific info. The blue and red describe the output from the model and verifier, respectively.
\begin{itemize}
\item[] 'east', 'southeast', 'south', 'southwest', 'west', 'northwest'\}, "forward()", "stop()"
\item[-] Relative direction: "turn_direction(x)" where $\mathrm{x} \in\{$ 'left', 'right', 'slightly left', 'slightly right'\}, "forward()", "stop()"
\item[-] Maximum straight road length: any positive integer
\end{itemize}

The action space argument accommodates the rule variants described in Section 4. For experiments shown in Figure 5, we use absolute direction action space during training and in-domain evaluation, while using the alternative rule for out-of-domain evaluation. We implement a maximum straight road length to limit the number of movable coordinates between turning points, preventing sequences of repetitive "forward()" actions. We conduct visual distribution shift experiments (Section 5.2) via training the model on New York City regions and evaluating the out-ofdomain performance on the worldwide navigation routes from the benchmark released by Yang et al. (2024a).

System Prompt ( $\boldsymbol{v}_{0}^{\text {in }}$ )
[Task Description]
You are an expert in navigation. You will receive a sequence of instructions to follow while observing your surrounding street views. You are also provided with your observation and action history in text. your goal is to take the action based on the current observation and instruction.
[Instruction]
\begin{itemize}
\item[1.] First, turn left to face east.
\item[2.] Move forward until you reach the next intersection where Hotel 32One is on your right behind.
\item[3.] Turn right to face north.
\item[4.] Move forward until you reach the next intersection where Dragon Gate Chinatown SF is on your right front.
\item[5.] Turn left to face east.
\item[6.] Move forward until the destination Café de la Presse is on your right.
\end{itemize}
[Current observation]
You observe a 2x2 grid of street view images with the following headings:
[front, right
back, left]
You need to identify if any of the landmarks in the instruction are visible in the street view grid.
[Action space]
\begin{itemize}
\item[-] "forward()": indicates moving forward for 1 step;
\item[-] "turn_direction(x)": indicates turn direction to the target heading, where $\mathrm{x} \in[$ 'north', 'northeast', 'east', 'southeast', 'south', 'southwest', 'west', 'northwest'];
\item[-] "stop()": indicates the navigation is finished;
\end{itemize}
[Observations and actions sequence]
\begin{itemize}
\item[] O_1: No landmarks nearby;
\item[] A_1: turn_direction(east)
\item[] O_2: No landmarks nearby;
\item[] A_2: forward()
\item[] O_3: No landmarks nearby;
\item[] A_3: forward()
\item[] O_4: You observe an image of 4 views; You observe an intersection
A_4:
\end{itemize}
[Output]
Your response should be a valid json file in the following format:
\{
"current observation": latest observation from the street view grid,
"current instruction": analyze the full instruction and identify the sentence to be executed,
"action": the action to be taken chosen from the action space,
\}
Appending model and verifier outputs to obtain $\boldsymbol{v}_{t}^{\text {in }}$
$\boldsymbol{v}_{t}^{\text {in }}=\left[\boldsymbol{v}_{0}^{\text {out }}, \boldsymbol{v}_{0}^{\text {ver }}, \boldsymbol{v}_{1}^{\text {out }}, \boldsymbol{v}_{1}^{\text {ver }}, \ldots, \boldsymbol{v}_{t-1}^{\text {out }}, \boldsymbol{v}_{t-1}^{\text {ver }}\right]$ \(\triangleright \boldsymbol{v}_{t}^{\text {in }}=\operatorname{concat}\left(\boldsymbol{v}_{0}^{\text {in }},\left[\boldsymbol{v}_{k}^{\text {out }}, \boldsymbol{v}_{k}^{\text {ver }}\right]_{k=0}^{t-1}\right)\)

Model output \(\left(\boldsymbol{v}_{t}^{\text {out }}\right)\)
\{
"current observation": "Hotel 32One is on my right behind; I observe an intersection",
"current instruction": "Turn right to face north.",
"action": "turn_direction(north)",
\}
Verifier Output \(\left(v_{t}^{\text {ver }}\right)\)
Correct solution. - Move to the next scene upon correct action

Figure 13: An example of our prompt update for constructing $\boldsymbol{v}_{t+1}^{\text {in }}$ using $\boldsymbol{v}_{t}^{\text {in }}, \boldsymbol{v}_{t}^{\text {out }}$ and $\boldsymbol{v}_{t}^{\text {ver }}$. This example provides an optional vision input for VLMs, adding a visual recognition challenge. The brown parts marks the task and related information, and the purple parts denote the state \(\left(s_{t}\right)\) specific info. The blue and red describe the output from the model and verifier, respectively.

System Prompt ( $\boldsymbol{v}_{0}^{\text {in }}$ )
[Task Description]
You are an expert in navgation. You will receive a sequence of instructions to follow. You are also provided with your observation and action histroy in text. Your goal is to first analyze the instruction and identify the next sentence to be executed. Then, you need to provide the action to be taken based on the current observation and instruction.
[Instruction]
\begin{itemize}
\item[1.] First, turn left to face east.
\item[2.] Move forward until you reach the next intersection where Hotel 32One is on your right behind.
\item[3.] Turn right to face north.
\item[4.] Move forward until you reach the next intersection where Dragon Gate Chinatown SF is on your right front.
\item[5.] Turn left to face east.
\item[6.] Move forward until the destination Café de la Presse is on your right.
\end{itemize}
[Action space]
\begin{itemize}
\item[-] "forward()": indicates moving forward for 1 step;
\item[-] "turn_direction(x)": indicates turn direction to the target heading, where $\mathrm{x} \in[$ 'north', 'northeast', 'east', 'southeast', 'south', 'southwest', 'west', 'northwest'];
\item[-] "stop()": indicates the navigation is finished;
\end{itemize}
[Observations and actions sequence]
\begin{itemize}
\item[] O_1: No landmarks nearby;
\item[] A_1: turn_direction(east)
\item[] O_2: No landmarks nearby;
\item[] A_2: forward()
\item[] O_3: No landmarks nearby;
\item[] A_3: forward()
\item[] O_4: Hotel 32One is on your right behind; You observe an intersection
A_4:
\end{itemize}
[Output]
Your response should be a valid json file in the following format:
\{
"current observation": latest observation from the street view grid,
"current instruction": analyze the full instruction and identify the sentence to be executed,
"action": the action to be taken chosen from the action space,
\}
Appending model and verifier outputs to obtain $\boldsymbol{v}_{t}^{\text {in }}$
$$
\boldsymbol{v}_{t}^{\text {in }}=\left[\boldsymbol{v}_{0}^{\text {out }}, \boldsymbol{v}_{0}^{\text {ver }}, \boldsymbol{v}_{1}^{\text {out }}, \boldsymbol{v}_{1}^{\text {ver }}, \ldots, \boldsymbol{v}_{t-1}^{\text {out }}, \boldsymbol{v}_{t-1}^{\text {ver }}\right] \quad \triangleright \boldsymbol{v}_{t}^{\text {in }}=\operatorname{concat}\left(\boldsymbol{v}_{0}^{\text {in }},\left[\boldsymbol{v}_{k}^{\text {out }}, \boldsymbol{v}_{k}^{\text {ver }}\right]_{k=0}^{t-1}\right)
$$
```
Model output ( (vt
{
"current observation": "Hotel 32One is on my right behind; I observe an intersection",
"current instruction": "Turn right to face north.",
"action": "turn_direction(north)",
}
Verifier Output ( (vt
Correct solution. ▷ Move to the next scene upon correct action
```

Figure 14: An example of our prompt update for constructing $\boldsymbol{v}_{t+1}^{\text {in }}$ using $\boldsymbol{v}_{t}^{\text {in }}, \boldsymbol{v}_{t}^{\text {out }}$ and $\boldsymbol{v}_{t}^{\text {ver }}$. The brown parts marks the task and related information, and the purple parts denote the state \(\left(s_{t}\right)\) specific info. The brown parts marks the task and related information, and the purple parts denote the state \(\left(s_{t}\right)\) specific info. The blue and red describe the output from the model and verifier, respectively.

