# A. Details on the General Points Environment

In this section, we demonstrate the design details for GeneralPoints mentioned in Section 4.1. We first present the data used for this environment (Appendix A.1). Then, we show examples of the environment's transition dynamics (Appendix A.2), followed by a description of key arguments and reward design specification (Appendix A.3).

# A.1. Data

GeneralPoints card quadruples are sampled from a deck of 52 standard poker cards. Each sampled quadruple is guaranteed to have at least one solution equals the target point, i.e. 24. We ensure this by using an expert solver during the sampling process.

# A.2. Detailed Examples on the Transition Dynamics

As shown in Figure 11 and Figure 12, we treat the system prompt as $\boldsymbol{v}_{0}^{\mathrm{in}}$ and then subsequently appending the future outputs $\boldsymbol{v}_{1: t}^{\text {out }}$ and verifier info $\boldsymbol{v}_{1: t}^{\text {ver }}$ into the prompt for getting the $t+1$ output. Figure 11 provides an example with the visual inputs, while Figure 12 shows the language only case.

# A.3. Additional Eetails on the Environmental Design

Arguments. The GeneralPoints environment supports the following configurable arguments:
\begin{itemize}
\item[-] Target point: Any positive integer
\item[-] Face cards rule: Two options
\begin{itemize}
\item[-] 'J', 'Q', and 'K' all count as '10'
\item[-] 'J', 'Q', and 'K' count as '11', '12', and '13' respectively
\end{itemize}
\item[-] Card sampling: Two options
\begin{itemize}
\item[-] Sample 4 cards without replacement from a deck of 52 poker cards
\item[-] Sample at least one card from 'J', 'Q', and 'K'
\end{itemize}
\item[-] Card color: Three options
\begin{itemize}
\item[-] Black suits only: . . .
\item[-] Red suits only: ✓
\item[-] All suits: », , , ↓.
\end{itemize}
\end{itemize}

For all experiments, we fix the target point at 24 . In Figure 5, training and in-domain evaluation use the rule where face cards count as '10'. For out-of-domain evaluation, we use the alternative face cards rule and require at least one face card, forcing calculations with numbers above 10 that are not encountered during training. For visual distribution shift experiments (Section 5.2), we train the model on black suits, and evaluate out-of-domain performance on red suits ✓, -

Reward design. An episode terminates when either a correct equation is generated or the maximum verification step of 5 is reached. The reward function is as follows:
\begin{itemize}
\item[-] $r=5$ : For generating a legal equation that equals the target point
\item[-] $r=-1$ : For legal equations using each card once but not equaling the target point
\item[-] $r=-1$ : For exceeding maximum verification step
\item[-] $r=-2$ : For legal equations containing numbers not among the given choices
\item[-] $r=-3$ : For all other illegal equations
\end{itemize}

In the vision-language variant (GeneralPoints-VL), an additional penalty of $r=-1.5$ is applied when the agent fails to correctly recognize the given cards.

