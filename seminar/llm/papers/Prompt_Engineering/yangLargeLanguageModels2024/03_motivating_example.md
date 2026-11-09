# 3 Motivating Example: Mathematical Optimization

We first demonstrate the potential of LLMs in serving as optimizers for mathematical optimization. In particular, we present a case study on linear regression as an example of continuous optimization, and on the Traveling Salesman Problem (TSP) as an example of discrete optimization. On both tasks, we see LLMs properly capture the optimization directions on small-scale problems merely based on the past optimization trajectory provided in the meta-prompt.

## 3.1 Linear Regression

In linear regression problems, the goal is to find the linear coefficients that probabilistically best explain the response from the input variables. We study the setting in which the independent and dependent variables $X$ and $y$ are both one-dimensional and an intercept $b$ is present, so that there are two one-dimensional variables $w, b$ to optimize over. In a synthetic setting, we sample ground truth values for one-dimensional variables $w_{\text {true }}$ and $b_{\text {true }}$, and generate 50 data points by $y=w_{\text {true }} x+b_{\text {true }}+\epsilon$, in which $x$ ranges from 1 to 50 and $\epsilon$ is the standard Gaussian noise. Our optimization starts from 5 randomly sampled \((w, b)\) pairs. In each step, we prompt an instructiontuned LLM with a meta-prompt that includes the best \(20(w, b)\) pairs in history and their sorted objective values. The meta-prompt then asks for a new ( $w, b$ ) pair that further decreases the objective value. A sample meta-prompt is shown in Figure 19 of Appendix C. 1 We prompt the meta-prompt 8 times to generate at most 8 new ( $w, b$ ) pairs in each step to improve optimization stability. Then we evaluate the objective value of the proposed pair and add it to history. We do black-box optimization: the analytic form does not appear in the meta-prompt text. This is because the LLM can often calculate the solution directly from the analytic form.

Table 2 summarizes the results with one of the following optimizer LLMs: text-bison, gpt-3.5-turbo, and gpt-4. We study three settings of $w_{\text {true }}$ and $b_{\text {true }}$ : within the starting region $[10,20] \times[10,20]$, "near outside" (each of $w_{\text {true }}$ and $b_{\text {true }}$ is outside the starting region but the distance is less than 10), and "far outside" (each of $w_{\text {true }}$ and $b_{\text {true }}$ is outside the starting region and the distance is greater than 10). We see:

\begin{table}

**Table 2: Linear regression by optimizer LLMs: the mean ± standard deviation of the number of steps and the number of unique \((w, b)\) pairs explored before reaching the global optima. Both $w$ and $b$ start from 5 random starting points in [10, 20]. We use temperature 1.0 for all models. We run each setting 5 times. The starting points are the same across optimizer LLMs but are different across 5 runs, and are grouped by: within the starting region, outside and close to the starting region, and outside and farther from the starting region. Bold numbers indicate the best among three LLMs in each setting.**
\begin{tabular}[t]{|l|l|l|l|l|l|l|l|}
\hline \multirow{2}{*}{$w_{\text {true }}$} & \multirow[t]{2}{*}{$b_{\text {true }}$} & \multicolumn{3}{|c|}{number of steps} & \multicolumn{3}{|c|}{number of unique ( $w, b$ ) pairs explored} \\
\hline & & text-bison & gpt-3.5-turbo & gpt-4 & text-bison & gpt-3.5-turbo & gpt-4 \\
\hline 15 & 14 & $5.8 \pm 2.6$ & 7.6 ± 4.5 & $\mathbf{4 . 0}_{\text {± } 1.5}$ & $40.0 \pm 12.4$ & $36.0 \pm 15.2$ & $\mathbf{1 7 . 2}_{\text {± } 5.1}$ \\
\hline 17 & 17 & $\mathbf{4 . 0}_{ \pm 1.8}$ & $12.6 \pm 6.0$ & $6.0 \pm 3.7$ & $33.4 \pm 11.7$ & $53.8 \pm 16.9$ & $\mathbf{2 6 . 0}_{ \pm} 10.6$ \\
\hline 16 & 10 & $\mathbf{3 . 8}_{\text {± } 2.2}$ & $10.4 \pm 5.4$ & $6.2 \pm 3.1$ & $30.2 \pm 13.4$ & $42.8 \pm 16.3$ & $\mathbf{2 4 . 2}_{\text {± } 8.2}$ \\
\hline 3 & 5 & $\mathbf{9 . 8}_{\text {± } 2.8}$ & $10.8 \pm 2.7$ & $12.2 \pm 2.0$ & $55.8 \pm 16.1$ & $39.6 \pm 10.1$ & $\mathbf{3 3 . 0}_{ \pm 4.0}$ \\
\hline 25 & 23 & $19.6 \pm 11.4$ & $26.4 \pm 18.3$ & $\mathbf{1 2 . 2}_{ \pm 3.7}$ & $104.0 \pm 52.3$ & $78.6 \pm 26.2$ & $\mathbf{4 4 . 2}_{\text {± } 8.3}$ \\
\hline 2 & 30 & $\mathbf{3 1 . 4} \pm 6.3$ & $42.8 \pm 9.7$ & $38.0 \pm 15.9$ & $126.4 \pm 17.7$ & $125.6 \pm 21.7$ & 99.0 ± 24.6 \\
\hline 36 & -1 & $\mathbf{3 5 . 8} \pm 6.4$ & $45.4 \pm 16.9$ & $50.4 \pm 18.8$ & $174.0 \pm 28.2$ & $142.2 \pm 31.2$ & $\mathbf{1 1 6 . 4} \pm 32.7$ \\
\hline
\end{tabular}
\end{table}
\begin{itemize}
\item[-] The number of unique \((w, b)\) pairs explored by each model is fewer than exhaustive search, indicating these models are able to to do black-box optimization: compare the numbers and propose a descent direction.
\item[-] The text-bison and gpt-4 models outperform gpt-3.5-turbo in convergence speed: they arrive at the optima with fewer steps. The gpt-4 model also outperforms in finding the optima with fewer explored unique points. Taking a closer look at the optimization trajectory, we see gpt-4 is the best at proposing a reasonable next step from the history: for example, when the history shows the objective values of \((w, b)=(8,7),(w, b)=(8,6)\), and \((w, b)=(8,5)\) are decreasing, it has a highest chance to propose \((w, b)=(8,4)\) for evaluation.
\item[-] The problem becomes harder for all models when the ground truth moves farther from the starting region: all models need more explorations and more steps.
\end{itemize}

## 3.2 Traveling Salesman Problem (TSP)

Next, we consider the Traveling Salesman Problem (TSP) (Jünger et al., 1995; Gutin \& Punnen, 2006), a classical combinatorial optimization problem with numerous algorithms proposed in literature, including heuristic algorithms and solvers (Rosenkrantz et al., 1977; Golden et al., 1980; Optimization et al., 2020; Applegate et al., 2006; Helsgaun, 2017), and approaches based on training deep neural networks (Kool et al., 2019; Deudon et al., 2018; Chen \& Tian, 2019; Nazari et al., 2018). Specifically, given a set of $n$ nodes with their coordinates, the TSP task is to find the shortest route that traverses all nodes from the starting node and finally returns to the starting node.

Our optimization process with LLMs starts from 5 randomly generated solutions, and each optimization step produces at most 8 new solutions. We present the meta-prompt in Figure 20 of Appendix C. 1 We generate the problem instances by sampling $n$ nodes with both $x$ and $y$ coordinates in [-100, 100]. We use the Gurobi solver (Optimization et al., 2020) to construct the oracle solutions and compute the optimality gap for all approaches, where the optimality gap is defined as the difference between the distance in the solution constructed by the evaluated approach and the distance achieved by the oracle solution, divided by the distance of the oracle solution. Besides evaluating OPRO with different LLMs including text-bison, gpt-3.5-turbo and gpt-4, we also compare OPRO to the following heuristics:
\begin{itemize}
\item[-] Nearest Neighbor (NN). Starting from an initial node, the solution is constructed with the nearest neighbor heuristic: At each step, among the remaining nodes that are not included in the current partial solution, NN selects the node with the shortest distance to the end node of the partial solution, and adds it as the new end node. The process finishes when all nodes have been added to the solution.
\item[-] Farthest Insertion (FI). One caveat of the nearest neighbor heuristic is that it does not take the distance between the start and end node into consideration when constructing partial solutions. To address this issue, FI aims to optimize the cost of inserting new nodes into the partial solution at each step. Define the minimal insertion cost of adding a new node $k$ as
\end{itemize}

\begin{table}

**Table 3: Results of the Traveling Salesman Problem (TSP) with different number of nodes $n$, where each $n$ contains 5 problems. "\# steps" calculates the mean ± standard error of optimization steps for successful runs that find the optimal solution. "\# successes" counts the number of problems that OPRO results in the optimal solution. When no optimal solution is found for any evaluated problem, the corresponding number of steps is N/A.**
\begin{tabular}[t]{|l|l|l|l|l|l|l|l|l|}
\hline \multirow{2}{*}{$n$} & \multicolumn{5}{|c|}{optimality gap (\%)} & \multicolumn{3}{|c|}{\# steps (\# successes)} \\
\hline & NN & FI & text-bison & gpt-3.5-turbo & gpt-4 & text-bison & gpt-3.5-turbo & gpt-4 \\
\hline 10 & 13.0 ± 1.3 & $3.2 \pm 1.4$ & $\mathbf{0 . 0}_{\text {± } 0.0}$ & $\mathbf{0 . 0}_{\text {± } 0.0}$ & $\mathbf{0 . 0}_{\text {± } 0.0}$ & $40.4 \pm 5.6$ (5) & 46.8 ± 9.3 (5) & \(\mathbf{9 . 6} \pm 3.0(\mathbf{5})\) \\
\hline 15 & 9.4 ± 3.7 & 1.2 ± 0.6 & $4.4 \pm 1.3$ & 1.2 ± 1.1 & $\mathbf{0 . 2}_{\text {± } 0.2}$ & N/A (0) & \(202.0 \pm 41.1(4)\) & \(\mathbf{5 8 . 5}_{ \pm 29.0}(\mathbf{4})\) \\
\hline 20 & $16.0 \pm 3.9$ & $\mathbf{0 . 2}_{\text {± } 0.1}$ & $30.4 \pm 10.6$ & $4.4 \pm 2.5$ & $1.4 \pm 0.6$ & N/A (0) & 438.0 ± 0.0 (1) & $\mathbf{1 9 5 . 5}_{ \pm} 127.6$ (2) \\
\hline 50 & 19.7 ± 3.1 & $\mathbf{9 . 8}_{\text {± } 1.5}$ & $219.8 \pm 13.7$ & 133.0 ± 6.8 & $11.0 \pm 2.6$ & N/A (0) & N/A (0) & N/A (0) \\
\hline
\end{tabular}
\end{table}
\(c(k)=\min _{(i, j)} d(i, k)+d(k, j)-d(i, j)\), where $i$ and $j$ are adjacent nodes in the current tour, and \(d(\cdot, \cdot)\) represents the distance between two nodes. At each step, FI adds a new node that maximizes the minimal insertion cost.

We present the results in Table 3. We randomly generate 5 problem instances for each number of nodes $n$. In addition to measuring the optimality gap, on problems where the LLM finds the optimal solutions, we also show the number of optimization steps taken to reach the global optimum. First, we observe that gpt-4 significantly outperforms gpt-3.5-turbo and text-bison across all problem sizes. Specifically, on smaller-scale problems, gpt-4 reaches the global optimum about 4× faster than other LLMs. On larger-scale problems, especially with $n=50$, gpt -4 still finds solutions with a comparable quality to heuristic algorithms, while both text-bison and gpt-3.5-turbo get stuck at local optima with up to 20× worse optimality gaps.

On the other hand, the performance of OPRO degrades dramatically on problems with larger sizes. When $n=10$, all LLMs find the optimal solutions for every evaluated problem; as the problem size gets larger, the OPRO optimality gaps increase quickly, and the farthest insertion heuristic starts to outperform all LLMs in the optimality gap.

Limitations. We would like to note that OPRO is designed for neither outperforming the stateof-the-art gradient-based optimization algorithms for continuous mathematical optimization, nor surpassing the performance of specialized solvers for classical combinatorial optimization problems such as TSP. Instead, the goal is to demonstrate that LLMs are able to optimize different kinds of objective functions simply through prompting, and reach the global optimum for some smallscale problems. Our evaluation reveals several limitations of OPRO for mathematical optimization. Specifically, the length limit of the LLM context window makes it hard to fit large-scale optimization problem descriptions in the prompt, e.g., linear regression with high-dimensional data, and traveling salesman problems with a large set of nodes to visit. In addition, the optimization landscape of some objective functions are too bumpy for the LLM to propose a correct descending direction, causing the optimization to get stuck halfway. We further elaborate our observed failure cases in Appendix A

