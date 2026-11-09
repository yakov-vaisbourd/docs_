# G Statistical Analysis

\begin{table}

**Table 6: Pairwise comparisons of six conditions on four metrics across 125 questions and two datasets. For each question and metric, the winning condition received a score of 100, the losing condition received a score of 0 , and in the event of a tie, each condition was scored 50 . These scores were then averaged over five evaluation runs for each condition. Results of Shapiro-Wilk tests indicated that the data did not follow a normal distribution. Thus, non-parametric tests (Wilcoxon signed-rank tests) were employed to assess the performance differences between pairs of conditions, with Holm-Bonferroni correction applied to account for multiple pairwise comparisons. The corrected p -values that indicated statistically significant differences are highlighted in bold.**
\begin{tabular}[t]{|l|l|l|l|l|l|l|l|l|l|l|}
\hline \multicolumn{3}{|c|}{} & \multicolumn{4}{|l|}{Podcast Transcripts} & \multicolumn{4}{|l|}{News Articles} \\
\hline & Condition 1 & Condition 2 & Mean 1 & Mean 2 & Z-value & p-value & Mean 1 & Mean 2 & Z-value & p-value \\
\hline \multirow{15}{*}{Comprehensiveness} & C0 & TS & 50.24 & 49.76 & -0.06 & 1 & 55.52 & 44.48 & -2.03 & 0.17 \\
\hline & C1 & TS & 51.92 & 48.08 & -1.56 & 0.633 & 58.8 & 41.2 & -3.62 & 0.002 \\
\hline & C2 & TS & 57.28 & 42.72 & -4.1 & <0.001 & 62.08 & 37.92 & -5.07 & <0.001 \\
\hline & C3 & TS & 56.48 & 43.52 & -3.42 & 0.006 & 63.6 & 36.4 & -5.63 & <0.001 \\
\hline & C0 & SS & 71.92 & 28.08 & -6.2 & <0.001 & 71.76 & 28.24 & -6.3 & <0.001 \\
\hline & C1 & SS & 75.44 & 24.56 & -7.45 & <0.001 & 74.72 & 25.28 & -7.78 & <0.001 \\
\hline & C2 & SS & 77.76 & 22.24 & -8.17 & <0.001 & 79.2 & 20.8 & -8.34 & <0.001 \\
\hline & C3 & SS & 78.96 & 21.04 & -8.12 & <0.001 & 79.44 & 20.56 & -8.44 & <0.001 \\
\hline & TS & SS & 83.12 & 16.88 & -8.85 & <0.001 & 79.6 & 20.4 & -8.27 & <0.001 \\
\hline & C0 & C1 & 53.2 & 46.8 & -1.96 & 0.389 & 51.92 & 48.08 & -0.45 & 0.777 \\
\hline & C0 & C2 & 50.24 & 49.76 & -0.23 & 1 & 53.68 & 46.32 & -1.54 & 0.371 \\
\hline & C1 & C2 & 51.52 & 48.48 & -1.62 & 0.633 & 57.76 & 42.24 & -4.01 & <0.001 \\
\hline & C0 & C3 & 49.12 & 50.88 & -0.56 & 1 & 52.16 & 47.84 & -0.86 & 0.777 \\
\hline & C1 & C3 & 50.32 & 49.68 & -0.66 & 1 & 55.12 & 44.88 & -2.94 & 0.016 \\
\hline & C2 & C3 & 52.24 & 47.76 & -1.97 & 0.389 & 58.64 & 41.36 & -3.68 & 0.002 \\
\hline \multirow{15}{*}{Diversity} & C0 & TS & 50.24 & 49.76 & -0.11 & 1 & 46.88 & 53.12 & -1.38 & 0.676 \\
\hline & C1 & TS & 50.48 & 49.52 & -0.12 & 1 & 54.64 & 45.36 & -1.88 & 0.298 \\
\hline & C2 & TS & 57.12 & 42.88 & -2.84 & 0.036 & 55.76 & 44.24 & -2.16 & 0.184 \\
\hline & C3 & TS & 54.32 & 45.68 & -2.39 & 0.1 & 60.16 & 39.84 & -4.07 & <0.001 \\
\hline & C0 & SS & 76.56 & 23.44 & -7.12 & <0.001 & 62.08 & 37.92 & -3.57 & 0.003 \\
\hline & C1 & SS & 75.44 & 24.56 & -7.33 & <0.001 & 64.96 & 35.04 & -4.92 & <0.001 \\
\hline & C2 & SS & 80.56 & 19.44 & -8.21 & <0.001 & 70.56 & 29.44 & -6.29 & <0.001 \\
\hline & C3 & SS & 80.8 & 19.2 & -8.3 & <0.001 & 69.12 & 30.88 & -5.53 & <0.001 \\
\hline & TS & SS & 82.08 & 17.92 & -8.43 & <0.001 & 67.2 & 32.8 & -4.85 & <0.001 \\
\hline & C0 & C1 & 49.76 & 50.24 & -0.13 & 1 & 39.68 & 60.32 & -3.61 & 0.003 \\
\hline & C0 & C2 & 46.32 & 53.68 & -1.5 & 0.669 & 40.96 & 59.04 & -3.14 & 0.012 \\
\hline & C1 & C2 & 44.08 & 55.92 & -3.27 & 0.011 & 50.24 & 49.76 & -0.22 & 1 \\
\hline & C0 & C3 & 44 & 56 & -2.6 & 0.065 & 41.04 & 58.96 & -3.47 & 0.004 \\
\hline & C1 & C3 & 45.44 & 54.56 & -2.98 & 0.026 & 49.52 & 50.48 & -0.01 & 1 \\
\hline & C2 & C3 & 48.48 & 51.52 & -0.96 & 1 & 50.96 & 49.04 & -0.39 & 1 \\
\hline \multirow{15}{*}{Empowerment} & C0 & TS & 40.96 & 59.04 & -4.3 & <0.001 & 42.24 & 57.76 & -3.32 & 0.012 \\
\hline & C1 & TS & 45.2 & 54.8 & -3.76 & 0.002 & 50 & 50 & -0.12 & 1 \\
\hline & C2 & TS & 47.68 & 52.32 & -2.2 & 0.281 & 49.52 & 50.48 & -0.22 & 1 \\
\hline & C3 & TS & 48.72 & 51.28 & -1.27 & 1 & 51.68 & 48.32 & -1.2 & 1 \\
\hline & C0 & SS & 42.96 & 57.04 & -3.71 & 0.003 & 42.72 & 57.28 & -3.12 & 0.022 \\
\hline & C1 & SS & 47.68 & 52.32 & -1.5 & 0.936 & 51.36 & 48.64 & -0.84 & 1 \\
\hline & C2 & SS & 50.72 & 49.28 & -0.55 & 1 & 49.84 & 50.16 & -0.2 & 1 \\
\hline & C3 & SS & 48.96 & 51.04 & -0.57 & 1 & 49.52 & 50.48 & -0.08 & 1 \\
\hline & TS & SS & 57.52 & 42.48 & -4.1 & <0.001 & 52.88 & 47.12 & -1.1 & 1 \\
\hline & C0 & C1 & 48.72 & 51.28 & -1.23 & 1 & 42.4 & 57.6 & -3.9 & 0.001 \\
\hline & C0 & C2 & 46.64 & 53.36 & -2.54 & 0.12 & 44.8 & 55.2 & -2.16 & 0.336 \\
\hline & C1 & C2 & 49.28 & 50.72 & -1.73 & 0.682 & 52 & 48 & -1.45 & 1 \\
\hline & C0 & C3 & 47.6 & 52.4 & -1.78 & 0.682 & 44.32 & 55.68 & -3.45 & 0.008 \\
\hline & C1 & C3 & 50 & 50 & 0 & 1 & 51.44 & 48.56 & -1.02 & 1 \\
\hline & C2 & C3 & 50.72 & 49.28 & -0.86 & 1 & 50.4 & 49.6 & -0.22 & 1 \\
\hline \multirow{15}{*}{Directness} & C0 & TS & 44.96 & 55.04 & -4.09 & <0.001 & 45.2 & 54.8 & -3.68 & 0.003 \\
\hline & C1 & TS & 47.92 & 52.08 & -2.41 & 0.126 & 46.64 & 53.36 & -2.91 & 0.04 \\
\hline & C2 & TS & 48.8 & 51.2 & -2.23 & 0.179 & 48.32 & 51.68 & -2.12 & 0.179 \\
\hline & C3 & TS & 48.08 & 51.92 & -2.23 & 0.179 & 48.32 & 51.68 & -2.56 & 0.074 \\
\hline & C0 & SS & 35.12 & 64.88 & -6.17 & <0.001 & 41.44 & 58.56 & -4.82 & <0.001 \\
\hline & C1 & SS & 40.32 & 59.68 & -4.83 & <0.001 & 45.2 & 54.8 & -3.19 & 0.017 \\
\hline & C2 & SS & 40.4 & 59.6 & -4.67 & <0.001 & 44.88 & 55.12 & -3.65 & 0.003 \\
\hline & C3 & SS & 40.48 & 59.52 & -4.69 & <0.001 & 45.6 & 54.4 & -2.86 & 0.043 \\
\hline & TS & SS & 43.6 & 56.4 & -3.96 & <0.001 & 46 & 54 & -2.68 & 0.066 \\
\hline & C0 & C1 & 46.96 & 53.04 & -2.87 & 0.037 & 47.6 & 52.4 & -2.17 & 0.179 \\
\hline & C0 & C2 & 48.4 & 51.6 & -2.06 & 0.197 & 48.48 & 51.52 & -1.61 & 0.321 \\
\hline & C1 & C2 & 49.84 & 50.16 & -1 & 0.952 & 49.28 & 50.72 & -1.6 & 0.321 \\
\hline & C0 & C3 & 48.4 & 51.6 & -1.8 & 0.29 & 47.2 & 52.8 & -2.62 & 0.071 \\
\hline & C1 & C3 & 49.76 & 50.24 & 0 & 1 & 48.8 & 51.2 & -1.29 & 0.321 \\
\hline & C2 & C3 & 50 & 50 & 0 & 1 & 48.8 & 51.2 & -1.84 & 0.262 \\
\hline
\end{tabular}
\end{table}
