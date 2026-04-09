# DMRG-LLM
Documents of LLM-Assisted Workflow for MPS/DMRG

Supplementary Materials for the following paper:

From Paper to Program: Accelerating Quantum Many-Body Algorithm Development via a Multi-Stage LLM-Assisted Workflow

https://arxiv.org/abs/2604.04089

To ensure full transparency and reproducibility, all materials associated with this study have been made publicly available in the GitHub repository \texttt{DMRG-LLM}~\cite{yizhou_dmrg_llm_2026}. The repository is organized as follows:

	\begin{itemize}
		\item \textbf{Markdown:} Complete, unedited transcripts of the conversations with the LLMs across all stages. This includes interactions with Kimi 2.5, Gemini 3.1 Pro Preview, GPT 5.4, and Claude Opus 4.6.
		\item \textbf{LaTeX:} The intermediate, mathematically rigorous technical specifications (e.g., the LLM-1 outputs) that bridged the theoretical source text to the final code.
		\item \textbf{Code:} The final, object-oriented Python codebase (\texttt{MPS}, \texttt{MPO}, and \texttt{DMRGEngine} classes) and the Jupyter notebooks used for the physical verification benchmarks.
		\item \textbf{FigPrompt:} The exact design briefs and prompts provided to Nano Banana 2 for the generation of the workflow and timeline diagrams.
	\end{itemize}
