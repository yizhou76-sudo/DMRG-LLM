Conversations with LLMs (Kimi 2.5, Gemini 3.1 Pro Preview, GPT 5.4, and Claude 4.6) are organized by topic: LaTeX notes are saved in 'TeX', and coding conversations in 'Code'.

=====================================
Directory TeX: Contains 5 .docx files (MPS-TeX-Kimi#1 through #5), representing the conversation thread that led to the final output main-MPS-Kimi-318.tex. MPS-TeX-Kimi-Claude#1 is the first round of LLM-assisted review (LLM-1), with subsequent files following the same numbering convention.

=====================================
Directory Code: MPS-Code-Claude#n-Fail(Pass).md corresponds to the n-th round of Claude-to-Claude iterative coding. MPS-Code-Claude-GPT#n-Pass(Fail) is the n-th round of Claude-to-GPT coding handoff that get pass (fail).

Prompt for LLM-1->LLM-2 run:
"This is a latex note on MPS and DMRG. Could you generate python codes according to it? Please compute both S=1/2 Heisenberg model and AKLT model, and make sure that a "scalable" matrix-free approach with LANCZOS is adapted. I work with Jupyter. Please show all figures and save them in directory "figureAKLT". Please stay strictly with the latex note implementation."

=====================================
Directory ZeroShot: Four LLMs were given the following prompt in a zero-shot setting:

"This is a review article on MPS-based DMRG. Could you generate python codes according to it, by using MPS/MPO finite-DMRG implementation. Please build Python codes from scracth, adhere strictly with the review article, and use only numpy + scipy + matplotlib. Please compute both S=1/2 Heisenberg model and AKLT model using two-site DMRG algorithm, and make sure that a "scalable" matrix-free approach with LANCZOS is adapted. I work with Jupyter and prefer to fewer cells. Please show all figures and save them in directory "figureAKLT". Begin with a quick test on small systems, then go to larger systems."

Step by Step. Test each Cell in order.

Note Added to Kimi Agent: Please test and debug the system until it is fully functional and produces accurate energy benchmarks. Please benchmark at larger system size up to L=20 and D=40 for S=1/2 Heisenberg model, and L=20 and D=8 for AKLT model.

======================================
Counts HITL feedback rounds for sucuessful LLM2 coders (except Kimi Agent, where is no HITL in debuging, only physics diagnosis), using the prompt as follows:
> Count the number of rounds involving debugging or physics diagnosis after the first code delivery.
> (For confirmation) In each round, how many conversation exchanges are there?
