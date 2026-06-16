# Paper-to-Program Many-Body Archive

This export folder contains the data and code artifacts associated with the paper-to-program many-body experiments.

## Layout

- `DMRG/` contains the DMRG paper-to-program case-study archive.
- `Pfaffian-MPS/` contains the Pfaffian/HFB-to-MPS paper-to-program case-study archive.
- Each case-study folder has its own `README.md` with detailed subfolder notes and suggested entry points.

## Parallel Archive Structure

Both case-study folders collect the same kinds of evidence:

- Prompts and transcripts recording direct baselines and staged workflow runs.
- Generated specifications linking source literature to implementation.
- Implementation code, notebooks, tests, benchmark scripts, and run-local outputs.
- Validation artifacts, figures or figure prompts, and workflow notes.
- Summary documents reporting pass/fail outcomes and reproducibility context.

## Folder Notes

- `DMRG/` keeps its historical subfolder names: `Markdown/`, `LaTeX/`, `Code/`, and `FigPrompt/`.
- `Pfaffian-MPS/` uses explicit workflow subfolders: `Prompts/`, `Specifications/`, `Agent-Runs/`, `Skills/`, and `Summaries/`.

Local virtual environments, Python caches, Claude local settings, `.DS_Store` files, duplicated source PDFs, and LaTeX auxiliary build files are excluded where applicable.
