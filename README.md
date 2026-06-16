# DMRG Paper-to-Program Archive

This folder contains the DMRG paper-to-program data-and-code archive. It preserves the historical directory layout while keeping all DMRG-specific materials separate from the companion `Pfaffian-MPS/` archive.

## Directory Layout

- `Code/`: generated DMRG implementations from GPT, Claude, Gemini, and Kimi runs.
- `Markdown/`: conversation transcripts, staged workflow transcripts, and zero-shot baseline records.
- `LaTeX/`: generated technical specifications and notes for the DMRG workflow.
- `FigPrompt/`: prompts used for figure generation.
- `Gemini-workflow-*.md`: manuscript, presentation, essay, and workflow reflection materials.

## Archive Contents

The DMRG case study converts Schollwock's DMRG review into matrix-product-state code and validates the implementations against the spin-1/2 Heisenberg chain and spin-1 AKLT model. The archive includes both specification-guided workflow runs and direct or zero-shot attempts.

Folder-local README files in `Code/` and `Markdown/` document naming conventions and transcript organization.

## Reading Order

Start with:

- `Markdown/README.md` for transcript organization.
- `Code/README.md` for generated-code organization.
- `Markdown/ZeroShot/` for direct baseline attempts.
- `Markdown/Code/` and `LaTeX/` for specification-guided workflow records.

## Notes

This folder is intentionally archival. File names preserve the original model and pass/fail labels where available. The companion Pfaffian-MPS case study is stored separately in `../Pfaffian-MPS/`.
