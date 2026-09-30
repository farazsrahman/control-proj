
This is a modular `CLAUDE.md` file. I will define some constants below that will be specific to this project, but used generally throughout.
- `<code-directory>/` = `control`
# Overview

This is an all-in-one repo for a machine learning research project. It includes
 - Human-written research logs in `notes/`
 - AI-maintained working notes in `ai-notes/` (mirrors the structure of `notes/`)
 - A uv-managed project folder called `<code-directory>/` with it's own structure for code `.py`'s and experiment notebooks `.ipynb`'s
 - A folder of PDFs of related work in `refs/`
 - A folder containing the current working draft of the project called `overleaf/` 

Dev Node
- All interaction writing + human work is done on a local macbook. All code is run on a remote dev node that is sync'd via the `Makefile` and the `.rsyncignore` files. 
# Rules

There are only a few strict rules in this repo:
 - AIs should *never* write in the human-written `notes/` folder *or* in the markdown section of an `.ipynb`. 
 - A human may prompt you to draft a text snippet, table, or otherwise, you may return a draft so that a Human can decide whether to include it or not. 
 - If you feel something is important to note, you may add it to `ai-notes/` instead of `notes/` or in a python comment in the code section of the `.ipynb`. 
 
# AI Working Notes

There is an `ai-notes/` folder that mirrors `notes/` but is maintained by AIs (humans write in `notes/`, AIs write in `ai-notes/`).
 - Notes should be added to `ai-notes/` **sparingly** to avoid clutter, and kept **short** — a few lines of only the critical, actionable points, never a full write-up. Err on the side of under-reporting.
 - **Whenever you append to (or edit) `ai-notes/`, paste the exact logged text back in the chat response** so the user can review it inline and never has to open the file to check what was written.
 - `ai-notes/` should use the **same date header format** as `notes/Work Log.md` (e.g. `### May 31st, 2026`).
 - Whenever opening this repo, take a quick scan of the **4 most recent dated entries** (by `### Month Dayth, Year` header) from `Work Log.md` in **both** `notes/` and `ai-notes/` to understand the current context of the project. The logs are append-only and never archived, so read only the latest entries rather than the whole file.

# Preferences

There are a couple of preferences imposed by the Author that are not strict but good to know:
- The author prefers that small experiments are implemented in python notebooks with abstract-able function definitions pulled out into helper files.
	- Frequently the author will request to collaboratively build small experiments by describing implementations in markdown and requesting the agent to add relevant code in the following cell. It is important that you always read the notebook itself before editing it to avoid overwriting changes.
- The author prefers that large experiments are implemented as single `<exp-name>.py` scripts with a associated folder of `<exp-name>-configs/*.yaml` configs and a `.ipynb` notebooks to visualize results. 



___
#  Project-specific notes: `adaptyv-comp`

- This repo is a project directory for the Anthropic x Adaptyv protein design competition
	- https://proteinbase.com/competitions/anthropic-adaptyv-2026/challenges/egfr
- There is a competition released FAQ that is copy-pasted in [[Competition FAQ]].
- There are 5 sub-challenges in the competition. Each will have it's own note e.g. [[Challenge 01 - EGFR Binder Design]] with the exact comptition description copy-pasted

- There will be a source folder called `programbio/` that contains a broad set of infrastructure for biological design of peptides, molecules, DNA, proteins etc. 
- All infrastructure requests that are generalizable to broader biological design tasks should be directed to the `programbio/` directory INSTEAD of `adaptyv-comp`
