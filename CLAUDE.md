
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
 - `<code-directory>/deps` is where all third party vendored code should sit. All files that were written in this project AND refer to `<code-directory>/deps` should live in `<code-directory>/api`. In other words, all code that interacts with third-party writtetn code should sit in the `<code-directory>/api/` directory, and those files should expose a nice interface for the rest of the codebase (e.g. files that sit in `<code-directory>/notebooks`) 
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
- In general all experimentation / prototypes should be done in notebooks, and scripts should ONLY be built when long training jobs or parameter sweeps need to be run that will last longer than a single dev session. The organization of outputs should be discussed to avoid cluttering the repo. 



___
#  Project-specific notes: `control-proj`
- This repository will generally leverage a lot of heavy simulation and reinforcement learning training infrastructure.
- Despite the heavy infrastructure, we will still try to keep experimnets self-contained in singular .ipynb jupyter notebooks or .py scripts
- When a experiment is run in a jupyter notebook, all plots and videos should be saved directly in the notebook to keep it self contained. This is also to ensure that random one-off videos do NOT CLUTTER the workspace.
