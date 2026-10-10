# Design for Jenga: Modular Block-Stacking Task (ManiSkill)

> AI transcription of the "Instructions for CLAUDE" section of the handwritten `notes/figs/Design for Jenga.pdf`, written for downstream coding agents.
> Text is OCR'd and lightly cleaned up. Diagrams are interpreted. Anything uncertain is tagged **[UNCERTAIN]** and collected under *Open questions*.
> The author's own note says the stability / construction-plan question must be **discussed before implementation**.

---

## Task overview

A Jenga-style modular design task for ManiSkill. `3x1x1` blocks (unit left undefined for now) lie randomly spread out on a tabletop surface. A `5x5x5` design area and a target design are specified. The goal is to get the manipulator to stack blocks within the design area to match the target design as closely as possible.

Sections below specify how designs are generated, how results are scored, and how the initial table is laid out.

---

## 1. Generating designs

**Representation**
- A design is a `5x5x5` integer matrix.
- Each `3x1x1` block that shows up in the design is represented by its own integer in `[1 ... 41]`, covering the up to 41 blocks that could show up (note `floor(125/3) = 41`).
- Most designs will be simpler, with `< 15` blocks. The block count is the knob for varying difficulty.
- Integer `0` means **no block** is placed at that location.

**Stability constraint (important)**
- The structure must be **stable** under gravity / the physics of the ManiSkill simulator.
- NOTE (a further constraint, to be considered and discussed before implementation): there should exist a path of blocks `(b_t)` such that each substructure is also stable, yielding a clean **construction plan** for the robot.

**Generation**
- Given these constraints, an LLM agent or a random combinatorial algorithm can come up with designs.

**Orientation**
- Blocks may be laid out either:
  - **horizontally** (`3x1` side down), in either direction, or
  - **vertically** (`1x1` side down).

---

## 2. Worked example

A design where **all blocks are laid horizontally and only three layers are used** (neither must be the case in general).

Top-down `5x5` views, with handwriting colour = layer: black = layer 1, red = layer 2, blue = layer 3.

| Layer | Colour | # blocks | Sketch interpretation |
|---|---|---|---|
| 1 | black | 4 | Hatched `3x1` bars. Top row spans cols 2-4; the rest look like vertical bars in cols 1, 3, 5. **[UNCERTAIN]** exact cells; bottom row looks partially hatched. |
| 2 | red | 3 | Row 2 cols 1-3, row 3 cols 3-5, row 4 cols 1-3 (all horizontal). |
| 3 | blue | 3 | Drawn over the red bars: roughly the left side (cols 1-2, rows 2-4) and row 3 cols 3-5. **[UNCERTAIN]** |

Total: 4 + 3 + 3 = **10 block design**.

Layer 2 (the one I can read confidently), `R` = occupied:

```
col:  1 2 3 4 5
row1: . . . . .
row2: R R R . .
row3: . . R R R
row4: R R R . .
row5: . . . . .
```

---

## 3. Scoring

Result = **number of cells where the built structure and the target design are both occupied** (plain cell overlap; the author's chosen v1 mechanism).

- A block occupies **exactly 3 cells**: the 3 cells containing the most of the block's mass (i.e. snap the block's pose to the grid and take the 3 cells along its long axis).
- Two blocks **may occupy the same cell** (occupancy is a set / boolean per cell, so overlap is not double counted or penalised).
- No shifting or alignment search in v1.

## 4. Initial layout

- **Half the table** is dedicated to blocks that are randomly laid out.
- The **other half** has a `5x5` grid on it as a visual marker of the base of the design space.
- The manipulator should start **somewhere in between**.

Sketch (isometric): a rectangular table. One half has ~6 scattered bars at varied orientations; the other half has the drawn 5x5 grid; the robot arm stands at the boundary, with a camera view-cone above it.

```
[ scattered 3x1x1 blocks ]   [robot]   [ 5x5 grid marker ]
        random layout      in-between   base of design space
```

---

## 5. v1 decisions (confirmed by author)

- **Robot and cameras:** same as `StackThree-v1` in `control/api/maniskill_tasks.py` (`panda_wristcam`, same sensor cameras). Do **not** reuse its cubes or task logic as a baseline: it uses different blocks and has no structured grid.
- **Block size:** the StackThree cubes (`half_size=0.02`, i.e. 4 cm) are the `1x1x1` unit. So a block is `12 x 4 x 4 cm`, and one grid cell is `4 cm`, making the design area `20 x 20 x 20 cm`.
- **Scoring:** plain cell overlap (section 3).
- **Stability:** a design is stable if, with all blocks **placed perfectly** at their design poses in the ManiSkill sim and physics stepped, the blocks deviate less than some minimum threshold from their placed poses. Threshold and number of settle steps are config constants to be chosen. The "construction plan" (every substructure stable) constraint is **not** in v1.

## 6. Implementation checklist

- [ ] New ManiSkill task env (own file in `control/api/`), reusing StackThree's robot and cameras.
- [ ] `3x1x1` bar actors (`12 x 4 x 4 cm`), 5x5 grid marker (4 cm cells) on one half of the table, random scatter on the other half, robot spawns between.
- [ ] Design representation: int array `(5, 5, 5)`, `0` = empty, `1..41` = block ids.
- [ ] Design generator (random, layer by layer) with the stability check from section 5.
- [ ] Horizontal (either direction) and vertical block orientations.
- [ ] Scorer: pose to 3 highest-mass cells, then cell overlap with target.
- [ ] Difficulty control via block count.
- [ ] Per CLAUDE.md: code touching ManiSkill goes in `control/api/`; experiments in notebooks.

## 7. Remaining open questions

Mostly resolved by section 8. Still agent's choice: block count per episode (suggest a few spare blocks), random yaw and collision-free sampling of table blocks (as in `StackThree`), seeded reproducibility. The layer 1 and 3 diagrams of the worked example are hard to read, so don't use the 10-block example as a test fixture.

---

## 8. Clarifications (from author, post-transcription)

Added after the handwritten notes were transcribed. Where these conflict with anything above, **these win**.

1. **Focus:** ignore the FloMo section at the top of the PDF. The Jenga task (from "Instructions for CLAUDE" onward) is what matters.
2. **Baseline:** `StackThree-v1` is **not** a good baseline for the task: it uses different blocks and has no structured grid.
3. **Robot and cameras:** should closely match StackThree: same robot (`panda_wristcam`) and same cameras.
4. **Block size:** the StackThree cubes are the `1x1x1` unit, so a design block is `3x1x1` of those cubes (`12 x 4 x 4 cm`; cell = `4 cm`).
5. **Scoring:** plain cell overlap, as the simplest mechanism for now.
6. **Cell occupancy:** a block occupies **exactly three cells**, the cells containing the most mass of the block.
7. **Shared cells:** two blocks **can** occupy the same cell.
8. **Stability:** defined as some minimum deviation when the blocks are perfectly placed in the ManiSkill environment and physics is run.
9. **Scope:** just build the **environment** for now. No policy, observation design for a policy, or reward work needed.
10. **Stability defaults:** the agent may pick the settle time and deviation threshold (suggested: ~1 s of sim time, each block moving under ~5 mm), as config constants.
11. **Design area placement:** decided by the agent, but it should sit **near the front of the robot** so the manipulator can reach all around it. The agent should also fix and document the `(5,5,5)` indexing convention (which corner is `[0,0,0]`, which axis is x).
12. **Reachability:** it is fine if some grasps are hard or impossible. Simple designs should be easy to build. Reachability / construction-plan constraints are deferred.

---

## 9. v1 Deliverable: agent-generated demonstration of the example design

**Goal:** a visual sanity check of the environment. A scripted agent builds the **10-block example design** (section 2) in the sim, and the author watches the video to inspect what the environment looks like (robot, cameras, table split, grid marker, blocks, stacking behaviour). Still environment-only: no policy, no learning.

**Reference pattern:** follow how the StackThree demo was produced (not its cubes or task logic, only the workflow):
- `solve_stack_three` in `control/api/maniskill_tasks.py` is a **motion-planned scripted solution** (`PandaArmMotionPlanningSolver`, `control_mode="pd_joint_pos"`) that uses **privileged simulator state** (object poses, grasp geometry from the block's oriented bounding box), not vision.
- It is recorded once for a fixed seed with ManiSkill's `RecordEpisode` wrapper into `<repo>/demos/<task>/motionplanning/` (`api.maniskill.REPO_DEMO_DIR`).
- The notebook (`control/notebooks/mani-skill-quickstart/stack_three.ipynb`) then loads it via `api.maniskill.load_demo`, replays the recorded actions open-loop from the demo's exact reset kwargs, and shows the frames with `api.video.show`.

**What to build**
1. The Jenga environment (sections 1-6 and 8), registered under its own task id.
2. A fixed, concrete **10-block target design** matching the example as closely as possible:
   - 4 blocks in layer 1, 3 in layer 2, 3 in layer 3, all laid flat (horizontal), as in section 2.
   - Layer 2 is the one read confidently (rows 2 and 4 cols 1-3, row 3 cols 3-5). Layers 1 and 3 are **[UNCERTAIN]**, so the agent should pick a layout consistent with the sketch that is stable and print the final `(5,5,5)` matrix. The author will confirm or correct it.
   - It must pass the stability check (section 5).
3. A scripted solver (like `solve_stack_three`, generalised to N blocks) that builds **bottom-up, layer by layer**: for each target block, grasp it from the scattered half of the table, move it over the grid, and place it at its design pose. Placement orientation (x or y direction along the 3-cell axis) follows the design. Grasp-search, as in `_pick_and_stack`, over the block's random yaw.
4. A recorded demonstration for a fixed seed, saved the same way as the StackThree one (small trajectory `.h5` + `.json`).
5. A notebook that (a) shows the reset scene, (b) replays the demo and displays the video **inline**, (c) prints the target `(5,5,5)` matrix, the achieved occupancy and the **cell-overlap score** from section 3.

**Rules from CLAUDE.md that apply**
- Third-party-facing code (ManiSkill env, solver, scorer) lives in `control/api/`; the notebook only imports from it.
- Plots and videos stay **inside the notebook**, with no stray video files in the workspace. The only committed artifact beyond code is the tiny demo trajectory, as with StackThree.
- Do **not** write markdown cells in the notebook (human only); explain with Python comments in code cells.

**Done when**
- The notebook runs top to bottom and shows the video of the robot assembling the design.
- The final score is printed. Ideal is a full match (30 occupied cells if no cell is shared); if the planner cannot place some blocks, report the partial score and which blocks failed.
- It is acceptable for hard grasps to fail (clarification 12), but the simple layers (layer 1 first) should work.

**Open for the agent to report back:** the exact `(5,5,5)` design used, the stability threshold/settle time chosen, the grid-marker pose and indexing convention, and any blocks the solver could not place.
