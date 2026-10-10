"""Random design generator + cheap geometric support heuristic for the Jenga task (pure numpy, no simulator).

Builds on `api.jenga_core` (design[i, j, k], k = 0 on the table, ids 1..41).

Support rules (`is_supported`), applied per block; cell units, equal block masses:
    1. Contacts: a block's bottom cells (all 3 for horizontal, the lowest one for vertical) touch the table
       (k = 0) or an occupied cell directly below. Its supporters are the blocks owning those cells below.
    2. Own balance: a horizontal block needs its middle cell supported, or both end cells (no cantilevers).
       A vertical block needs its bottom cell supported.
    3. Load balance (top-heavy check): the "exclusive load" of block b is b plus every block resting, directly
       or transitively, only on b and that load set. Its xy centre of mass must lie inside b's support
       rectangle (bounding box of b's contact cell squares, shrunk by `margin`). Blocks resting on several
       independent supports are not attributed to any one of them (heuristic; physics is the arbiter).
    4. Vertical blocks may not carry an exclusive load: anything on a 1x1 pillar must also rest elsewhere
       (e.g. a bridge across two pillars is fine, a T on a single pillar is not).

Generator (`random_design`): layers are filled bottom-up and never revisited, so construction order is
layer order; ids 1..N follow it, and every prefix of the id order also passes `is_supported` (each block is
checked as it is added). Vertical blocks are drawn with low probability (`p_vertical`) because a 1x1-footprint
pillar is tippy.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from api.jenga_core import BLOCK_LEN, GRID, MAX_BLOCKS, Block, design_to_blocks, empty_design, make_block

StabilityFn = Callable[[np.ndarray], bool]


def contacts(design: np.ndarray, block: Block) -> np.ndarray:
    """Bool mask over the block's bottom cells (3 if horizontal, 1 if vertical): touching table or a block."""
    bottom = block.cells[:1] if block.axis == 2 else block.cells
    if bottom[0, 2] == 0:
        return np.ones(len(bottom), dtype=bool)
    return design[bottom[:, 0], bottom[:, 1], bottom[:, 2] - 1] != 0


def supporters(design: np.ndarray, block: Block) -> set[int]:
    """Ids of blocks directly under `block` (empty if it rests on the table)."""
    bottom = block.cells[:1] if block.axis == 2 else block.cells
    if bottom[0, 2] == 0:
        return set()
    below = design[bottom[:, 0], bottom[:, 1], bottom[:, 2] - 1]
    return {int(v) for v in below if v}


def locally_supported(design: np.ndarray, block: Block) -> bool:
    """Rules 1-2: middle (or both ends) supported for horizontal blocks, bottom cell for vertical ones."""
    c = contacts(design, block)
    return bool(c[0]) if block.axis == 2 else bool(c[1] or (c[0] and c[2]))


def _loads_balanced(design: np.ndarray, blocks: list[Block], margin: float) -> bool:
    """Rules 3-4 over all blocks. `blocks` must be sorted bottom-up (as from design_to_blocks)."""
    sup = {b.id: supporters(design, b) for b in blocks}
    xy = {b.id: b.center[:2] for b in blocks}
    for n, b in enumerate(blocks):
        group = {b.id}
        for c in blocks[n + 1 :]:  # supporters always sit in a lower layer, so one pass suffices
            if sup[c.id] and sup[c.id] <= group:
                group.add(c.id)
        if b.axis == 2 and len(group) > 1:
            return False
        bottom = b.cells[:1] if b.axis == 2 else b.cells
        touch = bottom[contacts(design, b)][:, :2]
        lo, hi = touch.min(axis=0) + margin, touch.max(axis=0) + 1 - margin
        com = np.mean([xy[g] for g in group], axis=0)
        if np.any(com < lo) or np.any(com > hi):
            return False
    return True


def is_supported(design: np.ndarray, margin: float = 0.1) -> bool:
    """Cheap geometric pre-filter for physical stability (rules in the module docstring)."""
    blocks = design_to_blocks(design)
    return all(locally_supported(design, b) for b in blocks) and _loads_balanced(design, blocks, margin)


def _local_candidates(design: np.ndarray, k: int, max_height: int) -> tuple[list[Block], list[Block]]:
    """(horizontal, vertical) placements with bottom layer k that fit, are empty and pass rules 1-2."""
    horiz, vert = [], []
    for axis in (0, 1):
        for a in range(GRID - BLOCK_LEN + 1):
            for b in range(GRID):
                start = (a, b, k) if axis == 0 else (b, a, k)
                blk = make_block(0, start, axis)
                if not design[tuple(blk.cells.T)].any() and locally_supported(design, blk):
                    horiz.append(blk)
    if k + BLOCK_LEN <= max_height:
        for i in range(GRID):
            for j in range(GRID):
                blk = make_block(0, (i, j, k), 2)
                if not design[tuple(blk.cells.T)].any() and locally_supported(design, blk):
                    vert.append(blk)
    return horiz, vert


def _sample_once(
    num_blocks: int,
    rng: np.random.Generator,
    allow_vertical: bool,
    max_height: int,
    p_up: float,
    p_vertical: float,
    margin: float,
) -> np.ndarray | None:
    """One greedy bottom-up attempt; None on a dead end."""
    design, blocks, layer, in_layer = empty_design(), [], 0, 0
    while len(blocks) < num_blocks:
        placed = False
        # try the current layer (unless we randomly move up), then each higher layer in turn; moving up is only
        # voluntary once this layer holds its fair share of what is left, so large designs don't run out of room
        layers = list(range(layer, max_height))
        quota = -(-(num_blocks - len(blocks) + in_layer) // len(layers))
        if len(layers) > 1 and in_layer >= quota and rng.random() < p_up:
            layers = layers[1:] + layers[:1]
        for k in layers:
            horiz, vert = _local_candidates(design, k, max_height)
            if not allow_vertical:
                vert = []
            pools = [vert, horiz] if vert and (not horiz or rng.random() < p_vertical) else [horiz, vert]
            for pool in pools:
                for idx in rng.permutation(len(pool)):
                    blk = make_block(len(blocks) + 1, tuple(pool[idx].cells[0]), pool[idx].axis)
                    design[tuple(blk.cells.T)] = blk.id
                    if _loads_balanced(design, blocks + [blk], margin):
                        blocks.append(blk)
                        in_layer = in_layer + 1 if k == layer else 1
                        placed, layer = True, k
                        break
                    design[tuple(blk.cells.T)] = 0
                if placed:
                    break
            if placed:
                break
        if not placed:
            return None
    return design


def random_design(
    num_blocks: int,
    rng: None | int | np.random.Generator = None,
    *,
    allow_vertical: bool = True,
    stability_fn: StabilityFn | None = None,
    max_tries: int = 200,
    max_height: int = GRID,
    p_up: float = 0.3,
    p_vertical: float = 0.1,
    margin: float = 0.1,
) -> np.ndarray:
    """Random (5, 5, 5) design with exactly `num_blocks` blocks, ids 1..N in bottom-up build order.

    Each attempt is a greedy layer-by-layer build: at every step stay in the current layer (or, with
    probability `p_up` once the layer holds ceil(blocks left / layers left), move to the next layer that admits
    a block), pick a vertical block with probability
    `p_vertical` (else horizontal, either direction), and accept the first random placement passing
    `is_supported`. Attempts that dead-end, or that `stability_fn(design)` rejects, are resampled from the same
    rng stream, so a seed fully determines the output (given a deterministic `stability_fn`).
    `max_height` caps the number of layers used (smaller -> flatter, wider designs).
    """
    if not 0 <= num_blocks <= MAX_BLOCKS:
        raise ValueError(f"num_blocks must be in [0, {MAX_BLOCKS}], got {num_blocks}")
    rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    for _ in range(max_tries):
        design = _sample_once(num_blocks, rng, allow_vertical, max_height, p_up, p_vertical, margin)
        if design is not None and (stability_fn is None or stability_fn(design)):
            return design
    raise RuntimeError(f"no valid {num_blocks}-block design found in {max_tries} tries")
