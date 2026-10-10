"""Design representation for the Jenga modular-design task (pure numpy, no simulator).

Shared by the env (`api.jenga`), the design generator (`api.jenga_designs`) and the scripted solver.

Conventions (fixed, see ai-notes/Design for Jenga.md):
    design: int array of shape (5, 5, 5), indexed design[i, j, k]
        i -> world +x, j -> world +y, k -> world +z (k = 0 is the layer resting on the table)
        0 = empty, 1..41 = block id; each block id occupies exactly 3 contiguous collinear cells.
    A block's `axis` is the axis its 3 cells run along: 0 = x (horizontal), 1 = y (horizontal), 2 = z (vertical).
    Cell (i, j, k) is the cube [i, i+1] x [j, j+1] x [k, k+1] in cell units; the env maps cell units to
    world coordinates via its grid origin and CELL_SIZE.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

GRID = 5  # design area is GRID x GRID x GRID cells
BLOCK_LEN = 3  # a block is BLOCK_LEN x 1 x 1 cells
MAX_BLOCKS = GRID**3 // BLOCK_LEN  # 41
CELL_SIZE = 0.04  # metres; StackThree's cube (half_size=0.02) is the 1x1x1 unit


@dataclass(frozen=True)
class Block:
    id: int
    cells: np.ndarray  # [3, 3] int (i, j, k) cells, sorted along `axis`
    axis: int  # 0 = x, 1 = y, 2 = z

    @property
    def center(self) -> np.ndarray:
        """Block center in cell units (relative to the grid's [0, 0, 0] corner)."""
        return self.cells.mean(axis=0) + 0.5

    @property
    def layer(self) -> int:
        """Lowest layer the block touches (k of its bottom cell)."""
        return int(self.cells[:, 2].min())


def empty_design() -> np.ndarray:
    return np.zeros((GRID, GRID, GRID), dtype=np.int64)


def design_to_blocks(design: np.ndarray) -> list[Block]:
    """Parse a design into its blocks, sorted bottom-up (by lowest layer, then id). Raises if invalid."""
    design = np.asarray(design)
    if design.shape != (GRID, GRID, GRID):
        raise ValueError(f"design must have shape {(GRID,) * 3}, got {design.shape}")
    ids = np.unique(design[design != 0])
    if ids.size and (ids.min() < 1 or ids.max() > MAX_BLOCKS):
        raise ValueError(f"block ids must be in [1, {MAX_BLOCKS}], got {ids.tolist()}")

    blocks = []
    for bid in ids:
        cells = np.argwhere(design == bid)
        if len(cells) != BLOCK_LEN:
            raise ValueError(f"block {bid} occupies {len(cells)} cells, expected {BLOCK_LEN}")
        varying = np.flatnonzero(np.ptp(cells, axis=0))
        if len(varying) != 1:
            raise ValueError(f"block {bid} cells are not collinear: {cells.tolist()}")
        axis = int(varying[0])
        cells = cells[np.argsort(cells[:, axis])]
        if not np.array_equal(np.diff(cells[:, axis]), np.ones(BLOCK_LEN - 1)):
            raise ValueError(f"block {bid} cells are not contiguous: {cells.tolist()}")
        blocks.append(Block(int(bid), cells, axis))
    return sorted(blocks, key=lambda b: (b.layer, b.id))


def blocks_to_design(blocks: list[Block]) -> np.ndarray:
    """Inverse of design_to_blocks. Raises if two blocks claim the same cell."""
    design = empty_design()
    for b in blocks:
        idx = tuple(b.cells.T)
        if design[idx].any():
            raise ValueError(f"block {b.id} overlaps an existing block")
        design[idx] = b.id
    return design


def make_block(bid: int, start: tuple[int, int, int], axis: int) -> Block:
    """Block `bid` whose first cell is `start` and which extends BLOCK_LEN cells along +`axis`."""
    cells = np.tile(np.asarray(start), (BLOCK_LEN, 1))
    cells[:, axis] += np.arange(BLOCK_LEN)
    if cells.min() < 0 or cells.max() >= GRID:
        raise ValueError(f"block {bid} at {start} along axis {axis} leaves the grid")
    return Block(bid, cells, axis)


def occupancy(design: np.ndarray) -> np.ndarray:
    """Boolean (5, 5, 5) occupancy of a design."""
    return np.asarray(design) != 0


def overlap_score(target: np.ndarray, built_occupancy: np.ndarray) -> int:
    """v1 score: number of cells occupied in both the target design and the built structure."""
    return int(np.logical_and(occupancy(target), built_occupancy).sum())


def format_design(design: np.ndarray) -> str:
    """Layer-by-layer text view (k = 0 first). Rows are j (printed +y up), columns are i (+x right)."""
    out = []
    for k in range(GRID):
        layer = design[:, :, k]
        if not layer.any():
            continue
        out.append(f"layer k={k}")
        for j in reversed(range(GRID)):
            out.append("  " + " ".join(f"{v:2d}" if v else " ." for v in layer[:, j]))
    return "\n".join(out)
