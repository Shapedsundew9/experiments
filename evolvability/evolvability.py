"""Experiment in using evolvability as selection criteria in an evolutionary system.

Equilateral Four Sided Triangles (EFSTs) are a species on a mission. Thier goal is to find 
'The End' which is a single position in a 2-dimensional grid like universe. They feel the
glow of The End and are drawn to it. EFSTs cannot move but thier offspring can be deposited
in any of the four adjacent positions.
Unfortuately the universe in which they live has uninhabitable regions called dead zones.
A dead zone is made up of one or more adjacent grid positions.

EFSTs have 3 phases in thier life cycle.
1. Active phase: EFSTs are alive and reproduce.
2. Dormant phase: EFSTs are alive but do not reproduce.
3. Dead phase: EFSTs are dead and will not reproduce again.

An EFST may transition from the active phase to the dormant phase and back again many times.
However, once an EFST enters the dead phase it will never reproduce again.

There is no limit to how many EFSTs may occupy a single position in the universe.
"""
from typing import Self
from random import choice, randint
from itertools import count

from numpy import (
    zeros,
    uint8,
    int32,
    double,
    array,
    nanargmax,
    argmin,
    clip,
    ones,
    nan_to_num,
    where,
    full,
)
from numpy.linalg import norm
from numpy.typing import NDArray

from matplotlib import use

use("TkAgg")
import matplotlib.pyplot as plt


# Constants
SOTU: int = 512  # Size of the universe in positions
NUM_EFSTS: int = 10  # Number of EFSTs in the inital population
MAX_NUM_EFSTS: int = 1000  # Maximum number of EFSTs in the universe
THE_END = 255  # Value of the end position in the universe
ONES: NDArray[double] = ones(MAX_NUM_EFSTS, dtype=double)
OUTPUT = True


# Universe set up
universe: NDArray[uint8] = zeros((SOTU, SOTU), dtype=uint8)
the_end: NDArray[int32] = array((SOTU // 2, SOTU // 2), dtype=int32)
universe[*the_end] = THE_END
position: NDArray[int32] = zeros((MAX_NUM_EFSTS, 2), dtype=int32)

# Dead points
dead_points_x: list[int] = [320] * 64
dead_points_x.extend(range(320, 384))
dead_points_y: list[int] = list(range(320, 384))
dead_points_y.extend([320] * 64)
universe[dead_points_x, dead_points_y] = 1


# EFST class
class EFST:
    """EFST class."""

    def __init__(self, xy_position: NDArray[int32], parent: Self | None = None) -> None:
        """Initialize an EFST."""
        new_idx: int = next(idx)
        self.idx: int = (
            new_idx if new_idx < MAX_NUM_EFSTS else int(argmin(selection_fitness))
        )
        self.position: NDArray[int32] = xy_position
        position[self.idx] = xy_position
        fitness[self.idx] = self.fitness_function()
        num_better_offspring[self.idx] = int32(1)
        num_offspring[self.idx] = int32(1)
        best_ancestor_gap[self.idx] = int32(-1)
        if parent is None:
            self.parent: Self = self
            generation[self.idx] = int32(0)
            best_ancestor_fitness[self.idx] = fitness[self.idx]
            best_ancestor_generation[self.idx] = int32(0)
        else:
            self.parent = parent
            generation[self.idx] = generation[parent.idx] + 1
            best_ancestor_fitness[self.idx] = best_ancestor_fitness[parent.idx]
            best_ancestor_generation[self.idx] = best_ancestor_generation[parent.idx]
            if fitness[self.idx] > best_ancestor_fitness[parent.idx]:
                best_ancestor_gap[self.idx] = (
                    generation[self.idx] - best_ancestor_generation[self.idx]
                )
                best_ancestor_fitness[self.idx] = fitness[self.idx]
                best_ancestor_generation[self.idx] = generation[self.idx]
        self.offspring: list[Self] = []

    def __repr__(self) -> str:
        """Return a string representation of an EFST."""
        return f"EFST: [x, y]: {self.position}, {len(self.offspring)} f: {fitness[self.idx]:0.5f} s: {selection_fitness[self.idx]:0.5f} g: {generation[self.idx]}, idx: {self.idx}, ggf: {generation_gap_factor[self.idx]:0.5f}, nof: {num_offspring_factor[self.idx]:0.5f}, nwof: {num_worse_offspring_factor[self.idx]:0.5f}"

    def breed(self) -> Self:
        """Create a new EFST from the current EFST."""
        step: int = choice((-1, 1))
        new_position: NDArray[int32] = self.position + choice(
            (array((step, 0), dtype=int32), array((0, step), dtype=int32))
        )
        _new_efst: Self = EFST(new_position, self)
        self.offspring.append(_new_efst)

        # Determine evolvability
        if fitness[_new_efst.idx] > fitness[self.idx] and num_offspring[self.idx] > 1:
            num_better_offspring[self.idx] += 1
        num_offspring[self.idx] += 1
        return _new_efst

    def fitness_function(self) -> double:
        """Calculate the fitness of an EFST."""
        if universe[tuple(self.position)] == 1:
            return double(0.0)
        return 1 / (norm(self.position - the_end) + 1)


if OUTPUT:
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(0, SOTU)
    ax.set_ylim(0, SOTU)
    pp = ax.scatter(
        position[:, 0],
        position[:, 1],
        marker=".",
        color="red",
        linewidth=1,
        animated=True,
    )
    te = ax.scatter(
        [the_end[0]],
        [the_end[1]],
        marker="X",
        color="green",
        linewidth=1,
        animated=True,
    )
    dp = ax.scatter(
        dead_points_x,
        dead_points_y,
        marker="s",
        color="black",
        linewidth=1,
        animated=True,
    )
    plt.show(block=False)
    bg = fig.canvas.copy_from_bbox(fig.bbox)
    ax.draw_artist(pp)
    ax.draw_artist(te)
    ax.draw_artist(dp)
    fig.canvas.blit(fig.bbox)


# Breed the EFSTs
def shape_factor(
    values: NDArray[double] | NDArray[int32], num: int32, av: double | None = None
) -> NDArray[double]:
    average: double = av if av is not None else values.sum() / num
    if average == 0.0:
        return ONES
    factor: NDArray[double] = average / values
    return where(factor < 0.5, factor + 0.5, 1.0)


evo_list = []
for epoch in range(3000 * (not OUTPUT) + 1):
    idx = count()
    selection_fitness: NDArray[double] = zeros(MAX_NUM_EFSTS, dtype=double)
    fitness: NDArray[double] = zeros(MAX_NUM_EFSTS, dtype=double)
    num_offspring: NDArray[int32] = zeros(MAX_NUM_EFSTS, dtype=int32)
    num_better_offspring: NDArray[int32] = zeros(MAX_NUM_EFSTS, dtype=int32)
    best_ancestor_fitness: NDArray[double] = ones(MAX_NUM_EFSTS, dtype=double)
    best_ancestor_generation: NDArray[int32] = zeros(MAX_NUM_EFSTS, dtype=int32)
    best_ancestor_gap: NDArray[int32] = full(MAX_NUM_EFSTS, -1, dtype=int32)
    generation: NDArray[int32] = zeros(MAX_NUM_EFSTS, dtype=int32)
    position = zeros((MAX_NUM_EFSTS, 2), dtype=int32)

    # Create the initial population of EFSTs
    evolutions = count()
    efsts: list[EFST] = [
        EFST(
            array(
                (
                    randint(SOTU - NUM_EFSTS - 1, SOTU - 1),
                    randint(SOTU - NUM_EFSTS - 1, SOTU - 1),
                ),
                dtype=int32,
            )
        )
        for _ in range(NUM_EFSTS)
    ]

    while True:
        if OUTPUT:
            fig.canvas.restore_region(bg)
            pp.set_offsets(position)
            ax.draw_artist(pp)
            ax.draw_artist(te)
            ax.draw_artist(dp)
            fig.canvas.blit(fig.bbox)
            fig.canvas.flush_events()

        valid = fitness > 0
        num_valid = valid.sum()
        generation_gap_factor: NDArray[double] = shape_factor(
            generation - best_ancestor_generation,
            num_valid,
            best_ancestor_gap[best_ancestor_gap > -1].mean(),
        )
        num_offspring_factor: NDArray[double] = shape_factor(num_offspring, num_valid)
        num_worse_offspring_factor: NDArray[double] = shape_factor(
            num_offspring - num_better_offspring, num_valid
        )

        selection_fitness = (
            fitness
            * generation_gap_factor
            * num_offspring_factor
            * num_worse_offspring_factor
        )

        """positive_distance =  clip(nan_to_num(((generation - best_ancestor_generation).sum() / (fitness > 0).sum()) / (generation - best_ancestor_generation), nan=1.0), 0.0, 1.0)
        average_offspring_ratio = nan_to_num(num_better_offspring.sum() / num_offspring.sum(), nan=1.0)
        offspring_ratios = nan_to_num(num_better_offspring / num_offspring, nan=1.0)
        #print(f"Average offspring ratio: {average_offspring_ratio}, Sum of offspring ratios: {offspring_ratios.sum()}")
        #print(f"Sum of Positive distance: {positive_distance.sum()}, Sum of Fitness: {fitness.sum()}")
        selection_fitness = fitness * positive_distance * clip(offspring_ratios / average_offspring_ratio, 0.0, 1.0)
        """
        max_efst_idx = nanargmax(selection_fitness)
        max_efst_idx2 = nanargmax(fitness)
        if fitness[max_efst_idx] == 1.0:
            break
        if OUTPUT:
            print("Selection: ", efsts[max_efst_idx])
            print("Best: ", efsts[max_efst_idx2])
        new_efst: EFST = efsts[max_efst_idx].breed()
        if len(efsts) < MAX_NUM_EFSTS:
            efsts.append(new_efst)
        else:
            efsts[new_efst.idx] = new_efst
        next(evolutions)

    num_evo = next(evolutions) - 1
    print(f"{epoch}: Number of evolutions: {num_evo}")
    evo_list.append(num_evo)

print(f"Average number of evolutions: {sum(evo_list) / len(evo_list)}")
print(f"Max number of evolutions: {max(evo_list)}")
print(f"Min number of evolutions: {min(evo_list)}")
