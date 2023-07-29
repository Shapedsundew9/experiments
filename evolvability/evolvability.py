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

from numpy import zeros, uint8, int32, double, array, argmax, argmin
from numpy.linalg import norm
from numpy.typing import NDArray

from matplotlib import use
use('TkAgg')
import matplotlib.pyplot as plt


# Constants
SOTU: int = 512  # Size of the universe in positions
NUM_EFSTS: int = 10  # Number of EFSTs in the inital population
MAX_NUM_EFSTS: int = 1000  # Maximum number of EFSTs in the universe
THE_END = 255  # Value of the end position in the universe


# Universe set up
universe: NDArray[uint8] = zeros((SOTU, SOTU), dtype=uint8)
the_end: NDArray[int32] = array((SOTU // 2, SOTU // 2), dtype=int32)
universe[*the_end] = THE_END
idx = count()
selection_fitness: NDArray[double] = zeros(MAX_NUM_EFSTS, dtype=double)
position: NDArray[int32] = zeros((MAX_NUM_EFSTS, 2), dtype=int32)

# EFST class
class EFST:
    """EFST class."""

    def __init__(self,
                 xy_position: NDArray[int32],
                 parent: Self | None = None
            ) -> None:
        """Initialize an EFST."""
        new_idx: int = next(idx)
        self.idx: int = new_idx if new_idx < MAX_NUM_EFSTS else int(argmin(selection_fitness))
        self.position: NDArray[int32] = xy_position
        position[self.idx] = xy_position
        self.fitness: double = self.fitness_function()
        selection_fitness[self.idx] = self.fitness
        self.evolvability = double(1.0)
        if parent is None:
            self.parent: Self = self
            self.generation = 0
        else:
            self.parent: Self = parent
            self.generation = parent.generation + 1
        self.offspring: list[Self] = []

    def __repr__(self) -> str:
        """Return a string representation of an EFST."""
        return f"EFST: [x, y]: {self.position} f: {self.fitness:0.5f} s: {selection_fitness[self.idx]:0.5f} g: {self.generation} e: {self.evolvability:0.5f}, idx: {self.idx}"

    def breed(self) -> Self:
        """Create a new EFST from the current EFST."""
        vertical: int = choice((-1, 1))
        horizontal: int = choice((-1, 1))
        new_position: NDArray[int32] = self.position + array((horizontal, vertical), dtype=int32)
        _new_efst: Self = EFST(new_position, self)
        self.offspring.append(_new_efst)
        self.update_selection_fitness()
        if len(self.offspring) > NUM_EFSTS:
            evolvability = double((2*NUM_EFSTS - len(self.offspring)) / NUM_EFSTS)
            self.update_evolvability(evolvability)
        return _new_efst

    def update_parent_evolvability(self, evolvability: double = double(1.0)) -> None:
        if self.parent is not self and len(self.parent.offspring) > NUM_EFSTS:
            self.parent.update_parent_evolvability(evolvability / self.evolvability)
        else:
            for offspring in self.offspring:
                offspring.update_evolvability(self.evolvability)

    def update_evolvability(self, evolvability: double) -> None:
        self.evolvability: double = self.evolvability * evolvability
        for offspring in self.offspring:
            offspring.update_evolvability(self.evolvability)
        self.update_selection_fitness()
        self.update_parent_evolvability()

    def update_selection_fitness(self) -> None:
        selection_fitness[self.idx] = self.fitness * self.evolvability

    def fitness_function(self) -> double:
        """Calculate the fitness of an EFST."""
        return 1 / (norm(self.position - the_end) + 1)



fig, ax = plt.subplots(figsize=(10, 10))
ax.set_xlim(0, SOTU)
ax.set_ylim(0, SOTU)
pp = ax.scatter(position[:,0], position[:, 1], marker='.', color='red', linewidth=0, animated=True)

plt.show(block=False)
bg = fig.canvas.copy_from_bbox(fig.bbox)
# Create the initial population of EFSTs
efsts: list[EFST] = [
                        EFST(
                            array(
                                (
                                    randint(SOTU - NUM_EFSTS - 1, SOTU - 1),
                                    randint(SOTU - NUM_EFSTS - 1, SOTU - 1)
                                ),
                                dtype=int32
                            )
                        )
                        for _ in range(NUM_EFSTS)
                    ]

# Breed the EFSTs
ax.draw_artist(pp)
fig.canvas.blit(fig.bbox)
flag = False
evolutions = count()

while True:
    fig.canvas.restore_region(bg)
    pp.set_offsets(position)
    ax.draw_artist(pp)
    fig.canvas.blit(fig.bbox)
    fig.canvas.flush_events()
    max_efst_idx = argmax(selection_fitness)
    if selection_fitness[max_efst_idx] == 1.0:
        break
    print(efsts[max_efst_idx])
    new_efst: EFST = efsts[max_efst_idx].breed()
    if selection_fitness[new_efst.idx] == 0.5:
        flag = True
    print('New EFST: ', new_efst)
    if len(efsts) < MAX_NUM_EFSTS:
        efsts.append(new_efst)
    else:
        efsts[new_efst.idx] = new_efst
    next(evolutions)

print(f"Number of evolutions: {next(evolutions) - 1}")
