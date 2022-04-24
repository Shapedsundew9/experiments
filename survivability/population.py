from hurlabbab import hurlabbab
from numpy import pi, cos, sin, array, max, min, mean, empty, float32, zeros, int32
from numpy.random import uniform, choice
import matplotlib.pyplot as plt
from functools import partial
from copy import deepcopy


_STAT_BODY = {
    'values': [],
    'function': None
}

_STATS_BODY = {
    'max': deepcopy(_STAT_BODY),
    'min': deepcopy(_STAT_BODY),
    'mean': deepcopy(_STAT_BODY),
    'data': None
}

def _fitness_function(population_data):
    """Template fitness function.

    Args
    ----
    population_data (ndarray): Shape (nargs, *)

    Returns
    -------
    fitness (ndarray): Shape (*,)
    """
    raise NotImplementedError

def initialize(size, narg_lim):
    """Default function for initializing Hurlabbab arguments.

    Initialises the population data with a uniform distribution
    in the range narg_lim[x][0] <= arg[x][:] <= narg_lim[x][1].

    Args
    ----
    nargs (int): Number of hurlabbab arguments.
    size (int): Population size.
    narg_lim (iterable(tuple(float32, float32))) or None: List of (low, high) limits.

    Returns
    -------
    (ndarray): Shape (len(narg_lim), size) of float32 type.
    """
    narg = empty((len(narg_lim), size), dtype=float32)
    for arg, arg_lim in zip(narg, narg_lim):
        arg = uniform(arg_lim[0], arg_lim[1], size)
    return arg


class population():

    default_kwargs = {
        'crash_radius': 1.0,
        'crash_x': 0.0,
        'crash_y': 0.0,
        'generation': 0,
        'generation_limit': 500,
        'fitness_function': _fitness_function,
        'stats': {
            'fitness': deepcopy(_STATS_BODY),
            'generation': deepcopy(_STATS_BODY)
        },
        'tolerence': 1e-38,
        'done': False
    }

    def __init__(self, fitness_function, args, **kwargs):
        self.__dict__.update(**population.default_kwargs)
        self.__dict__.update(**kwargs)
        self.fitness_function = fitness_function
        self.args = args
        self.nargs, self.size = args.shape
        self.fitness = self.fitness_function(args)
        self.generation = zeros((self.size,), dtype=int32)
        self.clan = zeros((self.size,), dtype=int32)
        self.survivability = empty((self.size,), dtype=float32)
        self.stats['fitness']['data'] = self.fitness
        self.stats['generation']['data'] = self.fitness


    def survival_function(self):
        return self.fitness

    def stats(self):
        for stat in stats:
            stat

        self.statsbf.append(max(fitness))
        self.wf.append(min(fitness))
        self.mf.append(mean(fitness))
        generation = array([i.pg for i in self])
        self.hg.append(max(generation))
        self.lg.append(min(generation))
        self.mg.append(mean(generation))
        self.bg.append(self.g)

    def breed(self, show=False, save=False):
        self.g += 1
        self.extend([self.ff(i.hurl(self.g)) for i in self])
        survivors = sorted(self.sf(), key=lambda x:x.f, reverse=True)
        self.clear()
        self.extend(survivors)
        self.stats()
        self.done = self.done or self.g >= self.gl
        if show or save: self.plot(show)
        return self.done

    def plot(self, show=False):
        sz = 12 if show else 20
        fig = plt.figure(figsize=(sz, sz))

        # Spatial Distribution Chart
        ax1 = fig.add_subplot(2, 2, 1, title=f'Spatial Distribution (Total {self.n})')
        ax1.scatter([i.x for i in self], [i.y for i in self], marker='o', color='red')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')

        # Power Distribution Chart
        ax2 = fig.add_subplot(2, 2, 2, title=f'Power Distribution')
        ax2.hist([i.p for i in self], bins=min((100, int(self.n / 10))))
        ax2.set_xlabel('Individual Power')
        ax2.set_ylabel(f'Count (Total = {self.n})')
        ax2.yaxis.tick_right()
        ax2.yaxis.set_label_position("right")

        # Direction Distribution Chart
        ax3 = fig.add_subplot(2, 2, 3, title=f'Direction Distribution')
        ax3.hist([i.d for i in self], bins=min((100, int(self.n / 10))))
        ax3.set_xlabel('Individual Direction')
        ax3.set_ylabel(f'Count (Total = {self.n})')

        # Scale Distribution Chart
        ax4 = fig.add_subplot(2, 2, 4, title=f'Scale Distribution')
        ax4.hist([i.ps for i in self], bins=min((100, int(self.n / 10))), color='blue', alpha=0.5)
        ax4.hist([i.ds for i in self], bins=min((100, int(self.n / 10))), color='red', alpha=0.5)
        ax4.set_xlabel('Direction/Power Scale')
        ax4.set_ylabel(f'Count (Total = {self.n})')
        ax4.yaxis.tick_right()
        ax4.yaxis.set_label_position("right")

        fig.suptitle(f'Hurlabbab Population Generation {self.g} Statistics', fontsize=18)
        if show:
            plt.show()
        else:
            plt.savefig(f'population_gen_{self.g}.png')
            plt.close()


if __name__ == "__main__":
    p = population()
    p.breed(True)
    p.breed(True)
    p.breed(True)