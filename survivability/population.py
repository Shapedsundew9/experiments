from hurlabbab import hurlabbab
from numpy import pi, cos, sin, array, max, min, mean
from numpy.random import uniform, choice
import matplotlib.pyplot as plt
from functools import partial


def _ff(individual):
    individual.f = 1.0
    return individual

def _sf(population):
    for individual in population:
        individual.s = 1.0


class population(list):

    default_kwargs = {
        'n': 100,
        'cr': 1.0,
        'cx': 0.0,
        'cy': 0.0,
        'g': 0,
        'gl': 500,
        'ff': _ff,
        'sf': _sf,
        'bf': [0.0],
        'wf': [0.0],
        'mf': [0.0],
        'bg': [0],
        'hg': [0],
        'lg': [0],
        'mg': [0.0],
        'e': 1e-38,
        'done': False
    }

    def __init__(self, **kwargs):
        self.__dict__.update(**population.default_kwargs)
        self.__dict__.update(**kwargs)
        self.extend(tuple(hurlabbab(**self.crash_position()) for _ in range(self.n)))
        self.e = max((self.e, int(-self.n * 0.1)))

    def set_ff(self, ff):
            self.ff = partial(ff, p=self)
            for i in self: self.ff(i)

    def set_sf(self, sf):
            self.sf = partial(sf, p=self)
            self.sf()

    def crash_position(self):
        angle = uniform() * pi
        distance = uniform(-1.0, 1.0) * self.cr
        dx = cos(angle) * distance
        dy = sin(angle) * distance
        x = self.cx + dx
        y = self.cy + dy
        return {
            'x': x,
            'y': y
        }

    def stats(self):
        fitness = array([i.f for i in self])
        self.bf.append(max(fitness))
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