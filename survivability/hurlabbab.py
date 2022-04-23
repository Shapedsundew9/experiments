from numpy import exp, pi, sin, cos, array
from numpy.random import normal, uniform
from random import getrandbits
import matplotlib.pyplot as plt
from itertools import count


# Constants
N_BINS = 100000


class hurlabbab():

    default_kwargs = {
        'x': 0.0,
        'y': 0.0,
        'p': 1.0,
        'd': 2*pi,
        'ps': 0.2,
        'ds': 0.2,
        'ss': 0.1,
        'f': 1.0,
        's': 1.0,
        'a': None,
        'pg': 0
    }
    _exclude = (
        '__module__',
        '__main__',
        '__dict__',
        '__weakref__',
        '__doc__'
    )
    idx_gen = count()
    fossil_record = []


    def __init__(self, **kwargs):
        """Define"""
        self.__dict__.update(**hurlabbab.default_kwargs)
        self.d *= uniform(0.0, 1.0)
        self.p = self.fuzz_power()
        self.__dict__.update(**kwargs)
        self.id = next(hurlabbab.idx_gen)
        hurlabbab.fossil_record.append(self)

    def __repr__(self):
        return '\n'.join((k + ': ' + str(v) for k, v in self.__dict__.items() if k not in hurlabbab._exclude)) + '\n\n'

    def fuzz_power(self):
        """Scale the power by a random factor."""
        return self.p * exp(normal(scale=self.ps))

    def fuzz_direction(self):
        """Alter the direction bias by a random factor."""
        return (self.d + 2 * pi * normal(scale=self.ds)) % (2 * pi)

    def fuzz_scale(self):
        """Scale fuzzing."""
        return exp(normal(scale=self.ss))

    def hurl(self, pg):
        """Hurl a bab."""
        d = self.fuzz_direction()
        p = self.fuzz_power()
        return hurlabbab(
            x=self.x + cos(d) * p,
            y=self.y + sin(d) * p,
            p=self.fuzz_power(),
            d=self.fuzz_direction(),
            ps=self.ps * self.fuzz_scale(),
            ds=self.ds * self.fuzz_scale(),
            a=self.id,
            pg=pg

        )


    def plot(self, show=False):
        """Create plot of the landscape feature."""
        sz = 12 if show else 20
        fig = plt.figure(figsize=(sz, sz))
        d = array(tuple(self.fuzz_direction() for _ in range(N_BINS)))
        p = array(tuple(self.fuzz_power() for _ in range(N_BINS)))
        s = array(tuple(self.fuzz_scale() for _ in range(N_BINS)))
        x = cos(d) * p
        y = sin(d) * p

        # Direction Distribution Chart
        ax1 = fig.add_subplot(2, 2, 1, title=f'Direction Distribution ds = {self.ds}')
        ax1.hist(d, bins=100)
        ax1.set_xlabel('Radians')
        ax1.set_ylabel(f'Count (Total = {N_BINS})')

        # Power Distribution Chart
        ax2 = fig.add_subplot(2, 2, 2, title=f'Power Distribution ps = {self.ps}')
        ax2.hist(p, bins=100)
        ax2.set_xlabel('Distance')
        ax2.set_ylabel(f'Count (Total = {N_BINS})')
        ax2.yaxis.tick_right()
        ax2.yaxis.set_label_position("right")

        # Bab Position Distribution Chart
        ax3 = fig.add_subplot(2, 2, 3, title=f'Relative Bab Position Distribution from Parent X')
        hb = ax3.hexbin(x, y, cmap='inferno')
        ax3.plot([0.0], [0.0], marker='x', color='green')
        ax3.set_xlabel('Distance')
        ax3.set_ylabel('Distance')
        cb = fig.colorbar(hb, ax=ax3)
        cb.set_label(f'Count (Total = {N_BINS})')

        # Scale Distribution Chart
        ax4 = fig.add_subplot(2, 2, 4, title=f'Scale Distribution s = {self.ss}')
        ax4.hist(s, bins=100)
        ax4.set_xlabel('Scale')
        ax4.set_ylabel(f'Count (Total = {N_BINS})')
        ax4.yaxis.tick_right()
        ax4.yaxis.set_label_position("right")

        fig.suptitle(f'Hurlabbab {self.id} Statistics', fontsize=18)
        if show:
            plt.show()
        else:
            plt.savefig('stats_' + self.id + '.png')


if __name__ == "__main__":
    hurlabbab().plot(True)
