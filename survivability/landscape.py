"""Landscape.

The constants and functions used to define the landscape the Hurlabbabs crash land into.
The landscape includes the definition of the crash site and the distribution of Hurlabbabs.
"""
from numpy import pi, sqrt, linspace, meshgrid, cos, exp, sin, sqrt, abs
import matplotlib.pyplot as plt
from numba import jit

class landscape_feature():
    """Abstract class for a feature that can be added to the landscape."""

    default_kwargs = {
        'x': 0.0,
        'y': 0.0,
        'xw': 20.0,
        'yw': 20.0,
        'xns': 512,
        'yns': 512,
        'description': 'No description'
    }

    def __init__(self, **kwargs):
        self.__dict__.update(**landscape_feature.default_kwargs)
        self.__dict__.update(**kwargs)
        self.x_min = self.x - self.xw / 2.0
        self.y_min = self.y - self.yw / 2.0
        self.x_max = self.x + self.xw / 2.0
        self.y_max = self.y + self.yw / 2.0
        self.xs = self.xw / self.xns
        self.ys = self.yw / self.yns
        x = linspace(self.x_min, self.x_max, self.xns)
        y = linspace(self.y_min, self.y_max, self.yns)
        self.xx, self.yy = meshgrid(x, y)
        self.zz = self.function(self.xx, self.yy)

    def function(self, xx, yy):
        raise NotImplementedError

    def plot(self, show=False):
        """Create plot of the landscape feature."""
        sz = 12 if show else 20
        fig = plt.figure(figsize=(sz, sz))
        ax1 = fig.add_subplot(2, 2, 1)
        cf1 = ax1.contourf(self.xx, self.yy, self.zz, cmap='terrain')
        fig.colorbar(cf1, ax=ax1)
        ax2 = fig.add_subplot(2, 2, 2, projection='3d')
        ax2.plot_surface(self.xx, self.yy, self.zz, cmap='terrain')
        ax3 = fig.add_subplot(2, 2, 3)
        cf2 = ax3.contourf(self.xx, self.yy, self.zz, 100, cmap='terrain')
        fig.colorbar(cf2, ax=ax3)
        ax4 = fig.add_subplot(2, 2, 4, projection='3d')
        ax4.contour3D(self.xx, self.yy, self.zz, 50, cmap='binary')
        fig.suptitle(f'Landscape Feature: {self.description}', fontsize=18)
        if show:
            plt.show()
        else:
            plt.savefig(self.description.replace(' ', '_') + '.png')
            plt.close()

class mountain_cluster(landscape_feature):

    def __init__(self, scale=4.5):
        super().__init__(scale=scale, description='Mountain Cluster')

    # Numba this
    def function(self, xx, yy):
        t = sqrt(xx**2 + yy**2)
        s = 0.5 * t / self.scale + 0.5
        return -cos(xx / s) * cos(yy / s) * exp(-((xx / 5)**2) - ((yy / 5)**2))


class mountain_domain(landscape_feature):

    def __init__(self, scale=4.5):
        super().__init__(xw=1024, yw=1024, description='Mountain Domain')

    # Numba this
    def function(self, xx, yy):
        """'eggholder' function.

        See https://docs.scipy.org/doc/scipy/tutorial/optimize.html#global-optimization
        """
        return (-(yy + 47) * sin(sqrt(abs(xx/2 + (yy  + 47))))
                -xx * sin(sqrt(abs(xx - (yy  + 47)))))

if __name__ == "__main__":
    mountain_cluster().plot()
    mountain_domain().plot()
