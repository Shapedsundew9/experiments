from numpy import zeros, ones, float32, exp, array, concatenate, newaxis, empty
from numpy.random import default_rng
from tqdm import tqdm
from matplotlib import use
use('GTK3Cairo')
import matplotlib.figure as figure 


# Master Scaling Parameter
_MSP = 0.1


class hurlabbabs():

    def __init__(self, fitness_function, num_param, size=100):
        """Initialise the Hurlabbab population.
        
        Args
        ----
        fitness_function (callable): Takes a single numpy array of shape shape.
            Returns a single float32 value between 0.0 and 1.0 defining the fitness.
        num_param (int): The number of parameters to optimise.
        size (int): Must be >0. Size of the Hurlabbab population.
        """
        assert size > 0, "Population size must be > 0"

        # There is one set of parameters for every individual. Hurlabbabs
        # have one offspring each so need 2x the population size.
        # Children go into parents row + num row
        # Suriviving population is placed in first num rows.
        self.size = int(size)
        self.vector_shape = (self.size, num_param)
        self.fitness_function = fitness_function
        self._rng = default_rng()

        # TODO: More practical initial position in space
        self.position = zeros(self.vector_shape, dtype=float32)

        # Initialise hurling parameters
        # Babs are hurled along a vector that adjusts in magnitude and direction 
        self.magnitude_scale = self.fuzz(self.size)
        self.direction_scale = self.fuzz(self.size)
        self.vector = self._rng.uniform(-1.0, 1.0, size=self.vector_shape)

        # Calculating fitness from vector as position is 0.0
        self.fitness = array(tuple(map(self.fitness_function, self.vector)), dtype=float32)

    def fuzz(self, size, scale=_MSP):
        """Create a scaling vector for the population."""
        return exp(self._rng.normal(scale=scale, size=size).astype(float32))

    def survivors(self, bab_position, bab_fitness):
        """Determine the survivors from the current population and the new babs.
        
        Args
        ----
        bab_position (ndarray): Same as self.position but for the new babs.
        bab_fitness (array(float32)): Fitness of each bab. Same length as first
            bab_position dimensions.
        
        Returns
        -------
        (array(bool), array(bool)): Mask of the survivors (True) in the current population
            and the mask of the survivors in the new bab population. Both arrays have self.num
            elements. The total number of survivors must be self.num (i.e. half).
        """
        fitness = concatenate((self.fitness, bab_fitness))
        delta_fitness = fitness - fitness.min()
        weights = delta_fitness / delta_fitness.sum()
        survivors = self._rng.choice(self.size * 2, self.size, False, weights)
        mask = zeros((self.size * 2,), dtype=bool)
        mask[survivors] = True
        return mask[:self.size], mask[self.size:]

    def evolve(self):
        """Hurl a Bab.

        Reproduction with Hurlabbabs is asexual. Each individual in the population reproduces once each
        generation, hurling a bab(y) to a new position in the target space.

        A hurl from the same individual is not the same everytime

        After reproduction the entire population (2*self.num) are assessed and self.num are selected for survival
        to produce another generation.

        Babs inherit the parameters they were hurled with i.e. the fuzzed parent values.

        TODO: This saves fuzzing the parent values again but how does it influence the evolutionary path?
        """
        bab_direction_scale, bab_magnitude_scale, bab_vector, bab_position = self.reproduce()
        bab_fitness = array(tuple(map(self.fitness_function, bab_position)), dtype=float32)

        # Selection
        p_mask, b_mask = self.survivors(bab_position, bab_fitness)
        p_mask = ~p_mask

        # Surviving babs replace deceased individuals in the population
        self.position[p_mask] = bab_position[b_mask]
        self.vector[p_mask] = bab_vector[b_mask]
        self.fitness[p_mask] = bab_fitness[b_mask]
        self.magnitude_scale[p_mask] = bab_magnitude_scale[b_mask]
        self.direction_scale[p_mask] = bab_direction_scale[b_mask]

    def reproduce(self):
        """Reproduce.

        The base vector for the hurl is stored in self.vector and is immutable through the individuals lifetime.
        self.magnitude_scale and self.direction_scale are also immutable through the individuals lifetime.
        The magnitude of the vector is scaled by self.magnitude_scale which is fuzzed a little each time
        The direction of the vector is modified by adding a random vector scaled by self.direction_scale
        The random vector changes every hurl as does the direction_scale with a little fuzz.
        """
        bab_direction_scale = self.direction_scale * self.fuzz(self.size)
        bab_magnitude_scale = self.magnitude_scale * self.fuzz(self.size)
        bab_vector = (self.vector + self._rng.uniform(-1.0, 1.0, size=self.vector_shape)
            * bab_direction_scale[:, newaxis]) * bab_magnitude_scale[:, newaxis]
        bab_position = self.position + bab_vector
        return bab_direction_scale, bab_magnitude_scale, bab_vector, bab_position


def _bab_burst():
    """Visual validation of Hurlabbab behviour.
    
    This method generates sequences of 2D images of Hurlabbab positions
    in a 2D space using specific parameters for visual validation.

    Each sequence is a single rendered PNG figure with the name bab_burst_[x][y].pgn
    where x & y are the values of the initiating hurlabbab vector.

    Each sequence is an 8x8 grid of plots showing generation 0, 1, & 2 of hurlabbab
    evolution. Generation 0 are all clones of each other at position 0, 0.

    Going horizontally (x-axis) in the sequence of plots reduces the magnitude scale
    1/1, 1/2, 1/3, 1/4, 1/5, 1/6, 1/7, 1/8
    Going vertically (y-axis)  in the sequence of plots reduces the direction scale
    in the same way.

    'firebrick' is the initial hurlabbab (at position 0,0) i.e. generation 0
    'royalblue' are 100 babs from firebrick i.e. generation 1
    'lightcoral' is a random bab from royalblue i.e. an individual from generation 1
    'lightskyblue' are 100 babs from light coral i.e. generation 2

    In general the results show generations positions being closer to gen 0 with
    reducing magnitude scale and more tightly clustered with reducing direction scale.
    Thus, high magnitude scale and high direction scale (top, left) results in a
    broad cluster of gen 1 babs centred on the gen 0 vector end position. e.g. if 
    the gen0 vector is 0, 0 then gen 1 is distributed broadly around the gen 0 position.
    Low magnitude scale draws the gen 1 cluster closer to gen 0 and low direction scale 
    tightens the cluster. In the bottom right corner of the figure gen 1 is pretty
    much on top of gen 0.

    Gen 2 inherits the properties of gen 1 so if the chosen gen 1 hurlabbab is on the
    far edge of the gen 1 cluster from gen 0, gen 2 will most likely (but not certainly)
    be of a similar offset from its gen 1 parent.
    """
    gen_0 = hurlabbabs(lambda x: 1, 2, 100)
    gen_1 = hurlabbabs(lambda x: 1, 2, 100)
    gen_0.position = zeros((gen_0.size, 2), dtype=float32)
    gen_0.vector = empty((gen_0.size, 2), dtype=float32)
    for vector in tqdm(((0, 0), (1, 1), (-1, 1), (-1, -1), (1, -1)), desc='Plotting'):
        fig = figure.Figure(figsize=(12.0, 12.0))
        axs = fig.subplots(8, 8, sharex=True, sharey=True)
        gen_0.vector[:, 0] = vector[0]
        gen_0.vector[:, 1] = vector[1]
        for ms in range(8):
            for ds in range(8):
                gen_0.magnitude_scale = ones((gen_0.size,), dtype=float32) / (ms + 1)
                gen_0.direction_scale = ones((gen_0.size,), dtype=float32) / (ds + 1)
                gen_1.direction_scale, gen_1.magnitude_scale, gen_1.vector, gen_1.position = gen_0.reproduce()
                axs[ds, ms].scatter(gen_1.position[:,0], gen_1.position[:,1], marker='.', color='royalblue')
                gen_1.direction_scale[:] = gen_1.direction_scale[0]
                gen_1.magnitude_scale[:] = gen_1.magnitude_scale[0]
                gen_1.vector[:] = gen_1.vector[0,:]
                gen_1.position[:] = gen_1.position[0,:]
                _, __, ___, gen_2_position = gen_1.reproduce()
                axs[ds, ms].scatter(gen_2_position[:,0], gen_2_position[:,1], marker='.', color='lightskyblue')
                axs[ds, ms].scatter(gen_0.position[:,0], gen_0.position[:,1], marker='.', color='firebrick')
                axs[ds, ms].scatter(gen_1.position[:,0], gen_1.position[:,1], marker='.', color='lightcoral')
                axs[ds, ms].label_outer()
        fig.tight_layout()
        fig.savefig('bab_burst_' + str(vector[0]) + str(vector[1]) + '.png')

if __name__ == '__main__':
    _bab_burst()
