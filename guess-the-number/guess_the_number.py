# Guess The Number Experiment - Empirical Results
from functools import partial
from random import randint
from tqdm import trange, tqdm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, HTMLWriter, ImageMagickWriter
from numpy import array, digitize, bincount, arange
from numpy.random import shuffle


# Constants
LOW = 1
HIGH = 100
MAX_ATTEMPTS = 1000
NUM_GAMES = 1000
NUM_BARS = 100
BAR_STEP = int(MAX_ATTEMPTS / NUM_BARS)
HALF_BAR_STEP = int(0.5 * BAR_STEP)
BATCH_SIZE = 10
NUM_FRAMES = int(NUM_GAMES / BATCH_SIZE) + 1

# After much playing:
#   PillowWriter: Can only produce an infinite loop GIF. Is much quicker than ImageMagickWriter but produces a file 3x larger
#   PillowWriterNG: Is a hack but only reduces the looping to 2. Cannot find a way to play a GIF once this way.
#   HTMLWriter: Only produces HTML
#   ImageMagickWriter: Has a very slow save as it does 'convert' but smallest GIF by 3x. Need to loop at looping and progress bar options.
WRITER = ImageMagickWriter

# Over-ride the hard-coded loop=0 (infinite loop)
# See https://github.com/matplotlib/matplotlib/blob/9765379ce6e7343070e815afc0988874041b98e2/lib/matplotlib/animation.py#L513
class PillowWriterNG(PillowWriter):
    def finish(self):
            self._frames[0].save(
                self.outfile, save_all=True, append_images=self._frames[1:],
                duration=int(1000 / self.fps), loop=1)


def _check_attempt(attempt, answer):
    """Check the attempt.

    Args
    ----
    attempt (int): The attempt
    answer (int): The answer

    Returns
    -------
    tuple(bool, bool): (higher, lower)
        higher is True if answer is higher than attempt
        lower is True if answer is lower than attempt
        When both are False attempt == answer
    """
    return (answer > attempt, answer < attempt)


def new_game():
    """Create a new game.

    Returns
    -------
    callable(): A _check_attempt() with answer defined.
    """
    return partial(_check_attempt, answer=randint(LOW, HIGH))


def constant_strategy():
    """Random constant strategy.

    Randomly choose a constant in the range LOW to HIGH.

    Returns
    -------
    callable(): Strategy function returning a constant int in the
        range LOW to HIGH inclusive.
    """
    constant = randint(LOW, HIGH)
    def func(_, __):
        return constant
    return func


def random_strategy():
    """Random strategy.

    Randomly choose an integer in the range LOW to HIGH inclusive.

    Returns
    -------
    callable(): Strategy function returning a random integer in the range
        LOW to HIGH inclusive.
    """
    def func(_, __):
        return randint(LOW, HIGH)
    return func


def random_memory_strategy():
    """Random memory strategy.

    Randomly choose an integer in the range LOW to HIGH inclusive
    but do not repeat a choice.

    Returns
    -------
    callable(): Strategy function returning a random integer in the range
        LOW to HIGH inclusive with no replacement.
    """
    valid = arange(LOW, HIGH + 1)
    shuffle(valid)
    valid = list(valid)
    def func(_, __):
        return valid.pop()
    return func


def random_constrained_strategy():
    """Random constrained strategy.

    Randomly choose an integer in the range constrained by the feedback from
    previous choices.

    Returns
    -------
    callable(): Strategy function returning a random integer in the range
        constrained by the feedback from previous choices.
    """
    low = LOW
    high = HIGH
    last_choice = None
    def func(higher, lower):
        nonlocal last_choice, high, low
        if higher and not lower:
            low = last_choice
        elif not higher and lower:
            high = last_choice
        last_choice = randint(low, high)
        return last_choice
    return func


def optimal_strategy():
    """Optimal strategy.

    Binary search the range.

    Returns
    -------
    callable(): Binary search of the range.
    """
    low = LOW
    high = HIGH
    last_choice = None
    def func(higher, lower):
        nonlocal last_choice, high, low
        if higher and not lower:
            low = last_choice
        elif not higher and lower:
            high = last_choice
        last_choice = int((high - low) / 2) + low
        return last_choice
    return func


def play_game(strategy):
    """Play a game using the suplied strategy.

    Args
    ----
    strategy(callable): Callable takes no arguments and returns an integer.

    Returns
    -------
    int: Number of attempts to correctly choose the answer or 1000 if
        no correct choice was made.
    """
    game = new_game()
    guess = strategy()
    higher = lower = False
    for attempt in range(1, 1000):
        higher, lower = game(guess(higher, lower))
        if not higher and not lower:
            break
    return attempt


def experiment_one():
    """Execute experiment 1.

    Games are played by executing strategies. A strategy (functions with names
    *_strategy taking no parameters) defines a strategy function which is used to
    play a game. The strategy function takes two boolean parameters: higher &
    lower and returns a guess (an integer). Higher & lower is the feedback from the
    previous round of the current game. The game is a checking function defined by
    the new_game() function which returns higher & lower for each guess.
    """
    results = {}
    for strategy in filter(lambda x: '_strategy' in x, globals()):
        name = ' '.join(strategy.split('_')[:-1])
        func = globals()[strategy]
        results[name] = array([play_game(func) for _ in trange(0, MAX_ATTEMPTS, desc=f'Strategy {name}', ascii=True)])


    x = array(range(0, MAX_ATTEMPTS, BAR_STEP)) + HALF_BAR_STEP
    fig = plt.figure(figsize=(16, 6))
    ax = fig.add_axes([.05,.1,.9,.8])
    series = {name: ax.bar([], [], width=BAR_STEP, alpha=0.5) for name in results}
    pbar = tqdm(total=NUM_FRAMES, desc='Building full GIF', ascii=True)
    max_attempts = MAX_ATTEMPTS
    bar_step = BAR_STEP
    bins = tuple(range(10, NUM_BARS * BAR_STEP, BAR_STEP))
    plot_values = [array([x for x in values if x < max_attempts]) for values in results.values()]
    y_max = max([max(bincount(digitize(values, bins), minlength=NUM_BARS)) for values in plot_values])

    def animation(frame):
        batch = BATCH_SIZE * frame
        pbar.update(1)
        ax.clear()
        plt.xlim([0, max_attempts])
        plt.ylim([0, y_max])
        plt.xlabel("Number of Attempts")
        plt.ylabel("Number of Games")
        #plt.legend('upper center')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        for name in series:
            values = array([x for x in results[name][0:batch] if x < max_attempts])
            y = bincount(digitize(values, bins), minlength=NUM_BARS)
            ax.bar(x, y, width=bar_step, alpha=0.5, label=name)

    gif = FuncAnimation(fig, animation, frames=100)
    gif.save('full.gif', dpi=100, writer=WRITER(fps=25))

    pbar = tqdm(total=NUM_FRAMES, desc='Building zoom GIF', ascii=True)
    max_attempts = HIGH - LOW + 1
    x = array(range(0, max_attempts))
    bins = tuple(range(1, NUM_BARS))
    bar_step = 1
    plot_values = [array([x for x in values if x < max_attempts]) for values in results.values()]
    y_max = max([max(bincount(digitize(values, bins), minlength=NUM_BARS)) for values in plot_values])
    gif = FuncAnimation(fig, animation, frames=100)
    gif.save('zoom.gif', dpi=100, writer=WRITER(fps=25))


if __name__ == "__main__":
    experiment_one()