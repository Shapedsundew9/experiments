# Guess The Number Experiment - Empirical Results
from functools import partial
from random import randint
from tqdm import trange, tqdm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
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
        nonlocal last_choice
        if higher and not lower:
            low = last_choice
        elif not higher and lower:
            high = last_choice
        last_choice = randint(LOW, HIGH)
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
        nonlocal last_choice
        if higher and not lower:
            low = last_choice
        elif not higher and lower:
            high = last_choice
        last_choice = int((high - low + 1) / 2) + (low - 1)
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
        results[name] = array([play_game(func) for _ in trange(0, MAX_ATTEMPTS, desc=f'Strategy {name}')])


    x = array(range(0, MAX_ATTEMPTS, BAR_STEP)) + HALF_BAR_STEP
    fig = plt.figure(figsize=(16, 6))
    ax = fig.add_axes([.05,.1,.9,.8])
    plt.xlim([0, MAX_ATTEMPTS])
    plt.ylim([0, NUM_GAMES])
    series = {name: ax.bar([], [], width=BAR_STEP, alpha=0.5) for name in results}
    pbar = tqdm(total=NUM_FRAMES, desc='Building GIF')

    def animation(frame):
        batch = BATCH_SIZE * frame
        pbar.update(1)
        ax.clear()
        plt.xlim([0, MAX_ATTEMPTS])
        plt.ylim([0, NUM_GAMES])
        for name, bar in series.items():
            y = bincount(digitize(results[name][0:batch], range(10, NUM_BARS * BAR_STEP, BAR_STEP)), minlength=NUM_BARS)
            ax.bar(x, y, width=BAR_STEP, alpha=0.5)

    gif = FuncAnimation(fig, animation, frames=100)
    gif.save('video.gif', dpi=100, writer=PillowWriter(fps=25))


if __name__ == "__main__":
    experiment_one()