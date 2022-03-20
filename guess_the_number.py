# Guess The Number Experiment - Empirical Results
from functools import partial
from random import randint
from tqdm import trange

# Constants
LOW = 1
HIGH = 100
MAX_ATTEMPTS = 1000
NUM_GAMES = 1000


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
    """The random constant strategy.
    
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
    """The random strategy.
    
    Randomly choose an integer in the range LOW to HIGH inclusive.

    Returns
    -------
    callable(): Strategy function returning a random integer in the range
        LOW to HIGH inclusive.
    """
    def func(_, __):
        return randint(LOW, HIGH)
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
    for strategy in filter(lambda x: '_strategy' in x, dir()):
        name = ' '.join(strategy.split('_')[:-1])
        func = globals[strategy]
        results[name] = [play_game(strategy) for _ in trange(LOW, HIGH + 1, desc=f'Strategy {name}')]

    

if __name__ == "__main__":
    experiment_one()