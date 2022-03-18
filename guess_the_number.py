# Guess The Number Experiment - Empirical Results
from functools import partial
from random import randint

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
    return partial(_check_attempt, answer=randint(1, 100))
