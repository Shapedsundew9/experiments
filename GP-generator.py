"""Generate a Gene Pool (GP) using just
addition and subtraction codons.

The GP is limited to GC's with the interface:

Input: int, int
Output: int

Therefore GC graphs are limited too:

"""
from graph_tool import Graph

_GRAPHS = (
    {
        'A': [['I', 0, 2], ['I', 1, 2]],
        'B': [['I', 0, 2], ['A', 0, 2]],
        'O': [['B', 0, 2]]
    },
    {
        'A': [['I', 0, 2], ['I', 1, 2]],
        'B': [['A', 0, 2], ['I', 0, 2]],
        'O': [['B', 0, 2]]
    },
    {
        'A': [['I', 0, 2], ['I', 1, 2]],
        'B': [['A', 0, 2], ['A', 0, 2]],
        'O': [['B', 0, 2]]
    },
    {
        'A': [['I', 0, 2], ['I', 1, 2]],
        'B': [['I', 1, 2], ['A', 0, 2]],
        'O': [['B', 0, 2]]
    },
    {
        'A': [['I', 0, 2], ['I', 1, 2]],
        'B': [['A', 0, 2], ['I', 1, 2]],
        'O': [['B', 0, 2]]
    }
)
