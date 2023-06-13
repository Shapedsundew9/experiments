"""Generate a Gene Pool (GP) using just
addition and subtraction codons.

The GP is limited to GC's with the interface:

Input: int, int
Output: int

Therefore GC graphs are limited too:

"""
from copy import deepcopy
from random import choice
from itertools import count
from json import dump


_NUM_GENERATIONS = 100
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
_DEFAULT_GC = {
    'ancestor_a_ref': 0,
    'ancestor_b_ref': 0,
    'gca_ref': 0,
    'gcb_ref': 0,
    'ref': 0
}


reference = count()
gcs = [deepcopy(_DEFAULT_GC), deepcopy(_DEFAULT_GC)]
gcs[0]['ref'] = next(reference)
gcs[1]['ref'] = next(reference)
for generation in range(_NUM_GENERATIONS):
    gcs.append(
        {
            'ancestor_a_ref': 0,
            'ancestor_b_ref': 0,
            'gca_ref': choice(gcs),
            'gcb_ref': choice(gcs),
            'ref': next(reference)
        }
    )

with open('~/gcs.json', 'w', encoding='utf-8') as fptr:
    dump(gcs, fptr)
