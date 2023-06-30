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
from json import dump, load
from os.path import join

from egp_types.mGC import mGC
from egp_types.gc_type_tools import define_signature
from egp_stores.genomic_library import DATA_FILE_FOLDER
from egp_types.xgc_validator import LGC_json_load_entry_validator

_CREATOR = 'a508fd79-afea-4465-85e2-40d64cccd064'
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

# Load the codons
_add_subtract_codons = {
    "99d1719cd1f4cd610336e04540bb2670f9a0f495500c0e041d301645319a364c",  # int add
    "7684849a1e6e663de20cc19ebcfa233af9592d81f221ba902a4f618b32dae6ef"   # int subtract
}
abspath: str = join(DATA_FILE_FOLDER, 'codons.json')
with open(abspath, "r", encoding='utf-8') as file_ptr:
    gcs = {codon['signature']: codon for codon in load(file_ptr) if codon['signature'] in _add_subtract_codons}

for generation in range(_NUM_GENERATIONS):
    # Create a new generation by stacking  GCs on top of one another
    ngc = {
        'gca': choice(gcs)['signature'],
        'gcb': choice(gcs)['signature']
    }
    ngc.update({
        'ancestor_a': ngc['gca'],
        'ancestor_b': ngc['gcb'],
        'generation': max((gcs[ngc['gca']]['generation'], gcs[ngc['gcb']]['generation'])) + 1,  # Increment the generation
        'creator': _CREATOR,
        'graph': choice(_GRAPHS),
    })
    ngc = LGC_json_load_entry_validator.normalized(ngc)
    gcs[ngc['signature']] = ngc

with open('/home/shapedsundew9/gcs.json', 'w', encoding='utf-8') as fptr:
    dump(gcs, fptr, indent=4, sort_keys=True)
