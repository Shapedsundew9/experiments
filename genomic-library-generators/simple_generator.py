"""Generate a Genomic Library (GL) using just addition and subtraction codons.

The GL is limited to GC's with the interface:

Input: int, int
Output: int
"""
from random import choice
from json import dump, load
from os.path import join
from pprint import pformat

from tqdm import trange
from uuid import UUID

from egp_stores.genomic_library import DATA_FILE_FOLDER
from egp_types.xgc_validator import (
    LGC_json_load_entry_validator,
    LGC_entry_validator,
    LGC_json_dump_entry_validator,
)
from egp_types.ep_type import ep_type_lookup


_INT_TYPE_VALUE = ep_type_lookup["n2v"]["int"]
_CREATOR = UUID("a508fd79-afea-4465-85e2-40d64cccd064")
_NUM_GENERATIONS = 1000
_GRAPHS = (
    {
        "A": [["I", 0, _INT_TYPE_VALUE], ["I", 1, _INT_TYPE_VALUE]],
        "B": [["I", 0, _INT_TYPE_VALUE], ["A", 0, _INT_TYPE_VALUE]],
        "O": [["B", 0, _INT_TYPE_VALUE]],
    },
    {
        "A": [["I", 0, _INT_TYPE_VALUE], ["I", 1, _INT_TYPE_VALUE]],
        "B": [["A", 0, _INT_TYPE_VALUE], ["I", 0, _INT_TYPE_VALUE]],
        "O": [["B", 0, _INT_TYPE_VALUE]],
    },
    {
        "A": [["I", 0, _INT_TYPE_VALUE], ["I", 1, _INT_TYPE_VALUE]],
        "B": [["A", 0, _INT_TYPE_VALUE], ["A", 0, _INT_TYPE_VALUE]],
        "O": [["B", 0, _INT_TYPE_VALUE]],
    },
    {
        "A": [["I", 0, _INT_TYPE_VALUE], ["I", 1, _INT_TYPE_VALUE]],
        "B": [["I", 1, _INT_TYPE_VALUE], ["A", 0, _INT_TYPE_VALUE]],
        "O": [["B", 0, _INT_TYPE_VALUE]],
    },
    {
        "A": [["I", 0, _INT_TYPE_VALUE], ["I", 1, _INT_TYPE_VALUE]],
        "B": [["A", 0, _INT_TYPE_VALUE], ["I", 1, _INT_TYPE_VALUE]],
        "O": [["B", 0, _INT_TYPE_VALUE]],
    },
)

# Load the codons
# Whilst the signatures of all codons should be static during development they may change so here
# we find the integer addition & subtraction codons by looking at the fields that are now stable.
_NAMES = {"+", "-"}
_TYPES = [_INT_TYPE_VALUE]
abspath: str = join(DATA_FILE_FOLDER, "codons.json")
with open(abspath, "r", encoding="utf-8") as fptr:
    codons = load(fptr)

gcs = {
    bytes.fromhex(c["signature"]): LGC_json_load_entry_validator.normalized(c)
    for c in codons
    if c["meta_data"]["name"] in _NAMES
    and c["input_types"] == c["output_types"] == _TYPES
}
pgc = [c for c in codons if c["meta_data"]["name"] == "gc_stack_inverse"][0]
pgc_signature = bytes.fromhex(pgc["signature"])
gcs[pgc_signature] = LGC_json_load_entry_validator.normalized(pgc)

for generation in trange(_NUM_GENERATIONS):
    # Create a new generation by stacking  GCs on top of one another
    gcs_tuple = tuple(gcs.values())
    ngc = {
        "gca": choice(gcs_tuple)["signature"],
        "gcb": choice(gcs_tuple)["signature"],
        "creator": _CREATOR,
        "graph": choice(_GRAPHS),
        "pgc": pgc_signature,
    }
    ngc.update(
        {
            "ancestor_a": ngc["gca"],
            "ancestor_b": ngc["gcb"],
            "generation": max(
                (gcs[ngc["gca"]]["generation"], gcs[ngc["gcb"]]["generation"])
            )
            + 1,  # Increment the generation
            "code_depth": max(
                (gcs[ngc["gca"]]["code_depth"], gcs[ngc["gcb"]]["code_depth"])
            )
            + 1,
            "codon_depth": gcs[ngc["gca"]]["codon_depth"]
            + gcs[ngc["gcb"]]["codon_depth"],
            "num_codes": gcs[ngc["gca"]]["num_codes"]
            + gcs[ngc["gcb"]]["num_codes"]
            + 1,
            "num_codons": gcs[ngc["gca"]]["num_codons"] + gcs[ngc["gcb"]]["num_codons"],
        }
    )
    gcs[ngc["gca"]]["reference_count"] += 1
    gcs[ngc["gcb"]]["reference_count"] += 1
    assert LGC_entry_validator.validate(
        ngc
    ), f"\n{LGC_entry_validator.error_str()}\n------------\n{pformat(ngc, indent=4, width=120)}"
    ngc = LGC_entry_validator.normalized(ngc)
    gcs[ngc["signature"]] = ngc

with open("gcs.json", "w", encoding="utf-8") as fptr:
    dump(
        [LGC_json_dump_entry_validator.normalized(ngc) for ngc in gcs.values()],
        fptr,
        indent=4,
        sort_keys=True,
    )
print(f"{len(gcs)} GCs generated. See gcs.json for the results.")
