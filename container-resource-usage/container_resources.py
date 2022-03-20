"""Benchmarks of container sizes."""

from collections import namedtuple
import numpy as np
import sys
import gc


# From https://towardsdatascience.com/the-strange-size-of-python-objects-in-memory-ce87bdfbb97f
def actualsize(input_obj):
    memory_size = 0
    ids = set()
    objects = [input_obj]
    while objects:
        new = []
        for obj in objects:
            if id(obj) not in ids:
                ids.add(id(obj))
                memory_size += sys.getsizeof(obj)
                new.append(obj)
        objects = gc.get_referents(*new)
    return memory_size

def deltasize(input_obj, size):
    return actualsize(input_obj) - size

COUNT = 1000000

class empty():

    def __init__(self):
        pass


class small():

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


class slotted_small():

    __slots__ = ('x', 'y', 'z')

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


named = namedtuple('Bill', ('x', 'y', 'z'))


_empty = empty()
print(sys.getsizeof(4))
print(sys.getsizeof(400))
print(sys.getsizeof(400000))
print(sys.getsizeof(4000000000))
base_size = actualsize(_empty)
print(f'Size of empty() = {actualsize(_empty)}')
_tuple = [(i, i+1, i+2) for i in range(100000)]
print(f'Size of tuple(int, int, int) = {deltasize(_tuple, base_size)}')


"""
test_tuple = [(i, i+1, i+2) for i in range(COUNT)]
print(f'Size of {COUNT} tuple(int, int, int) = {actualsize(test_tuple):0.2f} MB')


test_named = [named(i, i+1, i+2) for i in range(COUNT)]
print(f'Size of {COUNT} namedtuple(int, int, int) = {actualsize(test_named):0.2f} MB')


test_small = [small(i, i+1, i+2) for i in range(COUNT)]
print(f'Size of {COUNT} small(int, int, int) = {actualsize(test_small):0.2f} MB')


test_small = [slotted_small(i, i+1, i+2) for i in range(COUNT)]
print(f'Size of {COUNT} slotted_small(int, int, int) = {actualsize(test_small):0.2f} MB')


test_tuple = [(np.int32(i), np.int32(i+1), np.int32(i+2)) for i in range(COUNT)]
print(f'Size of {COUNT} tuple(np.int32, np.int32, np.int32) = {actualsize(test_tuple):0.2f} MB')


test_array = [np.zeros((3,), dtype=np.int32) for i in range(COUNT)]
print(f'Size of {COUNT} ndarray[np.int32, np.int32, np.int32] = {actualsize(test_array):0.2f} MB')


test_dict = [dict(x=i+0, y=i+1, z=i+2) for i in range(COUNT)]
print(f'Size of {COUNT} dict(char:int, char:int, char:int) = {actualsize(test_dict):0.2f} MB')


test_dict = [dict(xxxxxxxxxxxxxx=i, yyyyyyyyyyyyyyyy=i+1, zzzzzzzzzzzzzzzz=i+2) for i in range(COUNT)]
print(f'Size of {COUNT} dict(str:int, str:int, str:int) = {actualsize(test_dict):0.2f} MB')
"""