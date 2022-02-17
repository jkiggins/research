import torch
import numpy as np

class VSweep:
    def __init__(self, values=None, head=None):
        self.values = values
        self.mode = None
        self.next = None
        # If there is no incoming head, then self is the head
        if head is None:
            head = self
        self.head = head


    def foreach(self, values):
        _next = VSweep(values, self.head)
        self.next = _next
        self.mode = 'foreach'

        return self.next


    def zip(self, values):
        _next = VSweep(values, self.head)
        self.next = _next
        self.mode = 'zip'

        return self.next


    def run(self):
        if self.mode == 'foreach':
            for v in self.values:
                for v2 in self.next.run():
                    if not hasattr(v2, '__iter__'):
                        v2 = [v2]
                    yield (v, *v2)

        elif self.mode == 'zip':
            for v, v2 in zip(self.values, self.next.run()):
                if not hasattr(v2, '__iter__'):
                    v2 = [v2]
                yield (v, *v2)

        elif self.mode is None:
            for v in self.values:
                if hasattr(v, '__iter__'):
                    yield (v,)
                else:
                    yield v

        else:
            raise ValueError("Unknown Mode: ", self.mode)


class ExpStorage:
    def __init__(self):
        self.db = []
        pass


    def store(self, v):
        self.db.append(v)


    def unique(self, col):
        def _hash_any(val):
            if type(val) == list:
                val = tuple(val)
            elif type(val) == dict:
                val = tuple(zip(dict.keys(), dict.values()))
            elif type(val) == np.ndarray:
                val = tuple(map(tuple, arr))

            return hash(val)


        all_col_values = []
        records = {}
        col_is_hashed = False

        for d in self.db:
            val = d[col]
            if type(val) in [list, tuple, dict, np.ndarray]:
                val = _hash_any(val)
                col_is_hashed = True
            
            all_col_values.append(val)

        unique = np.unique(np.array(all_col_values))

        print("Gathering records where db[{}] == {}".format(col, unique))

        for u in unique:
            if col_is_hashed:
                records[u] = self.gather(col, lambda x: _hash_any(x) == u)
            else:
                records[u] = self.gather(col, lambda x: np.isclose(x, u))

        return records
        
        
    def gather(self, col, match_fn):
        records = []

        for d in self.db:
            if match_fn(d[col]):
                records.append(d)

        return records
        

################### TESTS ######################

import pytest

def test_sweep():
    test_v1 = np.arange(10).astype(np.int32)
    test_v2 = np.flip(test_v1.copy())
    test_v3 = test_v2 + 10

    x = VSweep(test_v1.tolist())
    x = x.foreach(test_v2.tolist())
    x = x.zip(test_v3.tolist())

    outputs_gt = []
    for v in test_v1:
        for v2, v3 in zip(test_v2, test_v3):
            outputs_gt.append((v,v2,v3))
            
    for i, outputs in enumerate(x.head.run()):
        assert outputs == outputs_gt[i]


def test_exp_storage():

    # Test gathering records by unique numeric value
    db = ExpStorage()
    
    rand_rep_arr = np.random.random((10))
    rand_rep_arr = np.tile(rand_rep_arr, 10)

    for r in rand_rep_arr:
        db.store((r, np.random.random()))

    records = db.unique(0)
    assert len(records) == 10
    assert all([len(r) == 10 for k, r in records.items()])


    # Test gathering records by hash of list
    db = ExpStorage()
    
    rand_2d_arr = np.random.random((10,10))

    for row in rand_2d_arr:
        for i in range(10):
            db.store((row.tolist(), np.random.random()))

    records = db.unique(0)
    assert len(records) == 10
    assert all([len(r) == 10 for k, r in records.items()])

