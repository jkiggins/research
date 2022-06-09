import torch
import numpy as np
import torch
import random
import pickle

from pathlib import Path

from collections import OrderedDict

def seed_many():
    seed = 2340234234
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def load_or_run(fn, path, sim=False):
    path = Path(path)

    if path.exists() and (not sim):
        print('loading: ', path)
        db = ExpStorage(path)
    else:
        db = fn()
        db.save(path)

    return db
    

class VSweep:
    def __init__(self, values=None, head=None):
        self.values = values
        self.mode = None
        self.next = None
        self.head = head


    def foreach(self, values):
        next_head = self.head
        if next_head is None:
            next_head = self
            
        _next = VSweep(values, next_head)
        self.next = _next
        self.mode = 'foreach'

        return self.next


    def zip(self, values):
        next_head = self.head
        if next_head is None:
            next_head = self
        _next = VSweep(values, next_head)
        self.next = _next
        self.mode = 'zip'

        return self.next


    def __iter__(self):
        if self.head is None:
            return self.run()
        else:
            return self.head.__iter__()


    def __len__(self):
        if self.head is None:
            return self._compute_len()
        else:
            return len(self.head)


    def _compute_len(self):
        l = len(self.values)
        
        if self.next is None:
            pass
        elif self.mode == 'foreach':
            l = l * self.next._compute_len()
        elif self.mode == 'zip':
            l = self.next._compute_len()

        return l

        
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
    def __init__(self, path=None):
        self.experiments = {}
        
        if not (path is None):
            with open(path, 'rb') as fp:
                self.experiments = pickle.load(fp)

        self.prefix({})

        self.exp('main')


    def __iter__(self):
        return self.db.__iter__()

    def __len__(self):
        return len(self.db)

    def __getitem__(self, idx):
        return self.db[idx]

    def save(self, path):
        with open(path, 'wb') as fp:
            pickle.dump(self.experiments, fp)
        
    def exp(self, name):
        self.exp_name = name
        if not (self.exp_name in self.experiments):
            self.experiments[self.exp_name] = {'db': [], 'meta': {}}
            self.db = self.experiments[self.exp_name]['db']
            self.meta = self.experiments[self.exp_name]['meta']


    def prefix(self, d={}):
        self._prefix = d


    def store(self, v):
        if type(v) != dict:
            raise ValueError("Values to store must be dict, not ", type(v))

        for key in self._prefix:
            if key in v:
                raise ValueError("Provided key conflicts with prefix: {}".format(key))
            v[key] = self._prefix[key]

        self.db.append(v)


    def last(self):
        return self.db[-1]


    def _hashable(self, val):
        if type(val) == torch.Tensor:
            val = val.numpy()
            if len(val.shape) == 0:
                val = float(val)
            elif len(val.shape) == 1:
                val = val.tolist()

        if type(val) == list:
            val = tuple(val)
        elif type(val) == dict:
            val = tuple(zip(dict.keys(), dict.values()))
        elif type(val) == np.ndarray:
            val = tuple(map(tuple, val.tolist()))
            
        return val


    def flat(self):
        pivot_db = {}

        for d in self.db:
            for key, val in d:
                if not (key in pivot_db):
                    pivot_db[key] = []
                pivot_db[key].append(val)

        for key in pivot_db:
            pivot_db[key] = torch.as_tensor(pivot_db[key])

        return pivot_db


    def group_by(self, key, sort=False):
        """
        Group all stored records by a given key
        """

        # Records to return, where keys are the unique values of key across the dataset
        records = {}

        group_keys = []
        
        # Find all of the unique group_keys
        for d in self.db:
            if not (key in d):
                continue
            val = d[key]
            g_key = self._hashable(val)

            if not (g_key in group_keys):
                group_keys.append(g_key)

        # Use the gather function to grab records for each unique group key
        for g_key in group_keys:
            records[g_key] = self.gather(
                lambda x: hash(self._hashable(x[key])) == hash(g_key),
                filter = lambda x: key in x
            )

        if sort:
            sorted_keys = sorted(list(records.keys()))
            records_sorted = OrderedDict()
            for k in sorted_keys:
                records_sorted[k] = records[k]
            records = records_sorted

        return records

    
    def trace(self, key):
        arr = []
        for d in self:
            if key in d:
                arr.append(d[key])

        return arr


    def gather(self, match_fn, filter=None):
        """
        Build a new ExpStorage based on match_fn and filter functions
        """
        
        records = ExpStorage()

        for d in self.db:
            # If filter is defined, and returns False, ignore this entry
            if not (filter is None) and not filter(d):
                continue
            
            if match_fn(d):
                records.store(d)

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
        db.store({'r': r, 'rand': np.random.random()})

    records = db.group_by('r')
    assert len(records) == 10
    assert all([len(r) == 10 for k, r in records.items()])


    # Test gathering records by hash of list
    db = ExpStorage()
    
    rand_2d_arr = np.random.random((10,10))

    for row in rand_2d_arr:
        for i in range(10):
            db.store({'row': row.tolist(), 'rand': np.random.random()})

    records = db.group_by('row')
    assert len(records) == 10
    assert all([len(r) == 10 for k, r in records.items()])


    # Test gathering records by hash of list, then value
    db = ExpStorage()
    
    rand_2d_arr = np.random.random((10,10))
    rand_int_arr = np.random.randint((2), size=10)

    for row in rand_2d_arr:
        for i in range(10):
            db.store({'row': row.tolist(), 'rand-int': rand_int_arr[i],'rand': np.random.random()})

    records = db.group_by('row')
    assert len(records) == 10
    assert all([len(r) == 10 for k, r in records.items()])

    for key, r in records.items():
        records2 = r.group_by('rand-int')

        for key2, r2 in records2.items():
            num_entries_with_key = np.sum(rand_int_arr == key2)
            assert len(r2) == num_entries_with_key
