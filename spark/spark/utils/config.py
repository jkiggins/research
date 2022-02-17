import yaml

class Config:
    def __init__(self, path):
        self.path = path
        self._load()


    def _load(self):
        with open(self.path, 'r') as fp:
            cfg = yaml.safe_load(fp)

        self.cfg = cfg


    def __getitem__(self, key):
        return self.cfg[key]


    def __call__(self, key_addr, val=None):
        ret = self.cfg
        last = ret
        for key in key_addr.split('.'):
            last = ret
            ret = ret[key]

        if not (val is None):
            last[key] = val

        return ret


    def view(self, mod_dict):
        for k in mod_dict:
            self(k, mod_dict[k])
            
