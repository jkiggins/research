import oyaml as yaml

class Config:
    def __init__(self, path):
        self.path = path
        self._load()
        self._inherit()


    def _load(self):
        with open(self.path, 'r') as fp:
            cfg = yaml.safe_load(fp)

        self.cfg = cfg


    def bredth(self):
        _next = [(None, self.cfg)]
        while len(_next) != 0:
            parent, cfg = _next.pop(0)

            for k, v in cfg.items():
                if type(v) == dict:
                    _next.append((cfg, v))
                yield cfg, k, v


    def depth(self, cfg=None):
        if cfg is None:
            cfg = self.cfg

        keys = list(cfg.keys())
        values = list(cfg.values())

        for k, v in zip(keys, values):
            if type(v) == dict:
                yield from self.depth(v)
            yield cfg, k, v
        

    def _inherit(self):
        for parent, k, v in self.depth():
            if k == '__inherit__':
                Config.merge(parent, v)
                parent[k] == None


    @classmethod
    def merge(cls, src, new_cfg):
        _next = [(src, new_cfg)]

        while len(_next) != 0:
            src, new_cfg = _next.pop(0)
            
            for k in new_cfg:
                # Key exists in both
                key_in_both = (k in src) and (k in new_cfg)
                both_are_dict = key_in_both and (type(src[k]) == type(new_cfg[k]) == dict)
                key_in_new = not (k in src) and (k in new_cfg)
                
                if both_are_dict:
                    _next.append((src[k], new_cfg[k]))
                elif key_in_new:
                    src[k] = new_cfg[k]

        return src
                
        
    def __getitem__(self, key):
        return self.cfg[key]


    def __setitem__(self, key, value):
        self.cfg[key] = value



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



class Loader(yaml.SafeLoader):

    def __init__(self, stream):
        self._root = os.path.split(stream.name)[0]
        super(Loader, self).__init__(stream)

    def include(self, node):
        filename = os.path.join(self._root, self.construct_scalar(node))
        with open(filename, 'r') as f:
            return yaml.load(f, Loader)
        
Loader.add_constructor('!include', Loader.include)

################### tests ######################

import pytest
from pathlib import Path

@pytest.fixture
def save_path():
    import os
    
    spath = (Path(__file__).parent/".test").absolute()
    if not spath.exists():
        os.makedirs(str(spath))
    return spath


def test_config(save_path):
    yaml_str = """
classic_stdp: &classic_stdp
    mode: stdp
    tau_ca: 10000.0
    tau_ip3: 100.0
    alpha_pre: 1.0
    tau_kp: 100.0
    alpha_post: 1.0
    ca_th: 1.0

    u_step_params:
        mode: stdp
        ltd: 0.0
        ltp: 0.0

anti_stdp: &anti_stdp
  __inherit__: *classic_stdp
  alpha_pre: -1.0
  alpha_post: -1.0


ltp_bias: &ltp_bias
  __inherit__: *classic_stdp
  tau_ip3: 80
  
  u_step_params:
      ltd: -1.0
"""

    from pprint import pprint
    test_yaml_path = save_path/"test.yaml"
    with open(str(test_yaml_path), 'w') as fp:
        fp.write(yaml_str)

    cfg = Config(str(test_yaml_path))

    pprint(cfg.cfg)

    # Values in __inherit__ that aren't in the target section are populated
    assert cfg['anti_stdp']['mode'] == 'stdp'
    assert cfg['ltp_bias']['mode'] == 'stdp'

    # Values that the target overrides still have their value
    assert cfg['anti_stdp']['alpha_pre'] == -1.0
    assert cfg['ltp_bias']['tau_ip3'] == 80

    # Nested sections from __inherit__ not present in target are populated
    assert cfg['anti_stdp']['u_step_params']['ltd'] == 0.0

    # Overrides in nested sections are preserved
    assert cfg['ltp_bias']['u_step_params']['ltd'] == -1.0
