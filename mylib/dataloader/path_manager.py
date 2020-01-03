import os
import json


class PathManager:
    def __init__(self, cfg_path=None):
        self.environ = parse_environ(cfg_path)

    @property
    def base(self):
        return self.environ['DATASET']

    @property
    def info(self):
        import pandas as pd
        df = pd.read_csv(os.path.join('D:/ProgramData/github/ML_hw/3D-medical-image-classification/mylib/dataloader/info.csv'))
        return df

    @property
    def test(self):
        import pandas as pd
        df1 = pd.read_csv(os.path.join('D:/ProgramData/github/ML_hw/3D-medical-image-classification/mylib/dataloader/test.csv'))
        return df1

    @property
    def nodule_path(self):
        return os.path.join('D:/ProgramDataKaglle/new-data/train_val')
    
    @property
    def test_nodule_path(self):
        return os.path.join('D:/ProgramData/Kaglle/new-data/test')
    


def parse_environ(cfg_path=None):
    if cfg_path is None:
        cfg_path = os.path.join(os.path.dirname(__file__), "ENVIRON")
    assert os.path.exists(cfg_path), "`ENVIRON` does not exists."
    with open(cfg_path) as f:
        environ = json.load(f)
    return environ


PATH = PathManager()
