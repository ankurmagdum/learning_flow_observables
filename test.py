import json
import numpy as np
from itertools import product

with open('hyperparameters.json','r') as f:
    hyp = json.load(f)

for conf_id, conf in enumerate(product(*hyp.values())):
    print(conf_id, conf)

