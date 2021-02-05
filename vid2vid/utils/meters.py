# -----------------------------------------------------------------------
# Modified from DETR(https://github.com/??/???)
# -----------------------------------------------------------------------
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# -----------------------------------------------------------------------
import os
import numpy as np
import json
from collections import deque


class SmoothedValue(object):

    def __init__(self, window_size=20):
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
    

    def update(self, value):
        self.deque.append(value)
        self.count += 1
        self.total += value
    

    def reset(self, ):
        self.deque = deque(maxlen=20)
        self.total = 0.0
        self.count = 0
    

    @property
    def median(self):
        d = np.array(list(self.deque))
        return np.median(d)


    @property
    def avg(self):
        d = np.array(list(self.deque))
        return np.mean(d)
    

    @property
    def global_avg(self):
        return self.total / self.count
    

    @property
    def value(self):
        return self.deque[-1]
    

class Meter(object):

    def __init__(self, name, log_dir):
        self.name = name
        self.values = SmoothedValue()
        self.log_path = os.path.join(log_dir, name + '.json')
        self.log = ''
    

    def reset(self, ):
        self.values.reset()
    

    def write(self, value, step):
        self.values.update(value)
        log_dict = {'step': step, 
            'value': float(self.values.value),     
            'global_avg': float(self.values.global_avg),
            'meidan': float(self.values.median), 
            'avg': float(self.values.avg)}
        self.log += json.dumps(log_dict) + '\n'
    

    def flush(self, ):
        with open(self.log_path, 'a') as f:
            f.write(self.log)
            self.log = ''
