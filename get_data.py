import os
import pickle
from datetime import datetime

from skopt.benchmarks import branin, hart6

"""The six dimensional Hartmann function defined on the unit hypercube.

It has six local minima and one global minimum f(x*) = -3.32237 at
x* = (0.20169, 0.15001, 0.476874, 0.275332, 0.311652, 0.6573).

More details: <http://www.sfu.ca/~ssurjano/hart6.html>
"""
from sklearn.metrics.scorer import mean_absolute_error as error_func
import skopt
import hyperopt
import numpy as np
import time


def log(l, *a, **k):
    if l < 4:
        print(*a, **k)


benchmarks = {
    'branin': {
        'func': branin,
        'space_TPE': [hyperopt.hp.uniform('0', -5.0, 5.0), hyperopt.hp.uniform('1', -5.0, 5.0)],
        'space': [(-5.0, 5.0), (-5.0, 5.0)],
        'x': [np.pi, 2.275],
        'y': 0.397887
    },
    'hart6': {
        'func': hart6,
        'space_TPE': [hyperopt.hp.uniform('%d' % i, -1.0, 1.0) for i in range(6)],
        'space': [(-1.0, 1.0) for i in range(6)],
        'x': [0.20169, 0.15001, 0.476874, 0.275332, 0.311652, 0.6573],
        'y': -3.32237
    }
}


class Target:
    def __init__(self, error_level=1, benchmark='hart6', name=None, random_state=0):
        self.error_level = error_level
        self.records = []
        self.best = None
        self.benchmark = benchmark
        self.name = name
        self.random = np.random.RandomState(random_state)

    def get_space(self):
        if self.name == 'TPE':
            return benchmarks[self.benchmark]['space_TPE']
        else:
            return benchmarks[self.benchmark]['space']

    def target(self, x, error_level=None):
        input_time = time.time()
        if error_level is None:
            error_level = self.error_level
        log(4, 'loop', len(self.records))
        log(4, 'new_x', x)
        err_x = error_func(benchmarks[self.benchmark]['x'], x)
        log(4, 'error of x', err_x)
        res_true = benchmarks[self.benchmark]['func'](x)
        res = res_true + self.random.standard_normal() * error_level
        log(4, 'output score', res)
        log(4, 'true score', res_true)

        change = False
        if self.best is None:
            self.best = 0
            change = True
        elif res < self.records[self.best]['y_output']:
            self.best = len(self.records)
            change = True

        output_time = time.time()
        self.records.append({
            'input_time': input_time,
            'output_time': output_time,
            'idx': len(self.records),
            'x': x,
            'x_error': err_x,
            'y_true': res_true,
            'y_output': res,
            'best': self.best
        })
        if change:
            log(3, 'New Best:', self.name, self.records[-1])
        return res

    def final_score(self):
        return self.target(self.x, 0)


def run(error_level=0.05, benchmark='hart6', rseed=5, n_calls=100):
    methods = [
        {'name': 'BaysionOptimize', 'func': skopt.gp_minimize,
         'kwargs': {'n_calls': n_calls, 'random_state': rseed}},
        {'name': 'TPE', 'func': hyperopt.fmin,
         'kwargs': {'algo': hyperopt.tpe.suggest, 'max_evals': n_calls * 10, 'rstate': np.random.RandomState(rseed)}},
        {'name': 'GradientBoost', 'func': skopt.gbrt_minimize,
         'kwargs': {'n_calls': n_calls * 10, 'random_state': rseed}},
        {'name': 'RandomSearch', 'func': skopt.dummy_minimize,
         'kwargs': {'n_calls': n_calls * 1000, 'random_state': rseed}},
        {'name': 'DecisionTree', 'func': skopt.forest_minimize,
         'kwargs': {'n_calls': n_calls, 'random_state': rseed}}
    ]
    for m in methods:
        t = Target(error_level=error_level, benchmark=benchmark, name=m['name'])
        m['result'] = t
        m['func'](t.target, t.get_space(), **m['kwargs'])
    idx = 0
    name = 'pkl/Compare-{}-{}-{}.pkl'.format(benchmark, error_level, rseed)
    while os.path.exists(name):
        name = 'pkl/Compare-{}-{}-{}-{}.pkl'.format(benchmark, error_level, rseed, idx)
        idx += 1
    with open(name, 'wb') as fl:
        pickle.dump({
            'name': 'Compare Result',
            'setting': {
                'error_level': error_level,
                'benchmark': benchmark,
                'rseed': rseed,
                'n_calls': n_calls,
            },
            'data': methods,
        }, fl)


if __name__ == '__main__':
    for bench in ('hart6', 'branin'):
        for err in (0, 0.01, 0.1):
            for i in range(5):
                print('Run', bench, err, i)
                run(error_level=err, rseed=i, benchmark=bench)
