import numpy as np
from sklearn.metrics.scorer import mean_absolute_error
from hyperopt import fmin, tpe, hp, Trials
from skopt import gp_minimize
from skopt.benchmarks import branin as benchmark_func
"""Branin-Hoo function is defined on the square x1 ∈ [-5, 10], x2 ∈ [0, 15].

It has three minima with f(x*) = 0.397887 at x* = (-pi, 12.275),
(+pi, 2.275), and (9.42478, 2.475).

More details: <http://www.sfu.ca/~ssurjano/branin.html>
"""

space = {}
for x in range(6):
    name = 'x{}'.format(x)
    space[name] = hp.uniform(name, -1., 1.)
print(space)


def benchmark(kwargs):
    print('Get:', kwargs)
    return benchmark_func(kwargs)


space = []
for i in range(2):
    space.append(hp.uniform('{}'.format(i), -5, 5))
"""
Wanted x is (+pi, 2.275), y is 0.397887
"""

trial = Trials()
best = fmin(
    fn=benchmark,
    space=space,
    algo=tpe.suggest,
    max_evals=200,
    trials=trial,
    rstate=np.random.RandomState(0)
)

print('Best', best)

x = []
for i in range(2):
    x.append(best['{}'.format(i)])

print(benchmark(x))

print(trial.best_trial['result'])
