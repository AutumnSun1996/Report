import os

from get_data import *

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm


def plot_results(*methods, true_minimum=None, max_n_calls=np.inf, choice='x_error', x_mark='n', target_time=0,
                 max_time=1000):
    ax = plt.gca()
    ax.set_title("Convergence plot")
    if x_mark == 'n':
        ax.set_xlabel("Number of calls $n$")
    else:
        ax.set_xlabel("Time Consumption (seconds)")

    ax.set_ylabel(choice)
    ax.grid()
    colors = cm.viridis(np.linspace(0.25, 1.0, len(methods)))
    for result, color in zip(methods, colors):
        #         print(result)
        name = result['name']
        records = result['result'].records
        n_calls = int(np.min([len(records), max_n_calls]))
        mins = [records[r['best']][choice] for r in records[:n_calls]]
        if x_mark == 'n':
            ax.plot(range(1, n_calls + 1), mins, c=color, marker=".", markersize=12, lw=2, label=name)
        else:
            t0 = records[0]['output_time']
            time_consume = []
            for r in records[:n_calls]:
                time_consume.append(r['input_time'] - t0)
                t0 += r['output_time'] - r['input_time'] - target_time
                if time_consume[-1] > max_time:
                    mins = mins[:len(time_consume)]
                    break
            ax.plot(time_consume, mins, c=color, marker=".", markersize=12, lw=2, label=name)
    if true_minimum is not None:
        ax.axhline(true_minimum, linestyle="--",
                   color="r", lw=1,
                   label="True minimum")
    ax.legend(loc="best")
    return ax


if __name__ == '__main__':
    idx = 0
    names = sorted(os.listdir('pkl'))
    print(names[idx])
    with open('pkl/{}'.format(names[idx]), 'rb') as fl:
        data = pickle.load(fl)
    print(data['setting'])

    true_minimum = benchmarks[data['setting']['benchmark']]['y']
    plt.show(plot_results(*data['data'], true_minimum=true_minimum, max_n_calls=100, choice='y_output'))
    plt.show(plot_results(*data['data'], true_minimum=0., max_n_calls=100, choice='x_error'))
    # plt.show(plot_results(*data['data'], true_minimum=true_minimum, choice='y_output', x_mark='time', target_time=2, max_time=1000))
    # plt.show(plot_results(*data['data'], choice='x_error', x_mark='time', target_time=5, max_time=1000))


    # plt.show(plot_results(data['data'][3], data['data'][0], choice='y_true', x_mark='time', target_time=1, max_time=200))
    # plt.show(plot_results(data['data'][3], data['data'][0], choice='x_error', x_mark='time', target_time=1, max_time=200))
