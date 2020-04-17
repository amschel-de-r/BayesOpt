import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.animation import FuncAnimation
import numpy as np
import codecs
import json
import sys

# TODO make data upload simpler

def from_json(filename, arr_type):
    obj_text = codecs.open(filename, 'r', encoding='utf-8').read()
    b_new = json.loads(obj_text)
    arr = np.array(b_new)
    return np.core.records.fromarrays(arr.transpose(), dtype=arr_type)


predicted_dt = [('mean', 'd'), ('upper', 'd'), ('lower', 'd'), ('x', 'd')]
observed_dt = [('x', 'd'), ('y', 'd')]
allPredicted = []
allObserved = []
allAquisition = []
allNextBest = []
nQueries = int(sys.argv[1])

names = ["Cs", "Py"]
pyIndex = names.index("Py")
nLibs = 2
pynext_dt = [('x', 'd')]
def simpleJsonRead(lib):
    with open(f"DataOutput/nextbest{lib}.json", "r") as read_file:
        return json.load(read_file)

for i in range(nLibs):
    currentPredicted = []
    currentObserved = []
    currentAquisition = []
    for j in range(nQueries):
        ipredicted = from_json(f"DataOutput/predicted_test{names[i]}{j+1}.json", predicted_dt)
        iobserved = from_json(f"DataOutput/observed_test{names[i]}{j+1}.json", observed_dt)
        iaquisition = from_json(f"DataOutput/aquisition_test{names[i]}{j+1}.json", observed_dt)
        currentPredicted.append(ipredicted)
        currentObserved.append(iobserved)
        currentAquisition.append(iaquisition)
    allNextBest.append(simpleJsonRead(names[i]))
    allPredicted.append(currentPredicted)
    allObserved.append(currentObserved)
    allAquisition.append(currentAquisition)

objective = np.zeros(allPredicted[0][0]['x'].size, dtype=[('x', 'd'), ('y', 'd')])
x = allPredicted[0][0]['x']
objective['x'] = x
objective['y'] = -x*np.cos(-2*x)*np.exp(-(x/3))

fig = plt.figure(figsize=(48, 9))
fig.subplots_adjust(wspace=0, hspace=0)
outergs = gridspec.GridSpec(1, 2)
ax0s = []
ax1s = []
predlns = []
objlns = []
obslns = []
nextlns = []
aqlns = []
nextaqlns = []

for i in range(nLibs):
    gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outergs[i],
                                          wspace=0.0, hspace=0.0, height_ratios=[3, 1])

    ax0 = plt.subplot(gs[0], title=f"Posterior distribution {names[i]}")
    ax0.legend(loc='upper right')
    ax0s.append(ax0)

    predln, = ax0.plot([], [], label='Mean')
    objln, = ax0.plot(objective['x'], objective['y'], '--', color='gray', label='Objective function')
    obsln, = ax0.plot([], [], 'o', ms=10, mfc='white', mec='blue', mew=2, label='Observed values')
    nextln = ax0.axvline(x=0, label='Next Query Point', color='red')

    predlns.append(predln)
    objlns.append(objln)
    obslns.append(obsln)
    nextlns.append(nextln)

    ax1 = plt.subplot(gs[1], title=f"Aquisition Function {names[i]}")
    aqln, = ax1.plot([], [])
    nextaqln = ax1.axvline(x=0, label='Next Query Point', color='red')

    ax1s.append(ax1)
    aqlns.append(aqln)
    nextaqlns.append(nextaqln)


def init():
    for i in range(nLibs):
        predicted = allPredicted[i][0]
        aquisition = allAquisition[i][0]
        x_min = predicted['x'].min()
        x_max = predicted['x'].max()
        y_max = objective['y'].max() + 0.15 * (objective['y'].max() - objective['y'].min())
        y_min = objective['y'].min() - 0.15 * (objective['y'].max() - objective['y'].min())
        ax0s[i].set_xlim(x_min, x_max)
        ax0s[i].set_ylim(y_min, y_max)

        ax1s[i].set_xlim(aquisition['x'].min(), aquisition['x'].max())
        ax1s[i].set_ylim(0, aquisition['y'].max() * 1.05)
    plt.savefig("init.png")


def update(i):
    for j in range(nLibs):
        predicted = allPredicted[j][i]
        observed = allObserved[j][i]
        aquisition = allAquisition[j][i]

        next_query_point = allNextBest[j][i]
        predlns[j].set_data(predicted['x'], predicted['mean'])
        ax0s[j].collections.clear()
        ax0s[j].fill_between(predicted['x'], predicted['lower'], predicted['upper'], color="lightblue")
        obslns[j].set_data(observed['x'], observed['y'])
        x_data, y_data = nextln.get_data()
        x_data = [next_query_point for x in x_data]
        nextlns[j].set_data(x_data, y_data)

        ax1s[j].set_ylim(0, aquisition['y'].max() * 1.05)
        aqlns[j].set_data(aquisition['x'], aquisition['y'])
        ax1s[j].collections.clear()
        ax1s[j].fill_between(aquisition['x'], aquisition['y'], color="orange")
        x_data, y_data = nextaqln.get_data()
        x_data = [next_query_point for x in x_data]
        nextaqlns[j].set_data(x_data, y_data)
    # plt.savefig(f"{i}.png")


ani = FuncAnimation(fig, update, frames=nQueries, init_func=init, interval=2000)

plt.show()

# def plot(predicted, observed, aquistition, objective, filename):
#     next_query_point = aquistition[np.argmax(aquistition['y'])]['x']

#     fig = plt.figure(figsize=(16, 9))
#     fig.subplots_adjust(wspace=0, hspace=0)
#     gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
#     ax0 = plt.subplot(gs[0], title="Posterior distribution")
#     ax0.plot(predicted['x'], predicted['mean'], label='Mean')
#     ax0.plot(objective['x'], objective['y'], '--', color='gray', label='Objective function')
#     ax0.plot(observed['x'], observed['y'], 'o', ms=10, mfc='white', mec='blue', mew=2, label='Observed values')
#     ax0.fill_between(predicted['x'], predicted['lower'], predicted['upper'], color="lightblue")
#     ax0.axvline(x=next_query_point, label='Next Query Point', color='red')
#     ax0.legend(loc='upper right')

#     x_min = predicted['x'].min()
#     x_max = predicted['x'].max()
#     y_max = objective['y'].max() + 0.15 * (objective['y'].max() - objective['y'].min())
#     y_min = objective['y'].min() - 0.15 * (objective['y'].max() - objective['y'].min())

#     ax0.axis([x_min, x_max, y_min, y_max])
#     ax1 = plt.subplot(gs[1], title="Aquisition Function")
#     ax1.axis([aquistition['x'].min(), aquistition['x'].max(), 0, aquistition['y'].max() * 1.05])
#     ax1.plot(aquistition['x'], aquistition['y'])
#     ax1.fill_between(aquistition['x'], aquistition['y'], color="orange")
#     ax1.axvline(x=next_query_point, label='Next Query Point', color='red')
#     plt.show()
#     plt.savefig(filename)


# predicted_dt = [('mean', 'd'), ('upper', 'd'), ('lower', 'd'), ('x', 'd')]
# observed_dt = [('x', 'd'), ('y', 'd')]

# predicted = from_json(sys.argv[1], predicted_dt)
# observed = from_json(sys.argv[2], observed_dt)
# aquistition = from_json(sys.argv[3], observed_dt)

# objective = np.zeros(predicted['x'].size, dtype=[('x', 'd'), ('y', 'd')])
# x = predicted['x']
# objective['x'] = x
# objective['y'] = -x*np.cos(-2*x)*np.exp(-(x/3))

# plot(predicted, observed, aquistition, objective, sys.argv[4])
