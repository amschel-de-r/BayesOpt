

import subprocess
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from bayes_opt import UtilityFunction
from bayes_opt import BayesianOptimization
import json

# TODO allow to run from either folder

def blackbox(x):
    return -x*np.cos(-2*x)*np.exp(-(x/3))


def posterior(optimizer, x_obs, y_obs, grid):
    optimizer._gp.fit(x_obs, y_obs)

    mu, sigma = optimizer._gp.predict(grid, return_std=True)
    return mu, sigma


def log_gp(optimizer, x, i):
    x_obs = np.array([[res["params"]["x"]] for res in optimizer.res])
    y_obs = np.array([res["target"] for res in optimizer.res])
    
    mu, sigma = posterior(optimizer, x_obs, y_obs, x)
    
    
    utility_function = UtilityFunction(kind="ei", kappa=2.576, xi=0)
    utility = utility_function.utility(x, optimizer._gp, 0)
    pynextbest.append(optimizer.suggest(utility_function)['x'])

    x = list(x.flatten())
    mu = list(mu)
    upper = [mean + std for mean, std in zip(mu, sigma)]
    lower = [mean - std for mean, std in zip(mu, sigma)]
    er = list(zip(mu, upper, lower, x))
    qr = list(zip(x_obs.flatten(), y_obs))
    aq = list(zip(x, utility))

    with open(f'DataOutput/aquisition_testPy{i+1}.json', 'w') as json_file:
        json.dump(aq, json_file)
    
    with open(f"DataOutput/observed_testPy{i+1}.json", 'w') as json_file:
        json.dump(qr, json_file)

    with open(f"DataOutput/predicted_testPy{i+1}.json", 'w') as json_file:
        json.dump(er, json_file)


runs = 20

pbounds = {'x': (0, 8)}
optimizer = BayesianOptimization(blackbox, pbounds, 2, 0)
optimizer.probe({"x": 0}, lazy=True)
optimizer.probe({"x": 8}, lazy=True)
x = np.linspace(0, 8, 800).reshape(-1, 1)
utility_function = UtilityFunction(kind="ei", kappa=2.576, xi=0)
pynextbest = []


for i in range(runs):
    optimizer.maximize(init_points=0, n_iter=1, acq='ei', kappa=2.576, xi=0)
    log_gp(optimizer, x, i)
    print(i)


with open("DataOutput/pythonnextbest.json", 'w') as json_file:
    json.dump(pynextbest, json_file)

subprocess.run(f"dotnet run {runs};", check=True, shell=True)
    # # subprocess.run(f"cd ..; dotnet run Rob {i+1}", check=True, shell=True)
    # print(i)
print(runs)
subprocess.run(f"python PythonPlot/plot.py {runs} ", check=True, shell=True)
