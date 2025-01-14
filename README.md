# BayesOpt

## Current status
- Working optimisation, but no hyperparameter optimisation

## Gif comparison
- This lib (Cs) vs python bayes_opt lib (Py)
- Main difference: Py optimises noise parameter, Cs sticks with static noise parameter (currently v small)
![Alt Text](https://github.com/amschel-de-r/BayesOpt/blob/master/bayesopt.gif?raw=true)

## To Do
- Add hyperparameter optimisation (grid search)
- Test feasibility of L_BFGS_B for hyperparameters see dotnumerics [lib](https://github.com/davidsiaw/neuron/tree/master/DotNumerics/Optimization) ([website](http://www.dotnumerics.com/))
- Clean up variable names, namespaces, etc

## Sources used
- [bayes_opt python library](https://github.com/fmfn/BayesianOptimization)
- [underlying scikitlearn gaussian process regressor](https://github.com/scikit-learn/scikit-learn/tree/b194674c42d54b26137a456c510c5fdba1ba23e0/sklearn/gaussian_process)
- [GP C# library by koryakinp](https://github.com/koryakinp/GP)

