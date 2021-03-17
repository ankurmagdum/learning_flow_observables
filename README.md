# Deep learning fluid flow observables

This is a framework to find an optimal set of hyperparameters for neural networks, which are used to perform optimizations constrained by partial differential equations. 

The specific problem investigated is that of finding an optimal shape of an airfoil (RAE5243) in order to reduce drag while keeping the lift (somewhat) constant. The neural networks act as surrogates for highly expensive CFD simulations that make the task of optimization impractical. Even though the data required to train the neural networks is generated through the CFD simulations, the overall cost of optimization is drastically reduced.

The framework implemented in __main.py__ is constructed around the program __ismo__ (Iterative Surrogate Model Optimization) developed by Kjetil Lye.
