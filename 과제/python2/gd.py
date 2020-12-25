import numpy as np
def min_gd(fun, x0, grad, args=()):
    appropriate_small_value = 0.000001
    epsilon_stopping_criterion = np.full((len(x0)), appropriate_small_value, dtype=float) 

    alpha = 0.3
    beta = 0.8
    delta_x = -grad(x0, *args)

    while((np.abs(grad(x0,*args)) > epsilon_stopping_criterion).all()):
        #1 Descent direction ∆x = −∇f(x) 
        delta_x = -grad(x0,*args)

        #2 Line search. Choose a step size t > 0 via exact or backtracking line search
        t = 1
        while (fun(x0 + t * delta_x, *args) > fun(x0,*args) + alpha * t * (grad(x0, *args)).T @ delta_x):
            t *= beta

        #3 Update. x := x +t∆x 
        x0 = x0 + t * delta_x

    return x0