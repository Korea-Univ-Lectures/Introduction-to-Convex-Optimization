import numpy as np
import scipy.optimize as opt

"""
x = np.array([1, 2])
A = np.array([[2, 4], [5, 3]])
print(x.T @ A @ x)
"""

################################################
### part 1: unconstrained LP
################################################

# quadratic obj
def quad_obj(x,Q,b):
  return x.T@Q@x + b.T@x # NOTE(1): YOU NEED TO COMPLETE HERE

# size of matrix A and b
n=4

# generate random positive definite matrix Q
A=np.random.normal(size=(n,n))
Q=np.identity(n)+A.T@A

b=np.random.normal(size=n)

# random initial point
x0=np.random.normal(size=n)

# for part 1, we will use h1_qp function for solving QP
def h1_qp(f0,Q,b,x0):
  return opt.minimize(fun=f0,args=(Q,b),x0=x0)

res= h1_qp(f0=quad_obj,Q=Q,b=b,x0=x0)
print('optimization results from minimize function: ', res)


################################################
### part 2: constrained LP
################################################

# inequality constraints Ax<=b
A=np.array([[15, 6], [8, 7], [5, 18]]) # NOTE(2): YOU NEED TO COMPLETE HERE
b=np.array([15*6, 8*7, 18*5]) # NOTE(3): YOU NEED TO COMPLETE HERE

# objective vector

c=np.abs(np.random.normal(size=2)) * (-1) # NOTE(4): YOU NEED TO COMPLETE HERE

# NOTE(5): DEFINE ANY FUNCTIONS OR VARIABLES IF YOU NEED

def my_obj(x,c):
  return c@x

def my_minimization_function(c, A, b, bound, x0):
  return opt.linprog(c=c,A_ub=A, b_ub=b, bounds=(0, np.inf), x0=x0)

# initial point
x0=np.random.normal(size=len(c))

# for part 2, we will use h1_lp function for solving LP
def h1_lp(f0,c,x0,A,b):
  # f0 is objective function
  # c is objective vector, that is, we miminize c^Tx
  # x0 is initial point
  # A: constraint matrix Ax<=b
  # b: constraint vector Ax<=b
  return my_minimization_function(c, A, b, (0, np.inf), x0) # NOTE(6): YOU NEED TO COMPLETE HERE (YOU MAY ALSO USE opt.linprog INSTEAD OF opt.minimize)

# test your h1_lp function
reslp = h1_lp(my_obj, c, x0, A, b) # NOTE(7): YOU NEED TO COMPLETE HERE

print('optimization results: ', reslp)
