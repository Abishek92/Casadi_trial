# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 18:37:31 2020

@author: abishek
"""
from casadi import *
import numpy as np
import pandas as pd
from scipy.integrate import odeint
import matplotlib.pyplot as plot 



sim_time = 150

T = 5 #sampling time (min)
N = 150 #prediction horizon (min)
control_length = N/T
control_length = round(control_length)

u_max = 2.500 #Maximum limit on inuput
u_min = 0 #Maximum limit on inuput



## Disturbance 
Ra = np.ones(sim_time)


# # States of the system
x1 = SX.sym('x1')
x2 = SX.sym('x2')
x3 = SX.sym('x3')
x4 = SX.sym('x4')
x5 = SX.sym('x5')
states = vertcat(x1, x2, x3, x4, x5)
n_states = 5 

## Control input
u = SX.sym('u')
controls = vertcat(u)
n_controls = 1 

## disturbance
Ra = SX.sym('Ra');  

## Glucose and insulin dynamics
xdot = vertcat(-0.01*(x1-100)-(x1*x2)+Ra, -x2,-0.05*x3+(0.01*x4+0.005*x5),-0.01*x4+0.01*x5,-(0.01)*x5+u)

f = Function('f',[states,controls,Ra],[xdot]) ## Nonlinear function f(x,u)
U = SX.sym('U',n_controls,N) ## Decision variables
P = SX.sym('P',n_states+n_states+N) # initial and final value of states
X = SX.sym('X',n_states,(N+1)) ## States over the optimization 


## Optimal trajectory using control solution
f_opt = Function('f_opt',[U,P],[X]) 

## Propogation of state equation to get x_dot
X[:,0] = P[0:5] ## Initial state
for k in range(N):
    state = X[:,k]
    control = U[:,k]
    D = P[n_states + n_states + k]
    func_value = f(state,control,D)
    # Using Euler integration for propogating the model
    X[:,k+1] = state + (T*func_value)

## Weighting matrices for Optimization
Q = np.identity(1,dtype = float)
R = np.identity(1,dtype = float)    
    
g_ref = 110 # output reference value 
u_ref = 0.5 # output reference value (insulin reference in Units/min)

## Computing cost function 
J = 0 # initial value of objective function 
for i in range(N+1):
    y_val = X[:,i]
    u_val = X[:,i]/1000
    y_cost = T*(y_val - g_ref)
    u_cost = T*(u_val - u_ref)
    J = J + y_cost*Q*y_cost + u_cost*R*u_cost 

# ###############################
g = []    
# Constraint equations
for j in range(N+1):
    g.append([g,X[0,j]])
    # g += [X[:,j]]



# g = vertcat([g])
# # Optimization variables    
Opt_variables = U.T #transpose

## Nonlinear programming problem formulation

nlp_problem = {'f':J,'x':Opt_variables,'g':g,'p':P}


solver = nlpsol('solver', 'sqpmethod', nlp_problem)