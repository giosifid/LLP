#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created for LLP - ToN Submission 2022

@authors: D. Anderson, G. Iosifidis, D. J. Leith
"""
from random import random
import math

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import linprog, minimize, Bounds
from scipy.linalg import norm
from matplotlib import colors as clt


plt.rcParams['axes.facecolor'] = 'white'

plt.rc('font', size=14)          # controls default text sizes
plt.rc('axes', titlesize=14)     # fontsize of the axes title
plt.rc('axes', labelsize=14)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=13)    # fontsize of the tick labels
plt.rc('ytick', labelsize=13)    # fontsize of the tick labels
plt.rc('legend', fontsize=14)    # legend fontsize
plt.rc('figure', titlesize=15)   # fontsize of the figure title

def Project(x):  # projects the argument onto [-1,1]
 	if x > 1: return 1
 	if x < -1: return -1
 	return x
def Pve(x):     # projects the argument onto [0, infty]
 	if x < 0: return 0
 	return x
#
#
# - - - - - - - - - - - - - - P R O B L E M - - - - - - - - - - - - - - - - - 
#
# Problem: f_t(x)=c_tx, g_t(x)=\beta_tx+\gamma_t, x in X=[-1,1];
#
#
#
#
T = 10000  # time horizon
#
#
#



c=[0 for n in range(0,T)]
b=[0 for n in range(0,T)]
gam=[0 for n in range(0,T)]
c = [ -2 for n in range(0,T)] # cost c_t 
for n in range(0,T):
    b[n]=0
    gam[n]=-0.01
    if random()>(0.1/(n+1)**0.05):
        b[n]=1
        gam[n]=0


################### THIS part does only some initialization and can be used to create variuos prediction traces #################
################################################################################################################
flagA=0  # if =1, we add windows of offsets to the cost & constraint parameters
flagB=0  # if =1, we add windows of offsets to the errors
flagC=0  # if =1, we use prediction errors that diminsh with time
flagD=0  # if =1, we use the running averages of costs as predictions
flagE=0  # if =1, we use iid cost and constraint parameters
flagF=0  # if =1, we do no use predictions (setting all to zero)
#
#
#
#
#

#
#
#
#
if flagE==1: # change cost and constraints to iid   
    c=[1.02+0.12*random() for n in range(0,T)]
    b=[0.152+0.5*random() for n in range(0,T)]
    gam=[0.092 +0.005*random() for n in range(0,T)]    

th1=0
th2=0
th3=0
th4=0
if flagA>0: # adds iid offsets to costs and constraints in some windows
    th1=int(T/12)
    th2=int((4*T)/12)
    th3=int(3*T/5)
    th4=int(4.5*T/5)
for n in range(th1, th2):  
    c[n] = c[n]+ 0.5*random()  
    b[n] = b[n]-0.1*random()    
    gam[n] = gam[n]+ 0.04*random()
for n in range(th3, th4):  
    c[n] = c[n]- 0.5*random()   
    b[n] = b[n]+0.19*random()   
    gam[n] = gam[n]- 0.3*random()    
#
#
#
#
#
# - - - - - - - - - - P R E D I C T I O N   E R R O R S  - - - - - - - - - - -
#
c_er=max(c)/5
b_er=max(b)/5
g_er=max(gam)/5
kk=0.2  # set kk=0 to not add the sin-type offsets // kk>0 for sin-type pattern
eps_c = [c_er*random()+kk*math.sin(math.pi*n/12) for n in range(0,T)] # error for c_t
eps_b = [b_er*random()+kk*math.sin(math.pi*n/6) for n in range(0,T)]  # error for b_T
eps_g = [g_er*random()+kk*math.sin(math.pi*n/2) for n in range(0,T)]  # error for gam_t
#
#
th1=0
th2=0
th3=0
th4=0
th5=0
th6=0
if flagB>0:  # adds iid offsets to prediction errors in some windows
    th1=int(T/2)
    th2=int((4*T)/6)
    th3=int(T/8)
    th4=int((2*T)/8)
    th5=int(5*T/6)
    th6=int((16*T)/18)
for n in range(th1, th2):
    eps_c[n] = c_er*random() 
    eps_b[n] = -b_er*random()
    eps_g[n] = g_er*random()
#    
for n in range(th3, th4):
    eps_c[n] = -c_er*random() 
    eps_b[n] = b_er*random()
    eps_g[n] = -g_er*random()
#    
for n in range(th5, th6):
    eps_c[n] = c_er*random() 
    eps_b[n] = b_er*random()
    eps_g[n] = -g_er*random()    
#

if flagC==1: # uses predictions errors that diminish with time    
    eps_c=[c_er/(n+1) for n in range(0,T)]  
    eps_b=[b_er/(n+1) for n in range(0,T)]
    eps_g=[g_er/(n+1) for n in range(0,T)]            
    
# Predictions = Prediction Errors + Value
hc = [eps_c[n] + c[n] for n in range(0,T)]  
hb = [eps_b[n] + b[n] for n in range(0,T)]   
hgam = [eps_g[n] + gam[n] for n in range(0,T)] 

if flagD==1: # uses the running averages of costs/constraints as predictions
    hc=[sum(c[0:n])/(n+1) for n in range(0,T)]
    hb=[sum(b[0:n])/(n+1) for n in range(0,T)]
    hgam=[sum(gam[0:n])/(n+1) for n in range(0,T)]
    eps_c=[hc[n]-c[n] for n in range(0,T)] 
    eps_b=[hb[n]-b[n] for n in range(0,T)]
    eps_g=[hgam[n]-gam[n] for n in range(0,T)] 
    
    
if flagF==1: # do not use predictions at all    
    hc=[0 for n in range(0,T)]
    hb=[0 for n in range(0,T)]
    hgam=[0 for n in range(0,T)]
 
#
# We use L_2 distance to quantify the total prediction errors (in percentage)
obj_error = 100*norm(np.array(c)-np.array(hc))/norm(np.array(c))  
cons_error = 100*norm(np.array(b)-np.array(hb))/norm(np.array(b))
cons_error2 = 100*norm(np.array(gam)-np.array(hgam))/norm(np.array(gam))
tot_error = obj_error+cons_error
#
################################################################################################################
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
###############################################################################
# - - - - - - - - - - - -  B E N C H M A R K   P O L I C Y  - - - - - - - - - -
#
#
c_tot = sum(c[0:T])   # \sum_{t=0}^(T-1) c_t ; uses elements from 0 to T-1 
A_ub = np.array([b[0:T]]).reshape(T,1)  # we have T inequality constraints
A_ub = A_ub.reshape(T,1)
b_ub = -np.array(gam[0:T]).reshape(T,1)
b_ub = b_ub.reshape(T,1)
x0_bounds = (-1, 1)  # our set X is [-1,1] or [-10,10] or [-100,100]
bounds = [x0_bounds] 
result_bench = linprog(c_tot, A_ub=A_ub, b_ub=b_ub, bounds=bounds)
if result_bench.success==False:
    print('************* The problem is infeasible *************************')
x_star = result_bench.x[0]
FF = [x_star*c[n]  for n in range(0,T)]           # f_t(x^*)=c_tx^*
FF_T = [sum(FF[0:n+1])/(n+1) for n in range(0,T)] # sum_t f_t(x^*)/T
G = [x_star*b[n]+gam[n]  for n in range(0,T)]     # g_t(x^*)
GG = [sum(G[0:n+1]) for n in range(0,T)]          # cummulative constraint violation
GG_T = [sum(G[0:n+1])/(n+1) for n in range(0,T)]  # sum_t g_t(x^*)/T
#
#
################################################################################
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# - - - - - - - - - - -  - - - - - -    2022  L L P     - - - - - - - - - - - - -
#
my_bounds2=Bounds(0, 100000)  # bounds for lambda
my_bounds=bounds   # bounds for x
#
# Steps
sigma=abs(((max(c))**0.5)/(bounds[0][1]-bounds[0][0]) )
sigma=1
aa=20.9 #0.2921      ## 0.21 works well. # 0.921 works well
#aa=1.79 #1st of December
aa=8  # 2nd of Dec
beta=0.921   # must be in (0,1)
beta=0.1913   #1st of December
beta=0.05   #2nd of December
sigma_t=sigma #+4*max(c)**2
a_t=aa
ksi_t=0
G=bounds[0][1]*max(b)+max(gam)

#
# Initialization ------------
#
 

x_pred=[0.1 for nn in range(0,T)]
x_t=np.zeros(T)
sigma_t=np.array(x_pred)
l_t=np.zeros(T)
z_t=np.array(x_pred)
x_pred= np.random.randn(T)
h_t=0
hprev=0 # \sum h_i, with i up to t-1
sxprev=0 # \sum sigma_ix_i with i up to t-1
lambdabprev=l_t[0]*b[0]
cprev=c[0]
sigmaprev=sigma
bgzt=0
perfect=1
if perfect==1:
    hc=c # perfect predictions
    hb=b # perfect predictions 
    hgam=gam  # perfect predictions
    eps_c=np.zeros(T)
    eps_b=np.zeros(T)
    eps_g=np.zeros(T)
#Predictions = Prediction Errors + Value
if perfect==0:
    kk=1
    mm=0.5
    eps_c=np.ones(T)*(random()/mm)
    eps_b=np.ones(T)*(random()/kk)
    eps_g=np.ones(T)*(random()/kk)
    hc = [eps_c[n] + c[n] for n in range(0,T)]  
    hb = [eps_b[n] + b[n] for n in range(0,T)]   
    hgam = [eps_g[n] + gam[n] for n in range(0,T)] 

obj_error = 100*norm(np.array(c)-np.array(hc))/norm(np.array(c))  
cons_error = 100*norm(np.array(b)-np.array(hb))/norm(np.array(b))
cons_error2 = 100*norm(np.array(gam)-np.array(hgam))/norm(np.array(gam))
tot_error = obj_error+cons_error


for n in range(1,T-1):   # n (t) takes values from 0 (1) to T-2 (T-1)
    #
    Lambdat=l_t[n]*hb[n]
    x_t[n]=Project((sxprev - hc[n] - cprev - lambdabprev - l_t[n]*hb[n])/sigmaprev)
    #    
    cprev=cprev+c[n]    
    hprev=h_t
    h_t=h_t+norm(eps_c[n]+l_t[n]*eps_b[n]) 
    sigma_t[n]=sigma*((h_t)**0.5-(hprev)**0.5)
    sxprev=sxprev+sigma_t[n]*x_t[n]
    lambdabprev=lambdabprev+l_t[n]*b[n]
    sigmaprev=sigmaprev+sigma_t[n]
    #
    z_t[n]=(sxprev-lambdabprev-cprev)/sigmaprev
    #
    bgzt=bgzt+b[n]*z_t[n]+gam[n]
    #
    x_pred[n]=z_t[n-1]
    ksi_t=ksi_t+(b[n]*z_t[n]+gam[n] - hb[n]*x_pred[n]-hgam[n])**2
    a_t=aa/(max( n**beta, ((4*G*G)+ksi_t)**0.5 ))
    #a_t=aa/(n**beta)
    #
    x_pred[n]=z_t[n]   
    l_t[n+1]= Pve(a_t*( bgzt +(hb[n+1]*z_t[n]+hgam[n+1]) ))


# plot LLP with Non-Adaptive Steps
G0 = [(b[n]*x_t[n]+gam[n] ) for n in range(0,T-1)] # constraint violation
GG0 = [sum(G0[0:n+1]) for n in range(0,T-1)]       # total constraint violation
GG0_T = [sum(G0[0:n+1])/(n+1) for n in range(0,T-1)]  # average constraint violation
F0 = [x_t[n]*c[n] for n in range(0,T-1)]              # cost per slot 
FF0_T = [sum(F0[0:n+1])/(n+1) for n in range(0,T-1)]  # average cummulative (R_T)
D0 = [x_t[n]*c[n]-x_star*c[n]  for n in range(0,T-1)] # regret per slot 
DD0_T = [sum(D0[0:n+1])/(n+1) for n in range(0,T-1)]  # average regret cost (R_T/T)


plt.plot(DD0_T, 'k-', label='$R_T/T$' ) 
#plt.plot(FF0_T, 'k-', label='$\sum_{t}f_t(x_t)/T$' ) 
#plt.plot(GG0_T, 'b-', label='$\sum_{t}g_t(x_t)/T$' ) 
plt.plot(GG0, 'b-', label='$\sum_{t}g_t(x_t)$' ) 
if perfect==0:
    plt.title('LLP without Predictions', fontsize=18)
if perfect==1:
     plt.title('LLP with Perfect Gradient Predictions', fontsize=18)
plt.legend(loc='lower left' , framealpha=0.5, frameon=False, fontsize=15) 
ax = plt.gca()
ax.set_facecolor(clt.to_rgb('#F8F9FA'))
plt.show()
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# - - - - - - - - - - -  M O S P  BY  CHEN & GIANNAKIS - - - - - - - - - - - - 
################################################################################
my_bounds2=Bounds(0, 100000)  # bounds for lambda
my_bounds=bounds
l4_t=[0.0]
x4_t=[0.0]
bg=1/3
a_g=0.8*T**(-bg) # O(T^-1/3), multipliers affect the cost-constr trade off
mu_g=0.5*T**(-bg) # O(T^-1/3)
#
def my_lagrange2a(x4):
    myLag=((x4)**2)/(2*a_g)  # equation (8) of that paper
    return(myLag)

def my_lagrange2b(x4):
    myLag=c[n-1]*(x4-x4_t[n-1]) + l4_t[n]*(b[n-1]*x4+gam[n-1]) + ((x4-x4_t[n-1])**2)/(2*a_g)  # equation (8) of that paper
    return(myLag)        
        
for n in range(0,T-1): # n takes values from 0 to t-1
    if n==0:
        res=minimize(my_lagrange2a, [0], bounds=my_bounds ) #res.x[0] has the x
        x4_t=[res.x]  
    if n>0:
        res = minimize(my_lagrange2b, [0], bounds=my_bounds) #res.x[0] has the x
        x4_t=x4_t+[res.x]  
    m = l4_t[n] + mu_g*(b[n]*x4_t[n]+gam[n])  #l_n + mu_n \grad_l l(x_n,l_n)
    l4_t = l4_t + [Pve(m)]

G2 = [(b[n]*x4_t[n]+gam[n] ) for n in range(0,T-1)]     
GG2_T = [(sum(G2[0:n+1]))/(n+1) for n in range(0,T-1)]  # average constraint violation
GG2= [(sum(G2[0:n+1])) for n in range(0,T-1)]  # average constraint violation
F2 = [x4_t[n]*c[n]  for n in range(0,T-1)]
FF2_T = [sum(F2[0:n+1])/(n+1) for n in range(0,T-1)]   # average cost 
D0C = [x4_t[n]*c[n]-x_star*c[n]  for n in range(0,T-1)] # regret R_T
DD0C_T = [sum(D0C[0:n+1])/(n+1) for n in range(0,T-1)]  # average regret R_T/T

plt.plot(DD0C_T, 'k-', label='$R_T/T$' ) 
#plt.plot(FF2_T, 'k-', label='$\sum_{t}f_t(x_t)/T$' ) 
#plt.plot(GG2_T,  'b-', label='$\sum_{t}g_t(x_t)/T$' )
plt.plot(GG2,  'b-', label='$\sum_{t}g_t(x_t)$' )
plt.title('Chen et al, 2017',  fontsize=20)
plt.legend(loc='lower right' , framealpha=0.5, frameon=False) 
ax = plt.gca()
ax.set_facecolor(clt.to_rgb('#EDF9E8'))
plt.show()

                            


#
#
#
#
# - - - - - - - - - - -  AISTATS'20 by Valls et al - - - - - - - - - - - - 
################################################################################
my_bounds2=Bounds(0, 100000)  # bounds for lambda
my_bounds=bounds
l5_t=[0.0]
x5_t=[0.0]
a=0.5
#

def my_lagrange5b(x5):
    myLag5=c[n-1]*x5 + l5_t[n-1]*(b[n-1]*x5+gam[n-1]) + (n**a)*((x5-x5_t[n-1])**2)
    return(myLag5)   

def my_lagrange5c(l):
    myLag5=(n**a)*(l-l5_t[n-1])**2 - l*(b[n]*x5_t[n]+gam[n] )
    return(myLag5)        
        
for n in range(0,T-1):  
    if n==0:
        x5_t=[0]  
        l5_t=[1]
    if n>0:
        res = minimize(my_lagrange5b, [0], bounds=my_bounds)  
        x5_t=x5_t+[res.x]  
        res = minimize(my_lagrange5c, [0], bounds=my_bounds2) 
        l5_t=l5_t+[res.x]  

G3 = [(b[n]*x5_t[n]+gam[n] ) for n in range(0,T-1)]    
GG3_T = [(sum(G3[0:n+1]))/(n+1) for n in range(0,T-1)]   
GG3= [(sum(G3[0:n+1])) for n in range(0,T-1)]   
F3 = [x5_t[n]*c[n]  for n in range(0,T-1)]
FF3_T = [sum(F3[0:n+1])/(n+1) for n in range(0,T-1)]    
D0D = [x5_t[n]*c[n]-x_star*c[n]  for n in range(0,T-1)]  
DD0D_T = [sum(D0D[0:n+1])/(n+1) for n in range(0,T-1)]   

 

ax = plt.gca()

ax.set_ylim([-200, 200])


plt.plot(DD0D_T, 'k-', label='$R_T/T$' ) 
#plt.plot(FF3_T, 'k-', label='$\sum_{t}f_t(x_t)/T$' ) 
plt.plot(GG3,  'b-', label='$\sum_{t}g_t(x_t)$' )
#plt.plot(GG3_T,  'b--', label='$\sum_{t}g_t(x_t)/T$' )
plt.title('Valls et al. 2020', fontsize=20)
plt.legend(loc='lower left' , framealpha=0.5, frameon=False) 
ax = plt.gca()
ax.set_facecolor(clt.to_rgb('#FFF0F0'))
plt.show()





















#
#
#
#
# - - - - - - - - - - -  Sun et al 2017 - - - - - - - - - - - - 
################################################################################
#
l6_t=np.zeros(T)
x6_t=np.zeros(T)
# for doug's problem
B=1  # correct value is 4
D=1  # correct value is 1
G=1**0.5; # correct value is 2 or sqrt 2
#
# for my problem
B=4  # correct value is 4  maximum value of the divergence (the L2 distance here in X)
D=1.1  # correct value is 1. maximum value of constraint function
G=2; # correct value is sqrt 4, maximum value of the dual norm of the cost and constraint gradient
#
alpha=2
#mu=(B/(T*(D*D+G*G/alpha)))**0.5
mu=(B/(T*(D*D+(G*G/alpha))))**0.5
delta=2*G*G/alpha
#      

for n in range(1,T): 
    x6_t[n]=Project(x6_t[n-1]-mu*(c[n-1]+l6_t[n-1]*b[n-1]))    
    l6_t[n]=Pve(l6_t[n-1] + mu*(b[n]*x6_t[n-1]+gam[n]) - 2*delta*(mu**2)*l6_t[n-1])
 

G6 = [(b[n]*x6_t[n]+gam[n] ) for n in range(0,T-1)]    
GG6_T = [(sum(G6[0:n+1]))/(n+1) for n in range(0,T-1)]   
GG6 = [(sum(G6[0:n+1])) for n in range(0,T-1)]   
F6 = [x6_t[n]*c[n]  for n in range(0,T-1)]
FF6_T = [sum(F6[0:n+1])/(n+1) for n in range(0,T-1)]    
D0D6 = [x6_t[n]*c[n]-x_star*c[n]  for n in range(0,T-1)]  
DD0D_T6 = [sum(D0D6[0:n+1])/(n+1) for n in range(0,T-1)]   
test_T6 = [n**0.75 for n in range(0,T-1)]   
test_T7 = [5*n**0.75 for n in range(0,T-1)]   


plt.plot(DD0D_T6, 'k-', label='$R_T/T$' ) 
plt.plot(GG6,  'b-', label='$\sum_{t}g_t(x_t)$' )
plt.plot(test_T6,  'r--', label='$T^{3/4}$' )
plt.title('Sun et al. 2017', fontsize=20)
plt.legend(loc='upper left' , framealpha=0.5, frameon=False) 
ax = plt.gca()
ax.set_facecolor(clt.to_rgb('#f5f5dc'))#FFFAF0
plt.show()






if result_bench.success==False:
    print('*************The problem is infeasible *******************************')
else:
    print('The optimal solution x* is:')
    print(x_star)
    print('The percentage errors of predictions (costs and constraits) are:')
    print(obj_error)
    print(cons_error)
    print(cons_error2)

    
    
