#! /usr/bin/env python


import sys
import matplotlib.pyplot as plt

from numpy import log,diff,mean,argmin,sort,cumsum
from scipy.stats import expon,poisson


def C(A):
    m = mean(A)
    if m == 0:
        m = 1
    likelihood=expon.pdf(A,0,m)

    return -2*sum([log(i) for i in likelihood])



def sessionize2(x):
    n_=len(x)
    dx=diff(x)
    sdx=sort(dx)
    cdx=cumsum([dxi for dxi in sdx if dxi>0])
    xrange=cdx[-1]
    n=len(cdx)
    n0=n_-1-n
    max_lhd=-1e300
    max_l=-1
    for l in range(n-1):
        lhd=(l+1)*log((l+1)/cdx[l])+(n-l)*log((n-l)/(xrange-cdx[l]))
        if lhd>max_lhd:
            max_lhd=lhd
            max_l=l
    block_sum=0
    block_number=0
    block_start=0
    block_means=[]
    block_starts=[x[0]]
    block_numbers=[]
    for i in range(n_):
        block_sum+=x[i]
        block_numbers+=[block_number]
        if i<(n_-1) and dx[i]>sdx[max_l]:
            block_number+=1
            block_means+=[block_sum/(i-block_start+1)]#*(i-block_start+1)
            block_sum=0
            block_start=i+1
            block_starts+=[x[i+1]]
        elif i==n_-1:
            block_means+=[block_sum/(i-block_start+1)]#*(i-block_start+1)
    return block_means,block_starts,block_numbers



def PELT(list1,beta):
    cp = [[0]]
    R = [0]
    list1.sort()
    F = [-beta]
    ms =[0]
    D=list(diff(list1))
    for i in range( 1 , len(D) + 1 ):
        F_r =[F[j] + C(D[j:i]) + beta for j in R]
        tau = argmin(F_r)
        Rhold = [R[j] for j in range(len(F_r)) if F[j] + C(D[j:i])<F_r[tau]]
        F += [F_r[tau]]
        ms += [ mean (D[tau:i])]
        cp+=[cp[R[tau]]+[R[tau]]]
        R=list(set(Rhold))
        R+=[i]
        R.sort()
    return cp[-1]




def step_function_PELT(times2):
    times = times2

    beta =  2 * log(len(times))
    Wrapped = [i % 86400 for i in times2]
    Wrapped = list(set(Wrapped))
    Wrapped.sort()
    while len(Wrapped)>200:
        Wrapped2 = sessionize2(Wrapped)[1]
        Wrapped = Wrapped2
         #print (len(Wrapped),'length of sessionized times')
    Wrapped[-1] = 86400
    Wrapped.insert(0,0)
    cps = PELT( Wrapped,beta )
    cps+=[len(Wrapped)-1]
    cps.remove(0)
    cps = list(set(cps))
    cps.sort()
    rates = []
    for t in range(len(cps)-1):
        rate =(cps[t+1]-cps[t])/( Wrapped[cps[t + 1 ]] - Wrapped[cps[t]])
        if rate < 999999999999999999:
            rates+=[rate]
        else:
            rates += [999999999999999999]
    rates += [ 1 / (min(Wrapped) + (86400*2 - max(Wrapped) ))]
    rates = [i/len(Wrapped) for i in rates]
    return [Wrapped[i] for i in cps], rates, cps







def day_PELT(times, training_period):
    no_big_seasons = training_period / 7
    day = [int(i/ 86400) for i in times]
    day.sort()
    Count=[]
    for i in range(int(7)):
        Count += [sum([j == i for j in day]) / float(no_big_seasons)]

    Count2 = [sum(Count) / 7.0 for i in range(len(Count)) ]
    return (Count)







