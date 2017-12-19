#! /usr/bin/env python


import sys
import PELT_Seasonality_git
import datetime as dt
from scipy.optimize import minimize
from numpy import log,exp,diff,cumsum,mean,argmin,random,median
import matplotlib.pyplot as plt
from scipy.stats import expon,kstest, combine_pvalues
secsinday = 86400
season = 7
workweek=7
min_length = 200

class User():
    def __init__(self, firsttimeseen, workweek, trainingperiod, seasonal, geom):

        self.trainingdata = []
        self.trainingdata_for_training =[]
        self.transformed_data = [0]
        self.pvalues = []
        self.trainindatalength = 0
    

        ###seasonal
        self.dailycps = []
        self.dailyrates = []
        self.weekly_multipliers = [] #### Density function over a week.
        self.seasonal_rate = 0
        self.homo_rate = 0
        self.seasonal_pvalues = []
        self.homo_pvalues = []

        ### WE
        self.WE_alpha = 0
        self.WE_beta = 0
        self.WE_lambda = 0
        self.WE_pvalues = []

        ### HE
        self.HE_lambda = 0
        self.HE_alpha = 0
        self.HE_beta = 0
        self.HE_pvalues = []

    
        ###WS
        self.WS_cps = []
        self.WS_rates = []
        self.WS_cutoffs = []
        self.WS_pvalues = []

    
        self.start = int(dt.date.fromtimestamp(firsttimeseen).strftime('%s'))

        self.training_periods = 0


    ####  FOR EACH DATA TIME     
    def update_data(self, time, trainingperiod, workweek, seasonal, geom):
        if  (time - self.start) /secsinday > trainingperiod:
            if self.training_periods == 0:
                self.trainindatalength = len(self.trainingdata_for_training)
                if seasonal:
                    self.get_seasonal_parameters(workweek,trainingperiod)

                    self.get_transformed_training_data()
                    self.seasonal_rate = len(self.transformed_data) / max(self.transformed_data)
            
                else:
                    self.transformed_data = self.trainingdata
    

                if self.trainindatalength > min_length:
                    self.homo_rate =  len(self.trainingdata) / max(self.trainingdata)

                    self.get_self_exciting_parameters(geom,seasonal)
                self.training_periods += 1
            if seasonal:
                self.transformed_data += [ self.transform_seasonal(time-self.start) ]
            else:
                self.transformed_data += [time-self.start]
            
            if self.trainindatalength > min_length:
                if geom:
                    WS_p,H_p = self.update_pvalues(geom,seasonal)
                    return WS_p,H_p
                else:
                    WE_p,HE_p,WS_p,S_p,H_p = self.update_pvalues(geom,seasonal)
                    if WS_p != '':
                        return WS_p, HE_p, WE_p, S_p, H_p
                    else:
                        return ''
            
            self.trainingdata += [time-self.start]
            
        else:
            self.trainingdata_for_training += [time - self.start]
            self.trainingdata += [ time - self.start]
        return ''












  

    def transform_seasonal(self, t):


        
        No_seasons = int(t / (secsinday*season))
        No_days = int((t /secsinday) % workweek)
        time_of_day = t % secsinday
        Total = No_seasons
        Total += sum( [self.weekly_multipliers[day] for day in range(len(self.weekly_multipliers)) if day < No_days])
        if time_of_day < self.dailycutoffs[0]:
            Total += self.weekly_multipliers[No_days] * self.dailyrates[-1] * time_of_day

            return Total
        else:
            Total += self.weekly_multipliers[No_days] * self.dailyrates[-1] * self.dailycutoffs[0]
        for cps in range(1,len(self.dailycutoffs)):
            if time_of_day < self.dailycutoffs[cps]:
                Total += self.weekly_multipliers[No_days] * self.dailyrates[cps-1] * (time_of_day - self.dailycutoffs[cps-1])


                hold_t = t
                return Total
            else:
                Total += self.weekly_multipliers[No_days] * self.dailyrates[cps-1] * (self.dailycutoffs[cps] - self.dailycutoffs[cps-1])

        Total += self.weekly_multipliers[No_days] * self.dailyrates[-1] * (time_of_day - self.dailycutoffs[-1])

        return Total

 #### TRANSFORM THE DATA USING THE ESTIMATED SEASONALITY    
    def get_transformed_training_data(self):
        self.trainingdata.sort()
        for t in self.trainingdata:
            self.transformed_data += [self.transform_seasonal(t)]

            
 ####ESTIMATE THE PARAMETERS OF ALL MODELS FOR SELF-EXCITING BEHAVIOUR
    def get_self_exciting_parameters(self, geom, seasonal):
        if geom:
            L = len(self.transformed_data)
            l = len(self.transformed_data)
            self.prop_0 = float(l/L)
            self.prop_0=0
            self.transformed_data.sort()
            self.get_parameters_WS(geom)
        else:
            self.transformed_data = list(set(self.transformed_data))
            self.trainingdata  = list(set(self.trainingdata))
            self.trainingdata.sort()
            self.transformed_data.sort()
            self.get_parameters_HE()
            self.get_parameters_WE()
            self.get_parameters_WS(geom)


############FUCTIONS TO GET PARAMETERS

 ####ESTIMATING THE SEASONAL DENSITY FUNCTION AND AVERAGE NUMBER OF EVENTS PER DAY
    def get_seasonal_parameters(self, workweek, trainingperiod):
        self.dailycutoffs,self.dailyrates,self.dailycps =  PELT_Seasonality_git.step_function_PELT(self.trainingdata)
        weekly_multipliers_2 = PELT_Seasonality_git.day_PELT(self.trainingdata,trainingperiod)
        total_events= sum(weekly_multipliers_2)
        self.weekly_multipliers = [i/total_events for i in weekly_multipliers_2]



    def WE_lik(self, alpha , beta , lam ):
        term_1 = sum( [log( lam + alpha * exp( - beta * (self.transformed_data[ i + 1 ] - self.transformed_data[ i ])) ) for i in range( len (self.transformed_data ) - 1 ) ] )
        term_2 = - lam *self.transformed_data[-1]
        term_3 = ( alpha / beta )  * sum ( [ exp ( -beta * (self.transformed_data[ i+1 ] - self.transformed_data [ i ] ) ) -1  for i in range ( len ( self.transformed_data ) -1 )] )
        final = - ( term_1 + term_2 + term_3 )
        return final
    
    def get_parameters_WE(self):
        start = (log (1), log( 2-1 ), log( len(self.transformed_data) / max(self.transformed_data) ) )
        fun = lambda x: self.WE_lik( exp( x[0] ) , exp( x[1] ) + exp ( x[0] ) , exp( x[2] ) )
        a = minimize ( fun , start , method='Nelder-Mead' , tol=1e-6, options={'maxiter':1000,'maxfev':1000})
        var = a.x
        self.WE_alpha, self.WE_beta, self.WE_lambda =  [ exp (var[ 0 ] ), ( exp ( var[ 1 ]) ) + exp(var[ 0 ] ), exp(var [ 2 ])]
    



    def HE_lik(self, alpha , beta , lam ):
        A = [ 0 ]
        for i in range(1, len(self.transformed_data) - 1 ):
            A += [ exp ( -beta* (self.transformed_data[i+1] - self.transformed_data[i] ) ) * ( 1 + A[ -1 ]) ]
        term_1 = sum( [log( lam + alpha * A[ i ] ) for i in range( len (self.transformed_data) - 1 ) ] )
        term_2 = - lam * self.transformed_data[-1]
        term_3 = ( alpha / beta )  * sum ( [ exp ( -beta * (self.transformed_data[ -1 ] - self.transformed_data[ i ] ) ) -1  for i in range ( len ( self.transformed_data ) -1 )] )
        final = - ( term_1 + term_2 + term_3 )

        return final
    
    def get_parameters_HE(self):
        start = (log (1), log( 2-1 ), log( len(self.transformed_data)/max(self.transformed_data)) )
        fun = lambda x: self.HE_lik( exp( x[0] ) , exp( x[1] ) + exp ( x[0] ) , exp( x[2] ) )
        a = minimize ( fun , start , method='Nelder-Mead' , tol=1e-6, options={'maxiter':1000,'maxfev':1000})
        var = a.x
        self.HE_alpha, self.HE_beta, self.HE_lambda =  [ exp (var[ 0 ] ), ( exp ( var[ 1 ]) ) + exp(var[ 0 ] ), exp(var [ 2 ]) ]





    def get_parameters_WS(self, geom):
        self.WS_cutoffs ,self.WS_rates, self.WS_cps =  step_function_PELT_MLE( self.transformed_data, geom )




################ FUNCTIONS TO GET PVALUES


    def update_pvalues(self, geom, seasonal):
        waiting_time = self.transformed_data[-1] - self.transformed_data[-2]
        actual_waiting_time = self.trainingdata[-1] - self.trainingdata[-2]
        if geom:
            if waiting_time >= 0:
                if waiting_time == 0:
                    WS_p = random.uniform(0, self.get_pvalue_WS_geom(0))
                else:
                    WS_p =  self.get_pvalue_WS_geom(waiting_time)
                self.WS_pvalues += [ WS_p ]
                H_p = 1 - (1 - self.homo_rate)**(waiting_time+1)
                return WS_p,H_p
        else:
            if waiting_time > 0:
                WE_p = self.get_pvalue_WE(waiting_time)
                HE_p = self.get_pvalue_HE(waiting_time)
                S_p = self.get_pvalue_seasonal(waiting_time)
                H_p =self.get_pvalue_homo(waiting_time)
                WS_p = self.get_pvalue_WS(waiting_time)
                return WE_p, HE_p, WS_p, S_p, H_p
            else:
                return '','','','',''





    def get_pvalue_WE(self, waiting_time):
        EXP1 =  self.WE_lambda * waiting_time + (1-(self.WE_alpha / self.WE_beta) * (exp(-self.WE_beta * waiting_time))) #####CHECK
        return 1 - exp(-EXP1)
    


    def get_pvalue_HE(self, waiting_time):
        term_1 = self.HE_lambda * waiting_time
        current_time = self.transformed_data[-1]
        previous_time = self.transformed_data[-2]
        term_2 = - (self.HE_alpha / self.HE_beta) * sum( [exp(-self.HE_beta * (current_time-self.transformed_data[i])) -1  for i in range(len(self.transformed_data) - 1) ] )
        term_3 = - (self.HE_alpha / self.HE_beta) * sum( [exp(-self.HE_beta * (previous_time-self.transformed_data[i])) -1  for i in range(len(self.transformed_data) - 2) ] )
        EXP1 = term_1 + term_2 - term_3
        return 1 - exp(-EXP1)
    



    def get_pvalue_WS_geom(self, x):
        C = 0
        T = 1
        if len(self.WS_cutoffs) == 0:
            return 1 - (1 - self.WS_rates[0])**(x+1)
        
        if x <= self.WS_cutoffs[0]:
            return 1 - (1 - self.WS_rates[0])**(x+1)
        else:
            T = T * (1 - self.WS_rates[0])**(self.WS_cutoffs[0]+1)
            C += 1 - (1 - self.WS_rates[0])**(self.WS_cutoffs[0]+1)
        for i in range (1 , len ( self.WS_cutoffs ) ):
            if x > self.WS_cutoffs[i]:
                C += T * (1 - (1 - self.WS_rates[i])**( self.WS_cutoffs[i] - self.WS_cutoffs[i-1] ) )
                T = T * ( (1 - self.WS_rates[i])**( self.WS_cutoffs[i] - self.WS_cutoffs[i-1] ) )
            else:
                C += T * (1 - (1 - self.WS_rates[i])**( x - self.WS_cutoffs[i-1] ) )
                return C
        C += T * (1 - (1 - self.WS_rates[-1])**( x - self.WS_cutoffs[-1] ) )
        return C





    def get_pvalue_WS(self, x):
        C = 0
        T = 1

        if len(self.WS_cutoffs) == 0:
            return 1 - exp ( -self.WS_rates[0] * x )
        if x < self.WS_cutoffs[0]:
            return 1 - exp ( -self.WS_rates[0] * x )
        else:
            T = T * ( exp ( -self.WS_rates[0] * self.WS_cutoffs[0] ) )
            C += 1 - exp ( -self.WS_rates[0] * self.WS_cutoffs[0] )

        for i in range (1 , len ( self.WS_cutoffs ) ):
            if x > self.WS_cutoffs[i]:
                C += T * (1 - exp ( - self.WS_rates[i] * ( self.WS_cutoffs[i] - self.WS_cutoffs[i-1] ) ) )
                T = T * ( exp (-self.WS_rates[i] * (self.WS_cutoffs[i] - self.WS_cutoffs[i-1] ) ) )
            else:
                C += (T) * (1 - exp ( -self.WS_rates[i] * (x - self.WS_cutoffs[i-1] ) ) )
                return C
        C += (T) * ( 1 - exp ( -self.WS_rates[-1] * ( x - self.WS_cutoffs[-1] ) ) )
        return C

    def get_pvalue_seasonal(self,x):
        return 1 - exp( -self.seasonal_rate * x) 

    def get_pvalue_homo(self,x):
        return 1 - exp( -self.homo_rate * x) 





#####FUNCTIONS FOR PELT CHANGEPOINT DETECTION FOR SELF EXCITING BEHAVIOUR


def C2(list, censored):
    m = mean(list)
    if m == 0:
        m = 1
    
    p = 1/(m+1)
    log_like = [ t * log(1-p) + log(p) for t in list]
    return -2*sum(log_like)


def C(list,index):
    m = mean(list)
    if m == 0:
        m = 1
    log_like = [log(1/m) - 1/m * t for t in list]
    return -2*sum(log_like)


def step_function(times):
    n = len( times)
    PP = times
    cps = [0]
    rates = []
    cutoffs = []
    while cps[ -1 ] < n:
        j = cps[ -1 ]

        if j == 0:
            start = 0
        else:
            start = PP[ j - 1 ]
        max_lhd = 0
        for i in range( j, len( PP ) ):
            if PP[i] != start:
                lhd = (i+1-j) / ((PP[i] - start))
            else:
                lhd = 999999
            if lhd >= max_lhd:# and P[i]!= 0:
                l = i
                max_lhd = lhd
        cps += [l+1]
        rates += [ max_lhd ]

    return [PP[i-1] for i in cps], rates, cps


def PELT_MLE(list1, beta, geom):
    possible_steps = step_function(list1)[2]
    #possible_steps = [i + 1 for i in possible_steps[1:]]#i+1
    possible_steps += [0]
    possible_steps += [len(list1)-1]
    possible_steps +=  [min(i for i in range(len(list1)) if list1[i] != 0)  ]
    possible_steps = list(set(possible_steps))
    possible_steps.sort()

    cp=[[0]]
    R = [0]
    list1+=[0]
    list1.sort()
    F = [-beta]
    ms =[0]
    D=diff(list1)
    possible_steps.remove(0)


    for i in range(1,len(possible_steps)):
        if geom:
            F_r = [F[j] + C2(D[possible_steps[j]:possible_steps[i]],len(list1)-possible_steps[i]) + beta for j in R]
        else:
            F_r = [F[j] + C(D[possible_steps[j]:possible_steps[i]],0) + beta for j in R]
        tau = argmin(F_r)
        F += [F_r[tau]]


        R= [ p for p in range(len(possible_steps)) if possible_steps[p] <= possible_steps[i]]
        cp+=[cp[R[tau]]+[possible_steps[R[tau]]]]
    return cp[-1]


def step_function_PELT_MLE(times , geom):
    dy = diff( times )
    beta = 2 * log(len(times))
    sdy = [0] + sorted( dy )
    n = len( sdy )
    sdy.sort()
    Di =  list(diff( sdy ))
    
    P = [(Di[i]) * (n-i)  for i in range(len(Di))]
    PP = list((cumsum(P)))
    cps = (PELT_MLE(PP,beta, geom))
    
    



    cps+=[n]
    rates = []
    if geom:
        
        start_index = 0
        if (sdy[cps[1]]) == 0:
            rate =  ((cps[1]-cps[0]) / (n-cps[0]))
            rates += [rate]
            start_index = 1
        
        for i in range(start_index,len(cps)-1):
            m = mean(P[cps[i]:cps[i+1]])
            censored = n - cps[i+1]
            p = (1 / (1 + m))
            rate = p
            rates += [rate]
    else:
        for i in range(len(cps)-1):
            rate =(cps[i+1]-cps[i])/( sum (P[cps[i]:cps[i+1]]) )
            rates += [rate]


    cutoffs = []
    k=0
    cps.remove(0)
    cps.remove(n)

    return [sdy[i-1] for i in cps], rates, cps


def get_cdf(P):
    P.sort()
    cdfx=[i/10000 for i in range(10000)]
    cdfy =[]
    k=0
    for i in cdfx:
        while P[k] < i:
            k+=1
            if k>=len(P):
                break
        cdfy +=[k/len(P)]
        if k>=len(P):
            break
    while len(cdfy) < len(cdfx):
        cdfy+=[1]
    return cdfx,cdfy








