#! /usr/bin/env python

import datetime as dt
import sys
import json
import argparse
import numpy as np
from numpy import mean
import random
import matplotlib as mpl
import matplotlib.pyplot as plt
from Modelling_Network_Data_git import User



"""assumes seasonal period is one day"""


secsinday = 86400

"""defaults"""
workweek = 7
trainingperiod = 14
Nine80 = False
Separate_users = False
Mutual_Exciting = False
seasonal = False
geom = False
JSON = False

"""Parses command-line arguments and executes."""
formatter = argparse.ArgumentDefaultsHelpFormatter
commandlineargs = argparse.ArgumentParser(formatter_class = formatter, description = "The program reads from stdin and outputs to stdout.")

commandlineargs.add_argument('-t', nargs = 1, dest = "trainingperiod",
                             default = trainingperiod,
                             help ='training period in days (must be a multiple of work week)',
                             type = int)




commandlineargs.add_argument('-u', nargs = 1,  dest = "separate_users", default = Separate_users, metavar = 'PATH', help
                             = 'Set if you want to model each user separately include the list of users as a text file with each user on a separate line' )


commandlineargs.add_argument('-S', action = 'store_true', dest = "Seasonality",
                             default = seasonal, help = 'Set for seasonality')

commandlineargs.add_argument('-g',action = 'store_true', dest = "geometric",
                             default = geom, help = 'Set for geometric model, specifies no seasonality')


commandlineargs.add_argument('-d', nargs = 1, dest = "firstday",
                             help = 'time at midnight for the first day', required = True,
                             type = int)

args = commandlineargs.parse_args()





if args.firstday:
    #ensure midnight on the firstday
    day = int(dt.date.fromtimestamp(float(args.firstday[0])).strftime('%s'))


else:
    print >> sys.stderr, "Must specify unix time for first day."

if args.Seasonality:
    seasonal = True

if args.trainingperiod:
    trainingperiod = int(args.trainingperiod[0])
    if trainingperiod % workweek != 0 and seasonal:
        print >> sys.stderr, "Training period must be a multiple of workweek."
        sys.exit(1)


if args.separate_users:
    Users = []
    Separate_users = True
    text = open(args.separate_users[0])
    for line in text:
        Users += [line.strip()]



if args.geometric:
    seasonal = False
    geom = True



userhash = {}



for line in sys.stdin:

    if Separate_users:


        fields = line.rstrip('\r\n').split(' ')
        time = round(float(fields[0]),3)


        user = fields[1]
        if user not in Users:
            continue

        


        if user not in userhash:
            userhash[user] = User(day, workweek, trainingperiod, seasonal, geom )
        results = (userhash[user].update_data(time, trainingperiod, workweek, seasonal, geom))
        if results!= '':
            print (time,results)



    else:
        fields = line.rstrip('\r\n').split(' ')
        time = round(float(fields[0]),3)
        
        if len(userhash) == 0:
            userhash['all'] = User(day, workweek, trainingperiod, seasonal, geom)
        
        results = (userhash['all'].update_data(time, trainingperiod, workweek, seasonal, geom))
        if results!='':
            print (time,results)



