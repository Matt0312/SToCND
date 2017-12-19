# SToCND

### Python code for modelling seasonality and self-exciting behaviour in a sequence of event times

###Verion 1.0 uploaded of 1st December 2017

###Usage ./Network_model.py -h

###Documentation A description of the algorithms are given in the manuscript Statistical Modelling of Computer Network Traffic Event Times

### Output For each event in the test data output is. 
###For the discrete time mode: Time, Wold Step pvalue, Constant pvalue
###For the continuous model: Time, Wold Exponential pvalue, Hawkes Exponential pvalue, Wold Step pvalue, Seasonal Pvalue (0 if seaonal not implemented), (Constant pvalue)

###The Hawkes Step model was not included because of the computational costs. Please contact the authour if you want to implement this approach.


### Data Format, the data should come in the form of an event time and a user separated by a comma. An example is shown in LANL_data.txt

###Users.txt is an example of a list of different users that will be modelled sparately.

###Minimum length of training data is set to 200


###TEST EXAMPLE cat LANL_data.txt |./Network_model_git.py -t 28 -d 0 -u Users.txt -g


