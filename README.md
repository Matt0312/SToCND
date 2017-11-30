# SToCND

### Python code for modelling seasonality and self-exciting behaviour in a sequence of event times

###Verion 1.0 uploaded of 1st December 2017

###Usage ./Network_model.py

### ./Network_model.py -h


### Data Format, the data should come in the form of an event time and a user separated by a comma. An example is shown in LANL_data.txt

### Output For each event in the test data output is. 
###For the discrete time mode: Time, Wold Step pvalue, Constant pvalue
###For the continuous model: Time, Wold Exponential pvalue, Hawkes Exponential pvalue, Wold Step pvalue, Seasonal Pvalue (0 if seaonal not implemented), (Constant pvalue)

###Users.txt is an exmple of a list of different users that will be modelled sparately.

###TEST EXAMPLE cat LANL_data.txt |./Network_model_git.py -t 28 -d 0 -u ./Users.txt

##Documentation Some brief description of the algorithm and the model provided in the Documentation subfolder but mostly still under development, contact authors with any inquiries.
