# ML_autotrading
implement several Machine Learning algorithms to automate forex trading

#--------------------Logic behind this--------------------#
The main idea is to set up a financial cushion against "unpredictable times" a.k.a. "hello 2020"

The logic behind is simple enough:

1. Check if metatrader5 is open, else open it.
2. Load data from chosen currency pair.
3. call "ini()" function that begins the cycle.
4. check data with ML algorithms to make a decision: buy or sell
5. check operation every n seconds
6. register win/loss and call "ini()" 

steps 3 to 6 are the main logical loop.

Files in this repository are:
  pf_ML_trial.py------- Main program that uses the Machine Learning algorithms
  pf_naive_trial.py---- A naive approach answering the question "what if I buy/sell and repeat if WIN, do the opposite if LOSS?"
  pickle_training.py--- File to train pickles with n_samples

#------------------------Next Steps-----------------------#

fine tuning of the algorithms
improve "check()" function
keep experimenting with different ratios of profit/loss
improve risk managment
