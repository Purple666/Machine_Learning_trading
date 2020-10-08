from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import MetaTrader5 as mt5
import time

import os
import IPython
import IPython.display
import seaborn as sns

from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression
import pickle


#--------------------------------------- Part 1: set up connection ------------------------------------#
# connect to MetaTrader 5
if not mt5.initialize():
    print("initialize() failed")
    mt5.shutdown()


#--------------------------------------- Part 2: Get & Divide data -------------------------------------#
# ask for symbol
print('Enter a currency pair:')
curr_pair = input().upper()

# get currency pair rates
def update_info(curr_pair):
    global X_train, X_test, y_train, y_test, X_lately, df
    COIN_rates = mt5.copy_rates_from_pos(curr_pair, mt5.TIMEFRAME_M1, 0, 240)
    df = pd.DataFrame(COIN_rates)

    try:
        df.set_index('time',drop=True,inplace=True)
        df.drop(['spread','real_volume','open'],axis=1,inplace=True)
    except:
        pass
    forecast_col = 'close'
    forecast_out = 60
    df['label'] = df[forecast_col].shift(-forecast_out)

    X = np.array(df.drop(['label'], 1))
    X = preprocessing.scale(X)
    X_lately = X[-forecast_out:]
    y = np.array(df['label'])

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3)
    
    return X_train, X_test, y_train, y_test, X_lately, df


#-------------------------------------- Part 3: Define Functions --------------------------------------#
count_wins = 0
count_losses = 0

#-------------------------------------- WIN, LOSS, BUY, SELL  --NORMAL STUFF
def win(count_wins,count_losses,curr_pair):
    mt5.Close(curr_pair) # said ticket number is used in here
    print(f"WIN! New equitiy is: {mt5.account_info().equity}\ncalling __ini()__")
    ini(count_wins,count_losses,curr_pair)
    
def loss(count_wins,count_losses,curr_pair):
    mt5.Close(curr_pair) # said ticket number is used in here
    print(f"LOSS! New equitiy is: {mt5.account_info().equity}\ncalling __ini()__ in 2 seconds")
    time.sleep(2)
    ini(count_wins,count_losses,curr_pair)

def buy(count_wins,count_losses,j,curr_pair):
    bid = mt5.symbol_info(curr_pair).bid
    ask = mt5.symbol_info(curr_pair).ask
    buy_buy = mt5.Buy(curr_pair,0.02) #will open a new BUYING order
    print("bid: {}, ask: {}".format(bid,ask))
    check(1,count_wins,count_losses,j,curr_pair)

def sell(count_wins,count_losses,j,curr_pair):
    bid = mt5.symbol_info(curr_pair).bid
    ask = mt5.symbol_info(curr_pair).ask
    sell_sell = mt5.Sell(curr_pair,0.02) #will open a new SELLING order
    print("bid: {}, ask: {}".format(bid,ask))
    check(-1,count_wins,count_losses,j,curr_pair)

#--------------------------------------  CHECKER OF PROFITS (OR LOSSES)
def check(n,count_wins,count_losses,j,curr_pair):
    if n > 0:
        txt = "buy"
    else:
        txt="sell"
    while True:
        spread = round(mt5.symbol_info(curr_pair).ask - mt5.symbol_info(curr_pair).bid,7)
        print("\r",txt,", current profit: ",mt5.account_info().profit, ", spread: ",round(spread*10,4), flush=True)
        if mt5.account_info().profit >0.02:
            count_wins += 1
            win(count_wins,count_losses,curr_pair)
            break
        elif mt5.account_info().profit <= -0.1:
            count_losses += 1
            loss(count_wins,count_losses,curr_pair)
            break
        else:
            pass
        k = (1+j)*30
        print("checking price in: ",k," seconds.")
        time.sleep(k)
    return count_wins,count_losses

#--------------------------------------  INI OF THE BEGINNINGS
def ini(count_wins,count_losses,curr_pair):
    print(f"\ncurrent score is: {count_wins} wins, {count_losses} losses. Win percentage over total = {round(count_wins/(count_wins+count_losses+0.0000001),2)}")
    if not mt5.initialize():
        print("initialize() failed")
    pickle_my_pickles(count_wins,count_losses,curr_pair)

#--------------------------------------  MAGIC PICKLES
def pickle_my_pickles(count_wins,count_losses,curr_pair):
    global df_full_full, my_models, df_2, buysy, sellsy, j
    update_info(curr_pair)
    df_full_full = []
    my_models = ['ARDRegression_clf.pickle', 'HuberRegressor_clf.pickle',
                 'LinearRegression_clf.pickle', 'PassiveAggressiveRegressor_clf.pickle',
                 'BayesianRidge_clf.pickle', 'RANSACRegressor_clf.pickle', 'RidgeCV_clf.pickle',
                 'SGDRegressor_clf.pickle', 'TheilSenRegressor_clf.pickle','SupportVectorRegression_clf.pickle']
    # sometimes out due to being outliers: SupportVectorRegression_clf.pickle, 'PassiveAggressiveRegressor_clf.pickle'

    for model in my_models:
        pickle_in = open(model,'rb')
        clf = pickle.load(pickle_in)
        forecast_set = clf.predict(X_lately)
        df_full_full.append(forecast_set)
        
    df_2 = pd.DataFrame(df_full_full).T
    df_2 = df_2.rename(columns=lambda x: my_models[x][:-11])
    
    magic = df_2.describe().T
    mnm = magic['std'].mean()*0.5
    buysy = df.iloc[-1]['close'] + mnm
    sellsy = df.iloc[-1]['close'] - mnm
    print(buysy,sellsy)
    
    spread = round(mt5.symbol_info(curr_pair).ask - mt5.symbol_info(curr_pair).bid,7)
    df_2['weird_mean'] = df_2.T.mean()
    df_2[:30].plot(figsize=(16,9))
    xc = [0,30]
    yc = [df.iloc[-1]['close'],df.iloc[-1]['close']]
    y_up = [round((df.iloc[-1]['close'])+mnm,3),round((df.iloc[-1]['close'])+mnm,3)]
    y_down = [round((df.iloc[-1]['close'])-mnm,3),round((df.iloc[-1]['close'])-mnm,3)]
    plt.plot(xc, yc)
    plt.plot(xc, y_up)
    plt.plot(xc, y_down)
    plt.legend(loc=1)
    print("Take Profit for BUY: ",round(buysy,3))
    print("last value was: ",df.iloc[-1]['close'])
    print("Take Profit for SELL: ",round(sellsy,3))
#    plt.show()

    try:
        for i in range(12):
            j = i
            print(f"\nbuysy: {buysy}, weird mean: {df_2.weird_mean[i]},sellsy: {sellsy}")
            if df_2.weird_mean[i] < sellsy:
                print("Selling!")
                sell(count_wins,count_losses,j,curr_pair)
                break
            elif df_2.weird_mean[i] > buysy:
                print("Buying!")
                buy(count_wins,count_losses,j,curr_pair)
                break
            else:
                print(df_2.weird_mean[i],"no good, passing, please wait 5 seconds")
                time.sleep(5)
                pass
    except:
        pickle_my_pickles(count_wins,count_losses,curr_pair)

    return count_wins,count_losses, plt.show(), i


#-------------------------------------------  Part 4: Profits?  ------------------------------------------#
ini(0,0,curr_pair)
