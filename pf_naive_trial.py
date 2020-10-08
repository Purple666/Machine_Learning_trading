from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import MetaTrader5 as mt5
import time

import os
import datetime

import IPython
import IPython.display
import seaborn as sns


# connect to MetaTrader 5
if not mt5.initialize():
    print("initialize() failed")
    mt5.shutdown()

count_wins = 0
count_losses = 0

def win(n,count_wins,count_losses):
    mt5.Close('USDJPY') # said ticket number is used in here
    print(f"WIN! New equitiy is: {mt5.account_info().equity}\ncalling 'ini()")
    ini(n,count_wins,count_losses)
    
def loss(n,count_wins,count_losses):
    mt5.Close('USDJPY') # said ticket number is used in here
    print(f"LOSS! New equitiy is: {mt5.account_info().equity}\ncalling 'ini()")
    n = -n
    ini(n,count_wins,count_losses)

def buy(count_wins,count_losses):
    bid = mt5.symbol_info("USDJPY").bid
    ask = mt5.symbol_info("USDJPY").ask
    buy_buy = mt5.Buy('USDJPY',0.02) #will open a new BUYING order
    print("bid: {}, ask: {}".format(bid,ask))
    check(1,count_wins,count_losses)

def sell(count_wins,count_losses):
    bid = mt5.symbol_info("USDJPY").bid
    ask = mt5.symbol_info("USDJPY").ask
    sell_sell = mt5.Sell('USDJPY',0.02) #will open a new SELLING order
    print("bid: {}, ask: {}".format(bid,ask))
    check(-1,count_wins,count_losses)
    
def check(n,count_wins,count_losses):
    if n > 0:
        txt = "buy"
    else:
        txt="sell"
    while True:
        spread = round(mt5.symbol_info("USDJPY").ask - mt5.symbol_info("USDJPY").bid,7)
        print("\r",txt,", current profit: ",mt5.account_info().profit, ", spread: ",round(spread*10,4), flush=True)
        if mt5.account_info().profit > 0.03:
            count_wins += 1
            win(n,count_wins,count_losses)
            break
        elif mt5.account_info().profit < -0.21:
            count_losses += 1
            loss(n,count_wins,count_losses)
            break
        else:
            pass
        time.sleep(0.2)
    return count_wins,count_losses

def ini(n,count_wins,count_losses):
    print(f"current score is: {count_wins} wins, {count_losses} losses. Win percentage over total = {round(count_wins/(count_wins+count_losses+0.0000001),2)*100}%")
    if not mt5.initialize():
        print("initialize() failed")
    if n > 0:
        buy(count_wins,count_losses)
    else:
        sell(count_wins,count_losses)
    
    # define ini() which is a few models of unsupervised ML
    # main purpose, if ini() says price is going to be below current price... SELL
    # if ini() says price is going to be above current price.. BUY
ini(0,0,0)
