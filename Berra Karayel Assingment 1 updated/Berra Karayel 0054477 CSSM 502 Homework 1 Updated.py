#!/usr/bin/env python
# coding: utf-8

# ### Berra Karayel 0054477 CSSM 502 Homework 1 Updated

# In[18]:


import random
from datetime import datetime


# In[19]:


class Portfolio():
    def __init__(self):
        self.cash=0.0
        self.stocks={}
        self.mutualFunds={}
        self.log={}
        self.addtransaction("Your portfolio has been created!")
        
    def __str__(self):
        return "Cash: {} \n Stocks: {} \n Mutual Funds: {}".format(self.cash, self.stocks, self.mutualFunds)
        
    def history(self):
        return self.log
            
    def addtransaction(self,log_info):
        current_time=datetime.now()
        logtime = current_time.strftime("%d/%m/%Y %H:%M:%S")
        self.log["{} - {}".format(len(self.log),logtime)] = log_info
        
    def addCash(self,amount):
        self.cash=self.cash+amount
        self.addtransaction("{} $ added to your account. , Your Current Cash Balance: {} $".format(amount,self.cash ))
    
    def WithdrawCash(self,amount):
        if self.cash<amount:
            print('Your balance is insufficient! Enter a new amount: ')
        else:
            self.cash=-amount
            self.addtransaction("{} $ withdrawn from your account. , Your Current Cash Balance: {} $".format(amount,self.cash ))
            
    def buyStock(self, quantity, stock):
         cost = stock.price * quantity
         if cost > self.cash:
            print("You do not have enough balance for this transaction.")
         else:
             self.cash -= cost
             if stock.ticker in self.stocks:
                presentAmount = self.stocks[stock.ticker][0]
                cost_avg = ((presentAmount * self.stocks[stock.ticker][1]) + 
                                (quantity * stock.price)) / (presentAmount + quantity)
                self.stocks[stock.ticker] = (self.stocks[stock.ticker][0] + quantity, cost_avg)
             else:
                self.stocks[stock.ticker] = (quantity, stock.price)
             print("You have bought {} shares of {} stock. ".format(quantity,stock.ticker))
             self.addtransaction("You have bought {} shares of {} for {} $. Your Current Cash Balance: ${}$".format(quantity,stock.ticker,cost,self.cash))
             
             
    def sellStock(self, ticker, quantity):
        if ticker in self.stocks.keys():
            if self.stocks[ticker][0] >= quantity:
                received_cash= random.uniform(0.5,1.5) * quantity * self.stocks[ticker][1]
                self.cash += received_cash
                self.stocks[ticker] = (self.stocks[ticker][0] - quantity, self.stocks[ticker][1])
                print("You have sold {} shares of {} stock. ".format(quantity,ticker))
                self.addtransaction("You have sold {} shares of {} for {} $. Your Current Cash Balance is {}$".format(quantity,ticker,received_cash,self.cash))
            else: #wrong amount
                print("No suffficient stock has been found for this transaction")
        else:
            print("Please enter a valid stock ticker.")
         
    def buyMutualFund(self, quantity, mf):
        mf_cost = quantity #one share is equal to 1 dolar
        if mf_cost > self.cash:
            print("You do not have enough balance for this transaction. You have {} $ available. / You need {} $.".format(self.cash,mf_cost))
        else:
            self.cash -= mf_cost
            if mf.ticker in self.mutualFunds:
                self.mutualFunds[mf.ticker] = (self.mutualFunds[mf.ticker][0] + quantity, mf_cost)
            else:
                self.mutualFunds[mf.ticker] = (quantity, mf_cost)
            print("You have bought {} shares of {} mutual fund. ".format(quantity,mf.ticker))
            self.addtransaction("You have bought {} shares of {} for {} $. You Current Cash Balance is {}$".format(quantity,mf.ticker,mf_cost ,self.cash ))

    
    def sellMutualFund(self, ticker, amount):
        if ticker in self.mutualFunds.keys():
            if self.mutualFunds[ticker][0] >= amount:
                received_amount = random.uniform(0.9,1.2) * amount * self.mutualFunds[ticker][1]
                self.cash += received_amount
                self.mutualFunds[ticker] = (self.mutualFunds[ticker][0] - amount, self.mutualFunds[ticker][1])
                print("You have sold {} shares of {} mutual fund. ".format(amount,ticker))
                self.addtransaction("You have sold {} shares of {} for {} $ . Your Current Cash Balance: {} $".format(amount,ticker,received_amount,self.cash))
            else: 
                print("You don't have enough funds")
        else:
            print("Please enter a valid fund ticker.") 
    
    
    class Stock(object):
        def __init__(self, price, ticker):
          self.price = price
          self.ticker = ticker
         
    class MutualFund(object):
        def __init__(self, ticker):
            self.ticker = ticker


# In[20]:


portfolio = Portfolio() #Creates a new portfolio
portfolio.addCash(300.50) #Adds cash to the portfolio
s = Stock(20, "HFH") #Create Stock with price 20 and symbol "HFH"
portfolio.buyStock(5, s) #Buys 5 shares of stock s
mf1 = MutualFund("BRT") #Create MF with symbol "BRT"
mf2 = MutualFund("GHT") #Create MF with symbol "GHT"
portfolio.buyMutualFund(10.3, mf1) #Buys 10.3 shares of "BRT"
portfolio.buyMutualFund(2, mf2) #Buys 2 shares of "GHT"
print(portfolio) #Prints portfolio
#cash: $140.50
#stock: 5 HFH
#mutual funds: 10.33 BRT
# 2 GHT
portfolio.sellMutualFund("BRT", 3) #Sells 3 shares of BRT
portfolio.sellStock("HFH", 1) #Sells 1 share of HFH
portfolio.WithdrawCash(50) #Removes $50
portfolio.history() #Prints a list of all transactions
#ordered by time

