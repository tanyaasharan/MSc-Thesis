# -*- coding: utf-8 -*-
#
# BSE: The Bristol Stock Exchange
#
# Version 1.5; 02 Jan 2021 (the final version before switch to BSE2.x)
# Version 1.4; 26 Oct 2020 (Python 3.x)
# Version 1.3; July 21st, 2018 (Python 2.x)
# Version 1.2; November 17th, 2012 (Python 2.x)
#
# Copyright (c) 2012-2022, Dave Cliff
#
#
# ------------------------
#
# MIT Open-Source License:
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial
# portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
# LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# ------------------------
#
#
#
# BSE is a very simple simulation of automated execution traders
# operating on a very simple model of a limit order book (LOB) exchange
#
# major simplifications in this version:
#       (a) only one financial instrument being traded
#       (b) traders can only trade contracts of size 1 (will add variable quantities later)
#       (c) each trader can have max of one order per single orderbook.
#       (d) traders can replace/overwrite earlier orders, and/or can cancel
#       (d) simply processes each order in sequence and republishes LOB to all traders
#           => no issues with exchange processing latency/delays or simultaneously issued orders.
#
# NB this code has been written to be readable/intelligible, not efficient!

import sys
import math
import random
import time as chrono
import numpy as np
from datetime import datetime
import networkx as nx


from opinion_model import relative_disagreement_step
import o_prde
from graph import klemm_eguiluz_graph


# a bunch of system constants (globals)
bse_sys_minprice = 1                    # minimum price in the system, in cents/pennies
bse_sys_maxprice = 500                  # maximum price in the system, in cents/pennies
bse_sys_tape_len = 20                   # maximum number of events recorded on tape (most recent are retained)
bse_sys_blotter_len = 20                # maximum number of events recorded on tape (most recent are retained)
bse_sys_PRSH_landscape_mapper = True    # boolean that says whether PRSH is SHC or landscape mapper
# ticksize should be a param of an exchange (so different exchanges have different ticksizes)
# todo: change this, so ticksize no longer global. NB could do same for tape_len and blotter_len
ticksize = 1  # minimum change in price, in cents/pennies

_useNetworkCaveman = False


# an Order/quote has a trader id, a type (buy/sell) price, quantity, timestamp, and unique i.d.
class Order:

    def __init__(self, tid, otype, price, qty, time, qid):
        self.tid = tid  # trader i.d.
        self.otype = otype  # order type
        self.price = price  # price
        self.qty = qty  # quantity
        self.time = time  # timestamp
        self.qid = qid  # quote i.d. (unique to each quote)

    def __str__(self):
        return '[%s %s P=%03d Q=%s T=%5.2f QID:%d]' % \
               (self.tid, self.otype, self.price, self.qty, self.time, self.qid)


# Orderbook_half is one side of the book: a list of bids or a list of asks, each sorted best-first

class Orderbook_half:

    def __init__(self, booktype, worstprice):
        # booktype: bids or asks?
        self.booktype = booktype
        # dictionary of orders received, indexed by Trader ID
        self.orders = {}
        # limit order book, dictionary indexed by price, with order info
        self.lob = {}
        # anonymized LOB, lists, with only price/qty info
        self.lob_anon = []
        # summary stats
        self.best_price = None
        self.best_tid = None
        self.worstprice = worstprice
        self.session_extreme = None    # most extreme price quoted in this session
        self.n_orders = 0  # how many orders?
        self.lob_depth = 0  # how many different prices on lob?

    def anonymize_lob(self):
        # anonymize a lob, strip out order details, format as a sorted list
        # NB for asks, the sorting should be reversed
        self.lob_anon = []
        for price in sorted(self.lob):
            qty = self.lob[price][0]
            self.lob_anon.append([price, qty])

    def build_lob(self):
        lob_verbose = False
        # take a list of orders and build a limit-order-book (lob) from it
        # NB the exchange needs to know arrival times and trader-id associated with each order
        # returns lob as a dictionary (i.e., unsorted)
        # also builds anonymized version (just price/quantity, sorted, as a list) for publishing to traders
        self.lob = {}
        for tid in self.orders:
            order = self.orders.get(tid)
            price = order.price
            if price in self.lob:
                # update existing entry
                qty = self.lob[price][0]
                orderlist = self.lob[price][1]
                orderlist.append([order.time, order.qty, order.tid, order.qid])
                self.lob[price] = [qty + order.qty, orderlist]
            else:
                # create a new dictionary entry
                self.lob[price] = [order.qty, [[order.time, order.qty, order.tid, order.qid]]]
        # create anonymized version
        self.anonymize_lob()
        # record best price and associated trader-id
        if len(self.lob) > 0:
            if self.booktype == 'Bid':
                self.best_price = self.lob_anon[-1][0]
            else:
                self.best_price = self.lob_anon[0][0]
            self.best_tid = self.lob[self.best_price][1][0][2]
        else:
            self.best_price = None
            self.best_tid = None

        if lob_verbose:
            print(self.lob)

    def book_add(self, order):
        # add order to the dictionary holding the list of orders
        # either overwrites old order from this trader
        # or dynamically creates new entry in the dictionary
        # so, max of one order per trader per list
        # checks whether length or order list has changed, to distinguish addition/overwrite
        # print('book_add > %s %s' % (order, self.orders))

        # if this is an ask, does the price set a new extreme-high record?
        if (self.booktype == 'Ask') and ((self.session_extreme is None) or (order.price > self.session_extreme)):
            self.session_extreme = int(order.price)

        # add the order to the book
        n_orders = self.n_orders
        self.orders[order.tid] = order
        self.n_orders = len(self.orders)
        self.build_lob()
        # print('book_add < %s %s' % (order, self.orders))
        if n_orders != self.n_orders:
            return 'Addition'
        else:
            return 'Overwrite'

    def book_del(self, order):
        # delete order from the dictionary holding the orders
        # assumes max of one order per trader per list
        # checks that the Trader ID does actually exist in the dict before deletion
        # print('book_del %s',self.orders)
        if self.orders.get(order.tid) is not None:
            del (self.orders[order.tid])
            self.n_orders = len(self.orders)
            self.build_lob()
        # print('book_del %s', self.orders)

    def delete_best(self):
        # delete order: when the best bid/ask has been hit, delete it from the book
        # the TraderID of the deleted order is return-value, as counterparty to the trade
        best_price_orders = self.lob[self.best_price]
        best_price_qty = best_price_orders[0]
        best_price_counterparty = best_price_orders[1][0][2]
        if best_price_qty == 1:
            # here the order deletes the best price
            del (self.lob[self.best_price])
            del (self.orders[best_price_counterparty])
            self.n_orders = self.n_orders - 1
            if self.n_orders > 0:
                if self.booktype == 'Bid':
                    self.best_price = max(self.lob.keys())
                else:
                    self.best_price = min(self.lob.keys())
                self.lob_depth = len(self.lob.keys())
            else:
                self.best_price = self.worstprice
                self.lob_depth = 0
        else:
            # best_bid_qty>1 so the order decrements the quantity of the best bid
            # update the lob with the decremented order data
            self.lob[self.best_price] = [best_price_qty - 1, best_price_orders[1][1:]]

            # update the bid list: counterparty's bid has been deleted
            del (self.orders[best_price_counterparty])
            self.n_orders = self.n_orders - 1
        self.build_lob()
        return best_price_counterparty


# Orderbook for a single instrument: list of bids and list of asks

class Orderbook(Orderbook_half):

    def __init__(self):
        self.bids = Orderbook_half('Bid', bse_sys_minprice)
        self.asks = Orderbook_half('Ask', bse_sys_maxprice)
        self.tape = []
        self.tape_length = 100      # max number of records on tape (so we can do millions of orders without crashing)
        self.quote_id = 0           # unique ID code for each quote accepted onto the book
        self.ext_opinions = []      # contains all external opinions to be smoothed
        self.ext_opinions_length = 30 # 10 default for 10 seconds of external opinions to be stored
        self.ext_opinion = 0


# Exchange's internal orderbook

class Exchange(Orderbook):

    def add_order(self, order, verbose):
        # add a quote/order to the exchange and update all internal records; return unique i.d.
        order.qid = self.quote_id
        self.quote_id = order.qid + 1
        # if verbose : print('QUID: order.quid=%d self.quote.id=%d' % (order.qid, self.quote_id))
        if order.otype == 'Bid':
            response = self.bids.book_add(order)
            best_price = self.bids.lob_anon[-1][0]
            self.bids.best_price = best_price
            self.bids.best_tid = self.bids.lob[best_price][1][0][2]
        else:
            response = self.asks.book_add(order)
            best_price = self.asks.lob_anon[0][0]
            self.asks.best_price = best_price
            self.asks.best_tid = self.asks.lob[best_price][1][0][2]
        return [order.qid, response]

    def del_order(self, time, order, verbose):
        # delete a trader's quot/order from the exchange, update all internal records
        if order.otype == 'Bid':
            self.bids.book_del(order)
            if self.bids.n_orders > 0:
                best_price = self.bids.lob_anon[-1][0]
                self.bids.best_price = best_price
                self.bids.best_tid = self.bids.lob[best_price][1][0][2]
            else:  # this side of book is empty
                self.bids.best_price = None
                self.bids.best_tid = None
            cancel_record = {'type': 'Cancel', 'time': time, 'order': order}
            self.tape.append(cancel_record)
            # NB this just throws away the older items on the tape -- could instead dump to disk
            # right-truncate the tape so it keeps only the most recent items
            self.tape = self.tape[-self.tape_length:]

        elif order.otype == 'Ask':
            self.asks.book_del(order)
            if self.asks.n_orders > 0:
                best_price = self.asks.lob_anon[0][0]
                self.asks.best_price = best_price
                self.asks.best_tid = self.asks.lob[best_price][1][0][2]
            else:  # this side of book is empty
                self.asks.best_price = None
                self.asks.best_tid = None
            cancel_record = {'type': 'Cancel', 'time': time, 'order': order}
            self.tape.append(cancel_record)
            # NB this just throws away the older items on the tape -- could instead dump to disk
            # right-truncate the tape so it keeps only the most recent items
            self.tape = self.tape[-self.tape_length:]
        else:
            # neither bid nor ask?
            sys.exit('bad order type in del_quote()')

    def process_order2(self, time, order, verbose):
        # receive an order and either add it to the relevant LOB (ie treat as limit order)
        # or if it crosses the best counterparty offer, execute it (treat as a market order)
        oprice = order.price
        counterparty = None
        [qid, response] = self.add_order(order, verbose)  # add it to the order lists -- overwriting any previous order
        order.qid = qid
        if verbose:
            print('QUID: order.quid=%d' % order.qid)
            print('RESPONSE: %s' % response)
        best_ask = self.asks.best_price
        best_ask_tid = self.asks.best_tid
        best_bid = self.bids.best_price
        best_bid_tid = self.bids.best_tid
        if order.otype == 'Bid':
            if self.asks.n_orders > 0 and best_bid >= best_ask:
                # bid lifts the best ask
                if verbose:
                    print("Bid $%s lifts best ask" % oprice)
                counterparty = best_ask_tid
                price = best_ask  # bid crossed ask, so use ask price
                if verbose:
                    print('counterparty, price', counterparty, price)
                # delete the ask just crossed
                self.asks.delete_best()
                # delete the bid that was the latest order
                self.bids.delete_best()
        elif order.otype == 'Ask':
            if self.bids.n_orders > 0 and best_ask <= best_bid:
                # ask hits the best bid
                if verbose:
                    print("Ask $%s hits best bid" % oprice)
                # remove the best bid
                counterparty = best_bid_tid
                price = best_bid  # ask crossed bid, so use bid price
                if verbose:
                    print('counterparty, price', counterparty, price)
                # delete the bid just crossed, from the exchange's records
                self.bids.delete_best()
                # delete the ask that was the latest order, from the exchange's records
                self.asks.delete_best()
        else:
            # we should never get here
            sys.exit('process_order() given neither Bid nor Ask')
        # NB at this point we have deleted the order from the exchange's records
        # but the two traders concerned still have to be notified
        if verbose:
            print('counterparty %s' % counterparty)
        if counterparty is not None:
            # process the trade
            if verbose: print('>>>>>>>>>>>>>>>>>TRADE t=%010.3f $%d %s %s' % (time, price, counterparty, order.tid))
            transaction_record = {'type': 'Trade',
                                  'time': time,
                                  'price': price,
                                  'party1': counterparty,
                                  'party2': order.tid,
                                  'qty': order.qty
                                  }
            self.tape.append(transaction_record)
            # NB this just throws away the older items on the tape -- could instead dump to disk
            # right-truncate the tape so it keeps only the most recent items
            self.tape = self.tape[-self.tape_length:]


            return transaction_record
        else:
            return None

    # Currently tape_dump only writes a list of transactions (ignores cancellations)
    def tape_dump(self, fname, fmode, tmode):
        dumpfile = open(fname, fmode)
        # dumpfile.write('type, time, price\n')
        for tapeitem in self.tape:
            if tapeitem['type'] == 'Trade':
                dumpfile.write('Trd, %010.3f, %s\n' % (tapeitem['time'], tapeitem['price']))
        dumpfile.close()
        if tmode == 'wipe':
            self.tape = []
            
    def tape_dump2(self, trfile, time):
        """function records mean transactions for each second"""
        trades = []
        for tapeitem in self.tape:
            if tapeitem['type'] == 'Trade':
                trades.append(tapeitem['price'])
        if len(trades) > 0:
            price = np.mean(trades)
            std = np.std(trades)
            trfile.write('%06d, %.2f, %.2f, ' % (time, price, std))
            trfile.write('\n')

    # this returns the LOB data "published" by the exchange,
    # i.e., what is accessible to the traders
    

    def publish_lob(self, time, verbose):
        def micro_price_adjust(public_data):
            """Function to calculate the micro-price adjustment that is the foundation of the external opinion"""
            if public_data['asks']['n'] > 0 and public_data['bids']['n'] > 0:
                pa = public_data['asks']['best']
                pb = public_data['bids']['best']
                qa = public_data['asks']['lob'][0][1]
                qb = public_data['bids']['lob'][-1][1]
                
                mid_price = (pa + pb) / 2
                micro_price = ((pa * qb) / (qa + qb)) + ((pb * qa) / (qa + qb))
                
                
                #micro-price adjustment (delta_m)
                delta_m = micro_price - mid_price
            else:
                delta_m = 0
            #print(delta_m)
            
            return delta_m
        
        def update_external_opinions(self, public_data):
            """function converts the micro-price adjustment to external opinion"""
            delta_m = public_data['delta_m']
            X_e = math.tanh(delta_m * 0.2)
            #print(X_e)

            if len(self.ext_opinions) == self.ext_opinions_length:
                self.ext_opinions.pop(0)
            self.ext_opinions.append(X_e)
            
            return self.ext_opinions
        
        def get_inf(self):
            """function gets the influence of external opinion"""
            if len(self.ext_opinions) < self.ext_opinions_length:
                return 0.1
            else:
                sigma = np.std(self.ext_opinions)
                noise = 1 / (sigma+1)
                noise = min(noise, 2)
                inf = (abs(self.ext_opinion))**2
                inf = (inf*noise) + 0.1
                
                return inf
            
        
        public_data = {}
        public_data['time'] = time
        public_data['bids'] = {'best': self.bids.best_price,
                               'worst': self.bids.worstprice,
                               'n': self.bids.n_orders,
                               'lob': self.bids.lob_anon}
        public_data['asks'] = {'best': self.asks.best_price,
                               'worst': self.asks.worstprice,
                               'sess_hi': self.asks.session_extreme,
                               'n': self.asks.n_orders,
                               'lob': self.asks.lob_anon}
        public_data['QID'] = self.quote_id
        public_data['tape'] = self.tape
        
        #External opinion info published to all traders
        public_data['delta_m'] = micro_price_adjust(public_data)
        self.ext_opinions = update_external_opinions(self, public_data)
        self.ext_opinion = np.mean(self.ext_opinions)
        public_data['ext_opinions'] = self.ext_opinions
        public_data['ext_opinion'] = self.ext_opinion
        public_data['ext_inf'] = get_inf(self)
        
        if verbose:
            print('publish_lob: t=%d' % time)
            print('BID_lob=%s' % public_data['bids']['lob'])
            # print('best=%s; worst=%s; n=%s ' % (self.bids.best_price, self.bids.worstprice, self.bids.n_orders))
            print('ASK_lob=%s' % public_data['asks']['lob'])
            # print('qid=%d' % self.quote_id)

        return public_data


##################--Traders below here--#############


# Trader superclass
# all Traders have a trader id, bank balance, blotter, and list of orders to execute
class Trader:

    def __init__(self, ttype, tid, balance, params, time):
        self.ttype = ttype  # what type / strategy this trader is
        self.tid = tid  # trader unique ID code
        self.balance = balance  # money in the bank
        self.params = params # parameters/extras associated with this trader-type or individual trader.
        self.blotter = []  # record of trades executed
        self.blotter_length = 100   # maximum length of blotter
        self.orders = []  # customer orders currently being worked (fixed at 1)
        self.n_quotes = 0  # number of quotes live on LOB
        self.birthtime = time  # used when calculating age of a trader/strategy
        self.profitpertime = 0  # profit per unit time
        self.n_trades = 0  # how many trades has this trader done?
        self.lastquote = None  # record of what its last quote was


    def __str__(self):
        return '[TID %s type %s balance %s blotter %s orders %s n_trades %s profitpertime %s]' \
               % (self.tid, self.ttype, self.balance, self.blotter, self.orders, self.n_trades, self.profitpertime)


    def add_order(self, order, verbose):
        # in this version, trader has at most one order,
        # if allow more than one, this needs to be self.orders.append(order)
        if self.n_quotes > 0:
            # this trader has a live quote on the LOB, from a previous customer order
            # need response to signal cancellation/withdrawal of that quote
            response = 'LOB_Cancel'
        else:
            response = 'Proceed'
        self.orders = [order]
        if verbose:
            print('add_order < response=%s' % response)
        return response


    def del_order(self, order):
        # this is lazy: assumes each trader has only one customer order with quantity=1, so deleting sole order
        # todo: CHANGE TO DELETE THE HEAD OF THE LIST AND KEEP THE TAIL
        self.orders = []


    def bookkeep(self, trade, order, verbose, time):

        outstr = ""
        for order in self.orders:
            outstr = outstr + str(order)

        self.blotter.append(trade)  # add trade record to trader's blotter
        self.blotter = self.blotter[-self.blotter_length:] # right-truncate to keep to length

        # NB What follows is **LAZY** -- assumes all orders are quantity=1
        transactionprice = trade['price']
        if self.orders[0].otype == 'Bid':
            profit = self.orders[0].price - transactionprice
        else:
            profit = transactionprice - self.orders[0].price
        self.balance += profit
        self.n_trades += 1
        self.profitpertime = self.balance / (time - self.birthtime)

        if profit < 0:
            print(profit)
            print(trade)
            print(order)
            sys.exit('FAIL: negative profit')

        if verbose: print('%s profit=%d balance=%d profit/time=%d' % (outstr, profit, self.balance, self.profitpertime))
        self.del_order(order)  # delete the order


    # specify how trader responds to events in the market
    # this is a null action, expect it to be overloaded by specific algos
    def respond(self, time, lob, trade, verbose):
        return None


    # specify how trader mutates its parameter values
    # this is a null action, expect it to be overloaded by specific algos
    def mutate(self, time, lob, trade, verbose):
        return None


# Trader subclass Giveaway
# even dumber than a ZI-U: just give the deal away
# (but never makes a loss)
class Trader_Giveaway(Trader):

    def getorder(self, time, countdown, lob):
        if len(self.orders) < 1:
            order = None
        else:
            quoteprice = self.orders[0].price
            order = Order(self.tid,
                          self.orders[0].otype,
                          quoteprice,
                          self.orders[0].qty,
                          time, lob['QID'])
            self.lastquote = order
        return order


# Trader subclass ZI-C
# After Gode & Sunder 1993
class Trader_ZIC(Trader):

    def getorder(self, time, countdown, lob):
        if len(self.orders) < 1:
            # no orders: return NULL
            order = None
        else:
            minprice = lob['bids']['worst']
            maxprice = lob['asks']['worst']
            qid = lob['QID']
            limit = self.orders[0].price
            otype = self.orders[0].otype
            if otype == 'Bid':
                quoteprice = random.randint(minprice, limit)
            else:
                quoteprice = random.randint(limit, maxprice)
                # NB should check it == 'Ask' and barf if not
            order = Order(self.tid, otype, quoteprice, self.orders[0].qty, time, qid)
            self.lastquote = order
        return order


# Trader subclass Shaver
# shaves a penny off the best price
# if there is no best price, creates "stub quote" at system max/min
class Trader_Shaver(Trader):

    def getorder(self, time, countdown, lob):
        if len(self.orders) < 1:
            order = None
        else:
            limitprice = self.orders[0].price
            otype = self.orders[0].otype
            if otype == 'Bid':
                if lob['bids']['n'] > 0:
                    quoteprice = lob['bids']['best'] + 1
                    if quoteprice > limitprice:
                        quoteprice = limitprice
                else:
                    quoteprice = lob['bids']['worst']
            else:
                if lob['asks']['n'] > 0:
                    quoteprice = lob['asks']['best'] - 1
                    if quoteprice < limitprice:
                        quoteprice = limitprice
                else:
                    quoteprice = lob['asks']['worst']
            order = Order(self.tid, otype, quoteprice, self.orders[0].qty, time, lob['QID'])
            self.lastquote = order
        return order


# Trader subclass Sniper
# Based on Shaver,
# "lurks" until time remaining < threshold% of the trading session
# then gets increasing aggressive, increasing "shave thickness" as time runs out
class Trader_Sniper(Trader):

    def getorder(self, time, countdown, lob):
        lurk_threshold = 0.2
        shavegrowthrate = 3
        shave = int(1.0 / (0.01 + countdown / (shavegrowthrate * lurk_threshold)))
        if (len(self.orders) < 1) or (countdown > lurk_threshold):
            order = None
        else:
            limitprice = self.orders[0].price
            otype = self.orders[0].otype

            if otype == 'Bid':
                if lob['bids']['n'] > 0:
                    quoteprice = lob['bids']['best'] + shave
                    if quoteprice > limitprice:
                        quoteprice = limitprice
                else:
                    quoteprice = lob['bids']['worst']
            else:
                if lob['asks']['n'] > 0:
                    quoteprice = lob['asks']['best'] - shave
                    if quoteprice < limitprice:
                        quoteprice = limitprice
                else:
                    quoteprice = lob['asks']['worst']
            order = Order(self.tid, otype, quoteprice, self.orders[0].qty, time, lob['QID'])
            self.lastquote = order
        return order


# Trader subclass PRZI
# added 23 March 2021
# Dave Cliff's Parameterized-Response Zero-Intelligence (PRZI) trader
# see https://arxiv.org/abs/2103.11341
class Trader_PRZI(Trader):

    def __init__(self, ttype, tid, balance, params, time):
        # PRZI strategy defined by parameter "strat"
        # here this is randomly assigned
        # strat * direction = -1 => GVWY; =0 = > ZIC; =+1 => SHVR

        Trader.__init__(self, ttype, tid, balance, params, time)
        self.theta0 = 100           # threshold-function limit value
        self.m = 4                  # tangent-function multiplier
        self.strat = 1.0 - 2 * random.random() # strategy parameter: must be in range [-1.0, +1.0]
        self.strat = 1
        self.cdf_lut_bid = None     # look-up table for buyer cumulative distribution function
        self.cdf_lut_ask = None     # look-up table for seller cumulative distribution function
        self.pmax = None            # this trader's estimate of the maximum price the market will bear
        self.pmax_c_i = math.sqrt(random.randint(1,10))  # multiplier coefficient when estimating p_max

    def getorder(self, time, countdown, lob):

        # shvr_price tells us what price a SHVR would quote in these circs
        def shvr_price(otype, limit, lob):

            if otype == 'Bid':
                if lob['bids']['n'] > 0:
                    shvr_p = lob['bids']['best'] + 1   # BSE tick size is always 1
                    if shvr_p > limit:
                        shvr_p = limit
                else:
                    shvr_p = lob['bids']['worst']
                # print('>>buy shvr_price=%f' % shvr_p)
            else:
                if lob['asks']['n'] > 0:
                    shvr_p = lob['asks']['best'] - 1   # BSE tick size is always 1
                    if shvr_p < limit:
                        shvr_p = limit
                else:
                    shvr_p = lob['asks']['worst']
                # print('>>ask shvr_price=%f' % shvr_p)
            return shvr_p

        # calculate cumulative distribution function (CDF) look-up table (LUT)
        def calc_cdf_lut(strat, t0, m, dirn, pmin, pmax):
            # set parameter values and calculate CDF LUT
            # dirn is direction: -1 for buy, +1 for sell

            # the threshold function used to clip
            def threshold(theta0, x):
                t = max(-1*theta0, min(theta0, x))
                return t

            epsilon = 0.00001   #used to catch DIV0 errors
            verbose = False

            if (strat > 1.0) or (strat < -1.0):
                # out of range
                sys.exit('FAIL: PRZI.getorder() self.strat out of range\n')

            if (dirn != 1.0) and (dirn != -1.0):
                # out of range
                sys.exit('FAIL: PRZI.calc_cdf() bad dirn\n')

            if pmax < pmin:
                # screwed
                sys.exit('FAIL: pmax %s < pmin %s \n' % (pmax,pmin))

            dxs = dirn * self.strat

            if verbose:
                print('calc_cdf_lut: dirn=%d dxs=%d pmin=%d pmax=%d\n' % (dirn, dxs, pmin, pmax))

            p_range = float(pmax - pmin)
            if p_range < 1:
                # special case: the SHVR-style strategy has shaved all the way to the limit price
                # the lower and upper bounds on the interval are adjacent prices;
                # so cdf is simply the lower price with probability 1

                cdf=[{'price':pmin, 'cum_prob': 1.0}]

                if verbose:
                    print('\n\ncdf:', cdf)

                return {'strat': strat, 'dirn': dirn, 'pmin': pmin, 'pmax': pmax, 'cdf_lut': cdf}

            c = threshold(t0, m * math.tan(math.pi * (strat + 0.5)))

            # catch div0 errors here
            if abs(c) < epsilon:
                if c > 0:
                    c = epsilon
                else:
                    c = -epsilon

            e2cm1 = math.exp(c) - 1

            # calculate the discrete calligraphic-P function over interval [pmin, pmax]
            # (i.e., this is Equation 8 in the PRZI Technical Note)
            calp_interval = []
            calp_sum = 0
            for p in range(pmin, pmax + 1):
                p_r = (p - pmin) / (p_range)  # p_r in [0.0, 1.0]
                if self.strat == 0.0:
                    # special case: this is just ZIC
                    cal_p = 1 / (p_range + 1)
                elif self.strat > 0:
                    cal_p = (math.exp(c * p_r) - 1.0) / e2cm1
                else:  # self.strat < 0
                    cal_p = 1.0 - ((math.exp(c * p_r) - 1.0) / e2cm1)
                if cal_p < 0:
                    cal_p = 0   # just in case
                calp_interval.append({'price': p, "cal_p": cal_p})
                calp_sum += cal_p

            if calp_sum <= 0:
                print('calp_interval:', calp_interval)
                print('pmin=%f, pmax=%f, calp_sum=%f' % (pmin, pmax, calp_sum))

            cdf = []
            cum_prob = 0
            # now go thru interval summing and normalizing to give the CDF
            for p in range(pmin, pmax + 1):
                price = calp_interval[p-pmin]['price']
                cal_p = calp_interval[p-pmin]['cal_p']
                prob = cal_p / calp_sum
                cum_prob += prob
                cdf.append({'price': p, 'cum_prob': cum_prob})

            if verbose:
                print('\n\ncdf:', cdf)

            return {'strat': strat, 'dirn': dirn, 'pmin': pmin, 'pmax': pmax, 'cdf_lut': cdf}

        verbose = False

        if verbose:
            print('PRZI getorder: strat=%f' % self.strat)

        if len(self.orders) < 1:
            # no orders: return NULL
            order = None
        else:
            # unpack the assignment-order
            limit = self.orders[0].price
            otype = self.orders[0].otype

            # get extreme limits on price interval
            # lowest price the market will bear
            minprice = int(lob['bids']['worst'])  # default assumption: worst bid price possible is 1 tick
            # trader's individual estimate of highest price the market will bear
            maxprice = self.pmax    # default assumption
            if self.pmax is None:
                maxprice = int(limit * self.pmax_c_i + 0.5)     # in the absence of any other info, guess
                self.pmax = maxprice
            elif lob['asks']['sess_hi'] is not None:
                if self.pmax < lob['asks']['sess_hi']:          # some other trader has quoted higher than I expected
                    maxprice = lob['asks']['sess_hi']           # so use that as my new estimate of highest
                    self.pmax = maxprice

            # what price would a SHVR quote?
            p_shvr = shvr_price(otype, limit, lob)

            # it may be more efficient to detect the ZIC special case and generate a price directly
            # whether it is or not depends on how many entries need to be sampled in the LUT reverse-lookup
            # versus the compute time of the call to random.randint that would be used in direct ZIC
            # here, for simplicity, we're not treating ZIC as a special case...
            # ... so the full CDF LUT needs to be instantiated for ZIC (strat=0.0) just like any other strat value

            # use the cdf look-up table
            # cdf_lut is a list of little dictionaries
            # each dictionary has form: {'cum_prob':nnn, 'price':nnn}
            # generate u=U(0,1) uniform disrtibution
            # starting with the lowest nonzero cdf value at cdf_lut[0],
            # walk up the lut (i.e., examine higher cumulative probabilities),
            # until we're in the range of u; then return the relevant price

            # the LUTs are re-computed if any of the details have changed
            if otype == 'Bid':

                # direction * strat
                dxs = -1 * self.strat  # the minus one multiplier is the "buy" direction

                p_max = int(limit)
                if dxs <= 0:
                    p_min = minprice        # this is delta_p for BSE, i.e. ticksize =1
                else:
                    # shade the lower bound on the interval
                    # away from minprice and toward shvr_price
                    p_min = int(0.5 + (dxs * p_shvr) + ((1.0-dxs) * minprice))

                if (self.cdf_lut_bid is None) or \
                        (self.cdf_lut_bid['strat'] != self.strat) or\
                        (self.cdf_lut_bid['pmin'] != p_min) or \
                        (self.cdf_lut_bid['pmax'] != p_max):
                    # need to compute a new LUT
                    if verbose:
                        print('New bid LUT')
                    self.cdf_lut_bid = calc_cdf_lut(self.strat, self.theta0,
                                                    self.m, -1, p_min, p_max)

                lut = self.cdf_lut_bid

            else:   # otype == 'Ask'

                dxs = self.strat

                p_min = int(limit)
                if dxs <= 0:
                    p_max = maxprice
                else:
                    # shade the upper bound on the interval
                    # away from maxprice and toward shvr_price
                    p_max = int(0.5 + (dxs * p_shvr) + ((1.0-dxs) * maxprice))

                if (self.cdf_lut_ask is None) or \
                        (self.cdf_lut_ask['strat'] != self.strat) or \
                        (self.cdf_lut_ask['pmin'] != p_min) or \
                        (self.cdf_lut_ask['pmax'] != p_max):
                    # need to compute a new LUT
                    if verbose:
                        print('New ask LUT')
                    self.cdf_lut_ask = calc_cdf_lut(self.strat, self.theta0,
                                                    self.m, +1, p_min, p_max)

                lut = self.cdf_lut_ask

            if verbose:
                print('PRZI LUT =', lut)

            # do inverse lookup on the LUT to find the price
            u = random.random()
            for entry in lut['cdf_lut']:
                if u < entry['cum_prob']:
                    quoteprice = entry['price']
                    break

            order = Order(self.tid, otype,
                          quoteprice, self.orders[0].qty, time, lob['QID'])

            self.lastquote = order

        return order


# Trader subclass PRZI_SHC (ticker: PRSH)
# added 23 Aug 2021
# Dave Cliff's Parameterized-Response Zero-Intelligence (PRZI) trader
# but with adaptive strategy, as a k-point Stochastic Hill-Climber (SHC) hence PRZI-SHC.
# PRZI-SHC pronounced "prezzy-shuck". Ticker symbol PRSH pronounced "purrsh".

class Trader_PRZI_SHC(Trader):

    # how to mutate the strategy values when hill-climbing
    def mutate_strat(self, s, mode):
        if mode == 'gauss':
            sdev = 0.05
            newstrat = s
            while newstrat == s:
                newstrat = s + random.gauss(0.0, sdev)
                # truncate to keep within range
                newstrat = max(-1.0, min(1.0, newstrat))
        elif mode == 'uniform_whole_range':
            # draw uniformly from whole range
            newstrat = random.uniform(-1.0, +1.0)
        else:
            sys.exit('FAIL: bad mode in mutate_strat')
        return newstrat


    def strat_str(self):
        # pretty-print a string summarising this trader's strategies
        string = 'PRSH: %s active_strat=[%d]:\n' % (self.tid, self.active_strat)
        for s in range(0, self.k):
            strat = self.strats[s]
            stratstr = '[%d]: s=%f, start=%f, $=%f, pps=%f\n' % \
                       (s, strat['stratval'], strat['start_t'], strat['profit'], strat['pps'])
            string = string + stratstr

        return string


    def __init__(self, ttype, tid, balance, params, time):
        # if params == "landscape-mapper" then it generates data for mapping the fitness landscape

        verbose = True

        Trader.__init__(self, ttype, tid, balance, params, time)
        self.theta0 = 100           # threshold-function limit value
        self.m = 4                  # tangent-function multiplier
        self.k = 4                  # number of hill-climbing points (cf number of arms on a multi-armed-bandit)
        self.strat_wait_time = 7200  # how many secs do we give any one strat before switching? todo: make this randomized within some range
        self.strat_range_min = 0.0 # lower-bound on randomly-assigned strategy-value
        self.strat_range_max = 0.0 # upper-bound on randomly-assigned strategy-value
        self.active_strat = 0       # which of the k strategies are we currently playing? -- start with 0
        self.prev_qid = None        # previous order i.d.
        self.strat_eval_time = self.k * self.strat_wait_time   # time to cycle through evaluating all k strategies
        self.last_strat_change_time = time  # what time did we last change strategies?
        self.profit_epsilon = 0.0 * random.random()    # minimum profit-per-sec difference between strategies that counts
        self.strats = []              # strategies awaiting initialization
        self.pmax = None            # this trader's estimate of the maximum price the market will bear
        self.pmax_c_i = math.sqrt(random.randint(1,10))  # multiplier coefficient when estimating p_max
        self.mapper_outfile = None
        # differential evolution parameters all in one dictionary
        self.diffevol = {'de_state': 'active_s0',          # initial state: strategy 0 is active (being evaluated)
                         's0_index': self.active_strat,    # s0 starts out as active strat
                         'snew_index': self.k,             # (k+1)th item of strategy list is DE's new strategy
                         'snew_stratval': None,            # assigned later
                         'F': 0.8                          # differential weight -- usually between 0 and 2
        }

        start_time = time
        profit = 0.0
        profit_per_second = 0
        lut_bid = None
        lut_ask = None

        for s in range(self.k + 1):
            # initialise each of the strategies in sequence: for SHC, one random seed, then k-1 mutants of that seed
            # for DiffEvol, use draws from uniform distbn over whole range
            # the (k+1)th strategy is needed to hold s_new in differential evolution; it's not used in SHC.
            if s == 0:
                strategy = random.uniform(self.strat_range_min, self.strat_range_max)
            else:
                if self.params == 'SHC':
                    # simple stochastic hill climber: cluster other strats around strat_0
                    strategy = self.mutate_strat(self.strats[0]['stratval'], 'gauss')     # mutant of strats[0]
                elif self.params == 'DiffEvo':
                    # differential evolution: seed initial strategies across whole space
                    strategy = self.mutate_strat(self.strats[0]['stratval'], 'uniform_whole_range')
            self.strats.append({'stratval': strategy, 'start_t': start_time,
                                'profit': profit, 'pps': profit_per_second, 'lut_bid': lut_bid, 'lut_ask': lut_ask})

        if self.params == 'landscape-mapper':
            # replace seed+mutants set of strats with regularly-spaced strategy values over the whole range
            self.strats = []
            strategy_delta = 0.01
            strategy = -1.0
            k = 0
            self.strats = []
            while strategy <= +1.0:
                self.strats.append({'stratval': strategy, 'start_t': start_time,
                                    'profit': profit, 'pps': profit_per_second, 'lut_bid': lut_bid, 'lut_ask': lut_ask})
                k += 1
                strategy += strategy_delta
            self.mapper_outfile = open('landscape_map.csv', 'w')
            self.k = k
            self.strat_eval_time = self.k * self.strat_wait_time

        if verbose:
            print("PRSH %s %s\n" % (tid, self.strat_str()))


    def getorder(self, time, countdown, lob):

        # shvr_price tells us what price a SHVR would quote in these circs
        def shvr_price(otype, limit, lob):

            if otype == 'Bid':
                if lob['bids']['n'] > 0:
                    shvr_p = lob['bids']['best'] + ticksize   # BSE ticksize is global var
                    if shvr_p > limit:
                        shvr_p = limit
                else:
                    shvr_p = lob['bids']['worst']
            else:
                if lob['asks']['n'] > 0:
                    shvr_p = lob['asks']['best'] - ticksize   # BSE ticksize is global var
                    if shvr_p < limit:
                        shvr_p = limit
                else:
                    shvr_p = lob['asks']['worst']

            # print('shvr_p=%f; ' % shvr_p)
            return shvr_p

        # calculate cumulative distribution function (CDF) look-up table (LUT)
        def calc_cdf_lut(strat, t0, m, dirn, pmin, pmax):
            # set parameter values and calculate CDF LUT
            # strat is strategy-value in [-1,+1]
            # t0 and m are constants used in the threshold function
            # dirn is direction: 'buy' or 'sell'
            # pmin and pmax are bounds on discrete-valued price-range

            # the threshold function used to clip
            def threshold(theta0, x):
                t = max(-1*theta0, min(theta0, x))
                return t

            epsilon = 0.000001 #used to catch DIV0 errors
            verbose = False

            if (strat > 1.0) or (strat < -1.0):
                # out of range
                sys.exit('PRSH FAIL: strat=%f out of range\n' % strat)

            if (dirn != 'buy') and (dirn != 'sell'):
                # out of range
                sys.exit('PRSH FAIL: bad dirn=%s\n' % dirn)

            if pmax < pmin:
                # screwed
                sys.exit('PRSH FAIL: pmax %f < pmin %f \n' % (pmax, pmin))

            if verbose:
                print('PRSH calc_cdf_lut: strat=%f dirn=%d pmin=%d pmax=%d\n' % (strat, dirn, pmin, pmax))

            p_range = float(pmax - pmin)
            if p_range < 1:
                # special case: the SHVR-style strategy has shaved all the way to the limit price
                # the lower and upper bounds on the interval are adjacent prices;
                # so cdf is simply the limit-price with probability 1

                if dirn == 'buy':
                    cdf = [{'price':pmax, 'cum_prob': 1.0}]
                else: # must be a sell
                    cdf = [{'price': pmin, 'cum_prob': 1.0}]

                if verbose:
                    print('\n\ncdf:', cdf)

                return {'strat': strat, 'dirn': dirn, 'pmin': pmin, 'pmax': pmax, 'cdf_lut': cdf}

            c = threshold(t0, m * math.tan(math.pi * (strat + 0.5)))

            # catch div0 errors here
            if abs(c) < epsilon:
                if c > 0:
                    c = epsilon
                else:
                    c = -epsilon

            e2cm1 = math.exp(c) - 1

            # calculate the discrete calligraphic-P function over interval [pmin, pmax]
            # (i.e., this is Equation 8 in the PRZI Technical Note)
            calp_interval = []
            calp_sum = 0
            for p in range(pmin, pmax + 1):
                # normalize the price to proportion of its range
                p_r = (p - pmin) / (p_range)  # p_r in [0.0, 1.0]
                if strat == 0.0:
                    # special case: this is just ZIC
                    cal_p = 1 / (p_range + 1)
                elif strat > 0:
                    if dirn == 'buy':
                        cal_p = (math.exp(c * p_r) - 1.0) / e2cm1
                    else:   # dirn == 'sell'
                        cal_p = (math.exp(c * (1 - p_r)) - 1.0) / e2cm1
                else:   # self.strat < 0
                    if dirn == 'buy':
                        cal_p = 1.0 - ((math.exp(c * p_r) - 1.0) / e2cm1)
                    else:   # dirn == 'sell'
                        cal_p = 1.0 - ((math.exp(c * (1 - p_r)) - 1.0) / e2cm1)

                if cal_p < 0:
                    cal_p = 0   # just in case

                calp_interval.append({'price':p, "cal_p":cal_p})
                calp_sum += cal_p

            if calp_sum <= 0:
                print('calp_interval:', calp_interval)
                print('pmin=%f, pmax=%f, calp_sum=%f' % (pmin, pmax, calp_sum))

            cdf = []
            cum_prob = 0
            # now go thru interval summing and normalizing to give the CDF
            for p in range(pmin, pmax + 1):
                price = calp_interval[p-pmin]['price'] # todo: what does this do?
                cal_p = calp_interval[p-pmin]['cal_p']
                prob = cal_p / calp_sum
                cum_prob += prob
                cdf.append({'price': p, 'cum_prob': cum_prob}) #todo shouldnt ths be "price" not "p"?

            if verbose:
                print('\n\ncdf:', cdf)

            return {'strat':strat, 'dirn':dirn, 'pmin':pmin, 'pmax':pmax, 'cdf_lut':cdf}

        verbose = False

        if verbose:
            print('t=%f PRSH getorder: %s, %s' % (time, self.tid, self.strat_str()))

        if len(self.orders) < 1:
            # no orders: return NULL
            order = None
        else:
            # unpack the assignment-order
            limit = self.orders[0].price
            otype = self.orders[0].otype
            qid = self.orders[0].qid

            if self.prev_qid is None:
                self.prev_qid = qid

            if qid != self.prev_qid:
                # customer-order i.d. has changed, so we're working a new customer-order now
                # this is the time to switch arms
                # print("New order! (how does it feel?)")
                dummy = 1

            # get extreme limits on price interval
            # lowest price the market will bear
            # todo OR make it like maxprice code (below), i.e. don't start at absolute worst-case, instead estimate
            minprice = int(lob['bids']['worst'])  # default assumption: worst bid price possible as defined by exchange

            # trader's individual estimate highest price the market will bear
            maxprice = self.pmax # default assumption
            if self.pmax is None:
                maxprice = int(limit * self.pmax_c_i + 0.5) # in the absence of any other info, guess
                self.pmax = maxprice
            elif lob['asks']['sess_hi'] is not None:
                if self.pmax < lob['asks']['sess_hi']:        # some other trader has quoted higher than I expected
                    maxprice = lob['asks']['sess_hi']         # so use that as my new estimate of highest
                    self.pmax = maxprice

            # use the cdf look-up table
            # cdf_lut is a list of little dictionaries
            # each dictionary has form: {'cum_prob':nnn, 'price':nnn}
            # generate u=U(0,1) uniform disrtibution
            # starting with the lowest nonzero cdf value at cdf_lut[0],
            # walk up the lut (i.e., examine higher cumulative probabilities),
            # until we're in the range of u; then return the relevant price

            strat = self.strats[self.active_strat]['stratval']

            # todo: delete this! DCdebugging stuff
            # otype = 'Bid'
            # strat = -1

            # what price would a SHVR quote?
            p_shvr = shvr_price(otype, limit, lob)

            if otype == 'Bid':

                p_max = int(limit)
                if strat > 0.0:
                    p_min = minprice
                else:
                    # shade the lower bound on the interval
                    # away from minprice and toward shvr_price
                    p_min = int(0.5 + (-strat * p_shvr) + ((1.0 + strat) * minprice))

                lut_bid = self.strats[self.active_strat]['lut_bid']
                if (lut_bid is None) or \
                        (lut_bid['strat'] != strat) or\
                        (lut_bid['pmin'] != p_min) or \
                        (lut_bid['pmax'] != p_max):
                    # need to compute a new LUT
                    if verbose:
                        print('New bid LUT')
                    self.strats[self.active_strat]['lut_bid'] = calc_cdf_lut(strat, self.theta0, self.m, 'buy', p_min, p_max)

                lut = self.strats[self.active_strat]['lut_bid']

            else:   # otype == 'Ask'

                p_min = int(limit)
                if strat > 0.0:
                    p_max = maxprice
                else:
                    # shade the upper bound on the interval
                    # away from maxprice and toward shvr_price
                    p_max = int(0.5 + (-strat * p_shvr) + ((1.0 + strat) * maxprice))
                    if p_max < p_min:
                        # this should never happen, but just in case it does...
                        p_max = p_min


                lut_ask = self.strats[self.active_strat]['lut_ask']
                if (lut_ask is None) or \
                        (lut_ask['strat'] != strat) or \
                        (lut_ask['pmin'] != p_min) or \
                        (lut_ask['pmax'] != p_max):
                    # need to compute a new LUT
                    if verbose:
                        print('New ask LUT')
                    self.strats[self.active_strat]['lut_ask'] = calc_cdf_lut(strat, self.theta0, self.m, 'sell', p_min, p_max)

                lut = self.strats[self.active_strat]['lut_ask']

                
            verbose = False
            if verbose:
                print('PRZI strat=%f LUT=%s \n \n' % (strat, lut))
                # useful in debugging: print a table of lut: price and cum_prob, with the discrete derivative (gives PMF).
                last_cprob = 0.0
                for lut_entry in lut['cdf_lut']:
                    cprob = lut_entry['cum_prob']
                    print('%d, %f, %f' % (lut_entry['price'], cprob - last_cprob, cprob))
                    last_cprob = cprob
                print('\n');    
                
                # print ('[LUT print suppressed]')
                
            # todo: delete this! DCdebugging stuff
            # sys.exit()
            
            # do inverse lookup on the LUT to find the price
            u = random.random()
            for entry in lut['cdf_lut']:
                if u < entry['cum_prob']:
                    quoteprice = entry['price']
                    break

            order = Order(self.tid, otype, quoteprice, self.orders[0].qty, time, lob['QID'])

            self.lastquote = order

        return order


    def bookkeep(self, trade, order, verbose, time):

        outstr = ""
        for order in self.orders:
            outstr = outstr + str(order)

        self.blotter.append(trade)  # add trade record to trader's blotter
        self.blotter = self.blotter[-self.blotter_length:] # right-truncate to keep to length

        # NB What follows is **LAZY** -- assumes all orders are quantity=1
        transactionprice = trade['price']
        if self.orders[0].otype == 'Bid':
            profit = self.orders[0].price - transactionprice
        else:
            profit = transactionprice - self.orders[0].price
        self.balance += profit
        self.n_trades += 1
        self.profitpertime = self.balance / (time - self.birthtime)

        if profit < 0:
            print(profit)
            print(trade)
            print(order)
            sys.exit('PRSH FAIL: negative profit')

        if verbose: print('%s profit=%d balance=%d profit/time=%d' % (outstr, profit, self.balance, self.profitpertime))
        self.del_order(order)  # delete the order

        # Trader.bookkeep(self, trade, order, verbose, time) -- todo: calls all of the above?

        # todo: expand from here

        # Check: bookkeep is only called after a successful trade? i.e. no need to check re trade or not

        self.strats[self.active_strat]['profit'] += profit
        time_alive = time - self.strats[self.active_strat]['start_t']
        if time_alive > 0:
            profit_per_second = self.strats[self.active_strat]['profit'] / time_alive
            self.strats[self.active_strat]['pps'] = profit_per_second
        else:
            # if it trades at the instant it is born then it would have infinite profit-per-second, which is insane
            # to keep things sensible whne time_alive == 0 we say the profit per second is whatever the actual profit is
            self.strats[self.active_strat]['pps'] = profit


    # PRSH respond() asks/answers two questions
    # do we need to choose a new strategy? (i.e. have just completed/cancelled previous customer order)
    # do we need to dump one arm and generate a new one? (i.e., both/all arms have been evaluated enough)
    def respond(self, time, lob, trade, verbose):

        shc_algo = 'Simple'
        shc_algo = 'DiffEvo'

        # "Simple" is a very basic form of stochastic hill-climber (SHC) that v easy to understand and to code
        # it cycles through the k different strats until each has been operated for at least eval_time seconds
        # but a strat that does nothing will get swapped out if it's been running for no_deal_time without a deal
        # then the strats with the higher total accumulated profit is retained,
        # and mutated versions of it are copied into the other k-1 strats
        # then all counters are reset, and this is repeated indefinitely
        #
        # "DiffEvo" is a basic form of Differential Evolution. This maintains a population of at least four strats
        # iterates indefinitely on:
        #       shuffle the set of strats;
        #       name the first four strats s0 to s3;
        #       create new_strat=s1+f*(s2-s3);
        #       evaluate fitness of s0 and new_strat;
        #       if (new_strat fitter than s0) then new_strat replaces s0.
        #
        # todo: add in more shc_algo that are cleverer than this,
        # e.g. inspired by multi-arm-bandit algos like like epsilon-greedy, softmax, or upper confidence bound (UCB)

        verbose = False

        # first update each strategy's profit-per-second (pps) value -- this is the "fitness" of each strategy
        for s in self.strats:
            # debugging check: make profit be directly proportional to strategy, no noise
            # s['profit'] = 100 * abs(s['stratval'])
            # update pps
            pps_time = time - s['start_t']
            if pps_time > 0:
                s['pps'] = s['profit'] / pps_time
            else:
                s['pps'] = s['profit']


        if shc_algo == 'Simple':

            if verbose:
                # print('t=%f %s PRSH respond: shc_algo=%s eval_t=%f max_wait_t=%f' %
                #     (time, self.tid, shc_algo, self.strat_eval_time, self.strat_wait_time))
                dummy = 1

            # do we need to swap strategies?
            # this is based on time elapsed since last reset -- waiting for the current strategy to get a deal
            # -- otherwise a hopeless strategy can just sit there for ages doing nothing,
            # which would disadvantage the *other* strategies because they would never get a chance to score any profit.
            # when a trader does a deal, clock is reset; todo check this!!!
            # clock also reset when new a strat is created, obvs. todo check this!!! also check bookkeeping/proft etc

            # NB this *cycles* through the available strats in sequence

            s = self.active_strat
            time_elapsed = time - self.last_strat_change_time
            if time_elapsed > self.strat_wait_time:
                # we have waited long enough: swap to another strategy

                new_strat = s + 1
                if new_strat > self.k - 1:
                    new_strat = 0

                self.active_strat = new_strat
                self.last_strat_change_time = time

                if verbose:
                    print('t=%f %s PRSH respond: strat[%d] elapsed=%f; wait_t=%f, switched to strat=%d' %
                          (time, self.tid, s, time_elapsed, self.strat_wait_time, new_strat))

            # code below here deals with creating a new set of k-1 mutants from the best of the k strats

            # assume that all strats have had long enough, and search for evidence to the contrary
            all_old_enough = True
            for s in self.strats:
                lifetime = time - s['start_t']
                if lifetime < self.strat_eval_time:
                    all_old_enough = False
                    break

            if all_old_enough:
                # all strategies have had long enough: which has made most profit?

                # sort them by profit
                strats_sorted = sorted(self.strats, key = lambda k: k['pps'], reverse = True)
                # strats_sorted = self.strats     # use this as a control: unsorts the strats, gives pure random walk.

                if verbose:
                    print('PRSH %s: strat_eval_time=%f, all_old_enough=True' % (self.tid, self.strat_eval_time))
                    for s in strats_sorted:
                        print('s=%f, start_t=%f, lifetime=%f, $=%f, pps=%f' %
                              (s['stratval'], s['start_t'], time-s['start_t'], s['profit'], s['pps']))

                if self.params == 'landscape-mapper':
                    for s in self.strats:
                        self.mapper_outfile.write('time, %f, strat, %f, pps, %f\n' %
                              (time, s['stratval'], s['pps']))
                    self.mapper_outfile.flush()
                    sys.exit()

                else:
                    # if the difference between the top two strats is too close to call then flip a coin
                    # this is to prevent the same good strat being held constant simply by chance cos it is at index [0]
                    best_strat = 0
                    prof_diff = strats_sorted[0]['pps'] - strats_sorted[1]['pps']
                    if abs(prof_diff) < self.profit_epsilon:
                        # they're too close to call, so just flip a coin
                        best_strat = random.randint(0,1)

                    if best_strat == 1:
                        # need to swap strats[0] and strats[1]
                        tmp_strat = strats_sorted[0]
                        strats_sorted[0] = strats_sorted[1]
                        strats_sorted[1] = tmp_strat

                    # the sorted list of strats replaces the existing list
                    self.strats = strats_sorted

                    # at this stage, strats_sorted[0] is our newly-chosen elite-strat, about to replicate
                    # record it

                    # now replicate and mutate elite into all the other strats
                    for s in range(1, self.k):    # note range index starts at one not zero
                        self.strats[s]['stratval'] = self.mutate_strat(self.strats[0]['stratval'])
                        self.strats[s]['start_t'] = time
                        self.strats[s]['profit'] = 0.0
                        self.strats[s]['pps'] = 0.0
                    # and then update (wipe) records for the elite
                    self.strats[0]['start_t'] = time
                    self.strats[0]['profit'] = 0.0
                    self.strats[0]['pps'] = 0.0
                    self.active_strat = 0

                if verbose:
                    print('%s: strat_eval_time=%f, MUTATED:' % (self.tid, self.strat_eval_time))
                    for s in self.strats:
                        print('s=%f start_t=%f, lifetime=%f, $=%f, pps=%f' %
                              (s['stratval'], s['start_t'], time-s['start_t'], s['profit'], s['pps']))

        elif shc_algo == 'DiffEvo':
            # simple differential evolution

            # only initiate diff-evol once the active strat has been evaluated for long enough
            actv_lifetime = time - self.strats[self.active_strat]['start_t']
            if actv_lifetime >= self.strat_wait_time:

                if self.k < 4:
                    sys.exit('FAIL: k too small for diffevol')

                if self.diffevol['de_state'] == 'active_s0':
                    # we've evaluated s0, so now we need to evaluate s_new
                    self.active_strat = self.diffevol['snew_index']
                    self.strats[self.active_strat]['start_t'] = time
                    self.strats[self.active_strat]['profit'] = 0.0
                    self.strats[self.active_strat]['pps'] = 0.0

                    self.diffevol['de_state'] = 'active_snew'

                elif self.diffevol['de_state'] == 'active_snew':
                    # now we've evaluated s_0 and s_new, so we can do DE adaptive step
                    print('Differential evolution!!!!')
                    i_0 = self.diffevol['s0_index']
                    i_new = self.diffevol['snew_index']
                    fit_0 = self.strats[i_0]['pps']
                    fit_new = self.strats[i_new]['pps']

                    print('DiffEvol: t=%f, i_0=%d, i0fit=%f, i_n=%d, iNfit=%f' % (time, i_0, fit_0, i_new, fit_new))

                    if fit_new >= fit_0:
                        # new strat did better than old strat0, so overwrite new into strat0
                        self.strats[i_0]['stratval'] = self.strats[i_new]['stratval']

                    # do differential evolution

                    # pick four individual strategies at random, but they must be distinct
                    stratlist = list(range(0, self.k))    # create sequential list of strategy-numbers
                    random.shuffle(stratlist)             # shuffle the list

                    # s0 is next iteration's candidate for possible replacement
                    self.diffevol['s0_index'] = stratlist[0]

                    # s1, s2, s3 used in DE to create new strategy, potential replacement for s0
                    s1_index = stratlist[1]
                    s2_index = stratlist[2]
                    s3_index = stratlist[3]

                    # unpack the actual strategy values
                    s1_stratval = self.strats[s1_index]['stratval']
                    s2_stratval = self.strats[s2_index]['stratval']
                    s3_stratval = self.strats[s3_index]['stratval']

                    # this is the differential evolution "adaptive step": create a new individual
                    new_stratval = s1_stratval + self.diffevol['F'] * (s2_stratval - s3_stratval)

                    # clip to bounds
                    new_stratval = max(-1, min(+1, new_stratval))

                    # record it for future use (s0 will be evaluated first, then s_new)
                    self.strats[self.diffevol['snew_index']]['stratval'] = new_stratval

                    print('DiffEvol: t=%f, s0=%d, s1=%d, (s=%f), s2=%d, (s=%f), s3=%d, (s=%f), sNew=%f' %
                          (time, self.diffevol['s0_index'],
                           s1_index, s1_stratval, s2_index, s2_stratval, s3_index, s3_stratval, new_stratval))

                    # DC's intervention for fully converged populations
                    # is the stddev of the strategies in the population equal/close to zero?
                    sum = 0.0
                    for s in range(self.k):
                        sum += self.strats[s]['stratval']
                    strat_mean = sum / self.k
                    sumsq = 0.0
                    for s in range(self.k):
                        diff = self.strats[s]['stratval'] - strat_mean
                        sumsq += (diff * diff)
                    strat_stdev = math.sqrt(sumsq / self.k)
                    print('t=,%f, MeanStrat=, %f, stdev=,%f' % (time, strat_mean, strat_stdev))
                    if strat_stdev < 0.0001:
                        # this population has converged
                        # mutate one strategy at random
                        randindex = random.randint(0, self.k - 1)
                        self.strats[randindex]['stratval'] = random.uniform(-1.0, +1.0)
                        print('Converged pop: set strategy %d to %f' % (randindex, self.strats[randindex]['stratval']))

                    # set up next iteration: first evaluate s0
                    self.active_strat = self.diffevol['s0_index']
                    self.strats[self.active_strat]['start_t'] = time
                    self.strats[self.active_strat]['profit'] = 0.0
                    self.strats[self.active_strat]['pps'] = 0.0

                    self.diffevol['de_state'] = 'active_s0'

                else:
                    sys.exit('FAIL: self.diffevol[\'de_state\'] not recognized')

        else:
            sys.exit('FAIL: bad value for shc_algo')



class Trader_ZIP(Trader):

    # ZIP init key param-values are those used in Cliff's 1997 original HP Labs tech report
    # NB this implementation keeps separate margin values for buying & selling,
    #    so a single trader can both buy AND sell
    #    -- in the original, traders were either buyers OR sellers

    def __init__(self, ttype, tid, balance, params, time):
        Trader.__init__(self, ttype, tid, balance, params, time)
        self.willing = 1
        self.able = 1
        self.job = None  # this gets switched to 'Bid' or 'Ask' depending on order-type
        self.active = False  # gets switched to True while actively working an order
        self.prev_change = 0  # this was called last_d in Cliff'97
        self.beta = 0.1 + 0.4 * random.random()
        self.momntm = 0.1 * random.random()
        self.ca = 0.05  # self.ca & .cr were hard-coded in '97 but parameterised later
        self.cr = 0.05
        self.margin = None  # this was called profit in Cliff'97
        self.margin_buy = -1.0 * (0.05 + 0.3 * random.random())
        self.margin_sell = 0.05 + 0.3 * random.random()
        self.price = None
        self.limit = None
        # memory of best price & quantity of best bid and ask, on LOB on previous update
        self.prev_best_bid_p = None
        self.prev_best_bid_q = None
        self.prev_best_ask_p = None
        self.prev_best_ask_q = None

    def getorder(self, time, countdown, lob):
        if len(self.orders) < 1:
            self.active = False
            order = None
        else:
            self.active = True
            self.limit = self.orders[0].price
            self.job = self.orders[0].otype
            if self.job == 'Bid':
                # currently a buyer (working a bid order)
                self.margin = self.margin_buy
            else:
                # currently a seller (working a sell order)
                self.margin = self.margin_sell
            quoteprice = int(self.limit * (1 + self.margin))
            self.price = quoteprice

            order = Order(self.tid, self.job, quoteprice, self.orders[0].qty, time, lob['QID'])
            self.lastquote = order
        return order

    # update margin on basis of what happened in market
    def respond(self, time, lob, trade, verbose):
        # ZIP trader responds to market events, altering its margin
        # does this whether it currently has an order to work or not

        def target_up(price):
            # generate a higher target price by randomly perturbing given price
            ptrb_abs = self.ca * random.random()  # absolute shift
            ptrb_rel = price * (1.0 + (self.cr * random.random()))  # relative shift
            target = int(round(ptrb_rel + ptrb_abs, 0))
            # #                        print('TargetUp: %d %d\n' % (price,target))
            return target

        def target_down(price):
            # generate a lower target price by randomly perturbing given price
            ptrb_abs = self.ca * random.random()  # absolute shift
            ptrb_rel = price * (1.0 - (self.cr * random.random()))  # relative shift
            target = int(round(ptrb_rel - ptrb_abs, 0))
            # #                        print('TargetDn: %d %d\n' % (price,target))
            return target

        def willing_to_trade(price):
            # am I willing to trade at this price?
            willing = False
            if self.job == 'Bid' and self.active and self.price >= price:
                willing = True
            if self.job == 'Ask' and self.active and self.price <= price:
                willing = True
            return willing

        def profit_alter(price):
            oldprice = self.price
            diff = price - oldprice
            change = ((1.0 - self.momntm) * (self.beta * diff)) + (self.momntm * self.prev_change)
            self.prev_change = change
            newmargin = ((self.price + change) / self.limit) - 1.0

            if self.job == 'Bid':
                if newmargin < 0.0:
                    self.margin_buy = newmargin
                    self.margin = newmargin
            else:
                if newmargin > 0.0:
                    self.margin_sell = newmargin
                    self.margin = newmargin

            # set the price from limit and profit-margin
            self.price = int(round(self.limit * (1.0 + self.margin), 0))

        # #                        print('old=%d diff=%d change=%d price = %d\n' % (oldprice, diff, change, self.price))

        # what, if anything, has happened on the bid LOB?
        bid_improved = False
        bid_hit = False
        lob_best_bid_p = lob['bids']['best']
        lob_best_bid_q = None
        if lob_best_bid_p is not None:
            # non-empty bid LOB
            lob_best_bid_q = lob['bids']['lob'][-1][1]
            if (self.prev_best_bid_p is not None) and (self.prev_best_bid_p < lob_best_bid_p):
                # best bid has improved
                # NB doesn't check if the improvement was by self
                bid_improved = True
            elif trade is not None and ((self.prev_best_bid_p > lob_best_bid_p) or (
                    (self.prev_best_bid_p == lob_best_bid_p) and (self.prev_best_bid_q > lob_best_bid_q))):
                # previous best bid was hit
                bid_hit = True
        elif self.prev_best_bid_p is not None:
            # the bid LOB has been emptied: was it cancelled or hit?
            last_tape_item = lob['tape'][-1]
            if last_tape_item['type'] == 'Cancel':
                bid_hit = False
            else:
                bid_hit = True

        # what, if anything, has happened on the ask LOB?
        ask_improved = False
        ask_lifted = False
        lob_best_ask_p = lob['asks']['best']
        lob_best_ask_q = None
        if lob_best_ask_p is not None:
            # non-empty ask LOB
            lob_best_ask_q = lob['asks']['lob'][0][1]
            if (self.prev_best_ask_p is not None) and (self.prev_best_ask_p > lob_best_ask_p):
                # best ask has improved -- NB doesn't check if the improvement was by self
                ask_improved = True
            elif trade is not None and ((self.prev_best_ask_p < lob_best_ask_p) or (
                    (self.prev_best_ask_p == lob_best_ask_p) and (self.prev_best_ask_q > lob_best_ask_q))):
                # trade happened and best ask price has got worse, or stayed same but quantity reduced
                # -- assume previous best ask was lifted
                ask_lifted = True
        elif self.prev_best_ask_p is not None:
            # the ask LOB is empty now but was not previously: canceled or lifted?
            last_tape_item = lob['tape'][-1]
            if last_tape_item['type'] == 'Cancel':
                ask_lifted = False
            else:
                ask_lifted = True

        if verbose and (bid_improved or bid_hit or ask_improved or ask_lifted):
            print('B_improved', bid_improved, 'B_hit', bid_hit, 'A_improved', ask_improved, 'A_lifted', ask_lifted)

        deal = bid_hit or ask_lifted

        if self.job == 'Ask':
            # seller
            if deal:
                tradeprice = trade['price']
                if self.price <= tradeprice:
                    # could sell for more? raise margin
                    target_price = target_up(tradeprice)
                    profit_alter(target_price)
                elif ask_lifted and self.active and not willing_to_trade(tradeprice):
                    # wouldn't have got this deal, still working order, so reduce margin
                    target_price = target_down(tradeprice)
                    profit_alter(target_price)
            else:
                # no deal: aim for a target price higher than best bid
                if ask_improved and self.price > lob_best_ask_p:
                    if lob_best_bid_p is not None:
                        target_price = target_up(lob_best_bid_p)
                    else:
                        target_price = lob['asks']['worst']  # stub quote
                    profit_alter(target_price)

        if self.job == 'Bid':
            # buyer
            if deal:
                tradeprice = trade['price']
                if self.price >= tradeprice:
                    # could buy for less? raise margin (i.e. cut the price)
                    target_price = target_down(tradeprice)
                    profit_alter(target_price)
                elif bid_hit and self.active and not willing_to_trade(tradeprice):
                    # wouldn't have got this deal, still working order, so reduce margin
                    target_price = target_up(tradeprice)
                    profit_alter(target_price)
            else:
                # no deal: aim for target price lower than best ask
                if bid_improved and self.price < lob_best_bid_p:
                    if lob_best_ask_p is not None:
                        target_price = target_down(lob_best_ask_p)
                    else:
                        target_price = lob['bids']['worst']  # stub quote
                    profit_alter(target_price)

        # remember the best LOB data ready for next response
        self.prev_best_bid_p = lob_best_bid_p
        self.prev_best_bid_q = lob_best_bid_q
        self.prev_best_ask_p = lob_best_ask_p
        self.prev_best_ask_q = lob_best_ask_q


# ########################---trader-types have all been defined now--################


# #########################---Below lies the experiment/test-rig---##################


# trade_stats()
# dump CSV statistics on exchange data and trader population to file for later analysis
# this makes no assumptions about the number of types of traders, or
# the number of traders of any one type -- allows either/both to change
# between successive calls, but that does make it inefficient as it has to
# re-analyse the entire set of traders on each call
def trade_stats(expid, traders, dumpfile, time, lob):

    # Analyse the set of traders, to see what types we have
    trader_types = {}
    for t in traders:
        ttype = traders[t].ttype
        if ttype in trader_types.keys():
            t_balance = trader_types[ttype]['balance_sum'] + traders[t].balance
            n = trader_types[ttype]['n'] + 1
        else:
            t_balance = traders[t].balance
            n = 1
        trader_types[ttype] = {'n': n, 'balance_sum': t_balance}

    # first two columns of output are the session_id and the time
    dumpfile.write('%s, %06d, ' % (expid, time))

    # Next three columns of output are the LOB best bid and best offer and micro-price (or 'None' if they're undefined)
    if lob['bids']['best'] is not None:
        dumpfile.write('%d, ' % (lob['bids']['best']))
    else:
        dumpfile.write('None, ')
    if lob['asks']['best'] is not None:
        dumpfile.write('%d, ' % (lob['asks']['best']))
    else:
        dumpfile.write('None, ')
    # if lob['delta_m'] is not None:
    #     dumpfile.write('%d, ' % (lob['delta_m']))
    # else:
    #     dumpfile.write('None, ')

    # total remaining number of columns printed depends on number of different trader-types at this timestep
    # for each trader type we print FOUR columns...
    # TraderTypeCode, TotalProfitForThisTraderType, NumberOfTradersOfThisType, AverageProfitPerTraderOfThisType
    for ttype in sorted(list(trader_types.keys())):
        n = trader_types[ttype]['n']
        s = trader_types[ttype]['balance_sum']
        dumpfile.write('%s, %d, %d, %f, ' % (ttype, s, n, s / float(n)))


    dumpfile.write('\n')
    
    
# opinion_stats()
# dump CSV statistics on opinion distribution to file for later analysis
def opinion_stats(expid, traders, exchange, dumpfile, time):
        trader_types = {}
        ext_op = exchange.ext_opinion
        
        for t in traders:
                ttype = traders[t].ttype
                if ttype in trader_types.keys():
                        trader_types[ttype]['opinions'].append(traders[t].opinion)
                        trader_types[ttype]['n'] = trader_types[ttype]['n'] + 1
                else:
                        opinions = [traders[t].opinion]
                        trader_types[ttype] = {'n':1, 'opinions': opinions}


        dumpfile.write('%06d, %.3f, ' % (time, ext_op))
        for ttype in sorted(list(trader_types.keys())):
                n = trader_types[ttype]['n']
                os = trader_types[ttype]['opinions']
                dumpfile.write('%s, %d,' % (ttype, n))
                for o in os:
                    dumpfile.write('%.3f, ' % o)
        dumpfile.write('\n');


# create a bunch of traders from traders_spec
# returns tuple (n_buyers, n_sellers)
# optionally shuffles the pack of buyers and the pack of sellers
def populate_market(traders_spec, traders, shuffle, social_net, verbose):

    def trader_type(robottype, name, parameters):
        if robottype == 'GVWY':
            return Trader_Giveaway('GVWY', name, 0.00, parameters, 0)
        elif robottype == 'ZIC':
            return Trader_ZIC('ZIC', name, 0.00, parameters, 0)
        elif robottype == 'SHVR':
            return Trader_Shaver('SHVR', name, 0.00, parameters, 0)
        elif robottype == 'SNPR':
            return Trader_Sniper('SNPR', name, 0.00, parameters, 0)
        elif robottype == 'ZIP':
            return Trader_ZIP('ZIP', name, 0.00, parameters, 0)
        elif robottype == 'PRZI':
            return Trader_PRZI('PRZI', name, 0.00, parameters, 0)
        elif robottype == 'PRSH':
            return Trader_PRZI_SHC('PRSH', name, 0.00, parameters, 0)
        elif robottype == 'OPRDE':
            return o_prde.Trader_O_PRDE('OPRDE', name, 0.00, parameters, 0)
        else:
            sys.exit('FATAL: don\'t know robot type %s\n' % robottype)

    def shuffle_traders(ttype_char, n, traders):
        for swap in range(n):
            t1 = (n - 1) - swap
            t2 = random.randint(0, t1)
            t1name = '%c%02d' % (ttype_char, t1)
            t2name = '%c%02d' % (ttype_char, t2)
            traders[t1name].tid = t2name
            traders[t2name].tid = t1name
            temp = traders[t1name]
            traders[t1name] = traders[t2name]
            traders[t2name] = temp
            
    def trader_graph(trader_ids, n_groups=12, groups_size=5, prob=0.1, rs=42, rewire_prob=0.3, seed=42, k_neighbors=None, _useKlemm=True, klemm_m=4, klemm_mu=0, _useFullyConnected=True):
        
        N = len(trader_ids)

        if(_useNetworkCaveman):
            """This function generates a graph mapping the traders relationships using
                the Relaxed Caveman Graph. This model uses """
            graph = nx.relaxed_caveman_graph(n_groups, groups_size, prob, rs)

        elif _useKlemm:
            graph = klemm_eguiluz_graph(num_nodes=N, m=klemm_m, mu=klemm_mu, seed=seed)

        elif _useFullyConnected:
            graph = nx.complete_graph(N)

        else:
            """This function generates a graph mapping the traders relationships using Watts-Strogatz small-world model."""
            if k_neighbors is None:
                k_neighbors = N - 1 # Fully connected
            if k_neighbors % 2 != 0:
                k_neighbors -= 1
            graph = nx.watts_strogatz_graph(n=N, k=k_neighbors, p=rewire_prob, seed=seed)
            
        random.shuffle(trader_ids)
        for idx, trader_id in enumerate(trader_ids):
            graph.nodes[idx]['t_id'] = trader_id
            
        return graph

    landscape_mapping = False   # set to true when mapping fitness landscape (for PRSH etc).
    prsh_optmzr = 'Simple'
    prsh_optmzr = 'DiffEvo'

    n_buyers = 0
    trader_ids = []
    for bs in traders_spec['buyers']:
        ttype = bs[0]
        for b in range(bs[1]):
            tname = 'B%02d' % n_buyers  # buyer i.d. string
            params = None
            if ttype == 'PRSH':
                # parameters matter to PRSH...
                if landscape_mapping:
                    params = 'landscape-mapper'
                else:
                    # params determines type of optimizer used
                    params = prsh_optmzr
            if ttype == 'OPRDE':
                # parameters matter to PRSH...
                if landscape_mapping:
                    params = 'landscape-mapper'
                else:
                    # params determines OPRDE starting values
                    params = {'prsh_optmzr': 'DiffEvo',
                              'start_opinion': random.uniform(-1, 1),
                              'locality':  random.uniform(0, 1),
                              'uncertainty': random.uniform(0.2, 2),
                              'lower_un_bound': 0.2,
                              'upper_un_bound': 2,
                             }
            traders[tname] = trader_type(ttype, tname, params)
            trader_ids.append(tname)
            n_buyers = n_buyers + 1

    if n_buyers < 1:
        sys.exit('FATAL: no buyers specified\n')

    if shuffle:
        shuffle_traders('B', n_buyers, traders)

    n_sellers = 0
    for ss in traders_spec['sellers']:
        ttype = ss[0]
        for s in range(ss[1]):
            tname = 'S%02d' % n_sellers  # buyer i.d. string
            params = None
            if ttype == 'PRSH':
                # parameters matter to PRSH...
                if landscape_mapping:
                    params = 'landscape-mapper'
                else:
                    # params determines type of optimizer used
                    params = prsh_optmzr
            if ttype == 'OPRDE':
                # parameters matter to PRSH...
                if landscape_mapping:
                    params = 'landscape-mapper'
                else:
                    # params determines OPRDE starting values
                    params = {'prsh_optmzr': 'DiffEvo',
                              'start_opinion': random.uniform(-1, 1),
                              'locality': random.uniform(0, 1),
                              'uncertainty': random.uniform(0.2, 2),
                              'lower_un_bound': 0.2,
                              'upper_un_bound': 2,
                             }
            traders[tname] = trader_type(ttype, tname, params)
            trader_ids.append(tname)
            n_sellers = n_sellers + 1

    if n_sellers < 1:
        sys.exit('FATAL: no sellers specified\n')

    if shuffle:
        shuffle_traders('S', n_sellers, traders)
        
    if social_net:
        trader_graph(trader_ids, n_groups=12, groups_size=5, prob=0.1, rs=42, rewire_prob=0.1, seed=42, k_neighbors=8, _useKlemm=False, klemm_m=4, klemm_mu=1.0, _useFullyConnected=False)
    else:
        graph = None

    if verbose:
        for t in range(n_buyers):
            bname = 'B%02d' % t
            print(traders[bname])
        for t in range(n_sellers):
            bname = 'S%02d' % t
            print(traders[bname])

    return {'n_buyers': n_buyers, 'n_sellers': n_sellers}, graph


# customer_orders(): allocate orders to traders
# parameter "os" is order schedule
# os['timemode'] is either 'periodic', 'drip-fixed', 'drip-jitter', or 'drip-poisson'
# os['interval'] is number of seconds for a full cycle of replenishment
# drip-poisson sequences will be normalised to ensure time of last replenishment <= interval
# parameter "pending" is the list of future orders (if this is empty, generates a new one from os)
# revised "pending" is the returned value
#
# also returns a list of "cancellations": trader-ids for those traders who are now working a new order and hence
# need to kill quotes already on LOB from working previous order
#
#
# if a supply or demand schedule mode is "random" and more than one range is supplied in ranges[],
# then each time a price is generated one of the ranges is chosen equiprobably and
# the price is then generated uniform-randomly from that range
#
# if len(range)==2, interpreted as min and max values on the schedule, specifying linear supply/demand curve
# if len(range)==3, first two vals are min & max, third value should be a function that generates a dynamic price offset
#                   -- the offset value applies equally to the min & max, so gradient of linear sup/dem curve doesn't vary
# if len(range)==4, the third value is function that gives dynamic offset for schedule min,
#                   and fourth is a function giving dynamic offset for schedule max, so gradient of sup/dem linear curve can vary
#
# the interface on this is a bit of a mess... could do with refactoring


def customer_orders(time, last_update, traders, trader_stats, os, pending, verbose):

    def sysmin_check(price):
        if price < bse_sys_minprice:
            print('WARNING: price < bse_sys_min -- clipped')
            price = bse_sys_minprice
        return price

    def sysmax_check(price):
        if price > bse_sys_maxprice:
            print('WARNING: price > bse_sys_max -- clipped')
            price = bse_sys_maxprice
        return price

    def getorderprice(i, sched, n, mode, issuetime):
        # does the first schedule range include optional dynamic offset function(s)?
        if len(sched[0]) > 2:
            offsetfn = sched[0][2]
            if callable(offsetfn):
                # same offset for min and max
                offset_min = offsetfn(issuetime)
                offset_max = offset_min
            else:
                sys.exit('FAIL: 3rd argument of sched in getorderprice() not callable')
            if len(sched[0]) > 3:
                # if second offset function is specfied, that applies only to the max value
                offsetfn = sched[0][3]
                if callable(offsetfn):
                    # this function applies to max
                    offset_max = offsetfn(issuetime)
                else:
                    sys.exit('FAIL: 4th argument of sched in getorderprice() not callable')
        else:
            offset_min = 0.0
            offset_max = 0.0

        pmin = sysmin_check(offset_min + min(sched[0][0], sched[0][1]))
        pmax = sysmax_check(offset_max + max(sched[0][0], sched[0][1]))
        prange = pmax - pmin
        stepsize = prange / (n - 1)
        halfstep = round(stepsize / 2.0)

        if mode == 'fixed':
            orderprice = pmin + int(i * stepsize)
        elif mode == 'jittered':
            orderprice = pmin + int(i * stepsize) + random.randint(-halfstep, halfstep)
        elif mode == 'random':
            if len(sched) > 1:
                # more than one schedule: choose one equiprobably
                s = random.randint(0, len(sched) - 1)
                pmin = sysmin_check(min(sched[s][0], sched[s][1]))
                pmax = sysmax_check(max(sched[s][0], sched[s][1]))
            orderprice = random.randint(pmin, pmax)
        else:
            sys.exit('FAIL: Unknown mode in schedule')
        orderprice = sysmin_check(sysmax_check(orderprice))
        return orderprice

    def getissuetimes(n_traders, mode, interval, shuffle, fittointerval):
        interval = float(interval)
        if n_traders < 1:
            sys.exit('FAIL: n_traders < 1 in getissuetime()')
        elif n_traders == 1:
            tstep = interval
        else:
            tstep = interval / (n_traders - 1)
        arrtime = 0
        issuetimes = []
        for t in range(n_traders):
            if mode == 'periodic':
                arrtime = interval
            elif mode == 'drip-fixed':
                arrtime = t * tstep
            elif mode == 'drip-jitter':
                arrtime = t * tstep + tstep * random.random()
            elif mode == 'drip-poisson':
                # poisson requires a bit of extra work
                interarrivaltime = random.expovariate(n_traders / interval)
                arrtime += interarrivaltime
            else:
                sys.exit('FAIL: unknown time-mode in getissuetimes()')
            issuetimes.append(arrtime)

            # at this point, arrtime is the last arrival time
        if fittointerval and ((arrtime > interval) or (arrtime < interval)):
            # generated sum of interarrival times longer than the interval
            # squish them back so that last arrival falls at t=interval
            for t in range(n_traders):
                issuetimes[t] = interval * (issuetimes[t] / arrtime)
        # optionally randomly shuffle the times
        if shuffle:
            for t in range(n_traders):
                i = (n_traders - 1) - t
                j = random.randint(0, i)
                tmp = issuetimes[i]
                issuetimes[i] = issuetimes[j]
                issuetimes[j] = tmp
        return issuetimes

    def getschedmode(time, os):
        got_one = False
        for sched in os:
            if (sched['from'] <= time) and (time < sched['to']):
                # within the timezone for this schedule
                schedrange = sched['ranges']
                mode = sched['stepmode']
                got_one = True
                break  # jump out the loop -- so the first matching timezone has priority over any others
        if not got_one:
            sys.exit('Fail: time=%5.2f not within any timezone in os=%s' % (time, os))
        return (schedrange, mode)

    n_buyers = trader_stats['n_buyers']
    n_sellers = trader_stats['n_sellers']

    shuffle_times = True

    cancellations = []

    if len(pending) < 1:
        # list of pending (to-be-issued) customer orders is empty, so generate a new one
        new_pending = []

        # demand side (buyers)
        issuetimes = getissuetimes(n_buyers, os['timemode'], os['interval'], shuffle_times, True)

        ordertype = 'Bid'
        (sched, mode) = getschedmode(time, os['dem'])
        for t in range(n_buyers):
            issuetime = time + issuetimes[t]
            tname = 'B%02d' % t
            orderprice = getorderprice(t, sched, n_buyers, mode, issuetime)
            order = Order(tname, ordertype, orderprice, 1, issuetime, chrono.time())
            new_pending.append(order)

        # supply side (sellers)
        issuetimes = getissuetimes(n_sellers, os['timemode'], os['interval'], shuffle_times, True)
        ordertype = 'Ask'
        (sched, mode) = getschedmode(time, os['sup'])
        for t in range(n_sellers):
            issuetime = time + issuetimes[t]
            tname = 'S%02d' % t
            orderprice = getorderprice(t, sched, n_sellers, mode, issuetime)
            # print('time %d sellerprice %d' % (time,orderprice))
            order = Order(tname, ordertype, orderprice, 1, issuetime, chrono.time())
            new_pending.append(order)
    else:
        # there are pending future orders: issue any whose timestamp is in the past
        new_pending = []
        for order in pending:
            if order.time < time:
                # this order should have been issued by now
                # issue it to the trader
                tname = order.tid
                response = traders[tname].add_order(order, verbose)
                if verbose:
                    print('Customer order: %s %s' % (response, order))
                if response == 'LOB_Cancel':
                    cancellations.append(tname)
                    if verbose:
                        print('Cancellations: %s' % cancellations)
                # and then don't add it to new_pending (i.e., delete it)
            else:
                # this order stays on the pending list
                new_pending.append(order)
    return [new_pending, cancellations]


# one session in the market
def market_session(sess_id, starttime, endtime, trader_spec, order_schedule, tdump, opfile, trfile, dump_all, verbose, _recordFrequencyInMin=5):

    orders_verbose = False
    lob_verbose = False
    process_verbose = False
    respond_verbose = False
    bookkeep_verbose = False
    populate_verbose = False

    strat_dump = open(sess_id + '_strats.csv', 'w')
    new_pr2a_frame = True

    # initialise the exchange
    exchange = Exchange()

    # create a bunch of traders
    # graph variable contains the graph representation of traders (social network)
    #if set to false none is returned and BSE runs as normal
    traders = {}
    trader_stats, graph = populate_market(trader_spec, traders, True, False, populate_verbose)

    # timestep set so that can process all traders in one second
    # NB minimum interarrival time of customer orders may be much less than this!!
    timestep = 1.0 / float(trader_stats['n_buyers'] + trader_stats['n_sellers'])

    duration = float(endtime - starttime)

    last_update = -1.0

    time = starttime

    pending_cust_orders = []

    if verbose:
        print('\n%s;  ' % sess_id)

    # frames_done is record of what frames we have printed data for thus far
    frames_done = set()


    current_time = 0
    while time < endtime:

        # how much time left, as a percentage?
        time_left = (endtime - time) / duration

        # if verbose: print('\n\n%s; t=%08.2f (%4.1f/100) ' % (sess_id, time, time_left*100))

        trade = None

        [pending_cust_orders, kills] = customer_orders(time, last_update, traders, trader_stats,
                                                       order_schedule, pending_cust_orders, orders_verbose)

        # if any newly-issued customer orders mean quotes on the LOB need to be cancelled, kill them
        if len(kills) > 0:
            # if verbose : print('Kills: %s' % (kills))
            for kill in kills:
                # if verbose : print('lastquote=%s' % traders[kill].lastquote)
                if traders[kill].lastquote is not None:
                    # if verbose : print('Killing order %s' % (str(traders[kill].lastquote)))
                    exchange.del_order(time, traders[kill].lastquote, verbose)

        # get a limit-order quote (or None) from a randomly chosen trader
        tid = list(traders.keys())[random.randint(0, len(traders) - 1)]
        order = traders[tid].getorder(time, time_left, exchange.publish_lob(time, lob_verbose))

        # if verbose: print('Trader Quote: %s' % (order))

        if order is not None:
            if order.otype == 'Ask' and order.price < traders[tid].orders[0].price:
                sys.exit('Bad ask')
            if order.otype == 'Bid' and order.price > traders[tid].orders[0].price:
                sys.exit('Bad bid')
            # send order to exchange
            traders[tid].n_quotes = 1
            trade = exchange.process_order2(time, order, process_verbose)
            if trade is not None:
                # trade occurred,
                # so the counterparties update order lists and blotters
                traders[trade['party1']].bookkeep(trade, order, bookkeep_verbose, time)
                traders[trade['party2']].bookkeep(trade, order, bookkeep_verbose, time)
                if dump_all:
                    trade_stats(sess_id, traders, tdump, time, exchange.publish_lob(time, lob_verbose))
                    

            # traders respond to whatever happened
            lob = exchange.publish_lob(time, lob_verbose)
            for t in traders:
                # NB respond just updates trader's internal variables
                # doesn't alter the LOB, so processing each trader in
                # sequence (rather than random/shuffle) isn't a problem
                traders[t].respond(time, lob, trade, respond_verbose)

            # log all the PRSH strategy info for this timestep?
            frame_rate = _recordFrequencyInMin * 60     # print one frame every this many simulated seconds

            if int(time) % frame_rate == 0 and int(time) not in frames_done:
                # todo: make this a separate function called here rather than coded inline

                line_str = 't=,%f, ' % time

                best_buyer_id = None
                best_buyer_prof = 0
                best_buyer_strat = 0
                best_seller_id = None
                best_seller_prof = 0
                best_seller_strat = 0

                for t in traders:
                    trader = traders[t]

                    # print('PRSH recording, t=%s' % trader)
                    if trader.ttype == 'PRSH' or trader.ttype == 'OPRDE':
                        line_str += 'id=,%s,' % trader.tid
                        # line_str += 'bal=$,%f, n_trades=,%d, n_strats=,2, ' % (trader.balance, trader.n_trades)

                        act_strat = trader.strats[trader.active_strat]['stratval']
                        act_prof = trader.strats[trader.active_strat]['pps']
                        
                        line_str += 'actvstrat=,%f, ' % act_strat
                        line_str += 'actvprof=,%f, ' % act_prof

                        if trader.tid[:1] == 'B':
                            # this trader is a buyer
                            if best_buyer_id is None or act_prof > best_buyer_prof:
                                best_buyer_id = trader.tid
                                best_buyer_strat = act_strat
                                best_buyer_prof = act_prof
                        elif trader.tid[:1] == 'S':
                            # this trader is a seller
                            if best_seller_id is None or act_prof > best_seller_prof:
                                best_seller_id = trader.tid
                                best_seller_strat = act_strat
                                best_seller_prof = act_prof
                        else:
                            # wtf?
                            sys.exit('unknown trader id type in market_session')

                line_str += 'best_B_id=,%s, best_B_prof=,%f, best_B_strat=,%f, ' % \
                            (best_buyer_id, best_buyer_prof, best_buyer_strat)
                line_str += 'best_S_id=,%s, best_S_prof=,%f, best_S_strat=,%f, ' % \
                            (best_seller_id, best_seller_prof, best_seller_strat)
                line_str += '\n'
                strat_dump.write(line_str)
                strat_dump.flush()

                # record that we've written this frame
                frames_done.add(int(time))
                
        #Code adapted from open-source: github.com/ken-neth/opinion_dynamics_BSE.git
        # Kenneth Lomas (2021) opinion_dynamics_BSE
        if len(exchange.ext_opinions) == exchange.ext_opinions_length:
            if time > current_time:
                exchange.tape_dump2(trfile, time)
                opinion_stats(sess_id, traders, exchange, opfile, time)
                current_time += 1
                
                # Opinion updates occur once every second
                lob = exchange.publish_lob(time, lob_verbose)
                relative_disagreement_step(0.05, 0.05, 0.1, traders, graph, lob)

        time = time + timestep

    # session has ended

    strat_dump.close()

    if dump_all:

        # dump the tape (transactions only -- not dumping cancellations)
        exchange.tape_dump(sess_id+'_transactions.csv', 'w', 'keep')

        # record the blotter for each trader
        bdump = open(sess_id+'_blotters.csv', 'w')
        for t in traders:
            bdump.write('%s, %d\n'% (traders[t].tid, len(traders[t].blotter)))
            for b in traders[t].blotter:
                bdump.write('%s, Blotteritem, %s\n' % (traders[t].tid, b))
        bdump.close()


    # write trade_stats for this session (NB end-of-session summary only)
    trade_stats(sess_id, traders, tdump, time, exchange.publish_lob(time, lob_verbose))



#############################

# # Below here is where we set up and run a series of experiments


if __name__ == "__main__":
    
    #logging execution time of code
    startTime = datetime.now()

    
    _useFullyConnected = True
    _useKlemm = False

    # set up common parameters for all market sessions
    n_days = 4 # 1000 days is good, but 3*365=1095, might as well go for three years.
    start_time = 0.0
    # day_6 = 60.0 * 60.0 * 24 * 6
    end_time = 60.0 * 60.0 * 24 * n_days
    duration = end_time - start_time
    day = 60.0 * 60.0 * 24.0
    day2_start = start_time + 1 * day 


    # schedule_offsetfn returns time-dependent offset, to be added to schedule prices
    def schedule_offsetfn(t):

        pi2 = math.pi * 2
        c = math.pi * 3000
        wavelength = t / c
        gradient = 100 * t / (c / pi2)
        amplitude = 100 * t / (c / pi2)
        offset = gradient + amplitude * math.sin(wavelength * t)
        return int(round(offset, 0))
    

    sup_pre = (75, 75)
    dem_pre = (125, 125)

    # post-shock
    sup_post = (60, 60)
    dem_post = (110, 110)

    supply_schedule = [
        {'from': start_time, 'to': day2_start, 'ranges': [sup_pre],  'stepmode': 'fixed'},
    {   'from': day2_start, 'to': end_time,   'ranges': [sup_post], 'stepmode': 'fixed'},
    ]

    demand_schedule = [
        {'from': start_time, 'to': day2_start, 'ranges': [dem_pre],  'stepmode': 'fixed'},
        {'from': day2_start, 'to': end_time,   'ranges': [dem_post], 'stepmode': 'fixed'},
    ]


    # supply_schedule = [{'from': start_time, 'to': day_6, 'ranges': [(75, 75)], 'stepmode': 'fixed'}, 
    #                    {'from': day_6, 'to': end_time, 'ranges': [(55, 55)], 'stepmode': 'fixed'}]
    
    # demand_schedule = [{'from': start_time, 'to': day_6, 'ranges': [(125, 125)], 'stepmode': 'fixed'},
    #                    {'from': day_6, 'to': end_time, 'ranges': [(145, 145)], 'stepmode': 'fixed'}]

    # Here is an example of how to use the offset function
    #
    # range1 = (10, 190, schedule_offsetfn)
    # range2 = (200,300, schedule_offsetfn)

    # Here is an example of how to switch from range1 to range2 and then back to range1,
    # introducing two "market shocks"
    # -- here the timings of the shocks are at 1/3 and 2/3 into the duration of the session.
    
    # supply_schedule = [ {'from':start_time, 'to':duration/3, 'ranges':[range1], 'stepmode':'fixed'},
    #                     {'from':duration/3, 'to':2*duration/3, 'ranges':[range2], 'stepmode':'fixed'},
    #                     {'from':2*duration/3, 'to':end_time, 'ranges':[range1], 'stepmode':'fixed'}
    #                   ]


    # The code below sets up symmetric supply and demand curves at prices from 50 to 150, P0=100

    # range1 = (75, 75)
    # supply_schedule = [{'from': start_time, 'to': end_time, 'ranges': [range1], 'stepmode': 'fixed'}]

    # range2 = (125, 125)
    # demand_schedule = [{'from': start_time, 'to': end_time, 'ranges': [range2], 'stepmode': 'fixed'}]
    
    # supply_schedule = [ {'from':start_time, 'to':duration/3, 'ranges':[range1], 'stepmode':'fixed'},
    #                     {'from':duration/3, 'to':2*duration/3, 'ranges':[range2], 'stepmode':'fixed'},
    #                     {'from':2*duration/3, 'to':end_time, 'ranges':[range1], 'stepmode':'fixed'}
    #                   ]

    # The code below sets up flat (perfectly elastic) supply and demand curves at prices of 50 and 150, P0=100

    #range1 = (60, 60)
    #supply_schedule = [{'from': start_time, 'to': end_time, 'ranges': [range1], 'stepmode': 'fixed'}]

    #range2 = (140, 140)
    #demand_schedule = [{'from': start_time, 'to': end_time, 'ranges': [range2], 'stepmode': 'fixed'}]

    # new customer orders arrive at each trader approx once every order_interval seconds
    order_interval = 5

    order_sched = {'sup': supply_schedule, 'dem': demand_schedule,
                   'interval': order_interval, 'timemode': 'drip-poisson'}


    # Use 'periodic' if you want the traders' assignments to all arrive simultaneously & periodically
    #               'order_interval': 30, 'timemode': 'periodic'}

    # buyers_spec = [('GVWY',10),('SHVR',10),('ZIC',10),('ZIP',10)]
    # sellers_spec = [('GVWY',10),('SHVR',10),('ZIC',10),('ZIP',10)]

    opponent = 'OPRDE'
    # opp_N = 30
    sellers_spec = [(opponent, 30)]
    buyers_spec = [(opponent, 30)]
    # sellers_spec = [('OPRDE', 30)]
    # buyers_spec = sellers_spec

    traders_spec = {'sellers':sellers_spec, 'buyers':buyers_spec}

    # run a sequence of trials, one session per trial

    verbose = True

    # n_trials is how many trials (i.e. market sessions) to run in total
    n_trials = 1

    # n_recorded is how many trials (i.e. market sessions) to write full data-files for
    n_trials_recorded = 1

    trial = 1

    while trial < (n_trials+1):
        # create unique i.d. string for this trial
        trial_id = 'sess_OPRDE_NDR_d%03d_i%02d_%04d' % (n_days, order_interval, trial)

        if trial > n_trials_recorded:
            dump_all = False
        else:
            dump_all = True
        if(_useNetworkCaveman):
            networkName = "caveman"
        elif (_useKlemm):
            networkName = "klemm"
        elif (_useFullyConnected):
            networkName = "fully_connected"
        else:
            networkName = "watts"
        odump=open(trial_id + '.' + networkName + '.' + 'opinions.csv','w')
        tr_dump=open(trial_id + '.' + networkName + '.' + 'transactions2.csv','w')

        tdump = open(trial_id + 'avg_balance.csv', 'w')
        market_session(trial_id, start_time, end_time, traders_spec, order_sched, tdump, odump, tr_dump, dump_all, verbose)
        tdump.close()
        trial = trial + 1
        
    print(datetime.now() - startTime)