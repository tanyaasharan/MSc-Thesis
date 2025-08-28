#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 12:23:42 2022

@author: Abe
"""
import random
import sys
import math
import numpy as np

from BSE import Trader, Order, ticksize
from opinion_model import external_opinion_step

class Trader_O_PRDE(Trader):

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
        elif mode == 'uniform_bounded_range':
            #draw uniformly from range around opinion
            opinion_min = self.opinion - self.uncertainty
            opinion_max = self.opinion + self.uncertainty
            newstrat = random.uniform(max(-1, opinion_min), min(1, opinion_max))
        else:
            sys.exit('FAIL: bad mode in mutate_strat')
        return newstrat


    def strat_str(self):
        # pretty-print a string summarising this trader's strategies
        string = 'OPRDE %s active_strat=[%d]:\n' % (self.tid, self.active_strat)
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
        self.strat_range_min = params['start_opinion'] # lower-bound on randomly-assigned strategy-value
        self.strat_range_max = params['start_opinion'] # upper-bound on randomly-assigned strategy-value
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
        
        self.net_opinion = params['start_opinion'] #network opinion that updates when traders "talk" together
        self.external_opinion = params['start_opinion']
        self.locality = params['locality']
        self.opinion = params['start_opinion'] #no LOB data at the start so set to network opinion value
        self.n_iter = 0
        
        self.uncertainty = params['uncertainty'] # uncertainty between [0, 2]
        self.lower_un_bound = params['lower_un_bound']
        self.upper_un_bound = params['upper_un_bound']
        self.tid = tid

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
                if self.params['prsh_optmzr'] == 'SHC':
                    # simple stochastic hill climber: cluster other strats around strat_0
                    strategy = self.mutate_strat(self.strats[0]['stratval'], 'gauss')     # mutant of strats[0]
                elif self.params['prsh_optmzr'] == 'DiffEvo' and len(self.params) == 1:
                    # differential evolution: seed initial strategies across whole space
                    strategy = self.mutate_strat(self.strats[0]['stratval'], 'uniform_whole_range')
                else:
                    # differential evolution: seed initial strategies across opinion space
                    strategy = self.mutate_strat(self.strats[0]['stratval'], 'uniform_bounded_range')
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

        #shc_algo = 'Simple'
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
                    opinion_min = max(-1, self.opinion - (self.uncertainty))
                    opinion_max = min(1, self.opinion + (self.uncertainty))
                    new_stratval = max(opinion_min, min(opinion_max, new_stratval))

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
                        self.strats[randindex]['stratval'] = self.mutate_strat(self.strats[randindex]['stratval'], 'uniform_bounded_range')
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
            
    
    #Code adapted from open-source: github.com/ken-neth/opinion_dynamics_BSE.git
    # Kenneth Lomas (2021) opinion_dynamics_BSE
    def set_opinion(self, lob, weight_net, prob, RDA_net, X_j):
        X_i = self.opinion
        delta_m = lob['delta_m']
        
        # tanh transformation function (bounds between -1 and 1)
        if self.locality == 'Dynamic':
            #X_e = lob['ext_opinions'][-1]
            X_e = lob['ext_opinion']
            n_e = lob['ext_inf']
            u_i = self.uncertainty
            
            if X_e == 0 or random.random() <= prob:
                weight_e, RDA_ext = 0, 0
                weight_net = 2 * weight_net
            else:
                weight_e, RDA_ext = external_opinion_step(X_i, u_i, X_e, n_e, prob)
                weight_e = weight_e * weight_net
                self.set_uncertainty(u_i + (weight_net * RDA_ext * (n_e - u_i)))
            
            # RDA_net and RDA_e act as dynamic weights on each term in equation
            # Locality can be seen as RDA_net / RDA_e
            combined_opinion = X_i + (weight_net * RDA_net * (X_j - X_i)) + (weight_e * RDA_ext * (X_e - X_i))
        else:
            m = self.locality
            X_net = X_i + (weight_net * RDA_net * (X_j - X_i))
            
            # transformed cubic function mapping
            X_e = X_i + ((abs(X_i) / 10) * (delta_m ** 3))
            
            combined_opinion = (m * X_net) + ((1 - m) * X_e)
            
        validated_update = combined_opinion
        if combined_opinion >= 1:
            # set to upper bound
            validated_update = 1
        elif combined_opinion <= -1:
            # set to lower bound
            validated_update = -1
            
        self.opinion = validated_update
        
        
    def set_uncertainty(self, updated_uncertainty):

        validated_update = updated_uncertainty

        if updated_uncertainty >= self.upper_un_bound:
            # set to upper bound
            validated_update = self.upper_un_bound
        elif updated_uncertainty <= self.lower_un_bound:
            # set to lower bound
            validated_update = self.lower_un_bound

        self.uncertainty = validated_update