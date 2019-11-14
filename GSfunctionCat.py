#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 11:16:37 2019

@author: stellaveazey
"""
# GS Propensity model

# Growing spheres on prognositc score model
# Nearest adversary based on treatment


from __future__ import print_function
from numpy import random as nprand
import math
import numpy as np
import pandas as pd
import random
from sklearn.metrics.pairwise import manhattan_distances, pairwise_distances
import time
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.neural_network import MLPClassifier, BernoulliRBM
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr
import itertools as it
from itertools import combinations
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
import pprint
from scipy.stats import mode
from sklearn.ensemble.forest import _partition_estimators, parallel_helper
from sklearn.tree._tree import DTYPE
from sklearn.externals.joblib import Parallel, delayed
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
import multiprocessing as mp
import statistics
import sys




# COST MEASURES TO TEST
def l1_norm(obs_to_interprete, observation):
    l1 = sum(map(abs, obs_to_interprete - observation))
    return l1
#returns sum of feature value differences

def weighted_l1(obs_to_interprete, observation):
    exp = sum(map(lambda x: math.exp(x**2), obs_to_interprete - observation))
    return exp
#sum of exp((diff. in feature value)^2) = feature weight

def penalized_l1(obs_to_interprete, observation):
    #nul
    GAMMA_ = 1.0 #weight associated with vector sparsity
    l1 = l1_norm(observation)
    #returns sum of feature value differences
    nonzeros = sum((obs_to_interprete - observation) != 0)
    #sum of nonzero feature values
    return GAMMA_ * nonzeros + l1
#returns penalized l1

def l2(obs1, obs2):
    return pairwise_distances(obs1.reshape(1, -1), obs2.reshape(1, -1), metric='euclidean')[0][0]
#returns pairwise distance
    


#distance = []
#for i in np.arange(np.size(a_, 0)):
#    d=l2(a_[i,outcomeIndex], obs_to_interprete[outcomeIndex].values)
#    distance.append(d)
    
   
        
###GENERATION SPHERE AUTOUR. TROUVER ENNEMIS PUIS PRENDRE LE PLUS PROCHE
def generate_inside_ball(center, d, propensityIndex, outcomeIndex, segment, n=10000):
    def norm(v):
        out = []
        for o in v:
            out.append(sum(map(lambda x: x**2, o))**(0.5)) #Euclidean norm
        return np.array(out)
            #loop through elements of v, take the square root of the sum of their squares and append to array 'out'      
    z = nprand.normal(0, 1, (n, 10)) 
# n*d normal samples with mean 0 and SD 1
    # d = number of columns in X
    #print("ball segment", segment)
    z2 = np.array([a * b / c for a, b, c in zip(z, nprand.uniform(*segment, n)**(1/float(10)),  norm(z))])
# prbability integral transform?
    # unit vector of z * (uniform ** (1/float(d)))
    #print("z:", z.shape, "center", center.shape)
    z3 = z2 + center
    zm0 = pd.DataFrame(z2).apply(statistics.mean, axis=0)
    zm1 = pd.DataFrame(z2).apply(max, axis=0)
    zm2 = pd.DataFrame(z2).apply(min, axis=0)
    zmean = [zm0, zm1, zm2, segment]
    return z3, segment, zmean


def generate_layer_with_score_function(X, score_function, center, d, n, segment, propensityIndex, outcomeIndex, catidx, catdat):
    # out = [] (not used)
    a_, seg, zmean = generate_inside_ball(np.asarray(center).reshape(1, -1), d, propensityIndex, outcomeIndex, segment, n=n)
    #print("a_ finished")
    #print(catidx)
    if catidx is not None:
        #print("if catidx != None")
        a_[:,catidx] = catdat
        #print("gen_layer_sf", segment, "n=", n, "d=", d)
        score_function_ = np.round(np.asarray(score_function(a_)))
        #print("score_function min max", min(score_function_), max(score_function_))
        a_ = np.concatenate((a_, np.round(score_function_.reshape(n, 1))), axis=1)
        return a_
    else:
        #print("gen_layer_sf", segment, "n=", n, "d=", d)
        score_function_ = np.round(np.asarray(score_function(a_)))
        #print("score_function min max", min(score_function_), max(score_function_))
        a_ = np.concatenate((a_, score_function_.reshape(n, 1)), axis=1)
        #print("a_ created")
        return a_, seg, zmean




def featred_random(obs_to_interprete, treat, ennemy, propensityIndex, outcomeIndex, score_function, idx):
    score_function_OBS = treat[idx]
    #print("sfobs:", score_function_OBS)
    #print("enemy psi:", ennemy[propensityIndex])
    #print("oti psi:", obs_to_interprete[propensityIndex])
    print("obs type:", type(obs_to_interprete), "enn type:", type(ennemy))
    print("pre moves")
    #enn = pd.DataFrame(ennemy[0]).iloc[propensityIndex].values.reshape(1, -1)      
    moves = map(abs, obs_to_interprete[propensityIndex] - ennemy[propensityIndex])
    moves = sorted(enumerate(moves), key=lambda x: x[1])
    out = ennemy.copy()
    for d in moves:
        new=out.copy()
        if d[1] > 0.0:
            new[d[0]] = obs_to_interprete[d[0]]
            class_new = np.round(score_function(np.asarray(new).reshape(1, -1)))
            class_new
            if class_new != score_function_OBS:
                out = new
                print("done!")
    return out





def seek_ennemies2(X, y, treat, score_function, obs_to_interprete, n_layer, step, enough_ennemies, fe, dfe, propensityIndex, outcomeIndex, idx, cat, catdat):
    #print(step)
    score_function_CLASS = treat[idx]
    ennemies = []
    #fe, dfe = distance_first_ennemy(X_orig, obs_to_interprete, score_function)
    #print("the dfe", dfe)
    step1 = dfe * step
    a0, a1 = 0, step1
    i = 0
    #print("a0, a1, step", a0, a1, step)
    #print("begin layer with score_function")
    if cat != None: 
        adversaries={}
        catidx = []
        # get index of categorical variables
        for c in cat:
            catidx.append(X.columns.get_loc(c))
        catidx = np.asarray(catidx)
        for p in np.arange(len(catdat.index)):
            ennemies = []
            step=step1
            a0, a1 = 0, step
            i=0
            #print("cat index=", p, "of", len(catdat.index))
            #print("a0, a1, step", a0, a1, step)
            layer_ = generate_layer_with_score_function(X, score_function, obs_to_interprete, d=X.shape[1], n=n_layer, segment=(a0, a1), propensityIndex=propensityIndex, outcomeIndex=outcomeIndex, catidx=catidx, catdat=p)
            layer_enn = [x for x in layer_ if x[-1] == 1-score_function_CLASS]
            j=0
            while len(layer_enn) > 0 and (len(adversaries[p]) < 1 or p in adversaries == False):
                if j == 1000 and len(layer_enn) == 10000:
                     D = pairwise_distances(layer_[:,outcomeIndex], pd.DataFrame(obs_to_interprete[outcomeIndex]).values.reshape(1, -1), metric='euclidean')
                     idxes = sorted(enumerate(D), key=lambda x:x[1]) 
                     adversaries[p] = [idxes[0]]
                #print('step pre divide', step)
                step = step / 100.0
                #print('step post divide', step)
                a1 = a1/100
                #print("begin layer with score_function in while loop 1", a0, a1)
                layer_ = generate_layer_with_score_function(X, score_function, obs_to_interprete, d=X.shape[1], n=n_layer, segment=(a0, a1), propensityIndex=propensityIndex, outcomeIndex=outcomeIndex, catidx=catidx, catdat=p)
                layer_enn = [x for x in layer_ if x[-1]== 1-score_function_CLASS]
                j += 1
                #print('zoom in')
            else:
                while len(ennemies) < 1:
                    #print("begin layer with score_function in else while loop 1")
                    layer_ = generate_layer_with_score_function(X, score_function, obs_to_interprete, d=X.shape[1], n=n_layer, segment=(a0, a1), propensityIndex=propensityIndex, outcomeIndex=outcomeIndex, catidx=catidx, catdat=p)
                    #print("end layer with score_function in else while loop")
                    layer_enn = [x for x in layer_ if x[-1] == 1-score_function_CLASS]
                    ennemies.extend(layer_enn)
                    i += 1
                    a0 += step
                    a1 += step
                    #print('zoom out', 'step:', step)
                    adversaries[p] = layer_enn
        #print("adversaries", adversaries)
        return adversaries
    else:
        ennemies = []
        step = step1
        a0, a1 = 0, step
        i=0
        layer_, seg, zmean = generate_layer_with_score_function(X, score_function, obs_to_interprete, d=X.shape[1], n=n_layer, segment=(a0, a1), propensityIndex=propensityIndex, outcomeIndex=outcomeIndex, catidx=None, catdat=None) # Using covariates that score_functionict outcome; n_layer=10000
        #print("end layer with score_function")
        layer_enn = [x for x in layer_ if x[-1] == 1-score_function_CLASS]
        j=0
        f=0
        while len(layer_enn) > 0:
            if j>100 and len(layer_enn) == 10000:
                D = pairwise_distances(layer_[:,outcomeIndex], obs_to_interprete[outcomeIndex].reshape(1, -1), metric='euclidean')
                idxes = sorted(enumerate(D), key=lambda x:x[1])
                i = idxes[0][0]
                ennemies.extend([layer_[i,:]])
                f = 1
                layer_enn = []
            else:
                step = step / 100.0
                a1 = a1/100
                layer_, seg, zmean = generate_layer_with_score_function(X, score_function, obs_to_interprete, d=X.shape[1], n=n_layer, segment=(a0, a1), propensityIndex=propensityIndex, outcomeIndex=outcomeIndex, catidx=None, catdat=None)
                layer_enn = [x for x in layer_ if x[-1] == 1-score_function_CLASS]
                j += 1
        else:
            while len(ennemies) < 1 and f!=1:
                #print("begin layer with score_function in else while loop 2")
                layer_, seg, zmean = generate_layer_with_score_function(X, score_function, obs_to_interprete, d=X.shape[1], n=n_layer, segment=(a0, a1), propensityIndex=propensityIndex, outcomeIndex=outcomeIndex, catidx=None, catdat=None)
                #print(pd.DataFrame(layer_).describe())
                #print("end layer with score_function in else while loop")
                layer_enn = [x for x in layer_ if x[-1] == 1-score_function_CLASS]
                ennemies.extend(layer_enn)
                i += 1
                if i == 3000:
                    sys.exit(zmean) 
                a0 += step
                a1 += step
                #print('zoom out') 
    #print('Final nb of iterations ', i, 'Final radius', (a0, a1))
    return ennemies
            
            
    
    
    
    


def growing_sphere_explanation(X, y, treat, score_function, obs_to_interprete, fe, dfe, propensityIndex, outcomeIndex,  idx, n_layer=10000, step=1/100000000.0, enough_ennemies=1, moving_cost=l2, cat=None, catdat=None):
    #print("begin seek_ennemies2")
    ennemies = seek_ennemies2(X, y, treat, score_function, obs_to_interprete, n_layer, step, enough_ennemies, fe, dfe, propensityIndex, outcomeIndex, idx, cat, catdat)
    print("ennemies type", type(ennemies))
    #print("end seek_ennemies2")
    #"#print('Final nb of iterations ', i, 'Final radius', (a0,a1))" & returns enemies
    #print("enemies:", ennemies)
    #print("adversaries:", ennemies)
    if cat != None:
        mc = []
        for u in np.arange(len(ennemies)):
            mc.append(moving_cost(pd.DataFrame(obs_to_interprete).iloc[propensityIndex].values, ennemies[u][0][propensityIndex]))
        nearest_idx = mc.index(min(mc))
        nearest_ennemy = ennemies[nearest_idx][0][:-1]
    else:
        nearest_ennemy = sorted(ennemies, key=lambda x: moving_cost(pd.DataFrame(obs_to_interprete).iloc[propensityIndex].values, ennemies[0][propensityIndex]))[0][:-1]
    #print(nearest_ennemy)
    return nearest_ennemy


def gs_main(X, y, treat, score_function, obs_to_interprete, propensityIndex, outcomeIndex, cat, idx, **kwargs):
    sparseAdv = []
    #layers = []
    #print("Start distance_first_enemy")
    dfe, fe = distance_first_ennemy2(X=X, observation=obs_to_interprete, propensityIndex=propensityIndex, outcomeIndex=outcomeIndex, score_function=score_function, treat=treat, idx=idx, n=1, cat=cat)
    #print("End distance_first_enemy fe:", fe, "dfe:", dfe)     
    if cat != None:
        Xcat = pd.DataFrame(X)[cat]
        XcatUnique = pd.DataFrame()
        for col in np.arange(np.size(Xcat, 1)):
            XcatUnique.insert(loc=int(col), column=col, value=np.unique(Xcat.iloc[:,col]))
        # Create dict; Key: Variable name; Value: Unique variable values
        catDict = XcatUnique.to_dict(orient="list")
        # Create DF with all combinations of unique categorical variable values
        XcatExpand = expand_grid(catDict)
        XcatExpand.columns=cat
        #print("grid expand")
        #print(XcatExpand.shape)
        # Create dataframes for all combinations of categorical variables
        dfs = {}
        for p in np.arange(len(XcatExpand.index)):
            dfs[p] = X[(X[cat] == XcatExpand.iloc[p,:]).all(axis=1)]
            #print("df created")         
        for j in range(len(dfs)):
            if dfs[j].shape[0] > 0:
                #print("DF", j, "has shape", dfs[j].shape)
                dat=dfs[j]
                #print("begin gs_explanation")
                step=1/100000.0
                enn = growing_sphere_explanation(X=dat, y=y, treat=treat, score_function=score_function, obs_to_interprete=obs_to_interprete, propensityIndex=propensityIndex, outcomeIndex=outcomeIndex, idx=idx, cat=cat, fe=fe, dfe=dfe, catdat=XcatExpand, step=step)
                #print("end growing_sphere_explanataion")
    #enn = nearest enemy
                sparseAdv.append(featred_random(obs_to_interprete, treat, enn, propensityIndex, outcomeIndex, score_function, idx))
                #print("end sparseADV")
                #layers.append(uni_layer)
    if cat == None:
        #print("cat=none gs_main", "gs_main:", dfe)
        step=1/100000.0
        enn = growing_sphere_explanation(X, y, treat, score_function=score_function, obs_to_interprete=obs_to_interprete, propensityIndex=propensityIndex, outcomeIndex=outcomeIndex, idx=idx, cat=cat, fe=fe, dfe=dfe, step=step)
        sparseAdv.append(featred_random(obs_to_interprete, treat, enn, propensityIndex, outcomeIndex, score_function, idx))
        #layers.append(uni_layer)
    return enn, sparseAdv


# DEFINE function that returns euclidean distance, the number of moves (number of different features) and the Pearson R2 between the observation of interest and nearest adversary
def interpretability_scores(obs_to_interprete, ennemy, propensityIndex, outcomeIndex):
    eucl = pairwise_distances(pd.DataFrame(obs_to_interprete).iloc[np.r_[propensityIndex]].values.reshape(1, -1), ennemy[propensityIndex].reshape(1, -1), metric='euclidean')[0][0]
    var_non0 = sum((obs_to_interprete[np.r_[propensityIndex]] - ennemy[propensityIndex]) != 0)
    pearson = pearsonr(obs_to_interprete[np.r_[propensityIndex]], ennemy[np.r_[propensityIndex]])[0]
    return {'distance': eucl, 'nb_directions_move': var_non0, 'pearson':pearson}
    
    
def interpretability_metrics(X, y, treat, score_function, obs_to_interprete, propensityIndex, outcomeIndex, cat, idx, **kwargs):
    #t1 = time.time()
    #print("interp metrics") 
    nearest_ennemy = gs_main(X, y, treat, score_function, obs_to_interprete, propensityIndex, outcomeIndex, cat, idx, **kwargs)
    #interpretability_method = the module - the main function of the corresponding file called thru benchmark_oneobs (mod.main in interpretability_metrics() arguments below)
    scores={}
    for i in range(len(nearest_ennemy)):
        scores[i] = interpretability_scores(obs_to_interprete, nearest_ennemy[i], propensityIndex, outcomeIndex)
    #scores['time'] = time.time() - t1
    return scores, nearest_ennemy


 
# From: https://pandas.pydata.org/pandas-docs/stable/user_guide/cookbook.html   
def expand_grid(data_dict):
    rows = it.product(*data_dict.values())
    return pd.DataFrame.from_records(rows, columns=data_dict.keys())


def distance_first_ennemy2(X, observation, propensityIndex, outcomeIndex, score_function, treat, idx, n=1, cat=None):
    if cat != None:
        #print("cat=none")
        # Subset unique values of categorical variables only
        Xcat = pd.DataFrame(X)[cat]
        XcatUnique = pd.DataFrame()
        for col in np.arange(np.size(Xcat, 1)):
            XcatUnique.insert(loc=int(col), column=col, value=np.unique(Xcat.iloc[:,col]))
        # Create dict; Key: Variable name; Value: Unique variable values
        catDict = XcatUnique.to_dict(orient="list")
        # Create DF with all combinations of unique categorical variable values
        XcatExpand = expand_grid(catDict)
        XcatExpand.columns=cat
        #print("grid expand")
        #print(XcatExpand.shape)
        
        # Create dataframes for all combinations of categorical variables
        dfs = {}
        for p in np.arange(len(XcatExpand.index)):
            dfs[p] = X[(X[cat] == XcatExpand.iloc[p,:]).all(axis=1)]
            #print("df created")
            
        # create dicts to hold instance with min distance and its values for each DF is dfs
        dist_dct={}
        out_dct={}
        # For each non null dataframe, calculate distance between values in dataframe and observation
        # Sort distances from smallest to greatest
        for j in np.arange(len(dfs)):
            if dfs[j].shape[0] > 0:
                #print("df", j, "has shape", dfs[j].shape)
                dat=dfs[j]
                D = pairwise_distances(dat.iloc[:,outcomeIndex], pd.DataFrame(observation).iloc[outcomeIndex].values.reshape(1, -1), metric='euclidean')
                #print("pairwise")
                idxes = sorted(enumerate(D), key=lambda x:x[1]) 
                out = []
                dists = []
                k = 0
                #print("begin while")
                # Loop through possible adversaries with smallest distance first
                # If score_functionicted treatment status (propensity score >/< .5) does not equal that of obs
                # then stop looping - found adversary with min distance
                # add distance of adversary with minimum distance to dist_dct for every DF in dfs
                # add values of adversary to out_dct
                # if all score_functionicted propensity score values equal the same as the obs
                while len(out) < 1 and k < len(idxes):
                    i = idxes
                    #if psm.score_functionict(dat.iloc[i[k][0], 0:10].values.reshape(1, -1))[0] != psm.score_functionict(observation[0:10].reshape(1, -1))[0]:
                    if np.round(score_function(X.iloc[i[k][0],:].values)) != treat[idx]:
                        out.append(X.iloc[i[k][0],:])
                        dists.append(pairwise_distances(X.iloc[i[k][0],outcomeIndex].values.reshape(1, -1), pd.DataFrame(observation[outcomeIndex]).values.reshape(1, -1), metric='euclidean')[0][0])
                        dist_dct[j] = dists
                        out_dct[j] = out
                        #print("k=", k, "i=", i[0])
                    else:
                        #print("no match")                            
                        k += 1
        return min(dist_dct.values())[0], out_dct[min(dist_dct, key=dist_dct.get)]
    elif cat==None:
        print("dfe2, no cat")
        observation=np.asarray(tuple(observation))
        D = pairwise_distances(X.iloc[:,outcomeIndex], pd.DataFrame(observation[outcomeIndex]).values.reshape(1, -1), metric='euclidean')
        D[D==0]=None
        # #print("D complete")
        idxes = sorted(enumerate(D), key=lambda x:x[1])
        out = []
        dists = []
        k = 0
        while len(out) < 1 and k < len(idxes):
            i = idxes
            if np.round(score_function(X.iloc[i[k][0], :].values.reshape(1, -1))) != treat[idx]:
                out.append(X.iloc[i[k][0],:])
                dists.append(pairwise_distances(X.iloc[i[k][0],outcomeIndex].values.reshape(1, -1), pd.DataFrame(observation[outcomeIndex]).values.reshape(1, -1))[0][0])
                    #print("k=", k, "i=", i[0])
            else:
                #print("else")
                #print("no match")                            
                k += 1
        return dists[0], out

    


def gs(observation, X, y, treat, propensityIndex, outcomeIndex, outcomeTrt, outcomeCtrl, idx, score_function, cat):
    X=pd.DataFrame(X)
    print("one")
    scores_c, nearest_ennemy1_c = interpretability_metrics(X=X, y=y, treat=treat, score_function=score_function, obs_to_interprete=observation, propensityIndex = propensityIndex, outcomeIndex=outcomeIndex, cat=cat, idx=idx)
    key_min = min(scores_c.keys(), key=(lambda k: scores_c[k]['distance']))
    #print('Minimum distance:', scores_c[key_min]['distance'])
    adversary = nearest_ennemy1_c[key_min]
    #print('Nearest adversary:', adversary)
    #print('Nearest adversary propensity score:', score_function(adversary[propensityIndex]))
    #print('Original observation propensity score:',      score_function(pd.DataFrame(observation).iloc[propensityIndex].values))
    # Individual treatment effect  
    treated = treat[idx]
#    def ite(original, counterfactual, treated, idx):
#        if treated==1:
#            y_trt=y[idx]
#            y_c=(counterfactual[outcomeIndex].reshape(1, -1))[0]
#        elif treated==0:
#            y_c=y[idx]
#            y_trt=outcomeTrt.predict(counterfactual[outcomeIndex].reshape(1, -1))[0]
#        else: 
#            print("Treated variable doesn't take on 0 or 1.")
#        return y_trt-y_c
#    ITE = ite(observation, adversary, treated=treated, idx=idx)
#    print("ITE:", ITE)
    return adversary, treated
    
    



    
