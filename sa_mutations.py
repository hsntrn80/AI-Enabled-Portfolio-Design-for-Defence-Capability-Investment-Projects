import random
import numpy as np

def mutate1(S): # mutate one gene
    S_new = S[:]
    numHubs = len(S)
    idx1 = random.randint(0, numHubs-1)
    ex_state = S[idx1]
    S_new[idx1] = (ex_state+1) % 2  # 0 to 1 and 1 to 0
    return S_new

def mutate2(S): # mutate 2 random genes
    S_new = S[:]
    numHubs = len(S)
    idx1, idx2 = random.sample(range(0, numHubs), 2)
    ex_state = S[idx1]
    S_new[idx1] = (ex_state+1) % 2  # 0 to 1 and 1 to 0
    ex_state = S[idx2]
    S_new[idx2] = (ex_state+1) % 2  # 0 to 1 and 1 to 0
    return S_new 

def neighborhood_solution(S):
    if random.uniform(0, 1) <= 0.5:  # mutate one 
        S_new = mutate1(S)
        action = 0
    else: # mutate two
        S_new = mutate2(S)
        action = 1
    return action, S_new