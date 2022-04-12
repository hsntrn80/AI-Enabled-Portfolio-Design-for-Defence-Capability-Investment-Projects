# %%
#A simple test instance generator
#records to .json file to be read in code running stage
#cost of the project and success probability of the project
import random
import numpy as np
import json
import time
import os

import torch
import torch.optim as optim


from deap import base
from deap import creator
from deap import tools


from sa_mutations import *
from reinforce import *

# %%
# reproducability
random.seed(60)
np.random.seed(60)
torch.manual_seed(60)

# %%
def Generate_instance(N, cost_min, cost_max, p_success_min, p_success_max, budget):
    problem=[]
    for i in range(N):
        cost= random.randint(cost_min, cost_max)
        p_success= random.uniform(p_success_min, p_success_max)
        problem.append([cost, p_success])
    return  [problem,budget]

# %%
def fitness_evaluator(problem_input, theta, alpha, replication, investment_portfolio):
    investment_portfolio = np.array(investment_portfolio)
    budget=problem_input[1]
    cost_vector=[problem_input[0][i][0] for i in range(len(investment_portfolio))]
    success_vector=[problem_input[0][i][1] for i in range(len(investment_portfolio))]
    if  sum(investment_portfolio*cost_vector)>budget: #budget feasibility check
        return (float('-inf'),)
    else:
        replication_list=[]
        for i in range(replication):
            random_scenario=[random.uniform(0,1) for i in range(N)] # will generate this based on num replication
            a=[int(x >= y) for x, y in zip(success_vector, random_scenario)]
            num_success=sum(a*investment_portfolio)
            replication_list.append(num_success)
        replication_list=np.array(replication_list)
        varSuccess = np.percentile(replication_list, alpha)
        cvarSuccess = replication_list[replication_list <= varSuccess].mean()
        return (theta*np.mean(replication_list)+ (1-theta)*cvarSuccess, )

# %%
def run_algorithm(case_id, problem_input, theta, alpha, replication, N):
    # 1 is for maximization -1 for minimization
    # Maximize total cost
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    def generateIndividual(N):
        # Generating initial indvidual that are in the range of given max-min cluster numbers
        individual = list(np.random.choice([0, 1], size=N))

        # print type (creator.Individual(individual))
        return creator.Individual(individual)

    toolbox = base.Toolbox()
    toolbox.register("individual", generateIndividual, N)
    # define the population to be a list of individuals
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)


    # ----------
    # Operator registration
    # ----------
    # register the goal / fitness function
    toolbox.register("evaluate", fitness_evaluator, problem_input, theta, alpha, replication)

    # register the crossover operator
    toolbox.register("mate", tools.cxTwoPoint)

    # register a mutation operator with a probability to
    # flip each attribute/gene of 0.05
    #
    # toolbox.register("mutate", swicthtoOtherMutation, indpb=0.4)
    # toolbox.register("mutate", swicthtoOtherMutation)

    # operator for selecting individuals for breeding the next
    # generation: each individual of the current generation
    # is replaced by the 'fittest' (best) of three individuals
    # drawn randomly from the current generation.

    toolbox.register("select", tools.selTournament, tournsize=10)

    # create an initial population of 100 individuals (where
    # each individual is a list of integers)

    start_time = time.time() #start time
    pop = toolbox.population(n=1)
    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    # Extracting all the fitnesses of
    fits = [ind.fitness.values[0] for ind in pop]
    best_cost=max(fits)
    S_best=tools.selBest(pop, 1)[0]


    # RL Parameters
    batch_size = 32
    gamma = 0.999
    eps_start = 1
    eps_end = 0.01
    eps_decay = 0.001
    target_update = 10
    memory_size = 100000
    lr = 0.001
    num_episodes = 100

    n_neighborhood_functions = 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    em = EnvironmentManager(device, toolbox.evaluate, S_best, n_neighborhood_functions)
    strategy = EpsilonGreedyStrategy(eps_start, eps_end, eps_decay)
    agent = Agent(strategy, em.num_actions_available(), device)
    memory = ReplayMemory(memory_size)

    policy_net = DQN(len(S_best),em.num_actions_available()).to(device)
    target_net = DQN(len(S_best),em.num_actions_available()).to(device)

    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()  # don't train target net
    optimizer = optim.Adam(params=policy_net.parameters(), lr=lr)

    steps_RL = 128
    best_rl_s_x = S_best
    best_rl_cost = best_cost
    tc_best = best_rl_cost
    new_best_found = False
    steps_SA = 1000

    print("lr:", lr)
    print("num episodes:", num_episodes)
    print("batch_size", batch_size)
    print("steps SA", steps_SA)
    print("steps RL", steps_RL)

    stats = {}
    for episode in range(1, num_episodes+1):
        new_best_found = False
        print(f"Episode {episode}")
        total_rewards = 0
        total_loss =0
        # em.reset()
        state = em.get_state()   # state is the solution S
        # run RL for a number of steps
        start_tstep = time.time() #start time
        stats[episode] = {}
        stats[episode]['rl'] = []
        for timestep in range(1, steps_RL+1):
            # select action
            action = agent.select_action(state, policy_net)
            # get reward
            new_cost, reward = em.take_action(action[0])
            total_rewards += reward.item()
            # get next state
            next_state = em.get_state()
            # store experience in replay memory
            memory.push(Experience(torch.FloatTensor(state)/N, action, torch.FloatTensor(next_state)/N, reward))
            # switch to next state
            state = next_state
            if new_cost >= best_rl_cost:
                best_rl_cost = new_cost
                best_rl_s_x = state
                new_best_found = True
                print(f"{bcolors.OKGREEN}Better Objective in RL {new_cost}{bcolors.ENDC}")
                #print(f"{bcolors.OKGREEN}{state[0]}{bcolors.ENDC}")
            # optimize policy network
            if memory.can_provide_sample(batch_size):
                experiences = memory.sample(batch_size)
                states, actions, rewards, next_states = extract_tensors(experiences)

                current_q_values = QValues.get_current(policy_net, states, actions)  # current q values q(s,a)
                next_q_values = QValues.get_next(target_net, next_states)  # max term of bellman equation
                target_q_values = (next_q_values * gamma) + rewards  # optimal q values --> q*(s,a)

                loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1))  # loss = q*(s,a) - q(s,a)
                total_loss += loss.item()
                # print("Loss ", loss.data)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            stats[episode]['rl'].append({'time_step':timestep, 'best_rl_cost':best_rl_cost, 'state_cost':new_cost})
        end_tstep = time.time() #start time
        stats[episode]['rl_duration'] = end_tstep - start_tstep
        if episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())
        # print(f"{bcolors.OKBLUE}Total Reward: {total_rewards:.2f} Total Loss: {total_loss:.2f}{bcolors.ENDC}")

        if new_best_found :
            s_x = best_rl_s_x[0]
            tc_best = best_rl_cost
            S_best = s_x
        else:
            s_x = state[0]

        tc_s_x = toolbox.evaluate(s_x)[0]
        tm = 5
        T=tm
        t0 = 100000
        a = -math.log(T/t0)
        y = steps_SA
        q=0
        #print("---SA---")
        stats[episode]['sa'] = []
        start_sa = time.time() #start time
        for t in range(y):
            action, s_prime = neighborhood_solution(s_x)
            tc_sprime = toolbox.evaluate(s_prime)[0]
            delta_E = tc_sprime - tc_s_x
            memory.push(Experience(torch.FloatTensor([s_x])/N, torch.tensor([action]), torch.FloatTensor([s_prime])/N, torch.FloatTensor([delta_E])))
            if delta_E >= 0:
                # print ("Better neighbour found at itr= {}".format(itr))
                s_x = s_prime
                tc_s_x = tc_sprime
                if tc_sprime >= tc_best:
                    print(f"{bcolors.OKGREEN}Better Objective in SA {tc_sprime} t= {t}{bcolors.ENDC}")
                    #print(f"{bcolors.OKGREEN}{s_x}{bcolors.ENDC}")
                    tc_best = tc_sprime
                    S_best = s_x
            else:
                r = random.uniform(0, 1)
                if r < math.exp(-delta_E / T):
                    s_x = s_prime
                    tc_s_x = tc_sprime

            T = tm * math.exp(a*q/y)
            q += 1
            stats[episode]['sa'].append({'iter':t, 'best_sa_cost':tc_best, 'state_cost':tc_sprime})
        assert t==y-1
        end_sa = time.time() #start time
        stats[episode]['sa_duration'] = end_sa - start_sa
        # SA finished
        best_rl_s_x = S_best
        best_rl_cost = tc_best
        em.setState(S_best)

    # save RL statistics
    os.makedirs("results", exist_ok=True)
    with open(f'results/{case_id}_stats.json', 'w') as fp:
        json.dump(stats, fp)

    pop[0] = creator.Individual(S_best)
    TCs = list(map(toolbox.evaluate, pop))
    for ind, tc in zip(pop, TCs):
        ind.fitness.values = tc
    best_ind = tools.selBest(pop, 1)[0]
    # print("Best individual is %s, %s" % (individual2cluster(best_ind), best_ind.fitness.values))
    return best_ind.fitness.values, best_ind


# %%
cost_min, cost_max=  1, 100     #integer values are preferable
p_success_min, p_success_max= 0.01, 0.8 #between 0 and 1
theta, alpha, replication = (1, 10, 15)
for idx, N in enumerate([20, 50, 100]):
    budget = int((cost_min+cost_max)/2.0 * N/2)
    problem_input = Generate_instance(N, cost_min, cost_max, p_success_min, p_success_max, budget)
    best_cost, best_ind = run_algorithm(idx, problem_input, theta, alpha, replication, N)
    print(idx, best_cost, best_ind)
