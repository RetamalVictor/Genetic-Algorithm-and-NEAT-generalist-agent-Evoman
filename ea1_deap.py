# Parts of this code have been retrieved from: https://deap.readthedocs.io/en/master/ -- the DEAP documentation

# imports framework
import random
import sys

sys.path.insert(0, "evoman")

from environment import Environment
from demo_controller import player_controller

import numpy as np
import os
from deap import creator, base, tools
from random import random

# not using visuals -- speeds up the process
os.environ["SDL_VIDEODRIVER"] = "dummy"

# create the number of neurons
n_hidden_neurons = 10

# set the experiment name and create a directory if needed
experiment_name = "ea1_deap_enemies278_new_run10"
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# initializes simulation in multi evolution mode, for multiple static enemies.
env = Environment(
    experiment_name=experiment_name,
    enemies=[2, 7, 8],
    multiplemode="yes",
    playermode="ai",
    player_controller=player_controller(n_hidden_neurons),
    enemymode="static",
    level=2,
    speed="fastest",
    randomini="yes",
)

# save the environment state in the directory
env.state_to_log()

# ------- PARAMETERS OF INTEREST -----------

# concerning the run of the entire evolution
npop = 100
gens = 24

# concerning what happens in one gen
mutation_probability = 0.25
individual_mutation_probability = 0.1
crossover_probability = 0.5
alpha = 0.5

# concerning the neural network
lower, upper = -1.0, 1.0  # bounds on weights
n_vars = (env.get_num_sensors() + 1) * n_hidden_neurons + (
    n_hidden_neurons + 1
) * 5  # the weights themselves


# ---- SCHEME ----
# representation: neural network (demo controller)
# parent selection: roulette wheel
# survivor selection: (mu, lambda)
# crossover: blend alpha (+ limit)
# mutation: uniform

# ----- Basic simulation function ----

# runs simulation and returns tuple(!) with only fitness
def simulation(x, env):
    f, p, e, t = env.play(pcont=x)  # convert into a np array before evaluating
    return (f,)


# ------ DEAP -----

# create fitness (only max because we only need to optimize one)
creator.create("FitnessMax", base.Fitness, weights=(1.0,))

# create individual (list version-- we turn this into an numpy array when it needs to be evaluated)
creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

# create an individual
toolbox = base.Toolbox()
toolbox.register("create_gene", np.random.uniform, lower, upper)
toolbox.register(
    "create_individual",
    tools.initRepeat,
    creator.Individual,
    toolbox.create_gene,
    n=n_vars,
)

# create the population and an evaluation method
toolbox.register("pop", tools.initRepeat, list, toolbox.create_individual)
toolbox.register("evaluate", simulation, env=env)

# after creating the population - initialize the functions for the type of evolution
toolbox.register("mate", tools.cxBlend, alpha=alpha)
toolbox.register(
    "mutate",
    tools.mutUniformInt,
    low=lower,
    up=upper,
    indpb=individual_mutation_probability,
)

# selecting
toolbox.register("select", tools.selRoulette)
toolbox.register("survivor", tools.selBest)

# for the statistics
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("max", np.max)
stats.register("mean", np.mean)
stats.register("std", np.std)


# ------ RUNNING THE EA --------


def main():
    # create a population of a certain size
    pop = toolbox.pop(n=npop)

    # to save the best solution
    hall_of_fame = tools.HallOfFame(1, similar=np.array_equal)

    # to save results
    logbook = tools.Logbook()

    for i in pop:
        fitness = toolbox.evaluate(i)  # (in numpy array form) evaluate every individual
        i.fitness.values = fitness  # add the fitness value to the individual

    # save the best solution
    hall_of_fame.update(pop)

    # log the results
    recording = stats.compile(pop)
    logbook.record(
        gen=0,
        evals=len(pop),
        mean=recording["mean"],
        max=recording["max"],
        std=recording["std"],
    )

    # print the results after a generation is done
    print("gen: 0, best: ", recording["max"], "evals:", len(pop))
    print("mean: ", recording["mean"], ", std: ", recording["std"])

    # for a fixed number of generations, start recording this
    for generation in range(gens):
        # select number of offspring based on parents --> population size * 4
        next_gen = toolbox.select(pop, npop * 4)
        next_gen = list(
            map(toolbox.clone, next_gen)
        )  # note - we make this a list so we can iterate over this

        # split the list in two (consecutive kids will be paired)
        first_children = next_gen[::2]
        second_children = next_gen[1::2]

        # crossover
        for child1, child2 in zip(first_children, second_children):
            if random() < crossover_probability:
                # crossover
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        # mutate
        for mutant in next_gen:
            if random() < mutation_probability:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # make sure it is between the bounds! (needs to be done because of blend crossover)
        for individual in next_gen:
            for i in range(len(individual)):
                if individual[i] > upper:
                    individual[i] = upper
                elif individual[i] < lower:
                    individual[i] = lower

        # see which ones don't have a fitness anymore:
        individual_no_fitness = [ind for ind in next_gen if not ind.fitness.valid]

        # calculate fitness again if needed
        for i in individual_no_fitness:
            fitness = toolbox.evaluate(i)
            i.fitness.values = fitness

        # decide which members are going to survive (mu, lambda)
        next_gen_survivors = toolbox.survivor(next_gen, npop)

        # record statistics:
        hall_of_fame.update(next_gen_survivors)

        recording = stats.compile(next_gen_survivors)
        logbook.record(
            gen=generation + 1,
            evals=len(next_gen_survivors),
            mean=recording["mean"],
            max=recording["max"],
            std=recording["std"],
        )

        print(
            "gen: ",
            generation + 1,
            ", best: ",
            recording["max"],
            ", total best: ",
            hall_of_fame.items[0].fitness.values[0],
        )
        print(
            "mean: ",
            recording["mean"],
            ", std: ",
            recording["std"],
            "nr evals: ",
            len(next_gen_survivors),
        )

        # the population is entirely replaced by the best offspring
        pop[:] = next_gen_survivors

    # ---- FINAL RESULTS -----
    gen, max, mean, std = logbook.select("gen", "max", "mean", "std")

    # write into file
    print("printing to file....")

    # basic structure of the result file
    file_results = open(experiment_name + "/results.txt", "a")
    file_results.write("\n\ngen best mean std")

    # write in the results
    for i in range(len(gen)):
        file_results.write(
            "\n"
            + str(gen[i])
            + " "
            + str(round(max[i], 6))
            + " "
            + str(round(mean[i], 6))
            + " "
            + str(round(std[i], 6))
        )

    # close file writer
    file_results.flush()
    file_results.close()

    # saves best solution
    np.savetxt(experiment_name + "/best.txt", hall_of_fame.items[0])

    print("Finished experiment!")

    return pop


# start the simulation
if __name__ == "__main__":
    main()
