################################
# EvoMan FrameWork - V1.0 2016 #
# Author: Karine Miras         #
# karine.smiras@gmail.com      #
################################

# imports framework
import sys, os
import numpy as np
import neat

sys.path.insert(0, "evoman")
from environment import Environment
from specialist_controller import NEAT_Controls
import pickle


# choose this for not using visuals and thus making experiments faster
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

n_hidden_neurons = 10


# runs simulation
def simulation(env, x):
    f, p, e, t = env.play(pcont=x)
    return f


# funtcion to evaluate fitness and save needed outputs
def eval_genomes(genomes, config):
    generation = []
    global gen
    for genome_id, genome in genomes:
        genome.fitness = 0
        genome.fitness = simulation(env, genome)
        generation.append(genome.fitness)
    fitness_gens.append(np.mean(generation))
    fitness_max.append(np.max(generation))
    fitness_std.append(np.std(generation))
    file_aux = open(f"{experiment_name}/EXP_{i+1}" + "/results.txt", "a")
    file_aux.write("\ngen best mean std")
    file_aux.write(
        f"\ngen-{gen}: "
        + str((fitness_max[-1]))
        + " "
        + str((fitness_gens[-1]))
        + " "
        + str((fitness_std[-1]))
    )
    file_aux.close()
    gen = gen + 1


# Main part of the algorithm, it runs the NEAT algo
def run(config_file):
    """
    It uses the config file named config-feedforward.txt
    """
    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_file,
    )

    # Create the population
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))

    # Run for up to 25 generations.
    winner = p.run(eval_genomes, 25)

    # Save the best solution with pickle
    with open(f"{experiment_name}/EXP_{i+1}/winner.pkl", "wb") as f:
        pickle.dump(winner, f)
        f.close()
    # show final stats
    print("\nBest genome:\n{!s}".format(winner))


if __name__ == "__main__":

    N_runs = 10

    experiment_name = "Neat_enemies_78"
    # create directory if it does not exist yet
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    # find config file and create a neat config
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-feedforward.txt")

    # initializes simulation in individual evolution mode, for single static enemy.
    env = Environment(
        experiment_name=experiment_name,
        enemies=[7, 8],
        playermode="ai",
        player_controller=NEAT_Controls(),
        enemymode="static",
        level=2,
        speed="fastest",
        multiplemode="yes",
        randomini="yes",
    )

    # default environment fitness is assumed for experiment
    env.state_to_log()  # checks environment state
    for i in range(N_runs):
        if not os.path.exists(f"{experiment_name}/EXP_{i+1}"):
            os.makedirs(f"{experiment_name}/EXP_{i+1}")
        # global variables for saving mean and max each generation
        fitness_gens = []
        fitness_max = []
        fitness_gens = []
        fitness_max = []
        fitness_std = []
        gen = 0
        run(config_path)
