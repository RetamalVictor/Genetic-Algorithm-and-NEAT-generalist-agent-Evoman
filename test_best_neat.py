import sys, os

sys.path.insert(0, "evoman")
from environment import Environment
from controller import Controller
import numpy as np
import neat
import random
import pickle

# CONFIGURATION
experiment_name = "Neat_enemies_278/EXP_10/best_results"

os.environ["SDL_VIDEODRIVER"] = "dummy"

if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)


class player_controller(Controller):
    def __init__(self, net):
        self.net = net

    def control(self, inputs, controller):
        # Normalises the input using min-max scaling
        inputs = (inputs - min(inputs)) / float(
            (max(inputs) - min(inputs))
        )  # Array with 20 inputs between 0 and 1

        output = self.net.activate(
            inputs
        )  # Activate the FF NN with the normalized 20 sensory inputs

        # takes decisions about sprite actions
        if output[0] > 0.5:
            left = 1
        else:
            left = 0

        if output[1] > 0.5:
            right = 1
        else:
            right = 0

        if output[2] > 0.5:
            jump = 1
        else:
            jump = 0

        if output[3] > 0.5:
            shoot = 1
        else:
            shoot = 0

        if output[4] > 0.5:
            release = 1
        else:
            release = 0

        return [left, right, jump, shoot, release]


def run_from_solution(config_path, genome_path):
    # Function for running the game in test mode (with a given solution)

    # Load requried NEAT config
    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    # Unpickle saved winner
    with open(genome_path, "rb") as f:
        genome = pickle.load(f)

    # Convert loaded genome into required data structure
    genomes = [(1, genome)]

    # Load the one genome that is loaded
    net = neat.nn.FeedForwardNetwork.create(genomes[0][1], config)

    # Call game with only the loaded genome
    env = Environment(
        experiment_name=experiment_name,
        playermode="ai",
        enemies=[1, 2, 3, 4, 5, 6, 7, 8],
        speed="fastest",
        player_controller=player_controller(net),
        enemymode="static",
        level=2,
        randomini="yes",
        multiplemode="yes",
    )

    # create a file for the best results
    file_results = open(experiment_name + "/best_test_results.txt", "a")
    file_results.write("f p e t")

    # play games and write in
    for i in range(5):
        f, p, e, t = env.play()
        file_results.write("\n" + str(f) + " " + str(p) + " " + str(e) + " " + str(t))

    # close file
    file_results.flush()
    file_results.close()


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-feedforward.txt")
    path = "Neat_enemies_278/EXP_10/winner.pkl"

    run_from_solution(config_path, genome_path=path)
