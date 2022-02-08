# the demo_controller file contains standard controller structures for the agents.
# you can overwrite the method 'control' in your own instance of the environment
# and then use a different type of controller if you wish.
# note that the param 'controller' received by 'control' is provided through environment.play(pcont=x)
# 'controller' could contain either weights to be used in the standard controller (or other controller implemented),
# or even a full network structure (ex.: from NEAT).
import sys, os
sys.path.insert(0, 'evoman') 
from environment import Environment
from controller import Controller
import numpy as np
import neat
import random
import pickle


# CONFIGURATION
experiment_name = 'dummy_NEAT_2'
mode = 'test' # Either train or test. In case of test there should be a pickle file present
generations = 100
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# List of enemies to beat. Multiple enemies will result in an iteration of separate training sessions (specialist training)
# Works in both train and test stages, given for testing a pickle file is present for each enemy in the structure: winner_x.pkl where x = enemy number.
enemies = [2] 


class player_controller(Controller):
	def __init__(self, net):
		self.net = net

	def control(self, inputs, controller):
		# Normalises the input using min-max scaling
		inputs = (inputs-min(inputs))/float((max(inputs)-min(inputs))) # Array with 20 inputs between 0 and 1

		output = self.net.activate(inputs) # Activate the FF NN with the normalized 20 sensory inputs

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


def run_from_solution(config_path, genome_path, enemy):
	# Function for running the game in test mode (with a given solution)

	# Load requried NEAT config
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    # Unpickle saved winner
    with open(genome_path, "rb") as f:
        genome = pickle.load(f)

    # Convert loaded genome into required data structure
    genomes = [(1, genome)]

    # Load the one genome that is loaded
    net = neat.nn.FeedForwardNetwork.create(genomes[0][1], config) 

    # Call game with only the loaded genome
    env = Environment(experiment_name=experiment_name,
						  playermode="ai",
						  enemies=[enemy],
					  	  speed="fastest",
					  	  player_controller=player_controller(net),
						  enemymode="static",
						  level=2,
						  randomini="yes")

    f, p, e, t = env.play()

    print('End of game, fitness was: {}'.format(f))



if __name__ == "__main__":
	local_dir = os.path.dirname(__file__)
	config_path = os.path.join(local_dir, "config-feedforward.txt")
	path = 'winner_2.pkl'
	for e in range(1, 9):
		run_from_solution(config_path, genome_path = path, enemy = e)

	
	