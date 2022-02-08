# script to test the best solutions and write them to a file
# just add the name of the file and how you want the new file to be called

import sys

sys.path.insert(0, 'evoman')
from environment import Environment
from demo_controller import player_controller
import os
import numpy as np

# no visuals
os.environ["SDL_VIDEODRIVER"] = "dummy"

# where to find the best solution
# run this for all enemies
file_name_base = 'ea1_deap_enemies278_new/ea1_deap_enemies278_new_run'

for i in range(1, 11):
    # where to save it
    experiment_name = file_name_base + str(i) + "/test_best"

    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    n_hidden_neurons = 10

    # initializes simulation in individual evolution mode, for single static enemy.
    env = Environment(experiment_name=experiment_name,
                      enemies=[1,2,3,4,5,6,7,8],
                      multiplemode="yes",
                      playermode="ai",
                      player_controller=player_controller(n_hidden_neurons),
                      enemymode="static",
                      level=2,
                      speed="fastest",
                      randomini="yes")

    # write it into columns of fitness - player life - enemy life - time
    file_results = open(experiment_name + '/best_test_results.txt', 'a')
    file_results.write('f p e t')
    bsol = np.loadtxt(file_name_base + str(i) + "/best.txt")

    # loads file with the best solution for testing
    for i in range(5):
        f, p, e, t = env.play(pcont=bsol)
        file_results.write(
            '\n' + str(f) + ' ' + str(p) + ' ' + str(e) + ' ' + str(t))

    file_results.flush()
    file_results.close()


