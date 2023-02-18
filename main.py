import argparse
import os
from connect_four.connect_four_game import ConnectFourGame
from neural_net_wrapper import NeuralNetworkWrapper
from train import Train
from human_play import HumanPlay


#parameters 
num_iterations = 4
num_games = 30
num_mcts_sims = 30
c_puct = 1
l2_val = 0.0001
momentum = 0.9
learning_rate = 0.01
t_policy_val = 0.0001
temp_init = 1
temp_final = 0.001
temp_thresh = 10
epochs = 10
batch_size = 128
dirichlet_alpha = 0.5
epsilon = 0.25
model_directory = "./connect_four/models/"
num_eval_games = 12
eval_win_rate = 0.55
load_model = 1
human_play = 0
resnet_blocks = 5
record_loss = 1
loss_file = "loss.txt"

# Read command line arguments : load_model and human_play
parser = argparse.ArgumentParser()
parser.add_argument("--load_model",
                    help="Bool to initialize the network with the best model.",
                    dest="load_model",
                    type=int,
                    default=load_model)

parser.add_argument("--human_play",
                    help="Bool to play as a Human vs the AI.",
                    dest="human_play",
                    type=int,
                    default=human_play)



if __name__ == '__main__':
    arguments = parser.parse_args()
    human_play = arguments.human_play
    load_model = arguments.load_model
    game = ConnectFourGame()
    net = NeuralNetworkWrapper(game)

    if load_model:
        file_path = model_directory + "best_model.meta" # Initialize the network with the best saved model
        if os.path.exists(file_path):
            net.load_model("best_model")
        else:
            print("Trained model doesn't exist. Starting from scratch.")
    else:
        print("Trained model not loaded. Starting from scratch.")

    if human_play: # If human_play=1, play as a human vs the AI 
        human_play = HumanPlay(game, net)
        human_play.play()
    else: # If human_play=0, training
        train = Train(game, net)
        train.start()
