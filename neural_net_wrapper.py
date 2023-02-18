from neural_net import NeuralNetwork
import numpy as np
import os


class NeuralNetworkWrapper(object):
   
    def __init__(self, game):
        self.game = game #contains the game state
        self.net = NeuralNetwork(self.game) 
        self.sess = self.net.sess
        self.model_directory = "./connect_four/models/"
        self.epochs = 50
        self.batch_size = 128
        self.record_loss = 1
        self.loss_file = "loss.txt"

    def predict(self, state):
        state = state[np.newaxis, :, :] 
        pi, v = self.sess.run([self.net.pi, self.net.v],
                              feed_dict={self.net.states: state,
                                         self.net.training: False}) #probabilities and state values

        return pi[0], v[0][0]

    def train(self, training_data):

        print("\nTraining the network.\n")
        for epoch in range(self.epochs):
            print("Epoch", epoch + 1)
            examples_num = len(training_data)
            
            for i in range(0, examples_num, self.batch_size):
                states, pis, vs = map(list,
                                      zip(*training_data[i:i + self.batch_size]))
                feed_dict = {self.net.states: states,
                             self.net.train_pis: pis,
                             self.net.train_vs: vs,
                             self.net.training: True}
                self.sess.run(self.net.train_op,
                              feed_dict=feed_dict)
                pi_loss, v_loss = self.sess.run(
                    [self.net.loss_pi, self.net.loss_v],
                    feed_dict=feed_dict)
                if self.record_loss:
                    if not os.path.exists(self.model_directory):
                        os.mkdir(self.model_directory)
                    file_path = self.model_directory + self.loss_file
                    with open(file_path, 'a') as loss_file:
                        loss_file.write('%f|%f\n' % (pi_loss, v_loss))
        print("\n")

    def save_model(self, filename="current_model"):
        if not os.path.exists(self.model_directory):
            os.mkdir(self.model_directory)
        file_path = self.model_directory + filename
        print("Saving model:", filename, "at", self.model_directory)
        self.net.saver.save(self.sess, file_path)

    def load_model(self, filename="current_model"):
        file_path = self.model_directory + filename
        print("Loading model:", filename, "from", self.model_directory)
        self.net.saver.restore(self.sess, file_path)
