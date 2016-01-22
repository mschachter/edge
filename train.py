import tensorflow as tf
import InputGenerator
import lstm

filename = 'there_is_no_file_yet.txt'

input_generator = Input_Generator(filname, batch_size, n_bptt)

n_unit = 64

graph = tf.Graph()
with graph.as_default():
    lstm_layer = lstm.LSTM_Layer(num_nodes)

    
