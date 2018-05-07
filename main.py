import tensorflow as tf
import numpy as np
from models import Model, ModelDescriptor
from training import BackpropTrainer, ReinforceTrainer
from objectives import LossFunction, RewardFunction
from readwrite import ImageDataReader

import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# Specify parameters

number_of_glimpses = 12
number_of_scales = 3
glimpse_width = 12
glimpse_height = 12
num_glimpse_fc = 128
num_loc_fc = 128
glimpse_network_output_dimensionality = 256
core_network_state_units = 256
number_of_actions = 10
location_dimensionality = 2
batch_size = 200
data_path = '/share/jproject/fg538/r-006-gpu-3/mnist_cluttered_train_data.tfrecords'

# Create DataReader object
datareader = ImageDataReader(data_path,
                             'train/image',
                             'train/label',
                             batch_size,
                             (60, 60, 1),
                             one_hot_labels=True,
                             one_hot_depth=10)

# Read images and labels
images, labels = datareader.read()

# Create placeholders for input
X = tf.placeholder(tf.float32, [None, 60, 60, 1], name='X')
y = tf.placeholder(tf.float32, [None, 10], name='labels')

# Create Attention Model Descriptor

model_descriptor = ModelDescriptor(number_of_glimpses,
                                   number_of_scales,
                                   glimpse_width,
                                   glimpse_height,
                                   num_glimpse_fc,
                                   num_loc_fc,
                                   glimpse_network_output_dimensionality,
                                   core_network_state_units,
                                   number_of_actions,
                                   location_dimensionality,
                                   batch_size)

# Initialize Attention Model
                                   
attention_model = Model(model_descriptor)

# Build Attention Model

model_out = attention_model(X)

# Define Objective Functions

objective1 = LossFunction(model_out['ACTIONS'], y)
objective2 = RewardFunction(model_out['ACTIONS'], y, batch_size, number_of_glimpses)
objective3 = LossFunction(model_out['BASELINES'], objective2.rewards, tf.losses.mean_squared_error)

# Define Training Operations

bp_train1 = BackpropTrainer(tf.trainable_variables(), objective1.loss, tf.train.GradientDescentOptimizer(0.1))
bp_train2 = BackpropTrainer(attention_model.baseline_network.variable_collection, objective3.loss, tf.train.GradientDescentOptimizer(0.1))
rl_train = ReinforceTrainer(attention_model.location_network.variable_collection,
                            0.1,
                            0.9,
                            batch_size,
                            model_out['MEANS'],
                            model_out['LOCATIONS'],
                            model_out['STATES'],
                            objective2.rewards,
                            model_out['BASELINES_REDUCED'])

# Define Metrics

correct_prediction = tf.equal(tf.argmax(model_out['ACTIONS'], 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
expected_reward = tf.reduce_mean(objective2.rewards)

saver = tf.train.Saver()

# Configure Session

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.33

# Define numpy arrays to store accuracy, loss and rewards for later use

train_acc = np.zeros([5000])
train_reward = np.zeros([5000])
train_loss = np.zeros([5000])

# Start the Session

with tf.Session(config=config) as sess:

    # Initialize global and local variables
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    # Define summary-writer
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter('log', sess.graph)

    # Initialize threads-coordinator
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # Start training
    print('Starting training...')
    for i in range(5000):
        
        batch_x, batch_lbl = sess.run([images, labels])
        
        sess.run([bp_train1.train_op, bp_train2.train_op, rl_train.train_op], feed_dict={X: batch_x, y: batch_lbl})
        acc, loss, reward, locs = sess.run([accuracy, objective1.loss, expected_reward, model_out['LOCATIONS']], 
                                     feed_dict={X: batch_x, y: batch_lbl}) 
        print(locs)
        train_acc[i] = acc
        train_reward[i] = reward
        train_loss[i] = loss
        
        s = sess.run(merged_summary, feed_dict={X: batch_x, y: batch_lbl})
        writer.add_summary(s, i)

        if (i+1) % 10 == 0:
            print('Step {}: train_accuracy={}, train_loss={}, expected_reward = {}'.format(i+1, acc, loss, reward))

        if (((i+1) % 10 == 0) and (acc > 0.90)):
            params = saver.save(sess, '/N/u/rrrane/Project/model/{}_{}.ckpt'.format(output, i+1))
            print('Model saved: {}'.format(params))

    coord.request_stop()
    coord.join(threads)
