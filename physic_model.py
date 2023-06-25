import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

'''dataset_path = 'C:\\Users\\chh36\\Desktop\\data.csv'
dataset = pd.read_csv(dataset_path)

event_nums = dataset['event_num'].unique()
seq_list = []

for num in event_nums:
    df = dataset[dataset['event_num'] == num]
    df = df.drop(columns='event_num')
    df = df.to_numpy()
    seq_list.append(df)'''

def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2*(i//2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)

n, d = 20, 6
pos_encoding = positional_encoding(n, d)
print(pos_encoding.shape)
pos_encoding = pos_encoding[0]

# Juggle the dimensions for the plot
pos_encoding = tf.reshape(pos_encoding, (n, d//2, 2))
pos_encoding = tf.transpose(pos_encoding, (2, 1, 0))
pos_encoding = tf.reshape(pos_encoding, (d, n))

def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]

def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask

x = tf.random.uniform((1, 3))
temp = create_look_ahead_mask(x.shape[1])
print(temp)