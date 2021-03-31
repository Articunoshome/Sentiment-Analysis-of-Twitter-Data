#/usr/bin/env python

import tensorflow as tf
import tensorflow_datasets as tfds
import pandas as pd
import matplotlib.pyplot as plt

from termcolor import colored

# Load data
print(colored("Loading train and test data", "yellow"))
train = pd.read_csv('new_train.csv')
test = pd.read_csv('new_test.csv')
print(colored("Data loaded", "green"))

print(train.head(5).to_numpy())
train_data = train.to_numpy().astype('U240')[train.to_numpy() != 'Clean_tweet']
train_data = train_data[train_data != 'Sentiment'].reshape(train_data.shape[0]//2, 2)
train_set = tf.data.Dataset.from_tensor_slices(train_data)

test_data = test.to_numpy().astype('U240')[test.to_numpy() != 'Clean_tweet']
test_data = test_data[test_data != 'Sentiment'].reshape(test_data.shape[0]//2, 2)
test_set = tf.data.Dataset.from_tensor_slices(test_data)

labels = {'Neutral': 1.0, 'Positive': 2.0, 'Negative': 0.0}

# Create Vocabulary Set
print(colored("Encoding Text", "yellow"))

tokenizer = tfds.features.text.Tokenizer()

vocabulary_set = set()
for text_tensor, _ in train_set:
    some_tokens = tokenizer.tokenize(text_tensor.numpy())
    vocabulary_set.update(some_tokens)

vocab_size = len(vocabulary_set)
vocab_size

'''
The encoder's encode method takes in a string of text and returns a list of integers.
'''

encoder = tfds.features.text.TokenTextEncoder(vocabulary_set)
#encoder = tfds.features.text.SubwordTextEncoder(vocabulary_set)

def encode_fn(text_tensor, label):
    encoded_text = encoder.encode(text_tensor.numpy())
    return encoded_text, labels[label.numpy().decode('utf-8')]

def encode_map_fn(text, label):
    # py_func doesn't set the shape of the returned tensors.
    encoded_text, label = tf.py_function(encode_fn, inp=[text, label], Tout=(tf.int64, tf.float32))

    # `tf.data.Datasets` work best if all components have a shape set
    #  so set the shapes manually:
    encoded_text.set_shape([None])
    label.set_shape([])

    return encoded_text, label

encoded_train_set = train_set.map(lambda ex: encode_map_fn(ex[0], ex[1]))
encoded_test_set = test_set.map(lambda ex: encode_map_fn(ex[0], ex[1]))

BUFFER_SIZE = 2000
BATCH_SIZE = 200
TAKE_SIZE = 500

train_data = encoded_train_set.skip(TAKE_SIZE).shuffle(BUFFER_SIZE)
train_data = train_data.padded_batch(BATCH_SIZE, padded_shapes=([None],[]))

test_data = encoded_test_set.take(TAKE_SIZE)
test_data = test_data.padded_batch(500//8, padded_shapes=([None],[]))

# Since we have introduced a new token encoding (the zero used for padding), the vocabulary size has increased by one.
vocab_size += 1

print(colored("Encoded data", "green"))

# Define the model
print(colored("Creating the LSTM model", "yellow"))

model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(vocab_size + 1, 64))
model.add(tf.keras.layers.LSTM(64,return_sequences=True))
model.add(tf.keras.layers.LSTM(64,return_sequences=True))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)))
#model.add(tf.keras.layers.GlobalAveragePooling1D())
model.add(tf.keras.layers.Dense(3))

#model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

print(colored("Compiled the LSTM model", "green"))

model.summary()

# Training the model
print(colored("Training the LSTM model", "yellow"))
history = model.fit(train_data, epochs=14)
print(colored(history.history, "green"))

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
plt.savefig('rnn_acc.png')
print("Accuracy graph saved")

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
plt.savefig('rnn_loss.png')
print("Loss graph saved")


# Testing the model
print(colored("Testing the LSTM model", "yellow"))
loss, accuracy = model.evaluate(test_data)
print(colored("Test accuracy: {}".format(accuracy), "red"))

# Saving the model
model.save('rnn_model.h5')
