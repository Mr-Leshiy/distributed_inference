import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Input
import numpy as np
import tensorflow as tf
import copy


keras.utils.set_random_seed(42)

# Create a simple model
main_model = Sequential(
    [
        Input(shape=(64,), name="input_layer"),
        Dense(64, activation="relu", name="hidden_layer"),
        Dense(1, activation="sigmoid", name="output_layer"),
    ]
)

# Compile the model
main_model.compile(optimizer="adam", loss="binary_crossentropy")

input = np.ones(
    (1, 64),
    dtype=np.float32,
)

# split second layer into two models
tensor1, tensor2 = tf.split(
    main_model.layers[0].weights[0], num_or_size_splits=2, axis=1
)

model_1 = Dense(32, activation="relu", name="hidden_layer")
model_1(input)
model_1.set_weights(
    [
        tensor1,
        main_model.layers[0].weights[1][:32],
    ]
)

model_2 = Dense(32, activation="relu", name="hidden_layer")
model_2(input)
model_2.set_weights(
    [
        tensor2,
        main_model.layers[0].weights[1][32:],
    ]
)

# last layer into the separate model
model_3 = main_model.layers[1]

# evaluate
layer_1_res = tf.concat([model_1(input), model_2(input)], axis=1)
splitted_res = model_3(layer_1_res)

original_res = main_model(input)

print(splitted_res, original_res)
