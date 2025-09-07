import keras
from keras.models import Sequential
from keras.layers import Dense, Input
import tensorflow as tf
import tf2onnx
import onnx
import onnxruntime
import numpy as np

# Create a simple model
model = Sequential(
    [
        Input(shape=(64,), name="input_layer"),
        Dense(64, activation="relu", name="hidden_layer"),
        Dense(1, activation="sigmoid", name="output_layer"),
    ]
)

# Compile the model
model.compile(optimizer="adam", loss="binary_crossentropy")

# convert to ONNX format
input_signature = [
    tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype, name="input")
]
model.output_names = ["output"]
onnx_model, _ = tf2onnx.convert.from_keras(
    model, input_signature=input_signature, opset=16
)
onnx.checker.check_model(onnx_model)
# save model in ONNX format
onnx.save(onnx_model, "model.onnx")

model = onnx.load("model.onnx")
print(f"{model.graph.node}")

session = onnxruntime.InferenceSession("model.onnx")
input = np.ones(
    (1, 64),
    dtype=np.float32,
)
result = session.run(
    None,
    {"input": input},
)
print(f"{result}")
