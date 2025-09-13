import numpy
import onnx
from onnx import numpy_helper


model = onnx.load("model.onnx")

print(model)


# for val in model.graph.initializer:
#     print(val.name)
#     print(f"array shape: {numpy_helper.to_array(val).shape}")
