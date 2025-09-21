from python import Python
from src.model import OnnxModel


def main():
    onnx = Python.import_module("onnx")
    model = OnnxModel(onnx.load("model.onnx"))
    print(model.name())
