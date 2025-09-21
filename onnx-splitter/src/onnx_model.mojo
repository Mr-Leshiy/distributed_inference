from python import Python, PythonObject


struct OnnxModel:
    var inner: PythonObject

    def __init__(out self, val: PythonObject):
        onnx = Python.import_module("onnx")
        if Python.type(val) is not onnx.ModelProto:
            raise "Provided val is not an ONNX model type"
        onnx.checker.check_model(val)
        self.inner = val

    def name(self) -> String:
        return String(self.inner.graph.name)

    def run(self, inputs: PythonObject) -> PythonObject:
        onnx_reference = Python.import_module("onnx.reference")
        return onnx_reference.ReferenceEvaluator(self.inner).run(None, inputs)
