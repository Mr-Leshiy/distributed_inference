from src.onnx_model import OnnxModel
from python import Python, PythonObject
from testing import *


# C = A + B ONNX function
def sum_function(
    domain: String, fname: String, opset_imports: PythonObject
) -> PythonObject:
    onnx = Python.import_module("onnx")

    return onnx.helper.make_function(
        domain=domain,
        fname=fname,
        inputs=["A", "B"],
        outputs=["C"],
        nodes=[onnx.helper.make_node("Sum", ["A", "B"], ["C"])],
        opset_imports=opset_imports,
    )


# C = A * B ONNX function
def mul_function(
    domain: String, fname: String, opset_imports: PythonObject
) -> PythonObject:
    onnx = Python.import_module("onnx")

    return onnx.helper.make_function(
        domain=domain,
        fname=fname,
        inputs=["A", "B"],
        outputs=["C"],
        nodes=[onnx.helper.make_node("Mul", ["A", "B"], ["C"])],
        opset_imports=opset_imports,
    )


def test_sum_model():
    onnx = Python.import_module("onnx")
    numpy = Python.import_module("numpy")

    onnx = Python.import_module("onnx")
    A = onnx.helper.make_tensor_value_info("A", onnx.TensorProto.FLOAT, [1])
    B = onnx.helper.make_tensor_value_info("B", onnx.TensorProto.FLOAT, [1])
    C = onnx.helper.make_tensor_value_info("C", onnx.TensorProto.FLOAT, [1])
    node = onnx.helper.make_node("Sum", ["A", "B"], ["C"])
    graph = onnx.helper.make_graph(
        nodes=[node], name="test model", inputs=[A, B], outputs=[C]
    )
    # a + b ONNX model
    model = OnnxModel(onnx.helper.make_model(graph))

    input: PythonObject = {
        "A": numpy.array([4], dtype=numpy.float32),
        "B": numpy.array([10], dtype=numpy.float32),
    }
    assert_equal_pyobj(
        model.run(
            {
                "A": numpy.array([4], dtype=numpy.float32),
                "B": numpy.array([10], dtype=numpy.float32),
            }
        )[0],
        numpy.array([14], dtype=numpy.float32),
    )


# Model: y = (a + b) * (c + d)
def test_model_1():
    onnx = Python.import_module("onnx")
    numpy = Python.import_module("numpy")

    custom_domain = "custom"
    opset_imports: PythonObject = [
        onnx.helper.make_opsetid("", 14),
        onnx.helper.make_opsetid(custom_domain, 1),
    ]

    f1 = sum_function(custom_domain, "A + B", opset_imports)
    f2 = sum_function(custom_domain, "C + D", opset_imports)
    f3 = mul_function(custom_domain, "(...) + (...)", opset_imports)

    graph = onnx.helper.make_graph(
        name="test model",
        nodes=[
            onnx.helper.make_node(
                f1.name, ["A", "B"], ["X1"], domain=custom_domain
            ),
            onnx.helper.make_node(
                f2.name, ["C", "D"], ["X2"], domain=custom_domain
            ),
            onnx.helper.make_node(
                f3.name, ["X1", "X2"], ["Y"], domain=custom_domain
            ),
        ],
        inputs=[
            onnx.helper.make_tensor_value_info(
                "A", onnx.TensorProto.FLOAT, [1]
            ),
            onnx.helper.make_tensor_value_info(
                "B", onnx.TensorProto.FLOAT, [1]
            ),
            onnx.helper.make_tensor_value_info(
                "C", onnx.TensorProto.FLOAT, [1]
            ),
            onnx.helper.make_tensor_value_info(
                "D", onnx.TensorProto.FLOAT, [1]
            ),
        ],
        outputs=[
            onnx.helper.make_tensor_value_info(
                "Y", onnx.TensorProto.FLOAT, [1]
            ),
        ],
    )
    model = OnnxModel(
        onnx.helper.make_model(
            graph=graph,
            functions=[f1, f2, f3],
            opset_imports=opset_imports,
        )
    )

    inputs = Python.dict()
    inputs["A"] = numpy.array([4], dtype=numpy.float32)
    inputs["B"] = numpy.array([10], dtype=numpy.float32)
    inputs["C"] = numpy.array([3], dtype=numpy.float32)
    inputs["D"] = numpy.array([6], dtype=numpy.float32)
    assert_equal_pyobj(
        model.run(inputs)[0],
        numpy.array([126], dtype=numpy.float32),
    )
