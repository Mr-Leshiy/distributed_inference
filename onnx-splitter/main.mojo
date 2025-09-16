from python import Python, PythonObject


def main():
    onnx = Python.import_module("onnx")
    model = onnx.load("model.onnx")
    print(model.graph.node)

    # var nodes: List[PythonObject] = []
    # for node in model.graph.node:
    #     for node_output in node.output:
    #         for output in model.graph.output:
    #             if node_output == output.name:
    #                 print(node)
    #                 nodes.append(node)

    # onnx.helper.make_graph(
    #     nodes=nodes,
    #     name="fucking hell",
    #     inputs=[],
    #     outputs=[],
    # )
