import networkx as nx
from typeguard import typechecked

from zigzag.classes.workload.Workload import Workload
from zigzag.classes.workload.dummy_node import DummyNode
from zigzag.classes.workload.layer_node import LayerNode


@typechecked
class ONNXWorkload(Workload):

    def __init__(self, **attr):
        """!  Collect all the algorithmic workload information here."""
        super().__init__(**attr)

        self.node_id_to_obj = {}
        self.node_list = []

    def add(self, node_id, node_obj: LayerNode | DummyNode):
        """!  Add a node object to the ONNX workload graph.
        This can be a different object based on if it's an "accelerateable" node or not.
        """
        self.node_list.append(node_obj)
        self.node_id_to_obj[node_id] = node_obj

        self.add_node(node_obj)
        edges = []
        for op, parents in node_obj.input_operand_source.items():
            for parent_id in parents:
                parent_node_obj = self.node_id_to_obj[parent_id]
                edges.append((parent_node_obj, node_obj))
                node_obj.input_operand_source[op] = parent_node_obj
            self.add_edges_from(edges)

    def topological_sort(self):
        return nx.topological_sort(self)

    def get_node_with_id(self, id):
        for node in self.nodes:
            if node.id == id:
                return node
        raise ValueError("DNNWorkload instance does not have a node with the requested id")
