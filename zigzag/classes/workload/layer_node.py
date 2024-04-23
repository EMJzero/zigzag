from copy import deepcopy
from math import gcd
import logging as _logging
import math


from typeguard import typechecked

from zigzag.classes.datatypes import (
    LayerDim,
    LayerOperand,
    MemoryOperand,
    LoopList,
    PrLoop,
    PrScalingFactors,
    UnrollFactor,
)
from zigzag.classes.mapping.spatial.SpatialMapping import SpatialMapping, SpatialMappingHint
from zigzag.classes.workload.workload_attributes import (
    InputOperandSource,
    LayerAttributes,
    LayerDimRelations,
    LayerDimSizes,
    LayerEquation,
    LayerOperandPrecision,
    LayerPadding,
    MemoryOperandLinks,
    LayerTemporalOrdering,
)
from zigzag.utils import json_repr_handler

logger = _logging.getLogger(__name__)


@typechecked
class LoopRelevancyInfo:
    """! Per LayerOperand, store the Relevant, Irrelevant LayerDims, and which LayerDims are Partially Relevant to each other
    # TODO move to somewhere else
    """

    def __init__(self):
        self.r_dims: dict[LayerOperand, list[LayerDim]] = dict()
        self.ir_dims: dict[LayerOperand, list[LayerDim]] = dict()
        self.pr_dims: dict[LayerOperand, dict[LayerDim, list[LayerDim]]] = dict()
        self.__orig_pr_loop: PrLoop

    def get_r_layer_dims(self, layer_operand: LayerOperand) -> list[LayerDim]:
        return self.r_dims[layer_operand]

    def get_ir_layer_dims(self, layer_operand: LayerOperand) -> list[LayerDim]:
        return self.ir_dims[layer_operand]

    def get_pr_layer_dims(self, layer_operand: LayerOperand) -> dict[LayerDim, list[LayerDim]]:
        return self.pr_dims[layer_operand]

    def create_pr_decoupled_relevancy_info(self) -> "LoopRelevancyInfo":
        """! remove the pr loop dict, and put the pr-related data dimension (e.g. IX and IY)
        to r and ir dict with "r" and "ir" tags
        # TODO this method requires the unaltered pr_loop, before going through `extract_relevancy_info`
        """
        new = LoopRelevancyInfo()
        new.r_dims = deepcopy(self.r_dims)
        new.ir_dims = deepcopy(self.ir_dims)
        for layer_op in self.pr_dims:
            new.r_dims[layer_op] += [pr_layer_dim.create_r_version() for pr_layer_dim in self.__orig_pr_loop]
            new.ir_dims[layer_op] += [pr_layer_dim.create_ir_version() for pr_layer_dim in self.__orig_pr_loop]
        return new

    @staticmethod
    def extract_relevancy_info(
        equation: LayerEquation, layer_dim_sizes: LayerDimSizes, pr_loop: PrLoop, pr_loop_list: LoopList
    ) -> "LoopRelevancyInfo":
        """!
        # TODO requires cleanup and documentation
        """
        self = LoopRelevancyInfo()
        self.__orig_pr_loop = pr_loop

        dimension_list = layer_dim_sizes.layer_dims()
        for layer_op in equation.get_contained_operands():
            r_loop_list = equation.get_r_layer_dims(layer_op)
            ir_loop_list = list(set(dimension_list).difference(r_loop_list))

            pr_loop_remove_flag = any(layer_dim in pr_loop for layer_dim in r_loop_list)
            if pr_loop_remove_flag:
                self.r_dims[layer_op] = [layer_dim for layer_dim in r_loop_list if layer_dim not in pr_loop_list]
                self.ir_dims[layer_op] = [
                    layer_dim
                    for layer_dim in ir_loop_list
                    if layer_dim not in pr_loop_list and layer_dim_sizes[layer_dim] != 1
                ]
                self.pr_dims[layer_op] = pr_loop
            else:
                self.r_dims[layer_op] = [layer_dim for layer_dim in r_loop_list if layer_dim_sizes[layer_dim] != 1]
                self.ir_dims[layer_op] = [layer_dim for layer_dim in ir_loop_list if layer_dim_sizes[layer_dim] != 1]
                self.pr_dims[layer_op] = {}

        return self


@typechecked
class LayerNode:
    """! Represents a single layer in a workload."""

    def __init__(self, layer_id: int, layer_attrs: LayerAttributes, node_name="", layer_type=None):
        """! The class constructor
        To construct each layer node, algorithm equation/dimension/indirect relation are parsed.
        This parser collects information of operand, loop dimension, and loop relevance.
        Equal-to-1 loop dimensions are eliminated.
        @param layer_id: The identifier (key) of the layer, as defined in the workload
        @param node_name: an optional name for the Node. E.g. the node's name from the onnx model.

        # TODO clean up this method. Too many lines for a clean init method.
        """
        self.id = layer_id
        self.layer_attrs = layer_attrs
        self.name = node_name
        self.type = layer_type

        # Parsed attributes
        self.equation: LayerEquation = layer_attrs.parse_equation()
        self.layer_dim_sizes: LayerDimSizes = layer_attrs.parse_layer_dim_sizes()
        self.operand_precision: LayerOperandPrecision = layer_attrs.parse_operand_precision()
        self.dimension_relations: LayerDimRelations | None = layer_attrs.parse_layer_dim_relations()
        self.user_spatial_mapping: SpatialMapping = layer_attrs.parse_spatial_mapping()
        self.user_spatial_mapping_hint: SpatialMappingHint = layer_attrs.parse_spatial_mapping_hint()
        self.core_allocation: int = layer_attrs.parse_core_allocation()
        self.memory_operand_links: MemoryOperandLinks = layer_attrs.parse_mem_operand_links()
        self.user_temporal_ordering: LayerTemporalOrdering | None = layer_attrs.parse_temporal_ordering()
        self.padding: LayerPadding | None = layer_attrs.parse_padding()
        self.constant_operands: list[LayerOperand] = layer_attrs.parse_constant_operands()
        pr_layer_dim_sizes: LayerDimSizes | None = layer_attrs.parse_pr_layer_dim_sizes()
        self.input_operand_source: InputOperandSource = layer_attrs.parse_operand_source()

        # Derived attributes
        self.layer_operands = self.equation.get_contained_operands()
        self.output_operand: LayerOperand = self.layer_operands[0]
        self.input_operands: list[LayerOperand] = self.layer_operands[1:]

        self.pr_loop, pr_loop_list, self.pr_scaling_factors = self.build_pr_funcs()
        self.pr_layer_dim_sizes = (
            LayerDimSizes({dim: self.calc_pr_dimension_size_total(dim) for dim in self.pr_loop})
            if (pr_layer_dim_sizes is None or len(pr_layer_dim_sizes) == 0)
            else pr_layer_dim_sizes
        )
        self.loop_relevancy_info = LoopRelevancyInfo.extract_relevancy_info(
            self.equation, self.layer_dim_sizes, self.pr_loop, pr_loop_list
        )
        self.pr_decoupled_relevancy_info = self.loop_relevancy_info.create_pr_decoupled_relevancy_info()

        # To compute
        self.operand_size_elem: dict[LayerOperand, UnrollFactor] = dict()

        self.extract_layer_info()

        # ! not used?
        # self.source_storage_level = layer_attrs.data.get("source_storage_level", {})
        # self.operand_source_dimension_mapping: OperandSrcDimMapping = layer_attrs.data.get(
        #     "operand_source_dimension_mapping", {}
        # )

        # Extract layer info, e.g. total operand size, total operand data reuse, total MAC operation, etc.

        # TODO get rid of all this
        self.loop_dim_list = list(self.layer_dim_sizes.layer_dims())
        self.operand_dimensionality_order = {
            layer_op: self.equation.get_r_layer_dims(layer_op) for layer_op in self.equation.get_contained_operands()
        }
        self.operand_loop_dim = self.loop_relevancy_info
        # self.operand_loop_dim_reform = operand_loop_dim_reform
        self.operand_list = self.layer_operands
        # self.input_operand_source = input_operand_source

        # Save the variable (non-constant) input operands
        self.variable_input_operands: list[LayerOperand] = [
            op for op in self.input_operands if self.constant_operands is None or op not in self.constant_operands
        ]

        # ! not used?
        # # Save the way an operand's tensor should be reshaped for interaction with other nodes.
        # self.operand_tensor_reshape: dict[str, list] = layer_attrs.get(
        #     "operand_tensor_reshape", {op: [] for op in self.operand_list}
        # )

    def build_pr_funcs(self) -> tuple[PrLoop, LoopList, PrScalingFactors]:
        """!
        # TODO requires documentation
        """
        if self.dimension_relations is not None and len(self.dimension_relations) > 0:
            pr_loop, pr_loop_list, pr_scaling_factors = self.dimension_relations.extract_pr_loop_info()
        else:
            pr_loop, pr_loop_list, pr_scaling_factors = {}, [], {}

        return pr_loop, pr_loop_list, pr_scaling_factors

    def __str__(self):
        return f"LayerNode_{self.id}"

    def __repr__(self):
        return str(self)

    def __jsonrepr__(self):
        """!  JSON representation used for saving this object to a json file."""
        return json_repr_handler(
            {
                "equation": self.equation,
                "equation_relations": self.dimension_relations,
                "loop_dimensions": self.layer_dim_sizes,
                "operand_precision": self.operand_precision,
                "core_allocation": self.core_allocation,
                "user_spatial_mapping": self.user_spatial_mapping,
                "memory_operand_links": self.memory_operand_links,
                # "source_storage_level": self.source_storage_level, # NOTE not used?
            }
        )

    def calc_tensor_size(self, layer_op: LayerOperand, layer_dim_sizes: LayerDimSizes):
        """!  Calculates the tensor size (nb of elements) for the given operand layer_op with the given loop dimension
        sizes layer_dim_sizes."""
        return math.prod(self.calc_tensor_dims(layer_op, layer_dim_sizes).values())

    def calc_tensor_dim(self, dim: LayerDim, layer_dim_sizes: LayerDimSizes):
        if dim in layer_dim_sizes:
            return layer_dim_sizes[dim]
        elif dim in self.pr_loop:
            related_dimension_sizes = [layer_dim_sizes[dimension] for dimension in self.pr_loop[dim]]
            scaling_factors = list(self.pr_scaling_factors[dim].values())
            assert (
                len(related_dimension_sizes) == len(scaling_factors) == 2
            ), "Shouldn't happen if partial relevancy checks in extract_pr_loop_info() are done correctly."
            args = (int(val) for pair in zip(scaling_factors, related_dimension_sizes) for val in pair)
            pr_dim_size = self.calc_pr_dimension_size(*args)
            # Clip this to the largest possible size for this partially relevant dimension (computed at initialization based on padding)
            pr_dim_size = min(self.pr_layer_dim_sizes[dim], pr_dim_size)
            return pr_dim_size
        elif dim in self.layer_dim_sizes:
            assert (
                self.layer_dim_sizes[dim] == 1
            ), "This line should only be reached when the dim has a size of 1 in the layer."
            return 1
        else:
            raise ValueError("Something went wrong in the initialization of the layer, or in the caller function.")

    def calc_tensor_dims(self, layer_op: LayerOperand, layer_dim_sizes: LayerDimSizes) -> dict[LayerDim, UnrollFactor]:
        """!
        # TODO requires documentation
        """
        out: dict[LayerDim, UnrollFactor] = dict()
        r_layer_dims = self.loop_relevancy_info.get_r_layer_dims(layer_op)
        pr_layer_dims = list(self.loop_relevancy_info.get_pr_layer_dims(layer_op).keys())
        for dim in r_layer_dims + pr_layer_dims:
            out[dim] = self.calc_tensor_dim(dim, layer_dim_sizes)
        return out

    def calc_pr_dimension_size_total(self, dim: LayerDim) -> int:
        """!  Compute the total pr dimension size of this node, taking padding into account.
        @param dim (str): The partially relevant dimension, e.g. 'IX'.
        @return int: The total partially relevant dimension size
        """
        related_dimension_sizes = [self.layer_dim_sizes[related_dim] for related_dim in self.pr_loop[dim]]
        # assumes this dict is ordered
        scaling_factors: list[int] = list(self.pr_scaling_factors[dim].values())
        assert (
            len(related_dimension_sizes) == len(scaling_factors) == 2
        ), "Shouldn't happen if partial relevancy checks in extract_pr_loop_info() are done correctly."
        args = (val for pair in zip(scaling_factors, related_dimension_sizes) for val in pair)
        total_pr_dim_size = self.calc_pr_dimension_size(*args)
        # Partially relevant loop dimensions can also have padding, so get the padding for this pr dimension and subtract
        padding = LayerPadding.DEFAULT if self.padding is None else self.padding[dim]
        total_pr_dim_size_without_padding = int(total_pr_dim_size - sum(padding))
        return total_pr_dim_size_without_padding

    @staticmethod
    def calc_pr_dimension_size(sa: int, A: int, sb: int, B: int):
        """!  Calculates the number of unique indices c generated by iterating through the indices
        a in range(0,A,1) and b in range(0,B,1) according to the equation c = sa * a + sb * b.
        sa and sb thus represent the scaling of a, resp. b.
        """
        return int(A * B - max(0, B - (sa / gcd(sa, sb))) * (A - (sb / gcd(sa, sb))))

    def extract_layer_info(self):
        """!  This function extract basic information for each layer node."""

        self.total_MAC_count = self.layer_dim_sizes.get_total_size()

        # each operand's size (Unit: # of data element)
        for layer_op in self.layer_operands:
            self.operand_size_elem[layer_op] = 1
            for r_layer_dim in self.loop_relevancy_info.get_r_layer_dims(layer_op):
                self.operand_size_elem[layer_op] *= self.layer_dim_sizes[r_layer_dim]
            for pr_layer_dim in self.loop_relevancy_info.get_pr_layer_dims(layer_op):
                self.operand_size_elem[layer_op] *= self.calc_tensor_dims(layer_op, self.layer_dim_sizes)[pr_layer_dim]

        # each operand's size (Unit: bit)
        operand_size_bit: dict[LayerOperand, int] = {}
        for layer_op, size_in_elem in self.operand_size_elem.items():
            operand_size_bit[layer_op] = size_in_elem * self.operand_precision[layer_op]
        self.operand_size_bit = operand_size_bit

        # each operand's total data reuse factor, which is total MAC Op/total operand size (in element),
        # i.e. each data element can be used to support how many MAC operation.
        operand_data_reuse: dict[LayerOperand, float] = {}
        for operand, size_in_elem in self.operand_size_elem.items():
            operand_data_reuse[operand] = self.total_MAC_count / size_in_elem
        self.operand_data_reuse = operand_data_reuse

    def get_operand_irrelevant_layer_dims(self, layer_op: LayerOperand) -> list[LayerDim]:
        """!  Return the irrelevant dimensions of layer operand 'layer_op'."""
        return self.loop_relevancy_info.get_ir_layer_dims(layer_op)

    def get_layer_operand(self, mem_op: MemoryOperand) -> LayerOperand:
        """!  Return the layer operand associated with the given memory operand for this layer.
        If there is no such memory operand, an error is raised.
        """
        layer_op = self.memory_operand_links.mem_to_layer_op(mem_op)
        if layer_op is not None:
            return layer_op
        raise ValueError(f"The memory operand {mem_op} is not present in layer {self}.")

    # ! not used?
    # def get_operand_storage_level(self, layer_op: OperandStr):
    #     """!  Return the memory level at which an input operand is stored.
    #     If this layer node has no information for the given operand, it returns None.
    #     """
    #     if layer_op not in self.source_storage_level:
    #         return None
    #     return self.source_storage_level[layer_op]

    @staticmethod
    def check_layer(layer: "LayerNode"):
        """!Check that the layer includes:
        - the core which it is allocated to
        If not, a ValueError is raised.
        If the layer in main_inputs is not set, False is returned
        @return: True if layer is set correctly
        """
        if layer is None:
            raise ValueError()
        if layer.core_allocation is None:
            logger.critical(f"Layer {layer} has no core allocation.")  # pylint: disable=W1203
            raise ValueError()
        return True
