workload_gemm = {
    0: {  # example Gemm layer
        "operator_type": "Gemm",
        "equation": "O[m][n]+=A[m][k]*B[k][n]",
        "dimension_relations": [],
        "loop_dim_size": {
            "M": 8*8*8,
            "K": 8*8*8,
            "N": 8*8*8,
        },
        "operand_precision": {"O": 32, "O_final": 8, "B": 8, "A": 8},
        "operand_source": {"B": [], "A": []},
        "constant_operands": ["B"],
        "operand_source_dimension_mapping": {},
    },
}

workload_tpu = {
    0: {  # conv1, stride 2
        "operator_type": "Conv",
        "equation": "O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][iy][ix]",
        "dimension_relations": ["ix=2*ox+1*fx", "iy=2*oy+1*fy"],
        "loop_dim_size": {
            "B": 1,
            "K": 64,
            "C": 3,
            "OY": 112,
            "OX": 112,
            "FY": 7,
            "FX": 7,
        },
        "operand_precision": {"O": 16, "O_final": 8, "W": 8, "I": 8},
        "operand_source": {"W": [], "I": []},
        "constant_operands": ["I", "W"],
    }
}

mapping_gemm = {
    "Gemm": {  # Gemm
        "spatial_mapping": {"D1": ("M", 1024*3), "D2": ("N", 4096), "D3": ("K", 1024)},
        #"temporal_ordering": [
        #    # Innermost loop
        #    ("K", 8),
        #    ("N", 8),
        #    ("M", 8),
        #    ("K", 8),
        #    ("N", 8),
        #    ("M", 8),
        #    # Outermost loop
        #],
        "core_allocation": 1,
        "memory_operand_links": {
            "O": "O",
            "B": "I2",
            "A": "I1",
        },
    },
}

mapping_tpu = {
    "default": {
        "core_allocation": 1,
        "spatial_mapping": {"D1": ("K", 32), "D2": ("C", 32)},
        "memory_operand_links": {"O": "O", "W": "I2", "I": "I1"},
    },
    "Add": {
        "core_allocation": 1,
        "spatial_mapping": {"D1": ("G", 32), "D2": ("C", 1)},
        "memory_operand_links": {"O": "O", "X": "I2", "Y": "I1"},
    },
    "Pooling": {
        "core_allocation": 1,
        "spatial_mapping": {"D1": ("G", 32), "D2": ("C", 1)},
        "memory_operand_links": {"O": "O", "W": "I2", "I": "I1"},
    },
}

#from zigzag.classes.hardware.architecture.memory_hierarchy import MemoryHierarchy
#from zigzag.classes.hardware.architecture.operational_unit import Multiplier
#from zigzag.classes.hardware.architecture.operational_array import MultiplierArray
#from zigzag.classes.hardware.architecture.memory_instance import MemoryInstance
#from zigzag.classes.hardware.architecture.accelerator import Accelerator
#from zigzag.classes.hardware.architecture.core import Core

from zigzag.stages.CostModelStage import CostModelStage
from zigzag.stages.MainStage import MainStage
from zigzag.stages.SpatialMappingGeneratorStage import SpatialMappingGeneratorStage
from zigzag.stages.WorkloadStage import WorkloadStage
from zigzag.stages.input_parser_stages import AcceleratorParserStage, WorkloadParserStage
from zigzag.stages.reduce_stages import MinimalLatencyStage, SumStage
from zigzag.stages.save_stages import CompleteSaveStage, PickleSaveStage, SimpleSaveStage
from zigzag.stages.LomaStage import LomaStage
from zigzag.stages.SalsaStage import SalsaStage

from zigzag.visualization.results.plot_cme import (
    bar_plot_cost_model_evaluations_breakdown,
)

from zigzag.inputs.examples.hardware.Eyeriss_like import accelerator as accelerator_eyeriss
from zigzag.inputs.examples.hardware.TPU_like import accelerator as accelerator_tpu
#from zigzag.inputs.examples.hardware.Gemm import accelerator as accelerator_gemm

dump_filename_pattern=f"../outputs/Our_HW-single_layer_?.json"
pickle_filename = f"../outputs/Our_HW-single_layer_cme.pickle"


# Initialize the logger
import logging as _logging

_logging_level = _logging.INFO
# _logging_format = "%(asctime)s - %(funcName)s +%(lineno)s - %(levelname)s - %(message)s"
_logging_format = "%(asctime)s - %(levelname)s - %(message)s"
_logging.basicConfig(level=_logging_level, format=_logging_format)

# Initialize the MainStage which will start execution.
# The first argument of this init is the list of stages that will be executed in sequence.
# The second argument of this init are the arguments required for these different stages.

#stages_eval = [  # Initializes the MainStage as entry point
#    WorkloadParserStage,  # Parses the manual definition into the workload
#    AcceleratorParserStage,  # Parses the accelerator
#    CompleteSaveStage,  # Saves all received CMEs information to a json
#    WorkloadStage,  # Iterates through the different layers in the workload
#    SpatialMappingGeneratorStage,  # Generates multiple spatial mappings (SM)
#    MinimalLatencyStage,  # Reduces all CMEs, returning minimal latency one
#    TemporalOrderingConversionStage,  # Converts defined temporal_ordering to temporal mapping
#    CostModelStage,  # Evaluates generated SM and TM through cost model
#]

stages_salsa = [  # Initializes the MainStage as entry point
    WorkloadParserStage,  # Parses the ONNX Model into the workload
    AcceleratorParserStage,  # Parses the accelerator
    SimpleSaveStage,  # Saves all received CMEs information to a json
    WorkloadStage,  # Iterates through the different layers in the workload
    SpatialMappingGeneratorStage,  # Generates multiple spatial mappings (SM)
    MinimalLatencyStage,  # Reduces all CMEs, returning minimal latency one
    SalsaStage,  # Find pseudo-optimal temporal mapping
    CostModelStage  # Evaluates generated SM and TM through cost model
]

mainstage = MainStage(
    stages_salsa,
    accelerator=accelerator_eyeriss,  # required by AcceleratorParserStage
    workload=workload_gemm,  # required by ONNXModelParserStage
    mapping=mapping_gemm,  # required by ONNXModelParserStage
    dump_filename_pattern=dump_filename_pattern,
    pickle_filename=pickle_filename,
    opt='EDP',
    loma_lpf_limit=6,  # required by LomaStage
    loma_show_progress_bar=True,  # shows a progress bar while iterating over temporal mappings
    salsa_iteration_number=1000,
    salsa_start_temperature=0.05,
    salsa_opt_criterion="latency",
    salsa_number_of_core=8
)

# Launch the MainStage
answers = mainstage.run()
# Plot the energy and latency breakdown of our cost model evaluation
cme = answers[0][0]
save_path = "../outputs/breakdown.png"
bar_plot_cost_model_evaluations_breakdown([cme], save_path=save_path, xtick_rotation=0)
from zigzag.visualization.results.print_mapping import print_mapping
print_mapping(cme)
mem_names = [ml.memory_instance.name for ml in cme.mem_level_list]
stall_slacks = cme.SS_comb_collect
print("Stall and slack per port of each memory instance:")
for mem_name, ports_ss in zip(mem_names, stall_slacks):
    print(f"  {mem_name}: {ports_ss}")
print(f"Latency: {cme.latency_total2:.3e}")