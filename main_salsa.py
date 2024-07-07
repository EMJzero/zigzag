import logging as _logging
import re

from zigzag.stages.CostModelStage import CostModelStage
from zigzag.stages.MainStage import MainStage
from zigzag.stages.SpatialMappingGeneratorStage import SpatialMappingGeneratorStage
from zigzag.stages.WorkloadStage import WorkloadStage
from zigzag.stages.WorkloadParserStage import WorkloadParserStage
from zigzag.stages.AcceleratorParserStage import AcceleratorParserStage
from zigzag.stages.reduce_stages import MinimalLatencyStage, MinimalEDPStage, MinimalEnergyStage, SumStage
from zigzag.stages.save_stages import CompleteSaveStage, PickleSaveStage, SimpleSaveStage
from zigzag.stages.LomaStage import LomaStage
from zigzag.stages.SalsaStage import SalsaStage
from zigzag.parser.arguments import get_arg_parser

parser = get_arg_parser()
args = parser.parse_args()

# Initialize the logger
_logging_level = _logging.INFO
_logging_format = "%(asctime)s - %(name)s.%(funcName)s +%(lineno)s - %(levelname)s - %(message)s"
_logging.basicConfig(level=_logging_level, format=_logging_format)

hw_name = args.accelerator.split(".")[-1]
wl_name = re.split(r"/|\.", args.model)[-1]
if wl_name == "onnx":
    wl_name = re.split(r"/|\.", args.model)[-2]
experiment_id = f"{hw_name}-{wl_name}"
pkl_name = f"{experiment_id}-saved_list_of_cmes"

# Initialize the MainStage which will start execution.
# The first argument of this init is the list of stages that will be executed in sequence.
# The second argument of this init are the arguments required for these different stages.
mainstage = MainStage(
    [
        WorkloadParserStage,
        AcceleratorParserStage,
        SimpleSaveStage,
        PickleSaveStage,
        SumStage,
        CompleteSaveStage,
        WorkloadStage,
        SpatialMappingGeneratorStage,
        MinimalEDPStage,
        SalsaStage,  # Find pseudo-optimal temporal mapping
        CostModelStage,
    ],
    accelerator=args.accelerator,
    workload=args.model,
    mapping=args.mapping,
    dump_folder=f"outputs/{experiment_id}",
    pickle_filename=f"outputs/{pkl_name}.pickle",
    enable_mix_spatial_mapping_generation = True,
    loma_lpf_limit=6,
    loma_show_progress_bar=True,
    salsa_iteration_number=1000,
    salsa_start_temperature=0.05,
    salsa_opt_criterion="energy", #cannot be EDP, only energy and latency
    salsa_number_of_core=12,
)

# Launch the MainStage
mainstage.run()
