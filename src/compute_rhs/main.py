import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from jaxfluids import InputManager, InitializationManager, SimulationManager
from jaxfluids.data_types.ml_buffers import MachineLearningSetup, ParametersSetup, CallablesSetup
import jax.numpy as jnp
from jax2onnx import to_onnx

# SETUP SIMULATION
input_manager = InputManager("linear_advection.json", "numerical_setup.json")
initialization_manager = InitializationManager(input_manager)
sim_manager = SimulationManager(input_manager)

jxf_buffers = initialization_manager.initialization()

ml_setup = MachineLearningSetup(
    CallablesSetup(),
    ParametersSetup()
)

def compute_rhs_fn():
    return sim_manager.space_solver.compute_rhs(
        jxf_buffers.simulation_buffers.material_fields.conservatives,
        jxf_buffers.simulation_buffers.material_fields.primitives,
        jxf_buffers.simulation_buffers.material_fields.temperature,
        jxf_buffers.time_control_variables.physical_simulation_time,
        jxf_buffers.time_control_variables.physical_timestep_size,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        ml_setup,
        False
    )
    
model = to_onnx(
    compute_rhs_fn,
    inputs=[],
    input_params=[],
)

# Save the model to a file

os.makedirs("../../onnxfiles/compute_rhs", exist_ok=True)

with open("../../onnxfiles/compute_rhs/compute_rhs.onnx", "wb") as f:
    f.write(model.SerializeToString())