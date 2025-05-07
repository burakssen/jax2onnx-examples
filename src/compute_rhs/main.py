import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from jaxfluids import InputManager, InitializationManager, SimulationManager
from jaxfluids.solvers.space_solver import SpaceSolver
from jaxfluids.solvers.positivity.positivity_handler import PositivityHandler
import jax.numpy as jnp
from jax2onnx import to_onnx

# SETUP SIMULATION
input_manager = InputManager("linear_advection.json", "numerical_setup.json")
initialization_manager = InitializationManager(input_manager)
sim_manager = SimulationManager(input_manager)

jxf_buffers = initialization_manager.initialization()

domain_information = input_manager.domain_information
material_manager = input_manager.material_manager
equation_manager = input_manager.equation_manager
halo_manager = input_manager.halo_manager
numerical_setup = input_manager.numerical_setup
gravity = input_manager.case_setup.forcing_setup.gravity
geometric_source = input_manager.case_setup.forcing_setup.geometric_source

positivity_handler = PositivityHandler(
            domain_information=domain_information,
            material_manager=material_manager,
            equation_manager=equation_manager,
            halo_manager=halo_manager,
            numerical_setup=numerical_setup,
            levelset_handler=None,
            diffuse_interface_handler=None,
        )

space_solver = SpaceSolver(
            domain_information=domain_information,
            material_manager=material_manager,
            equation_manager=equation_manager,
            halo_manager=halo_manager,
            numerical_setup=numerical_setup,
            gravity=gravity,
            geometric_source=geometric_source,
            levelset_handler=None,
            diffuse_interface_handler=None,
            positivity_handler=positivity_handler)

def compute_rhs_fn():
    return space_solver.compute_rhs(
        initialization_manager.material_manager.equation_information.conservatives_slices,
        jxf_buffers.simulation_buffers.material_fields.primitives,
        initialization_manager.material_manager.get_temperature(
            primitives=jxf_buffers.simulation_buffers.material_fields.primitives,
        ),
        0.0,
        0.01)
    


model = to_onnx(
    compute_rhs_fn,
    inputs=[],
    input_params=[],
)

# Save the model to a file

os.makedirs("../../onnxfiles/compute_rhs", exist_ok=True)

with open("../../onnxfiles/compute_rhs/compute_rhs.onnx", "wb") as f:
    f.write(model.SerializeToString())