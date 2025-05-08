from jax import numpy as jnp
import jax
import numpy as np
import onnxruntime as ort

# Define the function
def foo(x, y):
    return jnp.sin(x) + y

# Import to_onnx after defining the function
from jax2onnx import to_onnx

# Use proper inputs for the conversion
x = jnp.ones((2, 3), dtype=jnp.float32)
y = jnp.ones((2, 3), dtype=jnp.float32)

# Convert the model - using convert instead of to_onnx
model = to_onnx(
    foo,
    inputs=((2,3), (2,3)),
)

# Save the model to a file
with open("foo.onnx", "wb") as f:
    f.write(model.SerializeToString())

# Load the model using ONNX Runtime
ort_session = ort.InferenceSession("foo.onnx")

# Prepare the input data
input_data = {
    "var_0": np.ones((2, 3), dtype=np.float32),
    "var_1": np.ones((2, 3), dtype=np.float32),
}

# Run the model
ort_outputs = ort_session.run(None, input_data)

# Get JAX outputs for comparison
jax_outputs = foo(x, y)
print("JAX outputs:", jax_outputs)
print("ONNX outputs:", ort_outputs[0])  # First output tensor

# Check if the outputs are close
is_close = np.allclose(jax_outputs, ort_outputs[0], rtol=1e-3, atol=1e-5)
print("Outputs are close:", is_close)