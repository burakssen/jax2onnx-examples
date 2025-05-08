import jax
import jax.numpy as jnp
from jax2onnx import to_onnx
import os

def model_fn(x):
    steps = 5

    def body_func(index, args):
        x, counter = args
        x += 0.1 * x**2
        counter += 1
        return (x, counter)

    args = (x, 0)
    args = jax.lax.fori_loop(0, steps, body_func, args)

    return args

dummy_input = jnp.ones((2,), dtype=jnp.float32)

print(model_fn(dummy_input))

onnx_model = to_onnx(model_fn, inputs=[dummy_input])

os.makedirs("../../onnxfiles/fori_loop", exist_ok=True)

with open("../../onnxfiles/fori_loop/fori_loop.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())