#!/usr/bin/env python3
"""
Generic ONNX Model Runner

This script provides a flexible and easy-to-use interface for running ONNX models.
It supports both interactive and non-interactive modes, with options for
configuring inputs and saving outputs.

Examples:
    Run with a specific model file:
        python onnxrun.py --model model.onnx
    
    Run with a specific model and save outputs:
        python onnxrun.py --model model.onnx --output_dir ./outputs
    
    Run in non-interactive mode with input configuration file:
        python onnxrun.py --model model.onnx --input_config inputs.json
    
    Run in quiet mode (less verbose output):
        python onnxrun.py --model model.onnx --quiet
"""

import onnxruntime as ort
import numpy as np
import os
import json


class ONNXModelRunner:
    """A class for running inference with ONNX models."""

    def __init__(self, model_path, verbose=True):
        """Initialize the model runner with a model path.
        
        Args:
            model_path (str): Path to the ONNX model file
            verbose (bool): Whether to print detailed information
        """
        self.model_path = model_path
        self.verbose = verbose
        self.model = self.load_model()
        if self.model:
            self.input_info, self.output_info = self.get_model_info()

    def load_model(self):
        """Load the ONNX model from the specified path."""
        if not os.path.exists(self.model_path):
            if self.verbose:
                print(f"Model file not found: {self.model_path}")
            return None
        
        if self.verbose:
            print(f"Loading model from: {self.model_path}")
        
        try:
            # Create inference session with optimizations enabled
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            session = ort.InferenceSession(self.model_path, session_options)
            return session
        except Exception as e:
            if self.verbose:
                print(f"Error loading model: {e}")
            return None

    def get_model_info(self):
        """Get detailed information about the model's inputs and outputs."""
        if not self.model:
            return [], []
            
        input_info = []
        for input_node in self.model.get_inputs():
            input_info.append({
                'name': input_node.name,
                'shape': input_node.shape,
                'type': input_node.type,
            })
        
        output_info = []
        for output_node in self.model.get_outputs():
            output_info.append({
                'name': output_node.name,
                'shape': output_node.shape,
                'type': output_node.type,
            })
        
        if self.verbose:
            self.print_model_info(input_info, output_info)
            
        return input_info, output_info

    def print_model_info(self, input_info, output_info):
        """Print formatted information about the model's inputs and outputs."""
        print("\n=== Model Information ===")
        print("\nINPUTS:")
        for i, inp in enumerate(input_info):
            print(f"  Input #{i+1}:")
            print(f"    Name: {inp['name']}")
            print(f"    Shape: {inp['shape']}")
            print(f"    Type: {inp['type']}")
        
        print("\nOUTPUTS:")
        for i, out in enumerate(output_info):
            print(f"  Output #{i+1}:")
            print(f"    Name: {out['name']}")
            print(f"    Shape: {out['shape']}")
            print(f"    Type: {out['type']}")
        print("========================\n")

    @staticmethod
    def get_np_dtype_from_onnx_type(onnx_type):
        """Convert ONNX data type to numpy data type."""
        type_map = {
            'tensor(float)': np.float32,
            'tensor(float16)': np.float16,
            'tensor(double)': np.float64,
            'tensor(int8)': np.int8,
            'tensor(int16)': np.int16,
            'tensor(int32)': np.int32,
            'tensor(int64)': np.int64,
            'tensor(uint8)': np.uint8,
            'tensor(uint16)': np.uint16,
            'tensor(uint32)': np.uint32,
            'tensor(uint64)': np.uint64,
            'tensor(bool)': np.bool_,
        }
        return type_map.get(onnx_type, np.float32)

    def create_input_data(self, interactive=True, input_specs=None):
        """Create input data based on model input specifications.
        
        Args:
            interactive (bool): Whether to interactively prompt for input values
            input_specs (dict, optional): Dictionary with input specifications
                Example: {
                    "input_name": {
                        "data_source": "ones", # "random", "zeros", "ones", "custom"
                        "dynamic_dims": [0, 1], # list of dynamic dimension indices
                        "dim_values": [1, 3], # values for dynamic dimensions
                        "custom_value": 5.0 # if data_source is "custom"
                    }
                }
        
        Returns:
            dict: Input data mapping input names to numpy arrays
        """
        if not self.model:
            return {}
            
        inputs = {}
        
        for i, inp in enumerate(self.input_info):
            if self.verbose:
                print(f"\nPreparing input #{i+1}: {inp['name']}")
            
            # Get input specs for this input
            specs = None
            if input_specs and inp['name'] in input_specs:
                specs = input_specs[inp['name']]
            
            # Determine input shape
            shape = inp['shape']
            concrete_shape = []
            
            # Handle dynamic dimensions (labeled as 'None' or negative values)
            for dim_idx, dim in enumerate(shape):
                if dim is None or (isinstance(dim, int) and dim < 0):
                    if specs and 'dim_values' in specs and dim_idx < len(specs['dynamic_dims']):
                        # Get value from specs
                        value_idx = specs['dynamic_dims'].index(dim_idx)
                        dim_val = specs['dim_values'][value_idx]
                    elif interactive:
                        # Get value interactively
                        dim_val = int(input(f"  Enter value for dynamic dimension {dim_idx}: "))
                    else:
                        # Default to 1 for dynamic dims when not interactive and no specs
                        dim_val = 1
                    concrete_shape.append(dim_val)
                else:
                    concrete_shape.append(dim)
            
            # Get numpy dtype from ONNX type
            dtype = self.get_np_dtype_from_onnx_type(inp['type'])
            
            # Determine data source
            data_source = "ones"  # Default
            custom_value = None
            
            if specs and 'data_source' in specs:
                data_source = specs['data_source']
                if data_source == 'custom' and 'custom_value' in specs:
                    custom_value = specs['custom_value']
            elif interactive:
                # Ask for input data source interactively
                data_source_input = input(f"  How would you like to provide data for input '{inp['name']}'?\n"
                                         f"  1. Random data\n"
                                         f"  2. All zeros\n"
                                         f"  3. All ones\n"
                                         f"  4. Custom value (fill all elements with the same value)\n"
                                         f"  Enter choice (1-4): ")
                
                data_source_map = {
                    '1': 'random',
                    '2': 'zeros',
                    '3': 'ones',
                    '4': 'custom'
                }
                data_source = data_source_map.get(data_source_input, 'ones')
                
                if data_source == 'custom':
                    custom_input = input("  Enter value to fill with: ")
                    try:
                        if 'float' in str(dtype):
                            custom_value = float(custom_input)
                        elif 'int' in str(dtype):
                            custom_value = int(custom_input)
                        elif 'bool' in str(dtype):
                            custom_value = custom_input.lower() in ['true', '1', 't', 'y', 'yes']
                    except ValueError:
                        if self.verbose:
                            print(f"  Invalid value for the data type. Using ones instead.")
                        data_source = 'ones'
            
            # Create input data based on the data source
            if data_source == 'random':
                if 'float' in str(dtype):
                    inputs[inp['name']] = np.random.randn(*concrete_shape).astype(dtype)
                else:
                    # For integer types, use random integers in a reasonable range
                    low = -10 if 'int' in str(dtype) and not 'uint' in str(dtype) else 0
                    high = 10
                    inputs[inp['name']] = np.random.randint(low, high, size=concrete_shape).astype(dtype)
            elif data_source == 'zeros':
                inputs[inp['name']] = np.zeros(concrete_shape, dtype=dtype)
            elif data_source == 'ones':
                inputs[inp['name']] = np.ones(concrete_shape, dtype=dtype)
            elif data_source == 'custom' and custom_value is not None:
                inputs[inp['name']] = np.full(concrete_shape, custom_value, dtype=dtype)
            else:
                # Default to ones
                inputs[inp['name']] = np.ones(concrete_shape, dtype=dtype)
                
            if self.verbose:
                print(f"  Created input with shape: {inputs[inp['name']].shape}, dtype: {inputs[inp['name']].dtype}")
        
        return inputs

    def run_inference(self, inputs):
        """Run inference with the provided inputs.
        
        Args:
            inputs (dict): Dictionary mapping input names to numpy arrays
            
        Returns:
            dict: Dictionary mapping output names to numpy arrays
        """
        if not self.model:
            return {}
            
        try:
            output_names = [output.name for output in self.model.get_outputs()]
            outputs = self.model.run(output_names, inputs)
            return dict(zip(output_names, outputs))
        except Exception as e:
            if self.verbose:
                print(f"Error during inference: {e}")
            return {}

    def get_output_info(self, outputs):
        """Get detailed information about the model outputs.
        
        Args:
            outputs (dict): Dictionary mapping output names to numpy arrays
            
        Returns:
            dict: Dictionary with output information
        """
        output_info = {}
        
        for name, output in outputs.items():
            info = {
                'shape': output.shape,
                'dtype': str(output.dtype),
                'stats': {}
            }
            
            # Calculate statistics for numerical outputs
            if np.issubdtype(output.dtype, np.number):
                info['stats'] = {
                    'min': float(output.min()),
                    'max': float(output.max()),
                    'mean': float(output.mean())
                }
                
            # Sample values
            flat_output = output.flatten()
            sample_size = min(5, len(flat_output))
            info['sample_values'] = flat_output[:sample_size].tolist()
            
            # For classification models, include top predicted classes
            if len(output.shape) > 0:  # Make sure it's not a scalar
                if (len(output.shape) == 1 and output.shape[0] <= 1000) or \
                   (len(output.shape) == 2 and output.shape[1] <= 1000):
                    # For 1D output or 2D output with reasonable number of classes
                    if len(output.shape) == 1 or output.shape[0] == 1:
                        top_indices = np.argsort(flat_output)[-5:][::-1].tolist()
                        info['top_indices'] = top_indices
                        info['top_values'] = flat_output[top_indices].tolist()
            
            output_info[name] = info
            
        return output_info

    def save_outputs(self, outputs, output_dir=None):
        """Save outputs to files.
        
        Args:
            outputs (dict): Dictionary mapping output names to numpy arrays
            output_dir (str, optional): Directory to save outputs to
            
        Returns:
            dict: Dictionary mapping output names to file paths
        """
        if not output_dir:
            output_dir = os.getcwd()
            
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        file_paths = {}
        
        for name, output in outputs.items():
            # Clean the name to make it a valid filename
            clean_name = ''.join(c if c.isalnum() else '_' for c in name)
            file_path = os.path.join(output_dir, f"{clean_name}.npy")
            
            # Save the numpy array
            np.save(file_path, output)
            file_paths[name] = file_path
            
            if self.verbose:
                print(f"Saved output '{name}' to {file_path}")
                
        # Also save output info in JSON format
        output_info = self.get_output_info(outputs)
        info_path = os.path.join(output_dir, "output_info.json")
        
        with open(info_path, 'w') as f:
            json.dump(output_info, f, indent=2)
            
        if self.verbose:
            print(f"Saved output info to {info_path}")
            
        return file_paths

    def display_outputs(self, outputs):
        """Display the model outputs in a user-friendly way."""
        if not outputs:
            return
            
        print("\n=== Model Outputs ===")
        for name, output in outputs.items():
            print(f"\nOutput: {name}")
            print(f"  Shape: {output.shape}")
            print(f"  Dtype: {output.dtype}")
            
            # Display statistics for numerical outputs
            if np.issubdtype(output.dtype, np.number):
                print(f"  Min: {output.min()}")
                print(f"  Max: {output.max()}")
                print(f"  Mean: {output.mean()}")
                
            # Show a sample of the output (first few elements)
            flat_output = output.flatten()
            sample_size = min(5, len(flat_output))
            print(f"  Sample values: {flat_output[:sample_size]}")
            
            # For classification models, try to show the top predicted classes
            # Check if output might represent class probabilities
            if len(output.shape) > 0:  # Make sure it's not a scalar
                if (len(output.shape) == 1 and output.shape[0] <= 1000) or \
                   (len(output.shape) == 2 and output.shape[1] <= 1000):
                    # For 1D output or 2D output with reasonable number of classes
                    if len(output.shape) == 1 or output.shape[0] == 1:
                        top_indices = np.argsort(flat_output)[-5:][::-1]
                        print(f"  Top 5 indices: {top_indices}")
                        print(f"  Top 5 values: {flat_output[top_indices]}")
        print("=====================")


def run_model(model_path, interactive=True, input_specs=None, output_dir=None, verbose=True):
    """Run an ONNX model and return the outputs.
    
    Args:
        model_path (str): Path to the ONNX model file
        interactive (bool): Whether to interactively prompt for input values
        input_specs (dict, optional): Dictionary with input specifications
        output_dir (str, optional): Directory to save outputs to
        verbose (bool): Whether to print detailed information
        
    Returns:
        dict: Dictionary mapping output names to numpy arrays
    """
    runner = ONNXModelRunner(model_path, verbose=verbose)
    if not runner.model:
        return {}
        
    inputs = runner.create_input_data(interactive=interactive, input_specs=input_specs)
    
    if verbose:
        print("\nRunning inference...")
        
    outputs = runner.run_inference(inputs)
    
    if outputs and verbose:
        runner.display_outputs(outputs)
        
    if output_dir:
        runner.save_outputs(outputs, output_dir)
        
    if verbose:
        print("\nInference completed!")
        
    return outputs


if __name__ == "__main__":
    import argparse
    
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description="Generic ONNX Model Runner")
    
    # Required arguments
    parser.add_argument("--model", type=str, help="Path to the ONNX model file", required=False)
    
    # Optional arguments
    parser.add_argument("--output_dir", type=str, help="Directory to save outputs", default=None)
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--quiet", action="store_true", help="Suppress detailed output")
    parser.add_argument("--input_config", type=str, help="JSON file with input configurations", default=None)
    
    args = parser.parse_args()
    
    # Print banner
    print("===== Generic ONNX Model Runner =====\n")
    
    # Get model path from arguments or prompt
    model_path = args.model
    if not model_path:
        model_path = input("Enter the path to the ONNX model file: ")
    
    # Load input configurations if provided
    input_specs = None
    if args.input_config:
        try:
            with open(args.input_config, 'r') as f:
                input_specs = json.load(f)
            print(f"Loaded input configurations from {args.input_config}")
        except Exception as e:
            print(f"Error loading input config: {e}")
    
    # Determine interactive mode
    interactive = args.interactive
    
    # If no explicit mode specified and no input config, default to interactive
    if not args.interactive and not input_specs:
        interactive = True
    
    # Run the model
    outputs = run_model(
        model_path=model_path,
        interactive=interactive,
        input_specs=input_specs,
        output_dir=args.output_dir,
        verbose=not args.quiet
    )