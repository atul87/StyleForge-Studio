import os
import sys
import unittest

# Ensure the parent directory (NST_Code) and workspace lib folder are in the python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
workspace_dir = os.path.dirname(parent_dir)
lib_dir = os.path.join(workspace_dir, 'lib')

if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
if lib_dir not in sys.path:
    sys.path.insert(0, lib_dir)

class TestONNXModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.onnx_path = os.path.join(parent_dir, 'static', 'models', 'nst.onnx')
        cls.onnx_data_path = os.path.join(parent_dir, 'static', 'models', 'nst.onnx.data')

    def test_onnx_files_exist(self):
        # Verify the ONNX files are in the static/models folder
        self.assertTrue(os.path.exists(self.onnx_path), f"ONNX file not found at {self.onnx_path}")
        self.assertTrue(os.path.exists(self.onnx_data_path), f"ONNX weights data file not found at {self.onnx_data_path}")
        
        # Verify file size of weights is significant
        self.greater_size = os.path.getsize(self.onnx_data_path) > 10 * 1024 * 1024 # > 10MB
        self.assertTrue(self.greater_size)

    def test_onnx_graph_validation(self):
        try:
            import onnx
        except ImportError:
            self.skipTest("onnx package not installed. Skipping model validation.")

        # Load and validate the ONNX model graph structure
        model = onnx.load(self.onnx_path)
        onnx.checker.check_model(model)
        
        # Inspect model inputs and outputs
        graph = model.graph
        inputs = graph.input
        outputs = graph.output
        
        # Verify we have content, style inputs, and a stylized output
        input_names = [inp.name for inp in inputs]
        output_names = [out.name for out in outputs]
        
        print(f"\nONNX Model Inputs: {input_names}")
        print(f"ONNX Model Outputs: {output_names}")
        
        # Checking that we have at least inputs and outputs
        self.assertTrue(len(inputs) >= 1)
        self.assertTrue(len(outputs) >= 1)

if __name__ == '__main__':
    unittest.main()
