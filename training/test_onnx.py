import onnxruntime as ort
import numpy as np
import sys

def verify_onnx(onnx_path):
    print(f"Loading ONNX model from {onnx_path}")
    session = ort.InferenceSession(onnx_path)
    
    # EPyMARL MLP input is (obs) [batch_size, 7] and (hidden_state_in) [batch_size, 64]
    dummy_obs = np.random.rand(1, 7).astype(np.float32)
    dummy_hidden = np.zeros((1, 64), dtype=np.float32)
    
    inputs = {
        'obs': dummy_obs,
        'hidden_state_in': dummy_hidden
    }
    
    print("Running inference...")
    outputs = session.run(None, inputs)
    
    q_values = outputs[0]
    hidden_out = outputs[1]
    
    print(f"Q-values shape: {q_values.shape}")
    print(f"Q-values: {q_values}")
    
    if q_values.shape == (1, 2):
        print("Success! ONNX model loaded and executed correctly with expected output shape.")
    else:
        print(f"Error! Expected Q-values shape (1, 2) but got {q_values.shape}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_onnx.py <path_to_onnx_file>")
        sys.exit(1)
    verify_onnx(sys.argv[1])
