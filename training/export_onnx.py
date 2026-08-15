import torch
import sys
import os

# Ensure epymarl source is in path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "epymarl", "src"))
from modules.agents.rnn_agent import RNNAgent
from types import SimpleNamespace

def export_to_onnx(checkpoint_dir, output_path):
    """
    Exports the trained individual agent policy (the RNN network) to ONNX format.
    Module C relies on this file to evaluate hardware performance.
    """
    print(f"Loading model from {checkpoint_dir}...")
    
    # In EPyMARL, models are saved in the checkpoint_dir (e.g., results/models/vdn_.../1/)
    # The agent network is typically named agent.th
    agent_path = os.path.join(checkpoint_dir, "agent.th")
    
    if not os.path.exists(agent_path):
        print(f"Error: Could not find agent.th in {checkpoint_dir}")
        sys.exit(1)
        
    # We construct the args namespace to initialize RNNAgent
    # Default EPyMARL settings for our env:
    args = SimpleNamespace(
        hidden_dim=128, # IQL uses 128
        rnn_hidden_dim=128,
        n_actions=2,       # Sleep or Sample
        use_rnn=False,     # Based on EPyMARL defaults for MLP
    )
    
    # EPyMARL by default adds agent_id (one-hot) to the observation if obs_agent_id=True.
    # For 4 agents, one-hot id is size 4. Mock env obs is size 3. Total input = 3 + 4 = 7.
    # Build model (matches epymarl RNNAgent exactly)
    agent = RNNAgent(input_shape=7, args=args)
    
    # Load weights
    agent.load_state_dict(torch.load(agent_path, map_location=lambda storage, loc: storage))
    agent.eval()
    
    # Dummy inputs for tracing: [batch_size, input_shape] and [batch_size, hidden_dim]
    dummy_obs = torch.randn(1, 7)
    dummy_hidden = torch.randn(1, args.rnn_hidden_dim)
    
    print(f"Exporting to {output_path}...")
    torch.onnx.export(
        agent,
        (dummy_obs, dummy_hidden),
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['obs', 'hidden_state_in'],
        output_names=['q_values', 'hidden_state_out'],
        dynamic_axes={'obs': {0: 'batch_size'}, 'hidden_state_in': {0: 'batch_size'}}
    )
    
    print(f"Successfully exported ONNX model to {output_path}")
    print("Module C can now use this file for hardware profiling.")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python export_onnx.py <checkpoint_dir_containing_agent.th> <output_onnx_path>")
        sys.exit(1)
    
    export_to_onnx(sys.argv[1], sys.argv[2])
