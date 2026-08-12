import os
import sys
import numpy as np
import onnxruntime as ort

# Add project root to sys.path to resolve imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from environment.pettingzoo_env import IoTSensorEnv

def simulate_deployment(onnx_path, episodes=1):
    print(f"Loading ONNX Model: {onnx_path}")
    session = ort.InferenceSession(onnx_path)
    
    # Initialize the real environment (Module A)
    env = IoTSensorEnv(scenario="stable")
    
    for episode in range(episodes):
        obs_dict, info_dict = env.reset()
        terminated = False
        truncated = False
        step = 0
        total_reward = 0
        samples_taken = 0
        
        print(f"\n--- Starting Episode {episode + 1} ---")
        
        while not (terminated or truncated):
            actions = {}
            for agent in env.agents:
                # EPyMARL adds one-hot agent ID to the observation.
                # We have 4 agents, so one-hot ID is size 4.
                # Total input shape is 3 (obs) + 4 (id) = 7
                base_obs = obs_dict[agent]
                
                agent_idx = int(agent.split('_')[1])
                one_hot = np.zeros(len(env.agents), dtype=np.float32)
                one_hot[agent_idx] = 1.0
                
                # Combine base observation and one-hot ID
                full_obs = np.concatenate([base_obs, one_hot]).astype(np.float32)
                full_obs = np.expand_dims(full_obs, axis=0) # Shape: (1, 7)
                
                # Dummy hidden state (since we use MLP, not RNN)
                hidden_in = np.zeros((1, 64), dtype=np.float32)
                
                # ONNX Inference
                inputs = {
                    'obs': full_obs,
                    'hidden_state_in': hidden_in
                }
                outputs = session.run(None, inputs)
                q_values = outputs[0][0]
                
                # Select action (argmax)
                action = int(np.argmax(q_values))
                actions[agent] = action
                
                if action == 1:
                    samples_taken += 1
            
            # Step the environment
            obs_dict, rewards_dict, terminations, truncations, info_dict = env.step(actions)
            
            # PettingZoo ParallelEnv returns dicts for rewards/terminations
            reward = rewards_dict[env.possible_agents[0]]
            total_reward += reward
            
            terminated = all(terminations.values())
            truncated = all(truncations.values())
            step += 1
            
        print(f"Episode {episode + 1} Finished!")
        print(f"Total Steps: {step}")
        print(f"Total Team Reward: {total_reward:.2f}")
        print(f"Total Samples Taken (across all agents): {samples_taken}")

if __name__ == "__main__":
    onnx_file = os.path.join(project_root, "training", "policy.onnx")
    if not os.path.exists(onnx_file):
        print(f"Error: ONNX file not found at {onnx_file}")
        sys.exit(1)
        
    simulate_deployment(onnx_file, episodes=3)
