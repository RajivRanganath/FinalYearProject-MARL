import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("training/policy.onnx")

def get_q(obs_array, agent_idx=0):
    agent_id = np.zeros(4, dtype=np.float32)
    agent_id[agent_idx] = 1.0
    obs = np.concatenate([obs_array, agent_id]).astype(np.float32)
    obs = np.expand_dims(obs, axis=0)
    hidden_in = np.zeros((1, 128), dtype=np.float32)
    outputs = session.run(None, {'obs': obs, 'hidden_state_in': hidden_in})
    return outputs[0][0]

print("=== Q-values for each agent across different observation states ===")
test_states = [
    ("High Battery (0.9), High Entropy Spike (0.8), No Neighbors Sampling (0.0)", [0.9, 0.8, 0.0]),
    ("High Battery (0.9), Low Entropy (0.1), No Neighbors Sampling (0.0)", [0.9, 0.1, 0.0]),
    ("Low Battery (0.1), High Entropy Spike (0.8), No Neighbors Sampling (0.0)", [0.1, 0.8, 0.0]),
    ("Low Battery (0.1), Low Entropy (0.1), No Neighbors Sampling (0.0)", [0.1, 0.1, 0.0]),
    ("High Battery (0.9), High Entropy Spike (0.8), High Neighbor Rate (0.75)", [0.9, 0.8, 0.75]),
    ("Dead Battery (0.01), High Entropy Spike (0.8), No Neighbors (0.0)", [0.01, 0.8, 0.0]),
]

for state_name, state in test_states:
    print(f"\n{state_name}:")
    for agent_idx in range(4):
        q = get_q(np.array(state, dtype=np.float32), agent_idx)
        preferred = "SAMPLE (1)" if np.argmax(q) == 1 else "SLEEP (0)"
        print(f"  Agent {agent_idx}: Q_sleep={q[0]:+.4f}, Q_sample={q[1]:+.4f} -> {preferred}")

print("\n=== Random state distribution test (1000 uniform samples per agent) ===")
for agent_idx in range(4):
    count_1 = 0
    for _ in range(1000):
        obs_array = np.random.uniform(0.0, 1.0, 3)
        if np.argmax(get_q(obs_array, agent_idx)) == 1:
            count_1 += 1
    print(f"Agent {agent_idx} Sample (act=1) rate: {count_1 / 10:.1f}% ({count_1} / 1000)")

