import sys
import numpy as np
sys.path.insert(0, '.')
from envs.drone_env import WhoopDroneEnv

env = WhoopDroneEnv()
obs, _ = env.reset(seed=0)
print('obs shape:', obs.shape)
print('action space:', env.action_space)
print('HOVER_THROTTLE:', round(env.HOVER_THROTTLE, 4))

# Zero delta action = hover throttle on every motor → drone should stay airborne
total_r = 0.0
for i in range(100):
    obs, r, term, trunc, info = env.step(np.zeros(4, dtype='f'))
    total_r += r
    if term or trunc:
        print(f'Crashed at step {i+1},  dist={info["distance_to_target"]:.3f} m')
        break
else:
    print(f'Survived 100 steps!  total_r={total_r:.2f}  dist={info["distance_to_target"]:.3f} m')

env.close()
