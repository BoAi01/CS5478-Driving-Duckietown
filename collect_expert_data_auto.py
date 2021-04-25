import argparse
import numpy as np
from gym_duckietown.envs import DuckietownEnv
import sys
sys.path.insert(0, '/home/aibo/duck/gym-duckietown/learning')

from utils.teacher import PurePursuitExpert

# declare the arguments
parser = argparse.ArgumentParser()

# Do not change this
parser.add_argument('--max_steps', type=int, default=1500, help='max_steps')

# You should set them to different map name and seed accordingly
parser.add_argument('--map-name', default='map5')
args = parser.parse_args()

seeds = {
"map1": [2, 3, 5, 9, 12],
"map2": [1, 2, 3, 5, 7, 8, 13, 16],
"map3": [1, 2, 4, 8, 9, 10, 15, 21],
"map4": [1, 2, 3, 4, 5, 7, 9, 10, 16, 18],
"map5": [1, 2, 4, 5, 7, 8, 9, 10, 16, 23]
}

seeds_map = seeds[args.map_name]

for i, seed in enumerate(seeds_map):
    env = DuckietownEnv(
        map_name=args.map_name,
        domain_rand=False,
        draw_bbox=False,
        max_steps=args.max_steps,
        seed=seed
    )
    expert = PurePursuitExpert(env=env)
    obs = env.reset()
    # env.render()
    total_reward = 0
    obs_all, actions_all = [], []

    for step in range(args.max_steps):
        action = expert.predict(None)
        action = list(action)
        obs, reward, done, info = env.step(action)
        obs_all.append(obs)
        actions_all.append(action)
        total_reward += reward
        # env.render()
        print('Seed %d [%d / %d], steps = %s, Timestep Reward=%.3f, Total Reward=%.3f' % (seed, i, len(seeds_map), env.step_count,
                                                                                     reward, total_reward))
    print(f"Seed {seed}, {i} / {len(seeds_map)}, Total Reward {total_reward:.5f}")

    np.save(f'/home/aibo/duck_dataset/obs/obs_{args.map_name}_seed{seed}.npy', np.array(obs_all))
    # np.DataFrame(data=actions_all).to_csv(f'home/tesseract/Desktop/{args.map_name}_seed{args.seed}.txt', index=False)
    np.savetxt(f'/home/aibo/duck_dataset/anno/{args.map_name}_seed{seed}.txt', np.array(actions_all), delimiter=',')

print(f'file all saved!')
