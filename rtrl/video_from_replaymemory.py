import rtrl
import numpy as np

# name = '191102_230743_1'  # race obstacles
# name = '191102_004927_1'  # race solo
name = '191102_004949_1'  # city
run = rtrl.load('/mnt/scratch/rmst/exp/' + name)


m = run.agent.memory.memory

obs, ac, rew, next_obs, done = zip(*m)

ac = np.asarray(ac)
rew = np.asarray(rew)
done = np.asarray(done)

core_obs, *_ = zip(*obs[-2000:])
vis_obs, vec_obs = zip(*core_obs)

import imageio
writer = imageio.get_writer('/mnt/scratch/rmst/test.mp4', fps=20)

for frame in vis_obs:
  writer.append_data(frame[0])

writer.close()