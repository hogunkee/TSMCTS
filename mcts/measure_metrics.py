import os
import numpy as np

def filter_log(log):
    success = False
    log = [float(t.replace('\n', '').split(' ')[-1]) for t in log if 'Score' in t]
    if log[-1]==log[-2]:
        if log[-1]>0.9:
            success = True
        log.pop(-1)
    log.append(success)
    return log

scenes = np.arange(40).tolist()
logs = [d for d in os.listdir('data') if d.startswith('mcts') or d.startswith('alphago')]
for logname in logs:
    print(logname)
    ep_length = []
    ep_success_length = []
    scenes = [s for s in os.listdir(os.path.join('data', logname)) is s.startswith('scene')]
    for scene in scenes:
        scene_dir = os.path.join('data', logname, scene)
        logfile = [f for f in os.listdir(scene_dir) if f.endswith('.log')][0]
        with open(os.path.join(scene_dir, logfile), 'r') as f:
            scores = filter_log(f.readlines())
        if scores[-1]:
            ep_success_length.append(len(scores[:-1]))
        ep_length.append(len(scores[:-1]))
    print('Success rate:', len(ep_success_length)/len(scenes))
    print('Success average length:', np.mean(ep_success_length))
    print('Average length:', np.mean(ep_length))
    print('-'*40)
