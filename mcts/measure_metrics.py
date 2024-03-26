import os
import numpy as np
from dateutil import parser

def filter_log_time(log):
    success = False
    times = [parser.parse(log[i-1].split(',')[0]) for i in range(len(log)) if 'Best Action' in log[i]]
    return times
def get_intervals(times):
    delta_time = []
    for i in range(len(times)-1):
        delta = times[i+1] - times[i]
        delta_time.append(delta)
    return delta_time
def filter_log(log):
    success = False
    log = [float(t.replace('\n', '').split(' ')[-1]) for t in log if 'Score' in t]
    if len(log)<2:
        return None
    if log[-1]==log[-2]:
        if log[-1]>0.9:
            success = True
        log.pop(-1)
    log.append(success)
    return log

scenes = np.arange(40).tolist()
logs = [d for d in os.listdir('data') if d.startswith('mcts') or d.startswith('alphago')]
for logname in logs:
    ep_length = []
    ep_success_length = []
    deltas = []
    scenes = [s for s in os.listdir(os.path.join('data', logname)) if s.startswith('scene')]
    if len(scenes)<5:
        continue
    print(logname)
    for scene in scenes:
        scene_dir = os.path.join('data', logname, scene)
        logfile = [f for f in os.listdir(scene_dir) if f.endswith('.log')]
        if len(logfile)==0:
            continue
        logfile = logfile[0]
        with open(os.path.join(scene_dir, logfile), 'r') as f:
            x = f.readlines()
        times = filter_log_time(x)
        deltas += get_intervals(times)
        scores = filter_log(x)
        if scores is None:
            continue
        if scores[-1]:
            ep_success_length.append(len(scores[:-1]))
        ep_length.append(len(scores[:-1]))
    print('Num episodes:', len(scenes))
    print('Average time:', np.mean(deltas).seconds)
    print('Success rate:', len(ep_success_length)/len(scenes))
    print('Success average length:', np.mean(ep_success_length))
    print('Average length:', np.mean(ep_length))
    print('-'*40)
