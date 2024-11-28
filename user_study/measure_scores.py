import os
import numpy as np

log_files = [l for l in os.listdir('logs') if l.endswith('.txt')]
except_names = ['dohyeongkim', 'test', 'hyeokjin']
log_files = [l for l in log_files if l.split('.')[0] not in except_names]
print('logs:', log_files)
print()

average_scores = []
failure_counts = []

scene_types = {}
with open('scene-types.txt', 'r') as sf:
    types_lines = sf.readlines()
    for tl in types_lines:
        sname, stype = tl.split('-')
        scene_types[sname] = stype[:-1]

scores_per_type = {'B':[], 'C':[], 'D':[], 'O':[]}
for log_file in log_files:
    scores = []
    fcount = 0
    with open('logs/'+log_file, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            if '_' in line:
                s = line.split('_')[1].split('.png')[0]
                scores.append(int(s))

                sname = line.split(':')[0]
                stype = scene_types[sname]
                scores_per_type[stype].append(int(s))
            elif 'X' in line:
                fcount += 1
                #scores.append(1000)
    average_scores.append(np.mean(scores)/1000)
    failure_counts.append(fcount)

print("Results")
print(average_scores)
print(failure_counts)
print('='*40)

print('Average Threshold:', np.mean(average_scores))
print('\tSTD:', np.std(average_scores))
print('Average Failure Cases:', np.mean(failure_counts))
print('Success Rate:', 100*(1-np.mean(failure_counts)/20))
print('='*40)
for k in scores_per_type:
    print('Type-%s:'%k, np.mean(scores_per_type[k])/1000)
    print('\t', np.std(np.array(scores_per_type[k])/1000))
print()
