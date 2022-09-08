import numpy as np
import json
from rich.progress import track

from netgross.network import undNetwork
from netgross import netplot
import cnets

ACTIVITY = "fullepileptic"
NAME = "inhibitory"
FILE = "activityphases/{activity}activity_{name}.json".format(activity=ACTIVITY, name=NAME)

with open(FILE) as ex:
    for i, line in enumerate(ex):
        times = json.loads(line)
        print(f"line {i} done")


binning = np.empty(250_100, dtype=object)
for _ in range(len(binning)):
    binning[_] = []

for neuron_index, times in track(enumerate(times), total=400):
    for time in times:
        time_index = int(time*100)
        binning[time_index].append(neuron_index)


with open(f"binned_{NAME}_{ACTIVITY}.json", 'w', encoding='utf-8') as f:
    for timestep in track(binning, total=len(binning)):
        json.dump(timestep, f, ensure_ascii=False, indent=4)