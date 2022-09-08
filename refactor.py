import numpy as np
import json
from rich.progress import track

from netgross.network import undNetwork
from netgross import netplot
import cnets

ACTIVITY = "metastable"
NAME = "excitatory"
FILE = "activityphases/{activity}activity_{name}.json".format(activity=ACTIVITY, name=NAME)

with open(FILE) as ex:
    # The file is single-line
    times_array = json.loads(ex.readline())

frames = {}

# The file is a list of lists of times so
# Cycles over the node index (first list is first node)
# And add this node to every frame in which it is active
for neuron_index, times in track(enumerate(times_array), total=len(times_array)):
    for time in times:
        time_index = int(time*100)
        already_firing = frames.get(time_index)
        # If a neuron is already active at that time, appends
        # else it starts the list
        if already_firing is None:
            frames[time_index] = [neuron_index]
        else:
            already_firing.append(neuron_index)

with open(f"binned_{NAME}_{ACTIVITY}.json", 'w', encoding='utf-8') as f:
    json.dump(frames, f, ensure_ascii=False, indent=4)