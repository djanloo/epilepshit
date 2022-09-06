import numpy as np
import json
from rich.progress import track

from netgross.network import undNetwork
from netgross import netplot
import cnets

NAME = "excitatory"
ACTIVITY = "normal"
FILE = "activityphases/{activity}activity_{name}.json".format(activity=ACTIVITY, name=NAME)


## Initializes a random network of 500 nodes with density 20%
# net = undNetwork.Random(500, 0.2)
# net.initialize_embedding(dim=2)
# net.cMDE([1.0, 0.5, 0.1], [0.5, 0.01, 0.0], [10, 1000, 3000])

with open(FILE) as ex:
    for i, line in enumerate(ex):
        times = json.loads(line)
        print(f"line {i} done")

# import matplotlib.pyplot as plt
# plt.hist(np.reshape(np.array(times, dtype=object), (-1,)))
# plt.show()
# exit()

binning = np.empty(250_100, dtype=object)
for _ in range(len(binning)):
    binning[_] = []

for neuron_index, times in track(enumerate(times), total=400):
    for time in times:
        time_index = int(time*100)
        print(f"Neuron {neuron_index} firing at {int(time*100)}")
        binning[time_index].append(neuron_index)


with open(f"binned_{NAME}_{ACTIVITY}.json", 'w', encoding='utf-8') as f:
    for timestep in track(binning, total=len(binning)):
        json.dump(timestep, f, ensure_ascii=False, indent=4)