import numpy as np
import json
from rich.progress import track

from netgross.network import undNetwork
from netgross import netplot
from matplotlib import pyplot as plt

NAME = "excitatory"
ACTIVITY = "metastable"
FILE = f"binned_{NAME}_{ACTIVITY}.json"


class Normal(undNetwork):

    def __init__(self):
        ## Initializes a random network of 500 nodes with density 20%
        self.net = undNetwork.Random(500, 0.2)
        self.net.initialize_embedding(dim=2)
        self.net.cMDE([1.0, 0.5, 0.1], [0.5, 0.01, 0.0], [10, 200, 500])

        self.file = open(FILE, "r")
    
    def update(self):
        for i, node in enumerate(self.net):
            if i < 300:
                node.value = 0.0
            else:
                node.value = 0.0
        #### SUPER ALERT
        # BINNING IS MADE HERE
        timesteps_per_frame = 100
        for repeat in range(timesteps_per_frame):
            try:
                firing_nodes_string = self.file.readline()
                firing_nodes = np.array(json.loads(firing_nodes_string)).reshape((-1,))
            except json.JSONDecodeError:
                continue
            else:
                for node in firing_nodes:
                    if node < 300:
                        self.net.nodes[node].value = 1.0
                    else:
                        self.net.nodes[node].value = -1.0

A = Normal()

netplot.plot_lines = False
netplot.scat_kwargs['cmap'] = 'viridis'

animation = netplot.animate_super_network(A, A.update,
                                            frames=100, interval=60, blit=False)
animation.save(f'{ACTIVITY}.gif',progress_callback = lambda i, n: print(f'Saving frame {i} of {n}', end='\r'), dpi=80)
# plt.show()
A.file.close()


