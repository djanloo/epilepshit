import numpy as np
import json
from rich.progress import track

from netgross.network import undNetwork
from netgross import netplot
from matplotlib import pyplot as plt

NAME = "excitatory"
ACTIVITY = "normal"
FILE = f"binned_{NAME}_{ACTIVITY}.json"


class Normal(undNetwork):

    def __init__(self):
        ## Initializes a random network of 500 nodes with density 20%
        self.net = undNetwork.Random(500, 0.2)
        self.net.initialize_embedding(dim=2)
        self.net.cMDE([1.0, 0.5, 0.1], [0.5, 0.01, 0.0], [10, 1000, 3000])

        self.file = open(FILE, "r")
    
    def update(self):
        for i, node in enumerate(self.net):
            if i < 300:
                node.value = -0.5
            else:
                node.value = 0.5  
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
                print("Frame found")
                for node in firing_nodes:
                    print(f"Firing {node}")
                    if node < 300:
                        self.net.nodes[node].value = 1.0
                        print(f"set node {node} to one but node values are: {self.net.nodes.value}")
                    else:
                        self.net.nodes[node].value = -1.0
        print(f"update. Node values are: {self.net.nodes.value}")

A = Normal()

netplot.plot_lines = False
netplot.scat_kwargs['cmap'] = 'viridis'

animation = netplot.animate_super_network(A, A.update,
                                            frames=100, interval=60, blit=False)
animation.save('cnetsa.gif',progress_callback = lambda i, n: print(f'Saving frame {i} of {n}', end='\r'), dpi=80)
# plt.show()
A.file.close()


