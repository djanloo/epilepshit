import numpy as np
import json
from rich.progress import track

from netgross.network import undNetwork
from netgross import netplot
from matplotlib import pyplot as plt

ACTIVITY = "metastable"
EXCFILE = f"binned_excitatory_{ACTIVITY}.json"
INHFILE =  f"binned_inhibitory_{ACTIVITY}.json"

class Normal(undNetwork):

    def __init__(self):

        ## Initializes a random network of 500 nodes with density 20%
        self.net = undNetwork.Random(500, 0.2)
        self.net.initialize_embedding(dim=2)
        self.net.cMDE([1.0, 0.5, 0.1], [0.5, 0.01, 0.0], [10, 200, 500])

        # Files
        self.exc_file = open(EXCFILE, "r")
        self.inh_file = open(INHFILE, "r")

        # Time of simulation displayed
        self.time = 0.0
    
    def update(self):

        print(f"Elapsed time: {self.time: .2f} us")

        for i, node in enumerate(self.net):

            # Neuron rest color: can be different from exc (0<i<400) and inh (400<i<500)
            if i < 400:
                node.value = 0.0
            else:
                node.value = 0.0

        #### SUPER ALERT
        # BINNING IS MADE HERE
        timesteps_per_frame = 1000

        for repeat in range(timesteps_per_frame):
            try:
                # Tries to load a line from both files
                # If weird shit happens just skips
                # And weird shit can mean no firing nodes in that frame
                firing_exc = np.array(json.loads(self.exc_file.readline())).reshape((-1,))
                firing_inh = np.array(json.loads(self.inh_file.readline())).reshape((-1,))
                
                # However it went, increase the animation display time
                self.time += 0.01 # One timestep (us)

            except json.JSONDecodeError:
                continue
            else:

                for node in firing_exc:
                    # Excitatory are 0 to 400
                    self.net.nodes[node].value = 1.0

                for node in firing_inh:
                    # First inhibitory is 400th of global network
                    self.net.nodes[node + 400].value = -1.0

A = Normal()

netplot.plot_lines = False
netplot.scat_kwargs['cmap'] = 'viridis'

animation = netplot.animate_super_network(A, A.update,
                                            frames=100, interval=60, blit=False)
animation.save(f'{ACTIVITY}.gif',progress_callback = lambda i, n: print(f'Saving frame {i} of {n}', end='\r'), dpi=80)
# plt.show()
A.exc_file.close()
A.inh_file.close()


