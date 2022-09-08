import numpy as np
import json
from rich.progress import track

from netgross.network import undNetwork
from netgross import netplot
from matplotlib import pyplot as plt

ACTIVITY = "metastable"
EXCFILE = f"binned_excitatory_{ACTIVITY}.json"
INHFILE =  f"binned_inhibitory_{ACTIVITY}.json"
TIMESTEP = 0.01 #us

class Normal(undNetwork):

    def __init__(self):

        # Try to visualize the inhibitory and excitatory neurons
        # as two distinct clusters
        # Build a full-two-cluster adj matrix then kill some element
        adjacency_matrix = np.zeros((500, 500), dtype=np.float32)
        adjacency_matrix[0:400, 0:400] = np.ones((400 ,400), dtype=np.float32)
        adjacency_matrix[400:500, 400:500] = np.ones((100,100), dtype=np.float32)
        adjacency_matrix[0:400, 400:500] = 2 * np.ones((400, 100), dtype=np.float32)
        adjacency_matrix[400:500, 0:400] = 2 * np.ones((100, 400), dtype=np.float32)
        # Kill an entry with probability 1-p: allegedly not correct but it's the first attempt
        # But first set the seed for reproducibility
        np.random.seed(42)
        for i in track(range(500), description="Killing strangers"):
            for j in range(i, 500):
                if np.random.uniform(0,1) > 0.2:
                    adjacency_matrix[i,j] = 0.0
                    adjacency_matrix[j,i] = 0.0


        np.fill_diagonal(adjacency_matrix , 0.0)

        ## Yes I didn't know how to spell adjaajahecncy
        self.net = undNetwork.from_adiacence(adjacency_matrix)
        self.net.initialize_embedding(dim=2)
        self.net.cMDE([1.0, 0.5, 0.1], [0.5, 0.01, 0.0], [10, 200, 500])

        # Files
        self.exc_file = open(EXCFILE, "r")
        self.inh_file = open(INHFILE, "r")

        # Time of simulation displayed
        self.time = 0.0
    
    def update(self):

        print(f"Elapsed time: {self.time: .1f} us")

        for i, node in enumerate(self.net):

            # Neuron rest color: can be different from exc (0<i<400) and inh (400<i<500)
            if i < 400:
                node.value = 0.0
            else:
                node.value = 0.0

        #### SUPER ALERT
        # BINNING IS MADE HERE
        binning_time = 1_000 # 10 us
        timesteps_per_frame = binning_time/TIMESTEP

        for repeat in range(int(timesteps_per_frame)):
            try:
                # However it went, increase the animation display time
                # Must be stated before the dangerous operation 
                # because catching the error prevents from tracing the timestep
                self.time += 0.01 # One timestep (us)

                # Tries to load a line from both files
                # If weird shit happens just skips
                # And weird shit can mean no firing nodes in that frame
                firing_exc = np.array(json.loads(self.exc_file.readline())).reshape((-1,))
                firing_inh = np.array(json.loads(self.inh_file.readline())).reshape((-1,))

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


