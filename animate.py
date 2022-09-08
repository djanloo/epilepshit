import numpy as np
import json
from rich.progress import track
from rich import print

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

        # The frame dictionary is stored in files
        # Tells which bastard is ON at a specific time
        with open(EXCFILE, "r") as ex_f:
            self.exc_frame_dictionary = json.load(ex_f)
        
        with open(INHFILE, "r") as in_f:
             self.inh_frame_dictionary = json.load(in_f)
        
        self.max_time_index = max(self.exc_frame_dictionary["max_time_index"],
                                    self.inh_frame_dictionary["max_time_index"])

        # Time of simulation displayed
        self.time = 0.0
        self.time_index = 0

        ## BINNING IS MADE HERE
        binning_time = 5 # 10 us
        self.timesteps_per_frame = binning_time/TIMESTEP
        print(f"One frame is [green] {self.timesteps_per_frame} [/green] timesteps")

    
    def update(self):

        print(f"Elapsed time: {self.time: .1f} us")

        # Refresh each neuron to be at rest, then turn ON the ones specified

        for i, node in enumerate(self.net):
            # Neuron rest color: can be different from exc (0<i<400) and inh (400<i<500)
            if i < 400:
                node.value = 0.0
            else:
                node.value = 0.0

        for single_frame_time in range(int(self.timesteps_per_frame)):

            self.time += TIMESTEP
            self.time_index += 1

            if self.time_index > self.max_time_index:
                print(f"[red] Time index ({self.time_index}) is over maximum value ({self.max_time_index}) [/red]")

            firing_exc = self.exc_frame_dictionary.get(str(self.time_index))
            if firing_exc is not None:
                for ex_neuron in firing_exc:
                    self.net.nodes[ex_neuron].value = 1.0

            firing_inh = self.inh_frame_dictionary.get(str(self.time_index))
            if firing_inh is not None:
                for inh_neuron in firing_inh:
                    self.net.nodes[inh_neuron + 400].value = -1.0

A = Normal()

netplot.plot_lines = False
netplot.scat_kwargs['cmap'] = 'viridis'

animation = netplot.animate_super_network(A, A.update,
                                            frames=100, interval=60, blit=False)
animation.save(f'{ACTIVITY}.gif',progress_callback = lambda i, n: print(f'Saving frame {i} of {n}', end='\n'), dpi=80)


