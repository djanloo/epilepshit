import numpy as np
import json
from rich.progress import track
from rich import print
import argparse

from netgross.network import undNetwork
import cnets
from netgross import netplot
from matplotlib import pyplot as plt

TIMESTEP = 0.01 #us
BINNING_TIME = 1.5#us
FRAMES = 100

# Parser to set stuff from cmdline
parser = argparse.ArgumentParser()
parser.add_argument('--activity')
args = parser.parse_args()

if args.activity is None:
    exit("Specify an activity you must")
else:
    activity = args.activity

EXCFILE = f"binned_excitatory_{activity}.json"
INHFILE =  f"binned_inhibitory_{activity}.json"

# Netplot setup
netplot.plot_lines = False
netplot.scat_kwargs['cmap'] = 'viridis'
netplot.scat_kwargs['vmin'] = -1.0
netplot.scat_kwargs['vmax'] = 1.0

class Normal(undNetwork):

    def __init__(self):

        # Try to visualize the inhibitory and excitatory neurons
        # as two distinct clusters
        # Build a full-two-cluster adj matrix then kill some element
        adjacency_matrix = np.zeros((500, 500), dtype=np.float32)

        # EXC-EXC submatrix
        adjacency_matrix[0:400, 0:400] = 0.8*np.ones((400 ,400), dtype=np.float32)
        # INH-INH submatrix
        adjacency_matrix[400:500, 400:500] =  np.ones((100,100), dtype=np.float32)

        # EXC_INH submatrices
        adjacency_matrix[0:400, 400:500] = np.ones((400, 100), dtype=np.float32)
        adjacency_matrix[400:500, 0:400] = adjacency_matrix[0:400, 400:500].T

        # Kill an entry with probability 1-p: allegedly not correct but it's the first attempt
        # But first set the seed for reproducibility
        np.random.seed(17)
        cnets.set_seed(17)
        for i in track(range(500), description="Killing strangers"):
            for j in range(i, 500):
                if np.random.uniform(0,1) > 0.2:
                    adjacency_matrix[i,j] = 0.0
                    adjacency_matrix[j,i] = 0.0

        np.fill_diagonal(adjacency_matrix , 0.0)

        ## Yes I didn't know how to spell adjaajahecncy
        self.net = undNetwork.from_adiacence(adjacency_matrix)
        self._turn_off_all()
        self.net.initialize_embedding(dim=2)
        self.net.cMDE([1.0, 0.5, 0.1], [0.8, 0.01, 0.0], [100, 200, 500])

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

        # Starts from first event
        self.inh_frame_dictionary.pop("max_time_index")
        self.time_index = min(int(tindex) for tindex in self.inh_frame_dictionary.keys())

        ## BINNING IS MADE HERE
        self.timesteps_per_frame = BINNING_TIME/TIMESTEP
        print(f"One frame is [green] {self.timesteps_per_frame} [/green] timesteps")
        self.frames_required = int(self.max_time_index/self.timesteps_per_frame)
        print(f"Total: {self.frames_required} frames available ({self.max_time_index*TIMESTEP} us)")

        # Other stats to trace
        self.frame_index = 0
        self.exc_firing_per_frame = np.zeros(FRAMES + 2)
        self.inh_firing_per_frame = np.zeros(FRAMES + 2)

        self._turn_off_all()

        # Check to see if I was drunk when I wrote netgross
        for i in range(self.net.N):
            if self.net.nodes[i].n != i:
                print("Yes, [red]you were drunk.[/red]")


    def _turn_off_all(self):
        for node in self.net.nodes:
            if node.n < 400:
                node.value = 0.5
            else:
                node.value = -0.5
    
    def update(self):
        # Refresh each neuron to be at rest, then turn ON the ones specified
        self._turn_off_all()  
        for single_frame_time in range(int(self.timesteps_per_frame)):

            self.time += TIMESTEP
            self.time_index += 1

            if self.time_index > self.max_time_index:
                print(f"[red] Time index ({self.time_index}) is over maximum value ({self.max_time_index}) [/red]")

            firing_exc = self.exc_frame_dictionary.get(str(self.time_index))
            if firing_exc is not None:
                for ex_neuron in firing_exc:
                    self.net.nodes[ex_neuron].value = 1.0
                    self.exc_firing_per_frame[self.frame_index] += 1

            firing_inh = self.inh_frame_dictionary.get(str(self.time_index))
            if firing_inh is not None:
                for inh_neuron in firing_inh:
                    self.net.nodes[inh_neuron + 400].value = -1.0
                    self.inh_firing_per_frame[self.frame_index] += 1
        
        self.frame_index += 1
        print(f"Elapsed time: {self.time: .1f} us")


A = Normal()

animation = netplot.animate_super_network(A, A.update,
                                            frames=FRAMES,#A.frames_required, 
                                            interval=100, blit=False)
animation.save(f'{activity}_binning_{BINNING_TIME}.gif',progress_callback = lambda i, n: print(f'Saving frame {i} of {n}', end='\n'), dpi=80)

# for i in range(FRAMES):
#     A.update()
n_ = np.arange(0,len(A.exc_firing_per_frame))
plt.step(n_, A.exc_firing_per_frame)
plt.step(n_, A.inh_firing_per_frame)
plt.xlabel("frame index")
plt.title(f"{activity}")
plt.show()
