import numpy as np
from matplotlib.collections import PolyCollection
import matplotlib.pyplot as plt

from neuron import h, gui
import LFPy


class Neuron:
    def __init__(self, swc:str):
        pass
        # self.model = self._instantiate_swc(swc)
        # print(self.model)

    #
    # @staticmethod
    # def _instantiate_swc(filename):
    #     # With help from https://github.com/ahwillia/PyNeuron-Toolbox/blob/master/PyNeuronToolbox/
    #     h.load_file('stdlib.hoc')
    #     h.load_file('import3d.hoc')
    #
    #
    #     cell = h.Import3d_SWC_read()
    #     cell.input(filename)
    #
    #
    #     i3d = h.Import3d_GUI(cell, 0)
    #     i3d.instantiate(None)
    #     swc_secs = list(i3d.swc.sections)
    #     sec_list = {1: cell.soma, 2: cell.axon, 3: cell.dend, 4: cell.apic}
    #
    #     return i3d

if __name__ == '__main__':
    neuron_name = 'data/swc/rat/Turner_n419.swc'
    # nu = Neuron()
    cell = LFPy.Cell(neuron_name, passive=True)
    synapse_parameters = {
        'idx' : cell.get_idx("soma[0]"), # segment index for synapse
        'e': 0.,  # reversal potential
        'syntype': 'ExpSyn',  # synapse type
        'tau': 2.,  # syn. time constant
        'weight': .1,  # syn. weight
        'record_current': True  # record syn. current
    }
    synapse = LFPy.Synapse(cell, **synapse_parameters)
    synapse.set_spike_times(np.linspace(5, 500, 20))

    electrode_parameters = {
        'sigma': 0.3,  # extracellular conductivity
        'x': np.array([0]),
        'y': np.array([0]),
        'z': np.array([50])
    }
    electrode = LFPy.RecExtElectrode(cell, **electrode_parameters)
    cell.simulate()


    zips = []
    for x, z in cell.get_idx_polygons(projection=('x', 'z')):
        zips.append(list(zip(x, z)))
    polycol = PolyCollection(zips,
                             edgecolors='none',
                             facecolors='gray')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.add_collection(polycol)
    ax.plot(cell.xmid[synapse.idx], cell.zmid[synapse.idx], 'ro')
    ax.axis(ax.axis('equal'))
    plt.show()

    fig = plt.figure()
    plt.subplot(311)
    plt.plot(cell.tvec, synapse.i)
    plt.subplot(312)
    plt.plot(cell.tvec, cell.somav)
    plt.subplot(313)
    plt.plot(cell.tvec, electrode.LFP.T)
    plt.show()