"""
Created on Thu Jan 16 17:36:44 2025

@author: Antonios
"""
# written by Antonios Dougalis, Feb 2025, Kuopio, Finland
# contact me at antoniosdougalis(at)gmail.com

from neuron import h, gui
from neuron.units import ms, mV
import numpy as np
import matplotlib.pyplot as plt
import os


morph_file = "HS14012020_001.ASC"
dll_file = r'C:\Users\anton\Documents\Computational modelling\195667-master\GitHub_Dougalis\nrnmech.dll'

morph_path = r'C:\Users\anton\Documents\Computational modelling\195667-master\morphs'
dll_path = r'C:\Users\anton\Documents\Computational modelling\195667-master\GitHub_Dougalis'

morph_file_path = os.path.join(morph_path, morph_file)
dll_file_path   = os.path.join(dll_path, dll_file)

os.chdir(dll_path)
print(os.getcwd())

# load help file
with open("help_file.py") as file:
    exec(file.read())
 
   
# NB: all density mechanims and point processes are automatically loaded in NEURON since THE nrnmech file is in the correct folder (Antonios)
# NO NEED TO EXPLICITLY CALL IT VIA e.g. h.load_file(dll_file)
    
class Morph_HumanNeuron:
    
    def __init__(self, morph_file_path): # Initialize the cell and load the morphology from the provided .asc file
        self.cell = self.load_morphology(morph_file_path)
    
    def load_morphology(self, morph_file_path): # Import the morphology from the .asc file (Neurolucida)
        h.load_file("import3d.hoc")
        cellmorph = h.Import3d_Neurolucida3()
        cellmorph.input(morph_file_path)
        imprt = h.Import3d_GUI(cellmorph, False)
        imprt.instantiate(None)
                
        return cellmorph
    
    
    def create_secList(self): # Create a list of all sections (compartments) in the cell
        secList  = h.SectionList( [sec for sec in h.allsec() ] )
                
        return secList  
    
    
    def plot_morphology(self, secList):
        
        theShape = h.Shape()
        theShape.show(0)
        for sec in secList:
            if 'soma' in sec.name():        
                theShape.color(1, sec = sec)
            elif 'apic' in sec.name():
                theShape.color(2, sec = sec) # make them 
            elif 'dend' in sec.name():
                theShape.color(3, sec = sec) # make them blue
            elif 'dend' or 'apic'in sec.name() and 'spine' in sec.name():
                theShape.color(4, sec = sec) # make them green        
        
        return theShape
    
    
    def total_length_of_neuron(self, secList):
        neuronLength = 0
        for sec in secList:
            neuronLength = neuronLength + sec.L
        
        return neuronLength
       
    def uninsert_mechanism(self): # uninsert ALL MECHANISMS from all sections in order to START OVER AGAIN
        for sec in h.allsec():
            for seg in range(sec.nseg):
                temp = sec.psection()['density_mechs']
                countTemp = len( list( temp.keys() ) ) # get the keys/()
                mechList = list( temp.keys() )
                for counti in range(countTemp):
                    sec.uninsert(mechList[counti])
                    
    def insert_mechanism(self, densityMech, section):
        # cell.uninsert_mechanism() # first clear any mechanisms that are already there so that they dont accumulat
        section = section
        for sec in h.allsec():               
            if 'soma' in sec.name() and 'soma' in section :
                sec.nseg = 1
                sec.cm = 1
                sec.Ra = 35.4
                if densityMech == ['hh']:
                    sec.insert(densityMech[0])  # Insert Hodgkin-Huxley mechanism
                    sec.gnabar_hh = 0.76
                    sec.gkbar_hh = 0.036
                    sec.gl_hh = 0.0003
                    sec.el_hh = -74.3
                    sec.ek = -97
                elif not densityMech:
                    densityMech = ['pas']
                    sec.insert(densityMech[0])  
                elif not densityMech=='hh': 
                    countDens = len(densityMech)
                    if countDens >= 1:
                        for mechi in range(countDens):
                            sec.insert(densityMech[mechi])  #Insert all the listed mechsnims in this section
                   
            elif 'axon' in sec.name() and 'axon' in section:
                sec.nseg = 10
                sec.cm = 1
                sec.Ra = 35.4
                for seg in sec:
                    if densityMech == ['hh']:
                        sec.insert(densityMech[0])  # Insert Hodgkin-Huxley mechanism 
                        sec.gnabar_hh = 0.76
                        sec.gkbar_hh = 0.036
                        sec.gl_hh = 0.00015
                        sec.el_hh = -74.3
                        sec.ek = -97 
                    elif not densityMech:
                        densityMech = ['pas']
                        sec.insert(densityMech[0])  
                    elif not densityMech=='hh':
                        countDens = len(densityMech)
                        if countDens >= 1:
                            for mechi in range(countDens):
                                sec.insert(densityMech[mechi])  #Insert all the listed mechsnims in this section
                    
            elif 'dend' in sec.name() and 'dend' in section:
                sec.nseg = 10
                sec.cm = 1
                sec.Ra = 100
                for seg in sec:
                    if densityMech == ['hh']:
                        sec.insert(densityMech[0])  # Insert Hodgkin-Huxley mechanism 
                        sec.gnabar_hh = 0.3
                        sec.gkbar_hh = 0.018
                        sec.gl_hh = 0.00015
                        sec.el_hh = -74.3
                        sec.ek = -97
                    elif not densityMech:
                        densityMech = ['pas']
                        sec.insert(densityMech[0])
                        sec.g_pas = 0.001
                        sec.e_pas = -54
                    elif not densityMech=='hh': 
                        countDens = len(densityMech)
                        if countDens >= 1:
                            for mechi in range(countDens):
                                sec.insert(densityMech[mechi])  #Insert all the listed mechsnims in this section    
                     
            elif 'apic' in sec.name() and 'apic' in section:
                sec.nseg = 10
                sec.cm = 1
                sec.Ra = 100
                for seg in sec:
                    if densityMech == ['hh']:
                        sec.insert(densityMech[0])  # Insert Hodgkin-Huxley mechanism 
                        sec.gnabar_hh = 0.3
                        sec.gkbar_hh = 0.018
                        sec.gl_hh = 0.00015
                        sec.el_hh = -74.3
                        sec.ek = -97
                    elif not densityMech:
                        densityMech = ['pas']
                        sec.insert(densityMech[0])
                        sec.g_pas = 0.001
                        sec.e_pas = -54
                    elif not densityMech=='hh': 
                        countDens = len(densityMech)
                        if countDens >= 1:
                            for mechi in range(countDens):
                                sec.insert(densityMech[mechi])  #Insert all the listed mechsnims in this section
                    
    def delete_spines(self):
        updated_secList = h.SectionList()
        
        for sec in h.allsec():
            updated_secList.append(sec)
            if 'spine' in sec.name():
                h.delete_section(sec=sec)
                # updated_secList.remove(sec=sec) # the section is automatically removewd from the sl, no need to use remove
                # updated_secList.printnames() # double check that this is as expected aboves 
                
        return updated_secList
    
    def load_model_parameters(self):
        # create a dictionary holding spine and synapse parameters
       
        model_parameters = {
        'noSpinesDist': 60,    
        'spineRa_Factor' : 3, 
        'spine_neck_diam' : 0.25, 'spine_neck_L' : 1.35, 'spine_head_Area' : 2.8, 'spine_neck_Ra' : 0,
        'rev_syn_AMPA' : 0, 'rev_syn_NMDA' : 0,
        'tau_1_AMPA' : 0.3,'tau_2_AMPA' : 1.8,
        'tau_1_NMDA' : 3, 'tau_2_NMDA' : 75,
        'N_NMDA' : 0.280112, 'GAMA_NMDA' : 0.08,
        'NMDA_W' : 0.0014,'AMPA_W' : 0.0007,
        'spine_head_X' : 1, 'dend_X' : 1, # position in spine where i will add the receptor
        
         }
        
        # define the spine_neck_Ra using the spineRa_Factor
        model_parameters['spine_neck_Ra'] = 35.4*model_parameters['spineRa_Factor']
        
        return model_parameters
    
    # function that calcluates the distance of each section from the soma
    def distance_from_soma(self, secList):
        disFromSoma = [ ]
        logDisSection = [ ] # use to keep the names of the sections that are allowed to have spines, i.e. those that are a certain fyrthr away from the soma
        section = list(secList)[0]
       
        model_parameters = cortical_cell.load_model_parameters()  
        
        h.distance(sec = section) # define section that is the origin (that is the soma)
        for sec in secList:
            tempDist = h.distance(section(0.5), sec(0.5))
            disFromSoma.append( tempDist ) # caluclate distance to soma
            if tempDist>model_parameters['noSpinesDist']:
                logDisSection.append( sec.name() ) 
       
        return disFromSoma, logDisSection
   
    
    def add_single_spines_AND_synapses(self, num_spines, section, syn_start, syn_interval, syn_noise, syn_number ):

        # # initialise and clear the spines so it is easier to run multiple times without accumulation
        del spines_neckList[:]
        del spines_headList[:]
        del synList[:]
        del netConList[:]
        del synStimList[:]
        
        # add the model parameters
        model_parameters = cortical_cell.load_model_parameters()  
        
        factor = 1/num_spines # for positioning the spines in a valid numerical segment of the section
        
        # Add NMDA receptors to each spine, add NetCon Object to control ALL the spines of a given segment
        for idx, sec in enumerate(list(secList)): 
            if section in sec.name():
            
                for i in range(num_spines):  # Create spines in section
                    
                    spine_neck =  h.Section(name= f'{sec}_spine_neck{i}') 
                    spine_head =  h.Section(name= f'{sec}_spine_head{i}') 
                    
                    spine_neck.L = model_parameters['spine_neck_L']
                    spine_neck.diam = model_parameters['spine_neck_diam']
                    spine_neck.Ra = model_parameters['spine_neck_Ra']
                    
                    spine_head_area = model_parameters['spine_head_Area']
                    spine_head.L = np.sqrt( (spine_head_area/(4*np.pi)) )
                    spine_head.diam = spine_head.L
                    spine_head.Ra = model_parameters['spine_neck_Ra']
                    
                    spine_neck.connect( sec( (i+1)* factor ) ) # connect the zero end of the spine neck to different part of the dendite
                    spine_head.connect( spine_neck(1) ) # conncet head and neck of the 
                    
                    
                    # initialise a new separate synStim for each spine of the neuron 
                    # (NB: all spines of the section will receive the same stimulus characteristics
                    # and the AMPA and NMDA receptors will be simultaneously activated)
                
                    synStim = h.NetStim( sec( (i + 1) * factor) )
                    synStim.start = syn_start # ms delay to start synaptic input
                    synStim.interval = syn_interval #ms inmter activation interval of synaptic input
                    synStim.noise = syn_noise # regularity(randomness of synaptic input)
                    synStim.number = syn_number # number of activations of synaptic input
                    synStimList.append(synStim)
            
                    hocL.append(h.SectionRef())  # Create a reference to the current section (spine) and add to hocL
                    
                    # insert AMPA receptor omechansim on the spine head
                    ampa_receptor = h.Exp2Syn(model_parameters['spine_head_X'], sec=spine_head)
                    synList.append( ampa_receptor )
              		
                    # note that every spine has a SEPARATE NETCON OBJECT FOR AMPA and a separate for NMDA receptor activation
                    netConObj = h.NetCon(synStimList[-1], synList[-1])
                    netConList.append( netConObj )
                  
              	    # Set AMPA receptor properties
                    synList[-1].e = model_parameters['rev_syn_AMPA']
                    synList[-1].tau1 = model_parameters['tau_1_AMPA']
                    synList[-1].tau2 = model_parameters['tau_2_AMPA']
                    netConList[-1].weight[0] = model_parameters['AMPA_W']
                    netConList[-1].delay = 0
                       
                    # Create NMDA receptor point process at the head of the spine
                    nmda_receptor = h.NMDA(model_parameters['spine_head_X'], sec=spine_head)
                    synList.append( nmda_receptor )
                    
                    netConObj = h.NetCon(synStimList[-1],synList[-1])
                    netConList.append( netConObj )
                       
                    # Set NMDA receptor properties
                    synList[-1].e = model_parameters['rev_syn_NMDA']  # Reversal potential (mV)
                    synList[-1].tau_r_NMDA = model_parameters['tau_1_NMDA']  # Rise time constant (ms)
                    synList[-1].tau_d_NMDA = model_parameters['tau_2_NMDA']  # Decay time constant (ms)
                    synList[-1].n_NMDA = model_parameters['N_NMDA']  # Additional parameter for NMDA receptor (e.g., scaling factor)
                    synList[-1].gama_NMDA = model_parameters['GAMA_NMDA']  # Another NMDA-specific parameter (e.g., scaling factor)
                    
                    netConList[-1].weight[0] = model_parameters['NMDA_W']
                    netConList[-1].delay = 0
                    
                    # Append the spine to the list
                    spines_neckList.append(spine_neck)
                    spines_headList.append(spine_head)
                   
        return hocL, synList, netConList, spines_neckList, spines_headList  


    def add_spines_AND_AMPA_NMDA_synapses(self, spineDensity, model_parameters, syn_start, syn_interval, syn_noise, syn_number ):
        
        # # initialise and clear the spines so it is easier to run multiple times without accumulation
        del spines_neckList[:]
        del spines_headList[:]
        del synList[:]
        del netConList[:]
        del synStimList[:]
        
        # add the model parameters
        model_parameters = cortical_cell.load_model_parameters()  
        
        # initialise section distance to saom calculation
        disFromSoma, _ = cortical_cell.distance_from_soma(secList)   
        
        # Add NMDA receptors to each spine, add NetCon Object to control ALL the spines of a given segment
        for idx, sec in enumerate(list(secList)):              
            
            num_spines = int(spineDensity*sec.L)
            factor = 1/num_spines # for positioning the spines in a valid numerical segment of the section
            
            # check the sitance to soma. If >30 mic inseert spoines in section otherwiose dont!
            if disFromSoma[idx]>model_parameters['noSpinesDist'] and 'axon' not in sec.name(): # insert spines in sections that are > 60 mic from soma and are not an axon
            
                for i in range(num_spines):  # Create spines in section
                    
                
                    spine_neck =  h.Section(name= f'{sec}_spine_neck{i}') 
                    spine_head =  h.Section(name= f'{sec}_spine_head{i}') 
                    
                    spine_neck.L = 1.35
                    spine_neck.diam = 0.25
                    spine_neck.Ra = 200
                    
                    spine_head_area = 2.8
                    spine_head.L = np.sqrt( (spine_head_area/(4*np.pi)) )
                    spine_head.diam = spine_head.L
                    spine_head.Ra = 200
                    
                    spine_neck.connect( sec( (i+1)* factor ) ) # connect the zero end of the spine neck to different part of the dendite
                    spine_head.connect( spine_neck(1) ) # conncet head and neck of the 
                    
                    # initilaise a new separate synStim for each spine of the neuron 
                    # (NB: all spines of the section will receive the same stimulus characteristics
                    # and the AMPA and NMDA receptors will be simultaneously activated)
                
                    synStim = h.NetStim( sec( (i + 1) * factor) )
                    synStim.start = syn_start # ms delay to start synaptic input
                    synStim.interval = syn_interval #ms inmter activation interval of synaptic input
                    synStim.noise = syn_noise # regularity(randomness of synaptic input)
                    synStim.number = syn_number # number of activations of synaptic input
                    synStimList.append(synStim)
            
                    hocL.append(h.SectionRef())  # Create a reference to the current section (spine) and add to hocL
                    
                    # insert AMPA receptor omechansim on the spine head
                    ampa_receptor = h.Exp2Syn(model_parameters['spine_head_X'], sec=spine_head)
                    synList.append( ampa_receptor )
                		
                    # note that every spine has a SEPARATE nETcON OBJECT FOR AMPA and a separate for NMDA receptor activation
                    netConObj = h.NetCon(synStimList[-1], synList[-1])
                    netConList.append( netConObj )
                    
                	# Set AMPA receptor properties
                    synList[-1].e = model_parameters['rev_syn_AMPA']
                    synList[-1].tau1 = model_parameters['tau_1_AMPA']
                    synList[-1].tau2 = model_parameters['tau_2_AMPA']
                    netConList[-1].weight[0] = model_parameters['AMPA_W']
                    netConList[-1].delay = 0
                       
                    # Create NMDA receptor point process at the head of the spine
                    nmda_receptor = h.NMDA(model_parameters['spine_head_X'], sec=spine_head)
                    synList.append( nmda_receptor )
                    
                    netConObj = h.NetCon(synStimList[-1],synList[-1])
                    netConList.append( netConObj )
                       
                    # Set NMDA receptor properties
                    synList[-1].e = model_parameters['rev_syn_NMDA']  # Reversal potential (mV)
                    synList[-1].tau_r_NMDA = model_parameters['tau_1_NMDA']  # Rise time constant (ms)
                    synList[-1].tau_d_NMDA = model_parameters['tau_2_NMDA']  # Decay time constant (ms)
                    synList[-1].n_NMDA = model_parameters['N_NMDA']  # Additional parameter for NMDA receptor (e.g., scaling factor)
                    synList[-1].gama_NMDA = model_parameters['GAMA_NMDA']  # Another NMDA-specific parameter (e.g., scaling factor)
                    
                    netConList[-1].weight[0] = model_parameters['NMDA_W']
                    netConList[-1].delay = 0
                    
                    # Append the spine to the list
                    spines_neckList.append(spine_neck)
                    spines_headList.append(spine_head)
                   
        return  hocL, synList, netConList, spines_neckList, spines_headList                   
    
# Function for running simulations
# initialising simulations (def: initialise_simulation)
# cretating and destroying IClamps for stimulation (def: create_stimIclamp )
# selecting compartments to plot (def: select_plot_compartment)
# specifying recording vectors (def: prepare_recording_vecs)
# converting hoc vectros to python (def: hvec_to_numpy)
# pythonic run of simulation with plotting via matplotlib (def: run_simulation, def:plot_results )
# hoc style NEURON simulation run with vbox plotting (def: graph_simulation_real_time)

def initialise_simulation(v_init, tstop, steps_per_ms, celcius):
           
    # Set simulation parameters
    h.v_init = v_init
    h.finitialize(h.v_init)
    h.tstop = tstop  # Set the simulation duration to 1000 ms
    h.steps_per_ms = steps_per_ms  # Set the number of steps per ms for high temporal resolution
    h.celsius = celcius  # Set the temperature to 37°C (for mammalian neurons)
    
def create_stimIclamp(secList, delay, duration, amplitude, section):
    
    # find the index of this section in secList
    for idx_section, sec in enumerate(list(secList)):
        if section in sec.name():  # or sec.name() == 'soma[0]' if the name is exact
            print(f"The index of soma[0] is: {idx_section}")
            break
    if 'IClamp' in list( sec.psection()['point_processes'].keys() ):
        temp_dict = sec.psection()['point_processes']
        del temp_dict['IClamp']
        
    sec = list(secList)[idx_section]
    
    # electrode at the soma
    elStim = h.IClamp(sec(0.5))
    elStim.delay = delay
    elStim.dur = duration
    elStim.amp = amplitude
    
    return elStim

def sec_str_2_plot(section, secList):
    
    tempSecList = [ ]
    for sec in list(secList):
        tempSecList.append(str(sec.name()))
    secIdx = tempSecList.index(section)  # use the sero in case there are other numbers in brackets, e.g. for spines
    sec2Plot = list(secList)[secIdx]
    
    return secIdx, sec2Plot

def select_plot_compartments(secList, section, spines_neckList, spines_headList):
    
    secIdx, _ = sec_str_2_plot(section, secList)
    
    # select voltage at particular section
    sm_seg = list(secList)[0](0.5)
    dend_seg = list(secList)[secIdx](0.5)
    spine_neck_seg = spines_neckList[-1](0.5)
    spine_head_seg = spines_headList[-1](0.5)
    
        
    return sm_seg, dend_seg, spine_neck_seg, spine_head_seg
 
def prepare_recording_vecs(secList, section, spines_neckList, spines_headList):
    
    # Initialise the holding cvetros for time and membrane potential from the sections of interest
    time, soma_v, dend_v, spine_neck_v, spine_head_v    = [ h.Vector() for _ in range(5) ]

    sm_seg, dend_seg, spine_neck_seg, spine_head_seg = select_plot_compartments(secList, section, spines_neckList, spines_headList)
    
    # record the result of the simulation
    soma_v.record( sm_seg._ref_v )  # Record voltage at the middle of the soma
    dend_v.record( dend_seg._ref_v)  # Record voltage at the middle of the dendrite
    spine_neck_v.record(spine_neck_seg._ref_v) # Record voltage at the neck of the spine
    spine_head_v.record(spine_head_seg._ref_v) # Record voltage at the head of the spine
    time.record(h._ref_t)  # Record time
    
        
    return time, soma_v, dend_v, spine_neck_v, spine_head_v, sm_seg, dend_seg, spine_neck_seg, spine_head_seg 

    
# definition that converts the hoc vectors into np.arrays for plotting
def hvec_to_numpy(*args):
    return [ np.array( vec.to_python() ) for vec in args] 


def run_simulation(secList, section, spines_neckList, spines_headList):
    
    time, soma_v, dend_v, spine_neck_v, spine_head_v, sm_seg, dend_seg, spine_neck_seg, spine_head_seg  = prepare_recording_vecs(secList, section, spines_neckList, spines_headList) 
    
       
    print(f"Starting simulation: Current time: {h.t} ms")
    h.run()
    print(f"End of simulation: Current time: {h.t} ms")
        
    t_vec, soma_V_vec, dend_V_vec, spine_neck_V_vec, spine_head_V_vec = hvec_to_numpy( time, soma_v, dend_v, spine_neck_v, spine_head_v )
    
    # Plot the results
    plot_results(t_vec, soma_V_vec, dend_V_vec, spine_neck_V_vec, spine_head_V_vec, sm_seg, dend_seg, spine_neck_seg, spine_head_seg)

    return t_vec, soma_V_vec, dend_V_vec, spine_neck_V_vec, spine_head_V_vec


def plot_results( t_vec, soma_V_vec, dend_V_vec, spine_neck_V_vec, spine_head_V_vec, sm_seg, dend_seg, spine_neck_seg, spine_head_seg):

    # Plot the results: membrane potentials of soma and dendrite
    plt.figure(figsize=(8, 6))
    
    plt.plot(t_vec, soma_V_vec,  label= 'Soma Voltage (mV)', color='k' )
    plt.plot(t_vec, dend_V_vec,  label= f'Dendrite {str(dend_seg)[:8]} Voltage (mV)', color='b' )
    plt.plot(t_vec, spine_neck_V_vec, label= f'Spine {str(spine_neck_seg) } Voltage (mV)', color='g' )
    plt.plot(t_vec, spine_head_V_vec, label= f'Spine {str(spine_head_seg) } Voltage (mV)', color='r' )
    
    # plt.legend()
    plt.xlabel('Time (ms)')
    plt.ylabel('Membrane Potential (mV)')
    plt.ylim([-70, -60])
    plt.grid(True)
    
    return plt.show()  
    

#%% load the morphology with mechanisms and parameters then add spines and AMPA and NMDA receptor:

# Initialize the class with the morphology file
cortical_cell = Morph_HumanNeuron(morph_file_path)

# create the section List
secList = cortical_cell.create_secList()

# uninsert mechanisms
cortical_cell.uninsert_mechanism()

# insert mechanisms
cortical_cell.insert_mechanism(densityMech= [ ], section = 'soma')
cortical_cell.insert_mechanism(densityMech= [ ], section = 'axon')

# syntax where a empty List for mechanisms is provided inserts passive properties to the section
cortical_cell.insert_mechanism(densityMech = [  ], section = 'dend') 
cortical_cell.insert_mechanism(densityMech = [  ] , section = 'apic')

# plot NEURON style morphology
theShape = cortical_cell.plot_morphology(secList)

#%% AUXILIARY: diagnostics on mechanism insertion 
   
# # check what has been inserted in the sections!
# if list( secList[0].psection()['density_mechs'].keys())[0] =='hh':
#     print(f' {secList[0]} has hh')
# else:
#     print(f' {secList[0]} has {secList[0].psection()["density_mechs"].keys() }  ')
    
# if list( secList[1].psection()['density_mechs'].keys() )[0] =='hh':
#     print(f'{secList[1]} has hh')
# else:
#     print(f' {secList[1]} has {secList[1].psection()["density_mechs"].keys() }  ')
    
# if list( secList[2].psection()['density_mechs'].keys())[0] =='hh':
#     print(f' {secList[2]} has hh')
# else:
#     print(f' {secList[2]} has {secList[2].psection()["density_mechs"].keys() }  ')
    
# if list( secList[-1].psection()['density_mechs'].keys())[0] =='hh':
#     print(f'{secList[-1]} has hh')
# else:
#     print(f' {secList[-1]} has {secList[-1].psection()["density_mechs"].keys() }  ')
    

# for sec in h.allsec():
#     if h.ismembrane('hh', sec=sec):
#         print(sec)   
        
#%% AUXILIARY: diagnostics on spines and receptor insertion

# print(spinesList)

# for sec in h.allsec():
#     new_secList.append(sec)

# # prin the proeprties of the section
# h.psection(sec=secList[0])
# h.psection(sec=new_secList[-1])

# # check that the mechsnioms have been insert
# # NB here we can access a lot of parameters and their values

# nmdaList = [ ]
# for sec in spinesList:
#     sec.psection()['point_processes']
    
#     if 'NMDA' in list(sec.psection()['point_processes'].keys()):
#         print(f'in section {sec} the NMDA mechanism is PRESENT')
#         nmdaList.append(1)
#     else:
#         nmdaList.append(0)

#%% Run All Individually

spines_neckList, spines_headList = [ [ ] for _ in range(2) ] # spines list
synStimList = [ ] # different SynStim objects
synList = [ ] # receptor list
netConList = [ ] # NetCOn object List
hocL = h.List() # Create a List object to store references to spine sections

# Function to initialize the simulation
v_init = -75; tstop = 100; steps_per_ms = 50; celcius = 37; section = 'soma[0]'
initialise_simulation(v_init, tstop, steps_per_ms, celcius)

# create IClamp
section = 'soma[0]'; delay = 20; duration = 10; amplitude = 2;
# stim_IClamp = create_stimIclamp(secList, delay=delay, duration=duration, amplitude=amplitude, section=section)

# delete spines  (dont call the output new_secList)
secList = cortical_cell.delete_spines()

#------- single section spines anmd receptors----------

# select randomly or choose a particular section to plot
RandSection = np.random.randint(0, len(list(secList)))
num_spines = 120; section = str(list(secList)[RandSection]); syn_start = 20; syn_interval = 50; syn_noise= 0; syn_number = 1
section = 'apic[48]'
hocL, synList, netConList, spines_neckList, spines_headList = cortical_cell.add_single_spines_AND_synapses(num_spines, section, syn_start, syn_interval, syn_noise, syn_number )

# select the compartments to plot
sm_seg, dend_seg, spine_neck_seg, spine_head_seg = select_plot_compartments(secList, section, spines_neckList, spines_headList)

# prepare recording vectors
time, soma_v, dend_v, spine_neck_v, spine_head_v, sm_seg, dend_seg, spine_neck_seg, spine_head_seg = prepare_recording_vecs(secList, section, spines_neckList, spines_headList)

# run simulations (python code and plotting)
t_vec, soma_V_vec, dend_V_vec, spine_neck_V_vec, spine_head_V_vec = run_simulation(secList, section, spines_neckList, spines_headList)

# plot morphology of neuron in python and higlight the dentirite that one works wit

theShape = h.Shape()    
theShape.show(False)
_, sec2Plot = sec_str_2_plot(section, secList)
theShape.color(4, sec = sec2Plot)



#%% AUXILIARY: check children connections

# list the conncetions to the soma
# somaConnected = secList[0].children()
# print(somaConnected)

# # Loop and disconncet the soma from any other structures.,...
# for sec in h.allsec():
#     if 'dend' in sec.name():  # Identify dendrites
#         sec.disconnect()  # Disconnect the dendrite from its parent (the soma)

# # check theat it has done it correctly
# h.topology()