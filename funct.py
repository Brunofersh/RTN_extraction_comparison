import numpy as np
import numpy as np
from scipy.optimize import fsolve
from scipy.optimize import root_scalar
from scipy.stats import poisson
import random
import math 
from itertools import combinations
import hashlib
import json
import pandas as pd

def generate_defect_and_tmcf_arrays(n_devices, trace_length,max_defects,min_defects,shift_mode,variation,tau_mode,min_tau,max_tau):
    #lambda_d = np.random.rand(0,10)
    #defect_array = np.random.poisson(lambda_d, n_devices)
    #defect_array = [defect_array] if isinstance(defect_array, (int, float)) else defect_array
    defect_array = np.empty(n_devices, dtype=object)
    tauc_matrix = np.empty(n_devices, dtype=object)
    taue_matrix = np.empty(n_devices, dtype=object)
    shift_matrix = np.empty(n_devices, dtype=object)
    for i in range(n_devices):
        #lambda_d=4*np.random.rand(1)
        defect_array[i] = random.randint(min_defects,max_defects)
        #tauc = np.random.uniform(low_limit_tauc, up_limit_tauc, defect_array[i]) # type: ignore
        #taue = np.random.uniform(low_limit_taue, up_limit_taue, defect_array[i]) # type: ignore
        
        #To sample in the log space
        if(tau_mode=="equal"):
            tau=10 ** np.random.uniform(np.log10(min_tau), np.log10(max_tau), 1)
            tauc=tau*np.ones(defect_array[i])
            taue=tau*np.ones(defect_array[i])
        elif(tau_mode=="sampled"):
            tauc = 10 ** np.random.uniform(np.log10(min_tau), np.log10(max_tau), defect_array[i]) # type: ignore
            taue = 10 ** np.random.uniform(np.log10(min_tau), np.log10(max_tau), defect_array[i]) # type: ignore
        if(shift_mode=="sampled"):
            shift = np.random.uniform(0, 1, defect_array[i]) # type: ignore
        elif(shift_mode=="variation"):
            shift = np.random.uniform(1-variation/2, 1+variation/2, defect_array[i])
        else:
            shift=np.ones(defect_array[i])

        taue_matrix[i], tauc_matrix[i], shift_matrix[i] = taue.tolist(), tauc.tolist(), shift.tolist()
    
    return defect_array, tauc_matrix, taue_matrix, shift_matrix



def calculate_state_t0(u_te, u_tc):
    """
    Calculates and returns a state value (0 or 1) or an array/list of state values.

    Args:
        u_te: A numerical value (float), a list, or a NumPy array.
        u_tc: A numerical value (float), a list, or a NumPy array with the same shape as u_te.

    Returns:
        An integer (0 or 1) if u_te and u_tc are floats, or a list/NumPy array of 0s and 1s.

    Raises:
        ValueError: If u_te and u_tc have incompatible shapes.
    """

    try:
        # Handle scalar (single) values
        if isinstance(u_te, (int, float)) and isinstance(u_tc, (int, float)):
            if u_te + u_tc == 0:
                return 0
            else:
                probability = u_te / (u_te + u_tc)
                return 0 if random.random() < probability else 1

        # Handle lists or NumPy arrays
        u_te = np.asarray(u_te)  # Ensure both are NumPy arrays
        u_tc = np.asarray(u_tc)

        if u_te.shape != u_tc.shape:
            raise ValueError("u_te and u_tc must be of the same type and shape.")

        zero_condition = u_te + u_tc == 0
        probabilities = np.where(zero_condition, 0, u_te / (u_te + u_tc))  
        states = np.where(np.random.random(size=probabilities.shape) < probabilities, 0, 1)

        return states 

    except Exception as e:
        print(f"Error calculating state: {e}")
        return None  # Or an array/list of appropriate error values (e.g., NaN) 




def genbits_multi(trace_length, Te_array, Tc_array, shift_array, trace_mode,n_events,tsampl=1e-6):
    """
    Generate bitstream without sampling (multi defect).

    Parameters:
    nbits (int): Number of bits to generate.
    tmcf (float): Time interval for checking the state.
    Te_array (list or ndarray): Emission time constants.
    Tc_array (list or ndarray): Capture time constants.
    tsampl (float): Time interval for sampling. Defaults to 1e-6.

    Returns:
    numpy.ndarray: Bitstream generated with sampling (multi defect).
    """ 

    if isinstance(Te_array, (int, list)) == isinstance(Tc_array, (int, list)): 
        Te_array = np.atleast_1d(np.array(Te_array)) # Fixed for single element numpy arrays
        Tc_array = np.atleast_1d(np.array(Tc_array)) # Fixed for single element numpy arrays

    # Initialize variables
    ndefts = Te_array.size  
    time = np.zeros(ndefts, dtype=float) 
    mag = np.zeros(trace_length, dtype=float) 
    state = np.empty(ndefts)

    


    # Calculate initial states
    for d in range(ndefts):
        state[d] = calculate_state_t0(Te_array[d], Tc_array[d])  
    

    for d in range(ndefts):
        num_e=0
        num_c=0
        event_flag=0
        if(trace_mode=="discrete_events"):
            while (event_flag==0):
                if(num_e>=n_events and num_c>=n_events):
                    event_flag=1
                if(event_flag==1):
                    continue
                if state[d] == 1:
                    tc = int(np.random.exponential(Tc_array[d])) 
                    time[d] += tc

                    
                    if(tc!=0):
                        if(time[d]>len(mag)):
                            mag=np.append(mag,np.zeros(int(time[d]-len(mag))+2))
                        mag[int(time[d])-1]-=shift_array[d]
                        state[d] = 0 # Switch to emmited
                        num_e+=1
                else:
                    get_time = time[d]  # Get the current event time 
                    te = int(np.random.exponential(Te_array[d]))  
                    time[d] += te
                    if(te!=0):
                        if(time[d]>len(mag)):
                            mag=np.append(mag,np.zeros(int(time[d]-len(mag))+2))
                        mag[int(time[d])-1]+=shift_array[d]
                        state[d] = 1 # Switch to captured
                        num_c+=1
        else:
            while (time[d] < trace_length):
                if state[d] == 1:
                    tc = int(np.random.exponential(Tc_array[d]))  
                    time[d] += tc
                    if(time[d]<trace_length):
                        mag[int(time[d])]-=shift_array[d]
                    state[d] = 0 # Switch to emmited
                else:
                    get_time = time[d]  # Get the current event time 
                    te = int(np.random.exponential(Te_array[d]))  
                    time[d] += te
                    if(time[d]<trace_length):
                        mag[int(time[d])]+=shift_array[d]
                    state[d] = 1 # Switch to captured
    if(trace_mode=="discrete_events"):
        mag=np.delete(mag,range(int(np.max(time[d]))+1,trace_length))    
    return mag