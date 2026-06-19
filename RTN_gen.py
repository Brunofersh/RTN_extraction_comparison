import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import json

from pathlib import Path

import funct as fn

# ==================================================
# Generation Parameters
# ==================================================

#----Trace Parameters----#
test_name=f"large_noise_test_d"
trace_mode="time_dependent" # "discrete_events" or "time_dependent"
n_events=5
trace_length=10000
n_traces=200
#----Current Shift Parameters----#
shift_mode="sampled" # "equal" or "sampled" or "variation"
variation=0.05
#----Dwell Times Parameters----#
tau_mode="sampled" # "equal" or "sampled" or "log_normal"
min_tau=1
max_tau=10000
#----Defect Number Parameters----#
min_defects=0
max_defects=0
min_defect_sweep=1
max_defect_sweep=5
#----Noise Parameters----#
noise_mode="sweep" # "set_noise" or "sweep"
SNR=20 #dB
#----Plot RTN Parameters----#
plot_traces=False

for defect in range(min_defect_sweep,max_defect_sweep+1):
    test_name=test_name+defect
    if(min_defects==0):
        min_defects=defect
    if(max_defects==0):
        max_defects=defect

    #----Sweep----#
    #for n_event in range(1,n_events+1):
    #for variation in np.arange(0.05,1.05,0.05):
    for SNR in range(-20,30,10): #<-- Change the sweep parameter here
        trace_length=int(trace_length)
        defect_array, tauc_matrix, taue_matrix, shift_matrix = fn.generate_defect_and_tmcf_arrays(n_traces, trace_length,max_defects,min_defects,shift_mode,variation,tau_mode,min_tau,max_tau)


        all_data = []  

        all_data = []

        for k in range(n_traces):

            # ---- generate signal ----
            mag_trace_clean = np.cumsum(fn.genbits_multi(trace_length,taue_matrix[k],tauc_matrix[k],shift_matrix[k],trace_mode,n_events))

            # ---- add noise based on SNR ----
            Ps = np.mean(mag_trace_clean**2)
            Pn = Ps / (10**(SNR/10))

            noise = np.random.normal(
                loc=0,
                scale=np.sqrt(Pn),
                size=len(mag_trace_clean)
            )

            mag_trace = mag_trace_clean + noise

            if(plot_traces):
                fig = go.Figure()

                fig.add_trace(go.Scatter(
                    x=np.arange(1, len(mag_trace) + 1),
                    y=mag_trace,
                    mode="lines",
                    name=f"Trace {k}"
                ))

            
            df_trace = pd.DataFrame({
                "sample": np.arange(len(mag_trace)),
                "mag_trace": mag_trace,
                "mag_trace_clean": mag_trace_clean,
            })


            df_trace["trace_id"] = k
            df_trace["trace_length"] = len(mag_trace)

            df_trace["taue"] = json.dumps(taue_matrix[k])
            df_trace["tauc"] = json.dumps(tauc_matrix[k])
            df_trace["shift"] = json.dumps(shift_matrix[k])
            df_trace["SNR"] = SNR
            df_trace["std_n"] = np.sqrt(Pn)


            if(trace_mode=="discrete_events" and shift_mode!="variation"):
                save_folder=f"data\\{test_name}\\events_{n_events}"
                save_folder_plots=f"plots\\{test_name}\\events_{n_events}"
            elif(shift_mode=="variation"):
                save_folder=f"data\\{test_name}\\variation_{variation}"
                save_folder_plots=f"plots\\{test_name}\\variation_{variation}"
            
            else:
                save_folder=f"data\\{test_name}"
                save_folder_plots=f"plots\\{test_name}"
            
            if(noise_mode=="sweep"):
                save_folder=f"data\\{test_name}\\SNR_{SNR}"
                save_folder_plots=f"plots\\{test_name}\\SNR_{SNR}"

            if(tau_mode=="log_normal"):
                save_folder=f"data\\{test_name}\\trace_length_{trace_length}"
                save_folder_plots=f"plots\\{test_name}\\trace_length_{trace_length}"


            # ---- save final CSV ----
            try:
                os.makedirs(save_folder)
            except FileExistsError:
                pass
            try:
                os.makedirs(save_folder_plots)
            except FileExistsError:
                pass
            df_trace.to_csv(f"{save_folder}\\rtn_params_{k}.csv", index=False)
            if(plot_traces):
                fig.write_html(f"{save_folder_plots}\\rtn_signal_{k}.html")