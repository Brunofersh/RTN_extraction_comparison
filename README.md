# RTN Level Extractor

These are a series of python files that generate synthetic and test both synthetic and real RTN traces.

## Synthetic RTN generator: RTN_gen.py

This is a recicled (but modified) deterministic synthetic RTN generator from the servicore repository, all of its generation functions are defined in the **funct.py** file. It offers many options that the user can set to generate different kinds of synthetic RTN traces:

### Trace Parameters

- **test_name** : simply the name of the test, the folder in which the RTN traces will be stored will be called like this.
- **trace_mode** : it can be set to "time_dependent" or "discrete events":
    - **time_dependent** : the trace will have a length determined by the user by the parameter **trace_length**.
    - **discrete_events** : the trace will have a length so that every RTN defect produces a number of events **n_events**, so the length is not fixed.
- **n_traces** : the number of traces generated for each data point of the sweep.

### Current Shift Parameters

- **shift_mode** : it can be set to "equal", "sampled" or "variation":
    - **equal** : all of the current shifts of the defects are equal to 1.
    - **sampled** : the current shifts will be sampled from a uniform distribution that goes from 0 to 1.
    - **variation** : samples the current shifts from a uniform distribution but the minimum will be 1-(0.5 · variation) and the maximum will be 1+(0.5 · variation) with **variation** being a parameter that the user can set.

### Dwell Times Parameters

- **tau_mode** : it can be set to "equal" or "sampled":
    - **equal** : all of the dwell times of the defects are equal, though the value is first sampled from a log-uniform distribution that goes from **min_tau** to **max_tau**.
    - **sampled** : the dwell times will be sampled from a log-uniform distribution that goes from **min_tau** to **max_tau**.

### Defect Number Parameters
- **min_defects**, **max_defects** : both can be set to an integer, it will determine the possible range of defects present in RTN traces. If one of them is 0, they will be set to **min_defect_sweep** or **max_defect_sweep**, respectively.
- **min_defect_sweep**, **max_defect_sweep** : both can be set to an integer. Normally, the RTN generation runs have the same parameters but for distinct number of defects present, because of this, these parameters are added.

### Noise Parameters
- **noise_mode** : can be set to "sweep" or "set_noise":
    - **sweep** : it notifies the code that you want to do an SNR sweep (more on that later).
    - **set_noise** : the traces will have signal to noise ratio set by the **SNR** parameter.

### Sweeps

Three main sweeps can be done: n_event sweep, variation sweep and SNR sweep. To change between sweeps is very simple but it is a manual process:

    for SNR in range(-20,30,10): #<-- Change the sweep parameter here

You can change the sweep parameter to n_event or variation (examples in the code).


## RTN Extraction Methods Files

### WTLP.py

This file implements the Weighted Time Lag Plot method, specifically, the function **WTLP_test** is used. This function receives the csv_path of the RTN trace, applies the method and returns the error metrics (only prepared for synthetic traces).

### kmeans.py
This file implements the K-means method, specifically, the function **kmeans_test** is used. This function receives the csv_path and optionally the maximum number of clusters of the RTN trace, applies the method and returns the error metrics. Has a "real trace" mode prepared for processing real traces, in that mode it only returns the estimated parameters.


### FHMM.py
This file implements the Factorial Hidden Markov Models method, specifically, the function **FHMM_test** is used. This function receives the csv_path of the RTN trace and some FHMM initialization parameters, then applies the method and returns the error metrics. Has a "real trace" mode prepared for processing real traces, in that mode it only returns the estimated parameters.

## Test file

Both synthetic and real traces can be tested using the "test.py" code, the mode can be changed from "synthetic" to "real". 