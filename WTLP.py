import numpy as np
import time
import csv
import random
import kaleido
import os
import math
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots
import pandas as pd
import shutil
from scipy.signal import argrelextrema
from scipy.ndimage import gaussian_filter
import warnings
import math
from scipy.ndimage import convolve

warnings.filterwarnings("ignore")
def num_sim(n1, n2):
  """ calculates a similarity score between 2 numbers """
  return 1 - abs(n1 - n2) / (n1 + n2+1e-15)


def fit_dwell_cdf(times, label="", n_tail_fraction=0.95):
    """
    Ajuste CDF empírica en escala log para tiempos de permanencia RTN.
    
    times             : array de dwell times (emisión o captura por separado)
    n_tail_fraction   : fracción de puntos a usar en el ajuste 
                        (excluye la cola derecha con pocos eventos, donde el 
                        estimador de S(t) es ruidoso)
    """
    t = np.sort(times)
    n = len(t)
    
    # CDF empírica con corrección de Hazen (evita ln(0))
    k      = np.arange(1, n + 1)
    S      = (n - k + 0.5) / n          # supervivencia empírica
    lnS    = np.log(S)
    
    # Usar solo la fracción estable de la cola
    n_fit  = max(3, int(n * n_tail_fraction))
    t_fit  = t[:n_fit]
    lnS_fit = lnS[:n_fit]
    
    # Regresión lineal forzada por el origen: lnS = m * t
    m_hat  = np.dot(t_fit, lnS_fit) / np.dot(t_fit, t_fit)
    tau    = -1.0 / m_hat
    
    # R² en escala log (bondad del ajuste → 1 = exponencial pura)
    lnS_pred = m_hat * t_fit
    ss_res   = np.sum((lnS_fit - lnS_pred) ** 2)
    ss_tot   = np.sum((lnS_fit - lnS_fit.mean()) ** 2)
    r2       = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0
    
    #print(f"{label}  τ = {tau:.4f}   R² = {r2:.4f}   n = {n}")
    
    return tau, r2, t, lnS, m_hat


def top_n_valores_e_indices(arr, n):
    flat = arr.flatten()
    unicos = np.unique(flat)
    ordenados = np.sort(unicos)[::-1]
    top_valores = ordenados[:n]
    
    resultado = []
    value_list=[]
    
    for val in top_valores:
        # coger solo la primera ocurrencia
        pos = np.argwhere(arr == val)[0]
        resultado.append(tuple(pos))
        value_list.append(val)
    
    return value_list,resultado



def WTLP_test(csv_path):
    trace=pd.read_csv(csv_path)

    trace_a=np.array(trace["mag_trace"])
    gt_shift=np.fromstring(trace["shift"][0][1:-2],dtype=float,sep=',')
    #print(gt_shift)
    gt_taue=np.fromstring(trace["taue"][0][1:-2],dtype=float,sep=',')
    gt_tauc=np.fromstring(trace["tauc"][0][1:-2],dtype=float,sep=',')
    std_n=trace["std_n"][0]
    data_array = np.array(trace_a,dtype=float)
    #print(std_n)
    npoints=len(data_array)


    vmax=np.max(data_array)*1.1
    vmin=np.min(data_array)*1.1

    if(vmax!=vmin):

        res=(vmax-vmin)/1e3

        data_array=np.trunc(data_array / res)*res

        if(vmax<0):
            vmax=np.max(data_array)*0.9
        else:
            vmax=np.max(data_array)*1.1
        if(vmin<0):
            vmin=np.min(data_array)*1.1
        else:
            vmin=np.min(data_array)*0.9

        #print(vmax)
        #print(vmin)

        # --- Parámetros (igual que antes) ---
        ncoords = math.ceil(((vmax - vmin) / res))
        if(std_n==0):
            std_n=(vmax-vmin)/100
        alp   = std_n / res

        ngauss = ncoords*2
        if ngauss % 2 == 0:
            ngauss += 1

        xymid = ngauss // 2


        #print(alp)

        # --- Kernel gaussiano (igual que antes) ---
        y, x = np.ogrid[:ngauss, :ngauss]
        gauss = np.exp(-((x - xymid)**2 + (y - xymid)**2) / (2 * alp**2))
        gauss /= gauss.max()
        gauss[gauss < 0.001] = 0

        # --- PASO 1: índices vectorizados (sin bucle) ---
        vi  = data_array[1:-1]          # V(i)
        vi1 = data_array[2:]            # V(i+1)

        ind1 = np.round((vi  - vmin) / res).astype(int)
        ind2 = np.round((vi1 - vmin) / res).astype(int)

        # Filtrar índices fuera de rango (por seguridad)
        mask = (ind1 >= 0) & (ind1 <= ncoords) & \
            (ind2 >= 0) & (ind2 <= ncoords)
        ind1, ind2 = ind1[mask], ind2[mask]

        # --- PASO 2: histograma 2D de transiciones ---
        blank = np.zeros((ncoords + 1, ncoords + 1), dtype=float)
        np.add.at(blank, (ind1, ind2), 1.0)

        # --- PASO 3: una sola convolución con el kernel ---
        blank = gaussian_filter(blank, sigma=alp, mode='constant', cval=0.0)

        blank=blank/(np.max(blank))
        #fig.show()
        x_indices = np.arange(blank.shape[1])* res + vmin
        y_indices = np.arange(blank.shape[0])* res + vmin


        peaks = np.zeros([ncoords+1,2], dtype=float)
        #print(peaks.shape)

        for i in range(ncoords+1):
            peaks[i,0] = blank[i, i]

        peaks[peaks<1e-5*np.max(peaks)]=0


        loc_max=argrelextrema(peaks[:,0], np.greater)

        peak_data=x_indices[loc_max]
        if(0.0 not in peak_data):
            peak_data=np.append(peak_data,0.0)


        data_clean=np.zeros(len(data_array))
        i_shift=np.zeros(len(data_array))
        shift=[]

        for value in range(len(data_array)):
            data_clean[value]=peak_data[np.argmin(np.abs(peak_data-data_array[value]))]
            if(value>0):
                i_shift[value]=data_clean[value]-data_clean[value-1]
                if((i_shift[value] != np.float64(0)) and (np.abs(i_shift[value]) not in shift)):
                    shift.append(np.abs(i_shift[value]))


        
        shift=np.float32(shift)
        rel_table=np.zeros((len(shift),2))
        rel=0
        for i in range(len(shift)):
            for j in range(len(shift)):
                if(i!=j):
                    if((num_sim(shift[i],shift[j])>=0.99)):
                        if(shift[i] not in rel_table):
                            rel_table[rel,0]=shift[i]
                            rel_table[rel,1]=shift[j]
                            shift[i]=0
                            rel+=1
                        
        shift = [i for i in shift if i != 0]            
        

        for value in range(len(shift)):
            if(rel_table[value,0]!=0):
                for i in range(len(i_shift)):
                    if(np.abs(i_shift[i])==rel_table[value,0]):
                        i_shift[i]=rel_table[value,1]*(i_shift[i]/np.abs(i_shift[i]))
        te_list = [[] for _ in range(len(shift))]
        tc_list = [[] for _ in range(len(shift))]
        t_last=np.zeros(len(shift))
        

        for value in range(len(data_array)):
            if(i_shift[value]<0):

                try:
                    defect_id=shift.index(np.abs(i_shift[value]))
                except ValueError:
                    defect_id=np.argmin(np.abs(shift-np.abs(i_shift[value])))

                tc_list[defect_id].append(value-t_last[defect_id])
                t_last[defect_id]=value
            elif(i_shift[value]>0):
                try:
                    defect_id=shift.index(np.abs(i_shift[value]))
                except ValueError:
                    defect_id=np.argmin(np.abs(shift-np.abs(i_shift[value])))
                te_list[defect_id].append(value-t_last[defect_id])
                t_last[defect_id]=value


        tauc_array=np.zeros(len(shift))
        taue_array=np.zeros(len(shift))

        for value in range(len(shift)):
            if(len(te_list[value])!=0 and len(tc_list[value])!=0):
                tauc_array[value],r2c,a,b,c=fit_dwell_cdf(np.array(tc_list[value]))
                taue_array[value],r2e,a,b,c=fit_dwell_cdf(np.array(te_list[value]))


        shift = [shift[i] for i in range(len(shift)) if tauc_array[i] != 0]
        tauc_array = [i for i in tauc_array if i != 0]
        taue_array = [i for i in taue_array if i != 0]

        shift_aux=np.array(shift)
        tauc_array_aux=np.array(tauc_array)
        taue_array_aux=np.array(taue_array)

        comparison_array=np.zeros((len(shift),len(gt_shift)))

        for i in range(len(shift)):
            for j in range(len(gt_shift)):
                
                comparison_array[i,j]=num_sim(shift[i],np.array(gt_shift[j]))


        e_n_defects=np.abs(len(gt_shift)-len(shift))
        if(len(shift)>=len(gt_shift)):
            val_array,top_index=top_n_valores_e_indices(comparison_array,len(gt_shift))
            e_shift=0
            e_taue=0
            e_tauc=0
            for i in range(len(gt_shift)):
                e_shift+=np.abs(shift[top_index[i][0]]-gt_shift[top_index[i][1]])/(np.abs(gt_shift[top_index[i][1]]))
                shift[top_index[i][0]]=0
                e_taue+=np.abs(taue_array[top_index[i][0]]-gt_taue[top_index[i][1]])/(np.abs(gt_taue[top_index[i][1]]))
                taue_array[top_index[i][0]]=0
                e_tauc+=np.abs(tauc_array[top_index[i][0]]-gt_tauc[top_index[i][1]])/(np.abs(gt_tauc[top_index[i][1]]))
                tauc_array[top_index[i][0]]=0
            e_shift+=e_n_defects*10
            e_taue+=e_n_defects*10
            e_tauc+=e_n_defects*10
                
        else:
            e_shift=0
            e_taue=0
            e_tauc=0
            val_array,top_index=top_n_valores_e_indices(comparison_array,len(shift))
            for i in range(len(shift)):
                e_shift+=np.abs(shift[top_index[i][0]]-gt_shift[top_index[i][1]])/(np.abs(gt_shift[top_index[i][1]]))
                gt_shift[top_index[i][1]]=0
                e_taue+=np.abs(taue_array[top_index[i][0]]-gt_taue[top_index[i][1]])/(np.abs(gt_taue[top_index[i][1]]))
                gt_taue[top_index[i][1]]=0
                e_tauc+=np.abs(tauc_array[top_index[i][0]]-gt_tauc[top_index[i][1]])/(np.abs(gt_tauc[top_index[i][1]]))
                gt_tauc[top_index[i][1]]=0
            e_shift+=e_n_defects*10
            e_taue+=e_n_defects*10
            e_tauc+=e_n_defects*10
            
        if(e_taue>max(len(shift),len(gt_shift))*10):
            e_taue=max(len(shift),len(gt_shift))*10
        if(e_tauc>max(len(shift),len(gt_shift))*10):
            e_tauc=max(len(shift),len(gt_shift))*10
        if(e_shift>max(len(shift),len(gt_shift))*10):
            e_shift=max(len(shift),len(gt_shift))*10
    else:
        e_n_defects=len(gt_shift)
        e_shift=e_n_defects*10
        e_taue=e_n_defects*10
        e_tauc=e_n_defects*10
        shift_aux=np.nan
        taue_array_aux=np.nan
        tauc_array_aux=np.nan

    return e_shift,e_taue,e_tauc,e_n_defects, shift_aux, taue_array_aux, tauc_array_aux
