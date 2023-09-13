#!/usr/bin/env python
# coding: utf-8

# In[1]:


import uproot
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math
import mplhep as hep
import hist


# In[2]:


#open root file
file1_a = uproot.open("analyzed/trim_0_0/Scurves_July5_0a/scurve-results.root")
file1_b = uproot.open("analyzed/trim_0_0/Scurves_July5_0b/scurve-results.root")
file2_a = uproot.open("analyzed/trim_32_0/Scurves_July5_0a/scurve-results.root")
file2_b =uproot.open("analyzed/trim_32_0/Scurves_July5_0b/scurve-results.root")
file3_a = uproot.open("analyzed/trim_63_0/Scurves_July5_0a/scurve-results.root")
file3_b = uproot.open("analyzed/trim_63_0/Scurves_July5_0b/scurve-results.root")
file4_a = uproot.open("analyzed/trim_32_1/Scurves_July5_0a/scurve-results.root")
file4_b = uproot.open("analyzed/trim_32_1/Scurves_July5_0b/scurve-results.root")
file5_a = uproot.open("analyzed/trim_63_1/Scurves_July5_0a/scurve-results.root")
file5_b = uproot.open("analyzed/trim_63_1/Scurves_July5_0b/scurve-results.root")
#file2.classnames()
#t1 = file1["scurve-results;1"]
#t1.keys()
t1a = file1_a["scurve-results;1"].arrays(["oh", "vfat", "channel", "threshold", "enc"], library = "pd")
t1a["dac"] = 0
t1a["FPGA"] = "a"
t2a = file2_a["scurve-results;1"].arrays(["oh", "vfat", "channel", "threshold", "enc"], library = "pd")
t2a["dac"] = 32
t2a["FPGA"] = "a"
t3a = file3_a["scurve-results;1"].arrays(["oh", "vfat", "channel", "threshold", "enc"], library = "pd")
t3a["dac"] = 63
t3a["FPGA"] = "a"
t4a = file4_a["scurve-results;1"].arrays(["oh", "vfat", "channel", "threshold", "enc"], library = "pd")
t4a["dac"] = -32
t4a["FPGA"] = "a"
t5a = file5_a["scurve-results;1"].arrays(["oh", "vfat", "channel", "threshold", "enc"], library = "pd")
t5a["dac"] = -63
t5a["FPGA"] = "a"
t1b = file1_b["scurve-results;1"].arrays(["oh", "vfat", "channel", "threshold", "enc"], library = "pd")
t1b["dac"] = 0
t1b["FPGA"] = "b"
t2b = file2_b["scurve-results;1"].arrays(["oh", "vfat", "channel", "threshold", "enc"], library = "pd")
t2b["dac"] = 32
t2b["FPGA"] = "b"
t3b = file3_b["scurve-results;1"].arrays(["oh", "vfat", "channel", "threshold", "enc"], library = "pd")
t3b["dac"] = 63
t3b["FPGA"] = "b"
t4b = file4_b["scurve-results;1"].arrays(["oh", "vfat", "channel", "threshold", "enc"], library = "pd")
t4b["dac"] = -32
t4b["FPGA"] = "b"
t5b = file5_b["scurve-results;1"].arrays(["oh", "vfat", "channel", "threshold", "enc"], library = "pd")
t5b["dac"] = -63
t5b["FPGA"] = "b"
all_array = pd.concat([t1a, t1b, t2a, t2b, t3a, t3b, t4a, t4b, t5a, t5b])
df = pd.DataFrame(all_array)
print(df)


# In[3]:


#first I need to calibrate and to obtain parameters of calibration. 
def DAC_calibration(dataframe):
    #fig, axs = plt.subplots(1, figsize=(4, 4))
    #plt.style.use(hep.style.CMS)
    #cut_oh = dataframe["oh"]==oh
    #cut_vfat = dataframe["vfat"] ==vfat
    fpga = dataframe["FPGA"].iloc[0]
    oh = dataframe["oh"].iloc[0]
    vfat = dataframe["vfat"].iloc[0]
    cut_dac = dataframe["dac"] == 0
    cut_fpga = dataframe["FPGA"] == fpga
    cut_oh = dataframe["oh"] == oh
    cut_vfat = dataframe["vfat"] == vfat
    raising_point = dataframe["threshold"] - dataframe["enc"]*(4)
    raising_point_vfat = raising_point[(cut_fpga) & (cut_oh) & (cut_vfat) & (cut_dac)]
    #here I comoute the average raising point for each VFAT (128 channels)
    raising_mean = raising_point_vfat.mean()
    i = dataframe["channel"].iloc[0]
    cut_channel = dataframe["channel"] == i
    df_channel = dataframe[(cut_fpga) & (cut_oh) & (cut_vfat) & (cut_channel)]
    #a = int(i // 16)
    #b = int(i % 16)
    #axs.scatter(df_channel["dac"], df_channel["threshold"])
    #here I fit with a line the five point for the trimming calibration
    p = np.polyfit(df_channel["dac"], df_channel["threshold"], 1)
    #then I compute the chisquared to see if the fit is good or there are problems.
    residuals = ((p[0]*df_channel["dac"] + p[1]) - (df_channel["threshold"]))
    chi_squared_partial = ((p[0]*(df_channel["dac"]) + p[1]) - (df_channel["threshold"]))**2
    chi_squared = np.sum(chi_squared_partial)
    #if the chi_squared is too high, then it's necessary to redo the fit.
    if np.max(residuals) > 0.5:
        chi_sq_lst =[0, 0, 0, 0, 0]
        df_channel["chisquared"] = chi_sq_lst
        #loop over the 5 dac values, to see where is the problem.
        for j in np.unique(df_channel["dac"]): 
            #selecting one point (out of 5) to esclude from the fit
            selection = df_channel["dac"] != j
            #selecting data for the other 4 point.
            dac_selected = df_channel["dac"][selection]
            #print(dac_selected)
            threshold_selected = df_channel["threshold"][selection]
            #print(threshold_selected)
            #plotting and fitting
            #axs.scatter(df_channel["dac"][selection], df_channel["threshold"][selection])
            p = np.polyfit(df_channel["dac"][selection], df_channel["threshold"][selection], 1)
            #computing the chi_squared with the 4 remaining point
            chi_squared_partial = ((p[0]*(dac_selected) + p[1]) - (threshold_selected))**2
            #print(chi_squared_partial)
            chi_squared = np.sum(chi_squared_partial)
            #we want to esclude the point that implies the lowest chi-squared
            df_channel["chisquared"][df_channel["dac"] == j] = chi_squared
        fit_cutting = df_channel["chisquared"].gt(np.min(df_channel["chisquared"]))
        chi_squared = np.min(df_channel["chisquared"])
        p = np.polyfit(df_channel["dac"][fit_cutting], df_channel["threshold"][fit_cutting], 1)
        #residuals = ((p[0]*df_channel["dac"] + p[1]) - (df_channel["threshold"]))
        #chi_squared_partial = ((p[0]*(df_channel["dac"]) + p[1]) - (df_channel["threshold"]))**2
        #chi_squared = np.sum(chi_squared_partial)
        #chi_squared_ch = chisquared2(p, df_channel)
        if math.isnan(p[0]):
            #print("ch: %d" %(i), df_channel[["dac", "threshold"]])
            #it happens when some threshold values are NaN
            p = np.polyfit(df_channel["dac"][~ np.isnan(df_channel["threshold"])], df_channel["threshold"][~ np.isnan(df_channel["threshold"])], 1)
            chi_squared_partial = ((p[0]*df_channel["dac"][~ np.isnan(df_channel["threshold"])] + p[1]) - df_channel["threshold"][~ np.isnan(df_channel["threshold"])])**2
            #print(chi_squared_partial)
            chi_squared = np.sum(chi_squared_partial)
    #axs.plot(df_channel["dac"], p[1] + (df_channel["dac"])*p[0], 'r-', label='fit')
    #residuals = (p[0]*df_channel["dac"] + p[1]) - (df_channel["threshold"])
    #cut_resid = (np.abs(residuals) < 0.4 ) 
    #axs.plot(df_channel["dac"], p[1] + (df_channel["dac"])*p[0], 'r-', label='fit')
    #axs.set_xlabel("DAC", fontsize = 30)
    #axs.set_ylabel("Threshold[fC]", fontsize = 30)
    #chi_squared_ch = chisquared2(p, df_channel, sel=)
    df_ch_dac = dataframe[(cut_fpga) & (cut_oh) & (cut_vfat) & (cut_channel) & (cut_dac)]
    #df_ch_dac["4threshold"][df_ch_dac["threshold"].isna()] = p[1]
    #print(raising_point_vfat[cut_channel])
    trimming_analog = ( df_ch_dac["threshold"] - df_ch_dac["enc"]*(4)) - raising_mean
    #print(trimming_analog)
    trimming_dig = (trimming_analog)/p[0]
    #axs.text(0.4, 0.85,"$p_1$ = %.2f, $p_0$ = %.2f \n $\u03c7^2$ = %.4f \n DAC = %.1f" %(p[0], p[1], chi_squared, trimming_dig), bbox=dict(facecolor='white', alpha=0.5), transform=axs.transAxes)
    #axs.set_title("ch. %d: trimming DAC calibration" %(i))
    #axs.tick_params(axis='both', which='major', labelsize=30)
    #print(chi_squared())
    #fig.savefig("fpga_%s_oh_%d_vfat_%d_trimming.png" %(fpga,oh, vfat))
    #plt.close(fig)
    return p

#df.groupby(["oh", "vfat"]).apply(DAC_calibration)
trimming_df = df.groupby(["FPGA","oh", "vfat", "channel"], as_index = False).apply(DAC_calibration)
#print(trimming_df)


# In[4]:


print(trimming_df)


# In[11]:


def compute_trimming_value(dataframe):
    fpga = dataframe["FPGA"].iloc[0]
    oh = dataframe["oh"].iloc[0]
    vfat = dataframe["vfat"].iloc[0]
    channel =dataframe["channel"].iloc[0]
    cut_fpga = dataframe["FPGA"] == fpga
    cut_oh = dataframe["oh"] == oh
    cut_vfat = dataframe["vfat"] == vfat
    cut_channel = dataframe["channel"] == channel
    average_df = t1.groupby(["FPGA", "oh", "vfat"], as_index = False)["raising_point"].mean()
    cut_fpga1 = average_df["FPGA"] == fpga
    cut_oh1 = average_df["oh"] == oh
    cut_vfat1 = average_df["vfat"] == vfat
    sel_channel = (cut_fpga) & (cut_oh) & (cut_vfat) & (cut_channel)
    sel_vfat = cut_fpga1 & cut_oh1 & cut_vfat1
    #print(raising_point[sel_channel])
    #print(average_df[sel_vfat]["raising_point"])
    #print(dataframe["raising_point"][sel_channel] )
    #print(average_df["raising_point"][sel_vfat])
    trimming_analog = dataframe["raising_point"][sel_channel].iloc[0] - average_df["raising_point"][sel_vfat].iloc[0]
    #print(trimming_analog)
    #print(float(trimming_analog[0]))
    #print(dataframe[sel_channel]["p"][0][0])
    m = (dataframe["p"][sel_channel].iloc[0])
    #print(trimming_analog.iloc[0])
    #print(m[0])
    trimming_dig = trimming_analog / (m[0])
    #print(trimming_dig)
    return trimming_dig

#df["raising_point"] = df["threshold"] - 4*df["enc"]
#average_df = df[df["dac"]==0].groupby(["FPGA", "oh", "vfat"], as_index = False)["raising_point"].mean()
#print(average_df)
p = trimming_df
t1 = pd.concat([t1a, t1b])
t1["p"] = p
t1["raising_point"] = t1["threshold"] - 4*t1["enc"]
#print(t1_def)
#print(t1["threshold"])
print(t1)
trimming_df1 = t1.groupby(["FPGA", "oh", "vfat", "channel"], as_index = False).apply(compute_trimming_value)


# In[12]:


print(trimming_df1)


# In[14]:


trimming_df1.columns = ["FPGA", "oh", "vfat", "channel", "trimming value"]
print(trimming_df1)


# In[24]:


trimming_list = trimming_df1["trimming value"].values.tolist()
print(trimming_list)
#t1["trimming"] = (trimming_list)
#print(t1)
for index, value in enumerate(trimming_list):
    if abs(value) > 63.:
        trimming_list[index] = 63 * np.sign(value)

trimming_polarization_list = []
trimming_amplitude_list = []
for element in trimming_list: 
    if not math.isnan(element): 
        trimming_amplitude_list.append(int(abs(element)))
        if np.sign(element) >= 0: 
            trimming_polarization_list.append(0)
        else:
            trimming_polarization_list.append(1)
    else:
        trimming_amplitude_list.append(0)
        trimming_polarization_list.append(0)


# In[25]:


print(trimming_polarization_list)
print(trimming_amplitude_list)


# In[26]:


trimming_df1["polarization"] = trimming_polarization_list
trimming_df1["amplitude"] = trimming_amplitude_list
print(trimming_df1)


# In[29]:


del trimming_df1["trimming value"]


# In[30]:


trimming_df1.to_csv("trimming_july05.csv")


# In[ ]:




