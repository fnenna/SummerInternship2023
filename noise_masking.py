#!/usr/bin/env python
# coding: utf-8

# In[1]:


import uproot
import awkward as ak
import hist
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import scipy.stats as sts
import matplotlib.mlab as mlab
from scipy.optimize import curve_fit
from scipy.special import factorial
import mplhep as hep
import argparse
plt.style.use(hep.style.ROOT)


# In[ ]:

def main(): 
	parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='Find the noisiest channel and mask them',
                    epilog='Text at the bottom of help')
	parser.add_argument('filename')           # positional argument
	#parser.add_argument('-c', '--count')      # option that takes a value
	args = parser.parse_args()
	#print(args.filename, args.count, args.verbose)
	run_file = args.filename
	root_file = "/eos/user/f/fnenna/" + run_file
	file = uproot.open(root_file)
	file.classnames()
	tree = file["outputtree"]
	t = tree.arrays(["OH", "VFAT", "CH", "digiStrip"], library = "pd")
	#t.groupby(["OH", "VFAT"]).apply(histo_hitsVSchannel)
	mask = t.groupby(["OH", "VFAT"]).apply(RMSanalysis).reset_index()
	print(mask)
	new_mask = mask.rename(columns = {0: "channel", "OH": "oh", "VFAT": "vfat"})
	#print(new_mask)
	new_mask1 = new_mask.drop(columns = "level_2")
	new_mask1["slot"] = new_mask1["vfat"].apply(lambda x: x**0)
	column_to_move = new_mask1.pop("slot")
	new_mask1.insert(0, "slot", column_to_move)
	new_mask1.to_csv("mask_channel_3.csv", index = False, sep = ";")


#t = tree.arrays(["OH", "VFAT", "CH", "digiStrip"], library = "pd", entry_stop = 60000)
#print(t)
#print(t["CH"])
#print(t[cut_oh&cut_vfat&cut_ch])
def histo_hitsVSchannel(arrays): 
    oh = arrays["OH"].iloc[0]
    vfat = arrays["VFAT"].iloc[0]
    cut_oh = arrays["OH"] == oh
    cut_vfat = arrays["VFAT"] == vfat
    #cut_ch = t["CH"] == ch
    selection = cut_oh & cut_vfat
    x = arrays[selection]["CH"]
    h = arrays[selection]["digiStrip"]
    #print(len(x))
    #print(len(h))
    f, axs = plt.subplots(1,2, figsize=(15, 7))
    bins = np.arange(-0.5, 128.5, 1)
    histo_heights, _ , _ = axs[0].hist(x, bins)
    axs[0].set_xlabel("channel")
    axs[0].set_ylabel("number of hits")
    #print(histo_heights)
    axs[1].hist(histo_heights, 30)
    axs[1].set_xlabel("number of hits per channel")
    axs[1].set_ylabel("count")
    f.suptitle("oh: %d, VFAT: %d" %(oh, vfat))
    f.savefig("histo_hitsVSchanneloh%dVfat%d"%(oh, vfat))
    plt.close()
    


def RMSanalysis(arrays):
    right = True 
    f, axs = plt.subplots(1,1, figsize = (20,10))
    mask = []
    masked = []
    oh = arrays["OH"].iloc[0]
    if oh > 3: 
        right = False
    if right:
        vfat = arrays["VFAT"].iloc[0]
        cut_oh = arrays["OH"] == oh
        cut_vfat = arrays["VFAT"] == vfat
    #cut_ch = t["CH"] == ch
        selection = cut_oh & cut_vfat
        x = arrays[selection]["CH"]
        h = arrays[selection]["digiStrip"]
    #print(len(x))
    #print(len(h))
        bins = np.arange(-0.5, 128.5, 1)
        histo_heights, histo_edges , _ = axs.hist(x, bins)
        axs.set_xlabel("channel")
        axs.set_ylabel("number of hits")
        bin_centers = (histo_edges[:-1] + histo_edges[1:]) / 2
    #print(len(histo_heights))
    #print(len(bin_centers))
        dataframe = pd.DataFrame({"heights": histo_heights, "channel":bin_centers})
        mean_value = np.mean(histo_heights)
    #print(mean_value)
    #print(histo_heights)
        rms = np.sqrt(np.sum(((histo_heights)-mean_value)**2)/((len(histo_heights))-1))
    #print(rms)
        df = pd.DataFrame({"number": [], "rms": []})
        number_list = [0]
        rms_list = [rms]
        histo_heights1 = histo_heights
        for i in range (1, 5):
                print("oh: %d, vfat%d" %(oh, vfat))
                print(histo_heights1) 
                histo_heights1 = histo_heights1[histo_heights1 < max(histo_heights1)]
                mean_value1 = np.mean(histo_heights1)
                rms1 = np.sqrt(np.sum((histo_heights1-mean_value)**2)/(len(histo_heights1)-1))
                rms_list.append(rms1)
                number_list.append(i)
                i += 1
        df = pd.DataFrame({"number": number_list, "rms": rms_list})
    #axs[1].scatter(df["number"], df["rms"])
    #axs[1].set_xlabel("number of channels ignored")
    #axs[1].set_ylabel("RMS")
        rms_reference = (df["rms"][df["number"] == 4].to_numpy())[0]
    #print(rms)
    #print(rms_reference)
        
        while rms>1.2*rms_reference:
        #aggiustare il programma per evitare che i Nan vengano processati. 
        #quindi mettere un if solo su quelli non Nan
                mean_value = np.mean(histo_heights)
                for i in range(0, len(dataframe["channel"][histo_heights == max(histo_heights)])):
                        masked.append(dataframe["channel"][histo_heights == max(histo_heights)].astype(int).iloc[i])
                        dataframe = dataframe[histo_heights < max(histo_heights)]
                        histo_heights = histo_heights[histo_heights < max(histo_heights)]
        #print(histo_heights)
                        rms = np.sqrt(np.sum((histo_heights-mean_value)**2)/(len(histo_heights)-1))
        #print(dataframe[histo_heights == max(histo_heights)]["channel"])
        plt.close()
    return pd.Series(masked, dtype = int)
        
    #print(df)
        
main()
