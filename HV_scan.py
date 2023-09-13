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
#import mplhep as hep
import os
import sys
#hep.style.use("CMS")

def gauss(x, A, mu, sigma): 
    return A/np.sqrt(2*np.pi*(sigma**2))*np.exp(- (x - mu)**2 /(sigma**2))

#perform the fit of an histogram with a gaussian function.
#input: data_frame, bin heights, bin borders, array of bound (A, mu, sigma) ([lowest values], [largest values])
def normal_fit(bin_heights, bin_borders, histo_name, bounds_array = (-np.inf, +np.inf)): 
    #calculate the bin centers, and then calculate the gauss function in the bin centers. 
    bin_centers = bin_borders[:-1] + np.diff(bin_borders) / 2
    popt, pcov = curve_fit(gauss, bin_centers, bin_heights, bounds= bounds_array)
    perr = np.sqrt(np.diag(pcov))
    bins = np.arange(bin_centers[0], bin_centers[-1], 0.01)
    histo_name.plot(bins, gauss(bins, *popt), 'r-', label='fit')
    return popt, perr

def saturated_fun(x, B, a, b): 
    #return B*(1-np.exp(-a*x + b))
    return (a*x+b)/(1+B*(a*x+b)) 
def saturated_expo(x, B, a, b): 
    return B*(1-np.exp(-a*x + b))

def ploteffVSeta(dataframe, fig, axs, colors): 
    chamber = dataframe["chamber"].iloc[0]
    eta = dataframe["eta"].iloc[0]
    a = chamber
    b = eta -1
    axs[a].errorbar(dataframe["HV"],  dataframe["efficiency"], yerr = dataframe["eff_error"], label = "eta%d" %eta, color = colors[b] )
    axs[a].set_xlabel("High Voltage" +r"$[\mu A]$", fontsize = 40)
    axs[a].set_ylabel("efficiency", fontsize = 40)
    axs[a].set_xlim([636, 711])
    axs[a].set_ylim([0.71, 1.])
    axs[a].set_title("chamber %d" %(chamber), fontsize = 40)
    axs[a].xaxis.set_tick_params(labelsize = 30)
    axs[a].yaxis.set_tick_params(labelsize = 30)
    bins = np.arange(630, 700, 0.1)
    #print(dataframe["efficiency"][dataframe["HV"]==700])
    average_plat = (ak.ravel(dataframe["efficiency"][dataframe["HV"]==700]) + ak.ravel(dataframe["efficiency"][dataframe["HV"]==680])) / 2
    #print(average_plat)
    axs[a].plot(bins, average_plat*(bins**0), '--', color = colors[b] )
    #textstr1 = r"at $700 \mu A: %.2f$" %dataframe["efficiency"][dataframe["HV"]==700]
    #props1 = dict(boxstyle='round', facecolor='white', alpha=0.5)
    # place a text box in upper left in axes coords
    #axs[a].text(0.5, 0.5, textstr1, transform=axs[a].transAxes, fontsize=20,
            #verticalalignment='center', horizontalalignment = "center", bbox=props1)
    axs[a].legend()

def main():
#here we compute the reconstruction efficiency (at 5sigma) for the 4 root file of HV scan.
    run_numbers = ["00000216", "00000217", "00000219", "00000221" ]
    HV_list = [700, 680, 660, 640]
    diction={"HV": [],"run_number": [], "chamber": [], "eta": [], "efficiency": [], "eff_error": []}
    for k in range(0, 4): 
        run_number = run_numbers[k]
        High_volt = HV_list[k]
        root_file = "/eos/cms/store/group/upgrade/GEM/TestBeams/ME0Stack/tracks/"+run_number+ ".root"
        file = uproot.open(root_file)
        file.classnames()
        t = file["trackTree"]
        #t.show()
        entry_stop1 = 600000
        data_propagated1 = t["partialProphitGlobalX"].array( entry_stop = entry_stop1)
        chamber = t["partialTrackChamber"].array( entry_stop = entry_stop1)
        chamber_rec = t["rechitChamber"].array( entry_stop = entry_stop1)
        data_reconstructed1 = t["rechitGlobalX"].array( entry_stop = entry_stop1)
        eta_partition_rec = t['rechitEta'].array( entry_stop = entry_stop1)
        eta_partition = t['partialProphitEta'].array( entry_stop = entry_stop1)
        N = 5
        for i in range(0, 4):
            fig, axs = plt.subplots(2, 2, figsize = (20, 20))
            cut_ch = (chamber == i)
            cut_rec0 = chamber_rec == i
            data_reconstructed_ch = data_reconstructed1[cut_rec0]
            data_propagated_ch = data_propagated1[cut_ch]
            eta_partition_ch = eta_partition[cut_ch]
            eta_partition_rec_ch = eta_partition_rec[cut_rec0]
            for j in range(1, 4):
                cut = eta_partition_ch == j
                cut_rec = eta_partition_rec_ch == j
                pairs = ak.cartesian((data_reconstructed_ch[cut_rec], data_propagated_ch[cut]))
                reconstructed, propagated = ak.unzip(pairs)
                delta_x = reconstructed - propagated
                a = (j-1)//2
                b = (j-1)%2
                fig_eff, ax1= plt.subplots(1, 1, figsize=( 10, 5 ))
                bin_heights, bin_borders, _ = ax1.hist(ak.flatten(delta_x), bins = int(2**((a+b-1)*(a+b))*len(delta_x)**(1/3)), range=(-4., 4.))
                popt, perr = normal_fit(bin_heights, bin_borders, ax1, bounds_array = ([0,-0.5, +0.3],[3*(max(bin_heights)), 0.5, 0.7]))
                sigma = popt[2]
                plt.close()
                #ax1.set_xlabel("x residual")
                #ax1.set_ylabel("dN/dx")
                #print("amplitude: %.2d pm %.2d" %(popt[0], perr[0]))
                #print("mu: %1.2f pm %1.2f" %(popt[1], perr[1]))
                #print("sigma: %1.2f pm %1.2f" %((popt[2]), perr[2]))
                #plt.plot(bin_centers, gauss(bin_centers, *popt), 'r-', label='fit')
                #(mu, sigma) = norm.fit(ak.flatten(delta_x))
            #fig1 = plt.figure( figsize=( 5, 7 ) )
                sigma = popt[2]
                data_propagated_ch = data_propagated1[cut_ch]
                matching = (np.min(np.abs(delta_x), axis = 1) <= N*sigma)
                #print(delta_x[matching])
                data_propagated1_matching = data_propagated_ch[matching]
                #print(data_propagated1_matching)
                #print(ak.flatten(data_propagated1_matching))
                #print(ak.min(data_propagated1_matching))
                matching_entries, bins_matching, _ = axs[a, b].hist(ak.ravel(data_propagated1_matching), bins = int((len(ak.ravel(data_propagated1_matching)))**(1/3)), ec='blue', fc='none', histtype='step', range = [-200, 200], label = "matching propagated x (%d sigma)" %N)
                axs[a, b].set_title("eta partition %d" %j)
                axs[a,b].set_xlabel("propagated x [mm] ( < 5sigma)")
                axs[a,b].set_ylabel("dN/dx")
                entries, bins, _ = axs[a, b].hist(ak.ravel(data_propagated_ch[cut]), bins = int((len(ak.ravel(data_propagated1_matching)))**(1/3)), ec='red', fc='none', histtype='step',range = [-200, 200], label = "propagated x")
                efficiency = (matching_entries.sum())/(entries.sum())
                eff_error = np.sqrt((efficiency*(1 - efficiency))/entries.sum())
                #print(efficiency)
                diction["run_number"].append(run_number)
                diction["chamber"].append(i)
                diction["eta"].append(j)
                diction["efficiency"].append(efficiency)
                diction["HV"].append(High_volt)
                diction["eff_error"].append(eff_error)
                #ax3.set_xlabel("propagated x [mm]")
                #ax3.set_ylabel("dN/dx")
                #ax1.legend((u"\u03bc = %1.2f \u00B1 %1.2f" %(popt[1], perr[1]), u"\u03C3: %1.2f \u00B1 %1.2f" %(popt[2], perr[2])), loc = "upper right" )
                #ax2.legend(loc='upper right', bbox_to_anchor=(1., 1.10))
                #print(matching_entries[0])
                #print(entries[0])
                #efficiency = (matching_entries[0].sum())/(entries[0].sum())
                #print("efficiency of layer %d: %.2f" %(i, efficiency) )
            fig.suptitle("Matching propagated hits chamber %d" %i, fontsize = 30)
            plt.close()
    fig, axs = plt.subplots(4, 1, figsize = (15,60) )
    colors = ["red", "green", "blue", "black"]
    df = pd.DataFrame.from_dict(diction)
    df.groupby(["chamber", "eta"]).apply(ploteffVSeta, fig, axs, colors)
    fig.tight_layout()
    fig.savefig("effVSHV_eta.png")
#here we do the HV scan ignoring noisy channels:
    diction={"HV": [],"run_number": [], "chamber": [], "eta": [], "efficiency": [], "eff_error": []}
    for k in range(0, 4): 
        run_number = run_numbers[k]
        High_volt = HV_list[k]
        root_file = "/eos/cms/store/group/upgrade/GEM/TestBeams/ME0Stack/tracks/"+run_number+ ".root"
        file = uproot.open(root_file)
        file.classnames()
        t = file["trackTree"]
        #t.show()
        entry_stop1 = 600000
        data_propagated0 = t["partialProphitGlobalX"].array( entry_stop = entry_stop1)
        chamber = t["partialTrackChamber"].array( entry_stop = entry_stop1)
        chamber_rec = t["rechitChamber"].array( entry_stop = entry_stop1)
        data_reconstructed0 = t["rechitGlobalX"].array( entry_stop = entry_stop1)
        eta_partition_rec0 = t['rechitEta'].array( entry_stop = entry_stop1)
        eta_partition0 = t['partialProphitEta'].array( entry_stop = entry_stop1)
        cut_noisy_rec = (data_reconstructed0 < 170) & ((data_reconstructed0 < 27) | (data_reconstructed0 > 32))
        cut_noisy_prop = data_propagated0 < 170
        print(cut_noisy_rec)
        print((cut_noisy_rec) & (chamber_rec == 0))
        print(chamber_rec)
        cut_gen = ak.count_nonzero((~(cut_noisy_rec)) & (chamber_rec == 0), axis=1) < 1
        data_reconstructed1 = data_reconstructed0[cut_gen]
        data_propagated1 = data_propagated0[cut_gen]
        print(ak.count_nonzero(~(cut_noisy_rec) & (chamber_rec == 0), axis=1))
    #print(rechitGlobalX)
    #print(partialProphitGlobalX)
        eta_partition1 = eta_partition0[cut_gen]
        eta_partition_rec1 = eta_partition_rec0[cut_gen]
        N = 5
        for i in range(0, 4):
            fig, axs = plt.subplots(2, 2, figsize = (20, 20))
            cut_ch = (chamber[cut_gen] == i)
            cut_rec0 = chamber_rec[cut_gen] == i
            data_reconstructed_ch = data_reconstructed1[cut_rec0]
            data_propagated_ch = data_propagated1[cut_ch]
            eta_partition_ch = eta_partition1[cut_ch]
            eta_partition_rec_ch = eta_partition_rec1[cut_rec0]
            for j in range(1, 4):
                cut = eta_partition_ch == j
                cut_rec = eta_partition_rec_ch == j
                pairs = ak.cartesian((data_reconstructed_ch[cut_rec], data_propagated_ch[cut]))
                reconstructed, propagated = ak.unzip(pairs)
                delta_x = reconstructed - propagated
                a = (j-1)//2
                b = (j-1)%2
                fig_eff, ax1= plt.subplots(1, 1, figsize=( 10, 5 ))
                bin_heights, bin_borders, _ = ax1.hist(ak.flatten(delta_x), bins = int(2**((a+b-1)*(a+b))*len(delta_x)**(1/3)), range=(-4., 4.))
                popt, perr = normal_fit(bin_heights, bin_borders, ax1, bounds_array = ([0,-0.5, +0.3],[3*(max(bin_heights)), 0.5, 0.7]))
                sigma = popt[2]
                plt.close()
                #ax1.set_xlabel("x residual")
                #ax1.set_ylabel("dN/dx")
                #print("amplitude: %.2d pm %.2d" %(popt[0], perr[0]))
                #print("mu: %1.2f pm %1.2f" %(popt[1], perr[1]))
                #print("sigma: %1.2f pm %1.2f" %((popt[2]), perr[2]))
                #plt.plot(bin_centers, gauss(bin_centers, *popt), 'r-', label='fit')
                #(mu, sigma) = norm.fit(ak.flatten(delta_x))
            #fig1 = plt.figure( figsize=( 5, 7 ) )
                sigma = popt[2]
                data_propagated_ch = data_propagated1[cut_ch]
                matching = (np.min(np.abs(delta_x), axis = 1) <= N*sigma)
                #print(delta_x[matching])
                data_propagated1_matching = data_propagated_ch[matching]
                #print(data_propagated1_matching)
                #print(ak.flatten(data_propagated1_matching))
                #print(ak.min(data_propagated1_matching))
                matching_entries, bins_matching, _ = axs[a, b].hist(ak.ravel(data_propagated1_matching), bins = int((len(ak.ravel(data_propagated1_matching)))**(1/3)), ec='blue', fc='none', histtype='step', range = [-200, 200], label = "matching propagated x (%d sigma)" %N)
                axs[a, b].set_title("eta partition %d" %j)
                axs[a,b].set_xlabel("propagated x [mm] ( < 5sigma)")
                axs[a,b].set_ylabel("dN/dx")
                entries, bins, _ = axs[a, b].hist(ak.ravel(data_propagated_ch[cut]), bins = int((len(ak.ravel(data_propagated1_matching)))**(1/3)), ec='red', fc='none', histtype='step',range = [-200, 200], label = "propagated x")
                efficiency = (matching_entries.sum())/(entries.sum())
                eff_error = np.sqrt((efficiency*(1 - efficiency))/entries.sum())
                #print(efficiency)
                diction["run_number"].append(run_number)
                diction["chamber"].append(i)
                diction["eta"].append(j)
                diction["efficiency"].append(efficiency)
                diction["HV"].append(High_volt)
                diction["eff_error"].append(eff_error)
                #ax3.set_xlabel("propagated x [mm]")
                #ax3.set_ylabel("dN/dx")
                #ax1.legend((u"\u03bc = %1.2f \u00B1 %1.2f" %(popt[1], perr[1]), u"\u03C3: %1.2f \u00B1 %1.2f" %(popt[2], perr[2])), loc = "upper right" )
                #ax2.legend(loc='upper right', bbox_to_anchor=(1., 1.10))
                #print(matching_entries[0])
                #print(entries[0])
                #efficiency = (matching_entries[0].sum())/(entries[0].sum())
                #print("efficiency of layer %d: %.2f" %(i, efficiency) )
            fig.suptitle("Matching propagated hits chamber %d" %i, fontsize = 30)
            fig.tight_layout()
            plt.close()
    fig, axs = plt.subplots(4, 1, figsize = (15,60) )
    colors = ["red", "green", "blue", "black"]
    df = pd.DataFrame.from_dict(diction)
    df.groupby(["chamber", "eta"]).apply(ploteffVSeta, fig, axs, colors)
    fig.tight_layout()
    fig.savefig("effVSHV_eta_noNoisy.png")

#Now we want to perform the HV scan using only hits from very straight tracks.
    diction={"HV": [],"run_number": [], "chamber": [], "eta": [], "efficiency": [], "eff_error": []}
    for k in range(0, 4): 
        run_number = run_numbers[k]
        High_volt = HV_list[k]
        root_file = "/eos/cms/store/group/upgrade/GEM/TestBeams/ME0Stack/tracks/"+run_number+ ".root"
        file = uproot.open(root_file)
        file.classnames()
        t = file["trackTree"]
        #t.show()
        entry_stop1 = 600000
        data_propagated0 = t["partialProphitGlobalX"].array( entry_stop = entry_stop1)
        chamber = t["partialTrackChamber"].array( entry_stop = entry_stop1)
        chamber_rec = t["rechitChamber"].array( entry_stop = entry_stop1)
        data_reconstructed0 = t["rechitGlobalX"].array( entry_stop = entry_stop1)
        eta_partition_rec0 = t['rechitEta'].array( entry_stop = entry_stop1)
        eta_partition0 = t['partialProphitEta'].array( entry_stop = entry_stop1)
        partialTrackSlopeX = t["partialTrackSlopeX"].array(entry_stop = entry_stop1)
        cut_gen = (partialTrackSlopeX < +0.002) & (partialTrackSlopeX > -0.002)
    #print(rechitGlobalX)
    #print(partialProphitGlobalX)
        data_reconstructed1 = data_reconstructed0
        data_propagated1 = data_propagated0[cut_gen]
        eta_partition1 = eta_partition0[cut_gen]
        eta_partition_rec1 = eta_partition_rec0
        N = 5
        for i in range(0, 4):
            fig, axs = plt.subplots(2, 2, figsize = (20, 20))
            cut_ch = (chamber[cut_gen] == i)
            cut_rec0 = chamber_rec == i
            data_reconstructed_ch = data_reconstructed1[cut_rec0]
            data_propagated_ch = data_propagated1[cut_ch]
            eta_partition_ch = eta_partition1[cut_ch]
            eta_partition_rec_ch = eta_partition_rec1[cut_rec0]
            for j in range(1, 4):
                cut = eta_partition_ch == j
                cut_rec = eta_partition_rec_ch == j
                pairs = ak.cartesian((data_reconstructed_ch[cut_rec], data_propagated_ch[cut]))
                reconstructed, propagated = ak.unzip(pairs)
                delta_x = reconstructed - propagated
                a = (j-1)//2
                b = (j-1)%2
                fig_eff, ax1= plt.subplots(1, 1, figsize=( 10, 5 ))
                bin_heights, bin_borders, _ = ax1.hist(ak.flatten(delta_x), bins = int(2**((a+b-1)*(a+b))*len(delta_x)**(1/3)), range=(-4., 4.))
                popt, perr = normal_fit(bin_heights, bin_borders, ax1, bounds_array = ([0,-0.5, +0.3],[3*(max(bin_heights)), 0.5, 0.7]))
                sigma = popt[2]
                plt.close()
                #ax1.set_xlabel("x residual")
                #ax1.set_ylabel("dN/dx")
                #print("amplitude: %.2d pm %.2d" %(popt[0], perr[0]))
                #print("mu: %1.2f pm %1.2f" %(popt[1], perr[1]))
                #print("sigma: %1.2f pm %1.2f" %((popt[2]), perr[2]))
                #plt.plot(bin_centers, gauss(bin_centers, *popt), 'r-', label='fit')
                #(mu, sigma) = norm.fit(ak.flatten(delta_x))
            #fig1 = plt.figure( figsize=( 5, 7 ) )
                sigma = popt[2]
                data_propagated_ch = data_propagated1[cut_ch]
                matching = (np.min(np.abs(delta_x), axis = 1) <= N*sigma)
                #print(delta_x[matching])
                data_propagated1_matching = data_propagated_ch[matching]
                #print(data_propagated1_matching)
                #print(ak.flatten(data_propagated1_matching))
                #print(ak.min(data_propagated1_matching))
                matching_entries, bins_matching, _ = axs[a, b].hist(ak.ravel(data_propagated1_matching), bins = int((len(ak.ravel(data_propagated1_matching)))**(1/3)), ec='blue', fc='none', histtype='step', range = [-200, 200], label = "matching propagated x (%d sigma)" %N)
                axs[a, b].set_title("eta partition %d" %j)
                axs[a,b].set_xlabel("propagated x [mm] ( < 5sigma)")
                axs[a,b].set_ylabel("dN/dx")
                entries, bins, _ = axs[a, b].hist(ak.ravel(data_propagated_ch[cut]), bins = int((len(ak.ravel(data_propagated1_matching)))**(1/3)), ec='red', fc='none', histtype='step',range = [-200, 200], label = "propagated x")
                efficiency = (matching_entries.sum())/(entries.sum())
                eff_error = np.sqrt((efficiency*(1 - efficiency))/entries.sum())
                #print(efficiency)
                diction["run_number"].append(run_number)
                diction["chamber"].append(i)
                diction["eta"].append(j)
                diction["efficiency"].append(efficiency)
                diction["HV"].append(High_volt)
                diction["eff_error"].append(eff_error)
                #ax3.set_xlabel("propagated x [mm]")
                #ax3.set_ylabel("dN/dx")
                #ax1.legend((u"\u03bc = %1.2f \u00B1 %1.2f" %(popt[1], perr[1]), u"\u03C3: %1.2f \u00B1 %1.2f" %(popt[2], perr[2])), loc = "upper right" )
                #ax2.legend(loc='upper right', bbox_to_anchor=(1., 1.10))
                #print(matching_entries[0])
                #print(entries[0])
                #efficiency = (matching_entries[0].sum())/(entries[0].sum())
                #print("efficiency of layer %d: %.2f" %(i, efficiency) )
            fig.suptitle("Matching propagated hits chamber %d" %i, fontsize = 30)
            fig.tight_layout()
            plt.close()
    fig, axs = plt.subplots(4, 1, figsize = (15,60) )
    colors = ["red", "green", "blue", "black"]
    df = pd.DataFrame.from_dict(diction)
    df.groupby(["chamber", "eta"]).apply(ploteffVSeta, fig, axs, colors)
    fig.tight_layout()
    fig.savefig("effVSHV_eta_noSlope.png")

#As last thing, we want to put together both the two filters.
    diction={"HV": [],"run_number": [], "chamber": [], "eta": [], "efficiency": [], "eff_error": []}
    for k in range(0, 4): 
        run_number = run_numbers[k]
        High_volt = HV_list[k]
        root_file = "/eos/cms/store/group/upgrade/GEM/TestBeams/ME0Stack/tracks/"+run_number+ ".root"
        file = uproot.open(root_file)
        file.classnames()
        t = file["trackTree"]
        #t.show()
        entry_stop1 = 600000
        data_propagated0 = t["partialProphitGlobalX"].array( entry_stop = entry_stop1)
        chamber = t["partialTrackChamber"].array( entry_stop = entry_stop1)
        chamber_rec = t["rechitChamber"].array( entry_stop = entry_stop1)
        data_reconstructed0 = t["rechitGlobalX"].array( entry_stop = entry_stop1)
        eta_partition_rec0 = t['rechitEta'].array( entry_stop = entry_stop1)
        eta_partition0 = t['partialProphitEta'].array( entry_stop = entry_stop1)
        partialTrackSlopeX = t["partialTrackSlopeX"].array(entry_stop = entry_stop1)
        cut_noisy_rec = (data_reconstructed0 < 170) & ((data_reconstructed0 < 27) | (data_reconstructed0 > 32))
        cut_noisy_prop = data_propagated0 < 170
        print(cut_noisy_rec)
        print((cut_noisy_rec) & (chamber_rec == 0))
        print(chamber_rec)
        cut_gen = (ak.count_nonzero((~(cut_noisy_rec)) & (chamber_rec == 0), axis=1) < 1) & (partialTrackSlopeX < +0.002) & (partialTrackSlopeX > -0.002)
    #print(rechitGlobalX)
    #print(partialProphitGlobalX)
        data_reconstructed1 = data_reconstructed0
        data_propagated1 = data_propagated0[cut_gen]
        eta_partition1 = eta_partition0[cut_gen]
        eta_partition_rec1 = eta_partition_rec0
        N = 5
        for i in range(0, 4):
            fig, axs = plt.subplots(2, 2, figsize = (20, 20))
            cut_ch = (chamber[cut_gen] == i)
            cut_rec0 = chamber_rec == i
            data_reconstructed_ch = data_reconstructed1[cut_rec0]
            data_propagated_ch = data_propagated1[cut_ch]
            eta_partition_ch = eta_partition1[cut_ch]
            eta_partition_rec_ch = eta_partition_rec1[cut_rec0]
            for j in range(1, 4):
                cut = eta_partition_ch == j
                cut_rec = eta_partition_rec_ch == j
                pairs = ak.cartesian((data_reconstructed_ch[cut_rec], data_propagated_ch[cut]))
                reconstructed, propagated = ak.unzip(pairs)
                delta_x = reconstructed - propagated
                a = (j-1)//2
                b = (j-1)%2
                fig_eff, ax1= plt.subplots(1, 1, figsize=( 10, 5 ))
                bin_heights, bin_borders, _ = ax1.hist(ak.flatten(delta_x), bins = int(2**((a+b-1)*(a+b))*len(delta_x)**(1/3)), range=(-4., 4.))
                popt, perr = normal_fit(bin_heights, bin_borders, ax1, bounds_array = ([0,-0.5, +0.3],[3*(max(bin_heights)), 0.5, 0.7]))
                sigma = popt[2]
                plt.close()
                #ax1.set_xlabel("x residual")
                #ax1.set_ylabel("dN/dx")
                #print("amplitude: %.2d pm %.2d" %(popt[0], perr[0]))
                #print("mu: %1.2f pm %1.2f" %(popt[1], perr[1]))
                #print("sigma: %1.2f pm %1.2f" %((popt[2]), perr[2]))
                #plt.plot(bin_centers, gauss(bin_centers, *popt), 'r-', label='fit')
                #(mu, sigma) = norm.fit(ak.flatten(delta_x))
            #fig1 = plt.figure( figsize=( 5, 7 ) )
                sigma = popt[2]
                data_propagated_ch = data_propagated1[cut_ch]
                matching = (np.min(np.abs(delta_x), axis = 1) <= N*sigma)
                #print(delta_x[matching])
                data_propagated1_matching = data_propagated_ch[matching]
                #print(data_propagated1_matching)
                #print(ak.flatten(data_propagated1_matching))
                #print(ak.min(data_propagated1_matching))
                matching_entries, bins_matching, _ = axs[a, b].hist(ak.ravel(data_propagated1_matching), bins = int((len(ak.ravel(data_propagated1_matching)))**(1/3)), ec='blue', fc='none', histtype='step', range = [-200, 200], label = "matching propagated x (%d sigma)" %N)
                axs[a, b].set_title("eta partition %d" %j)
                axs[a,b].set_xlabel("propagated x [mm] ( < 5sigma)")
                axs[a,b].set_ylabel("dN/dx")
                entries, bins, _ = axs[a, b].hist(ak.ravel(data_propagated_ch[cut]), bins = int((len(ak.ravel(data_propagated1_matching)))**(1/3)), ec='red', fc='none', histtype='step',range = [-200, 200], label = "propagated x")
                efficiency = (matching_entries.sum())/(entries.sum())
                eff_error = np.sqrt((efficiency*(1 - efficiency))/entries.sum())
                #print(efficiency)
                diction["run_number"].append(run_number)
                diction["chamber"].append(i)
                diction["eta"].append(j)
                diction["efficiency"].append(efficiency)
                diction["HV"].append(High_volt)
                diction["eff_error"].append(eff_error)
                #ax3.set_xlabel("propagated x [mm]")
                #ax3.set_ylabel("dN/dx")
                #ax1.legend((u"\u03bc = %1.2f \u00B1 %1.2f" %(popt[1], perr[1]), u"\u03C3: %1.2f \u00B1 %1.2f" %(popt[2], perr[2])), loc = "upper right" )
                #ax2.legend(loc='upper right', bbox_to_anchor=(1., 1.10))
                #print(matching_entries[0])
                #print(entries[0])
                #efficiency = (matching_entries[0].sum())/(entries[0].sum())
                #print("efficiency of layer %d: %.2f" %(i, efficiency) )
            fig.suptitle("Matching propagated hits chamber %d" %i, fontsize = 30)
            fig.tight_layout()
            plt.close()
    fig, axs = plt.subplots(4, 1, figsize = (15,60) )
    colors = ["red", "green", "blue", "black"]
    df = pd.DataFrame.from_dict(diction)
    df.groupby(["chamber", "eta"]).apply(ploteffVSeta, fig, axs, colors)
    fig.tight_layout()
    fig.savefig("effVSHV_eta_noNoisy&Slope.png")

main()
    