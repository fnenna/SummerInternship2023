#importing all the libraries necessary.
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
import argparse
import os
import sys
#to have a standard graphic visualization.
#hep.style.use(hep.style.CMS)
#run the pycode from terminal.

#we define a series of functions that I need for fittings.
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

def Average(lst):
    return sum(lst) / len(lst)
baseSmall = 235.2
baseLarge = 574.8
height = 787.9
n_eta = 8
n_strips = 384
err_constant= 12
increase = (baseLarge - baseSmall)/8
strip_pitch_array = []
spat_res_array = []
for eta in range (0,8):
    strip_pitch_eta =[]
    for length in np.arange((baseSmall + eta*increase),(baseSmall + (eta+1)*increase)+1, 0.01):
        strip_pitch_eta.append(length / n_strips)
    spat_res = Average(strip_pitch_eta) / np.sqrt(err_constant) 
    strip_pitch_array.append((((8-eta)+1), Average(strip_pitch_eta)))
    spat_res_array.append(((8-eta), spat_res))
    spat_res_array_used = spat_res_array[len(spat_res_array)//2 : len(spat_res_array)+1]
df_spat_res = pd.DataFrame(spat_res_array_used)

def plot_efficiency_chamber1(dataframe): 
    k = dataframe["chamber"].iloc[0]
    l = dataframe["eta"].iloc[0]
    a = k
    b = l - 1
    axs[a, b].errorbar(dataframe["multiple"]*dataframe["sigma"], dataframe["efficiency"], yerr = dataframe["error"], marker = ".", linestyle = "none")
    axs[a, b].set_title("Efficiency layer %d, eta %d" %(a, b+1))
    axs[a,b].set_xlabel("residual cut [mm]")
    axs[a, b].set_xlim([0., 15.])
    axs[a, b].set_ylim([0.5, 1.])
    axs[a, b].set_ylabel("efficiency")
    #popt, pcov = curve_fit(saturated_expo, dataframe["multiple"]*dataframe["sigma"], dataframe["efficiency"], bounds = ([0., 0., 0.], [1.2, 1000, 10]))
    #bins = np.arange((dataframe["multiple"]*dataframe["sigma"])[0], (dataframe["multiple"]*dataframe["sigma"])[-1], 0.01)
    #axs[a, b].plot(bins, saturated_expo(bins, *popt), 'r-', label='fit' )

def plot_efficiency_chamber(dataframe, fig, axs, run_number): 
    k = dataframe["chamber"].iloc[0]
    l = dataframe["eta"].iloc[0]
    a = k
    b = l - 1
    axs[a, b].errorbar(dataframe["multiple"]*dataframe["sigma"], dataframe["efficiency"], yerr = dataframe["error"], marker = ".", linestyle = "none")
    axs[a, b].set_title("Efficiency layer %d, eta %d" %(a, b+1))
    axs[a,b].set_xlabel("residual cut [mm]")
    axs[a, b].set_xlim([0., 15.])
    axs[a, b].set_ylim([0.5, 1.1])
    axs[a, b].set_ylabel("efficiency")
    popt, pcov = curve_fit(saturated_fun, dataframe["multiple"]*dataframe["sigma"], dataframe["efficiency"], bounds = ([1., 0., -40.], [2., 100, 40]))
    #bins = np.arange((dataframe["multiple"]*dataframe["sigma"])[0], (dataframe["multiple"]*dataframe["sigma"])[-1], 0.01)
    bins = np.arange(np.min(dataframe["multiple"]*dataframe["sigma"]), 25, 0.1)
    perr = np.sqrt(np.diag(pcov))
    axs[a, b].plot(bins, saturated_fun(bins, *popt), 'r-', label='fit' )
    bins = np.arange(0., 25, 0.1)
    axs[a, b].plot(bins, popt[0]**(-1)*(bins**0), '--', label='fit' )
    #let's try another function.
    #popt1, pcov1 = curve_fit(saturated_expo, dataframe["multiple"]*dataframe["sigma"], dataframe["efficiency"], bounds = ([0.6, 0., 0.], [1., 5., 1.]))
    #perr1 = np.sqrt(np.diag(pcov1))
    #bins = np.arange((dataframe["multiple"]*dataframe["sigma"])[0], (dataframe["multiple"]*dataframe["sigma"])[-1], 0.01)
    #bins = np.arange(0, 25, 0.1)
    #axs[a, b].plot(bins, saturated_expo(bins, *popt1), 'g-', label='fit_expo' )
    #axs[a, b].plot(bins, popt1[0]*(bins**0), '--', label='fit_expo' )
    textstr = '\n'.join((
        r'$a=%.2f \pm %.2f$' % (popt[1], perr[1]),
        r'$b=%.2f \pm %.2f$' % (popt[2], perr[2]),
        r'$satVal = %.4f \pm %.4f$' % (popt[0]**(-1), perr[0]/popt[0])))
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    # place a text box in upper left in axes coords
    axs[a,b].text(0.05, 0.95, textstr, transform=axs[a,b].transAxes, fontsize=14,
            verticalalignment='top', bbox=props)
    textstr1 = r"at $5\sigma: %.3f \pm %.3f$" %(dataframe["efficiency"][dataframe["multiple"]==5], dataframe["error"][dataframe["multiple"]==5])
    props1 = dict(boxstyle='round', facecolor='white', alpha=0.5)
    # place a text box in upper left in axes coords
    axs[a,b].text(0.5, 0.5, textstr1, transform=axs[a,b].transAxes, fontsize=20,
            verticalalignment='center', horizontalalignment = "center", bbox=props1)
    #textstr = '\n'.join((
        #r'$a=%.2f \pm %.2f$' % (popt1[1], perr1[1]),
        #r'$b=%.2f \pm %.2f$' % (popt1[2], perr1[2]),
        #r'$A = %.1f \pm %.1f$' % (popt1[0], perr1[0])))
    #props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    # place a text box in upper left in axes coords
    #axs[a,b].text(0.05, 0.95, textstr, transform=axs[a,b].transAxes, fontsize=14, verticalalignment='top', bbox=props)
    fig.tight_layout()
    fig.savefig(run_number + "/chi2/efficiencyVSsigma_layer_eta_ch.png")

def main():
    parser = argparse.ArgumentParser(
                        prog='ProgramName',
                        description='Analyse data from ME0 stack testbeam july 2023',
                        epilog='Text at the bottom of help')
    parser.add_argument('filename')           # positional argument
    #parser.add_argument('-c', '--count')      # option that takes a value
    args = parser.parse_args()
    #print(args.filename, args.count, args.verbose)
    run_file = args.filename   #ex. "00000342.root"
    root_file = "/eos/user/f/fnenna/" + run_file
    #We want to open the root file on python with uproot.
    file = uproot.open(root_file)
    file.classnames()
    t = file["trackTree"]
    #fix an entry_stop, if we don't want to take all the data
    entry_stop1 = 2000000
    run_number = (run_file[5:8])
    #build a multiple array.
    print(str(run_number))
    os.mkdir(str(run_number)+"/chi2")
    print(str(run_number))
    #we want to plot reconstructed hits and propagated hits together.
    #first, we create arrays for prop_hits and rec_hits:
    partialProphitGlobalX0 = t["partialProphitGlobalX"].array( entry_stop = entry_stop1)
    partialProphitEta0 = t["partialProphitEta"].array( entry_stop = entry_stop1)
    partialTrackChamber0 = t["partialTrackChamber"].array( entry_stop = entry_stop1)
    rechitGlobalX0 = t["rechitGlobalX"].array( entry_stop = entry_stop1)
    rechitEta0 = t["rechitEta"].array( entry_stop = entry_stop1)
    rechitChamber0 = t["rechitChamber"].array( entry_stop = entry_stop1)
    chi_squared = t["partialTrackChi2"].array( entry_stop = entry_stop1)
    cut_chi = chi_squared >= 0.12
    cut_chi2 = ak.count_nonzero(cut_chi, axis=1) < 1
    partialProphitGlobalX  = partialProphitGlobalX0[cut_chi2]
    partialProphitEta = partialProphitEta0[cut_chi2]
    partialTrackChamber = partialTrackChamber0[cut_chi2]
    rechitGlobalX = rechitGlobalX0[cut_chi2]
    rechitEta  =rechitEta0[cut_chi2]
    rechitChamber = rechitChamber0[cut_chi2]
    #we want to plot for each chamber. So we start a loop on the 4 chambers.
    for i in range(0, 4):
        cut = (partialTrackChamber == i)
        cut_rec = (rechitChamber == i)
        global_X = rechitGlobalX[cut_rec]
        #for each chamber we create a figure containing x-position distribution and eta distribution.
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        bin_height = ax1.hist(ak.ravel(global_X), bins = 50, ec='blue', fc='none', histtype='step', range=[-200, 200], label = "Reconstructed hits")
        ax1.set_xlabel("X position[mm]")
        ax1.set_ylabel("dN/dX")
        ax1.set_ylim([0, max(bin_height[0])+1000])
        eta = rechitEta[cut_rec]
        ax2.hist(ak.ravel(eta), bins =4,ec='blue', fc='none', histtype='step', range = [0.5, 4.5], label = "Reconstructed hits")
        ax2.set_xlabel("Eta partition")
        ax2.set_ylabel(r"$dN/d\eta$")
        fig.tight_layout()
        partGlobal_X = partialProphitGlobalX[cut]
        #fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        fig.suptitle("PropHit vs RecHit on the layer n.%d" %i)
        ax1.hist(ak.flatten(partGlobal_X), bins = 50, ec='red', fc='none', histtype='step', range=[-200, 200], label = "Propagated hits")
        partEta = partialProphitEta[cut]
        ax2.hist(ak.flatten(partEta), bins =4,ec='red', fc='none', histtype='step', range = [0.5, 4.5], label = "Propagated hits")
        ax1.legend(loc = "upper left")
        ax2.legend()
        fig.tight_layout()
        fig.savefig(run_number + "/chi2/prop_rec_hits_layer%d.png" %i)
    # At this point we want to compute residuals for each eta partition.
    data_propagated1 = t["partialProphitGlobalX"].array( entry_stop = entry_stop1)[cut_chi2]
    chamber = t["partialTrackChamber"].array( entry_stop = entry_stop1)[cut_chi2]
    chamber_rec = t["rechitChamber"].array( entry_stop = entry_stop1)[cut_chi2]
    data_reconstructed1 = t["rechitGlobalX"].array( entry_stop = entry_stop1)[cut_chi2]
    eta_partition_rec = t['rechitEta'].array( entry_stop = entry_stop1)[cut_chi2]
    eta_partition = t['partialProphitEta'].array( entry_stop = entry_stop1)[cut_chi2]
    fig_chamber, axs_chamber = plt.subplots(2, 2, figsize=( 20, 20 ))
    for i in range(0, 4):
        cut_layer = (chamber == i)
        cut_layer_rec = (chamber_rec == i)
        fig, axs = plt.subplots(2, 2, figsize=( 20, 20 ))
        fig.suptitle("layer n %d" %i)
        data_reconstructed1_1 =data_reconstructed1[cut_layer_rec]
        data_propagated1_1 = data_propagated1[cut_layer]
        eta_partition1 = eta_partition[cut_layer]
        data = {"partition": [], "sigma": [], "error_sigma": []}
        for j in range(1, 4): 
            cut_eta = (eta_partition[cut_layer] == j)
            cut_eta_rec = (eta_partition_rec[cut_layer_rec] == j)
            pairs = ak.cartesian((data_reconstructed1_1[cut_eta_rec], data_propagated1_1[cut_eta]))
            reconstructed, propagated = ak.unzip(pairs)
            delta_x = reconstructed - propagated
            a = (j-1) // 2
            b = (j-1) % 2
            delta_x_unique = ak.flatten(ak.min(delta_x, axis = 1), axis = 0)
            bin_heights, bin_borders, _ = axs[a,b].hist(delta_x_unique, bins = int(2**((a+b-1)*(a+b))*len(delta_x)**(1/3)), range=(-4., 4.))
            try:
                popt, perr = normal_fit(bin_heights, bin_borders, axs[a,b], bounds_array = ([0,-10, +0.1],[(3)*(max(bin_heights)), 10, 2]) )
            except Exception:
                print("Error for ch%d eta %d" %(i,j))
            axs[a, b].set_xlabel("x residual[mm]")
            axs[a, b].set_ylabel("dN/dx")
            textstr = '\n'.join((
                r'$\mu=%.2f \pm %.2f$' % (popt[1], perr[1]),
                r'$\sigma=%.2f \pm %.2f$' % (popt[2], perr[2]),
                r'$A = %.1f \pm %.1f$' % (popt[0], perr[0])))
            props = dict(boxstyle='round', facecolor='white', alpha=0.5)
            # place a text box in upper left in axes coords
            axs[a, b].text(0.05, 0.95, textstr, transform=axs[a, b].transAxes, fontsize=14,
                verticalalignment='top', bbox=props)
            axs[a, b].set_title("eta partition n. %d" %j)
            data["partition"].append(j)
            data["sigma"].append(popt[2])
            data["error_sigma"].append(perr[2])
            #fig.savefig(run_number + "/residuals_layer %d.png" %i)
            df = pd.DataFrame(data)
        c = i // 2
        d = i % 2
        axs_chamber[c, d].errorbar(df["partition"], df["sigma"], yerr = df["error_sigma"], fmt = ".", markersize = 10)
        axs_chamber[c, d].set_xlabel("eta partition")
        axs_chamber[c, d].set_ylabel("residuals sigma [mm]")
        axs_chamber[c, d].plot(df_spat_res[0], df_spat_res[1], color = "red", marker = "x", markersize = 15, linewidth = 0., label = r'$strippitch / \sqrt{12}$')
        axs_chamber[c, d].legend(frameon = True)
        axs_chamber[c, d].set_title("layer n.%d" %i)
        fig.tight_layout()
        fig.savefig(run_number + "/chi2/residuals_eta_layer%d.png" %i)
    #fig_chamber.savefig(run_number + "/spatialResolutionVSeta.png")
    fig_chamber.tight_layout()
    fig_chamber.savefig(run_number + "/chi2/spatial_resolution_vs_eta.png")
    #
    #
    #
    #Now we want to find the matching propagated hits: so the number of propagated hits that are within N*sigma 
    #at least one reconstructed hit in the corresponding event.
    data_propagated1 = t["partialProphitGlobalX"].array( entry_stop = entry_stop1)[cut_chi2]
    chamber = t["partialTrackChamber"].array( entry_stop = entry_stop1)[cut_chi2]
    chamber_rec = t["rechitChamber"].array( entry_stop = entry_stop1)[cut_chi2]
    data_reconstructed1 = t["rechitGlobalX"].array( entry_stop = entry_stop1)[cut_chi2]
    eta_partition_rec = t['rechitEta'].array( entry_stop = entry_stop1)[cut_chi2]
    eta_partition = t['partialProphitEta'].array( entry_stop = entry_stop1)[cut_chi2]
    N = 5
    for i in range(0, 4):
        fig, axs = plt.subplots(2, 2, figsize = (20, 20))
        cut_ch = (chamber == i)
        cut_rec0 = chamber_rec == i
        data_reconstructed_ch = data_reconstructed1[cut_rec0]
        data_propagated_ch = data_propagated1[cut_ch]
        eta_partition_ch = eta_partition[cut_ch]
        eta_partition_rec_ch = eta_partition_rec[cut_rec0]
        for j in range(1,4):
            cut = eta_partition_ch == j
            cut_rec = eta_partition_rec_ch == j
            pairs = ak.cartesian((data_reconstructed_ch[cut_rec], data_propagated_ch[cut]))
            reconstructed, propagated = ak.unzip(pairs)
            delta_x = reconstructed - propagated
            a = (j-1)//2
            b = (j-1)%2
            fig_eff, ax1= plt.subplots(1, 1, figsize=( 10, 5 ))
            bin_heights, bin_borders, _ = ax1.hist(ak.flatten(delta_x), bins = int(2**((a+b-1)*(a+b))*len(delta_x)**(1/3)), range=(-4., 4.))
            popt, perr = normal_fit(bin_heights, bin_borders, ax1, bounds_array = ([0,-0.5, +0.3],[3*(max(bin_heights)), 0.5, 1.5]))
            sigma = popt[2]
            plt.close()
            sigma = popt[2]
            data_propagated_ch = data_propagated1[cut_ch]
            matching = (np.min(np.abs(delta_x), axis = 1) <= N*sigma)
            data_propagated1_matching = data_propagated_ch[matching]
            matching_entries = axs[a, b].hist(ak.ravel(data_propagated1_matching), bins = int((len(ak.ravel(data_propagated1_matching)))**(1/3)), ec='magenta', fc='none', histtype='step', range = [-200, 200], label = "matching propagated x (%d sigma)" %N)
            axs[a, b].set_title("eta partition %d" %j)
            axs[a,b].set_xlabel("propagated x [mm] ( < 5sigma)")
            axs[a,b].set_ylabel("dN/dx")
            entries = axs[a, b].hist(ak.ravel(data_propagated_ch[cut]), bins = int((len(ak.ravel(data_propagated1_matching)))**(1/3)), ec='red', fc='none', histtype='step',range = [-200, 200], label = "propagated x")           
            efficiency = matching_entries[0] / entries[0]
            axs[a, b].hist(ak.ravel(data_reconstructed_ch[cut_rec]), bins = int((len(ak.ravel(data_propagated1_matching)))**(1/3)), ec='blue', fc='none', histtype='step',range = [-200, 200], label = "reconstructed x" )
            axs[a, b].legend(loc='upper left')
        fig.suptitle("Matching propagated hits chamber %d" %i, fontsize = 30)
        fig.tight_layout()
        fig.savefig(run_number + "/chi2/matching_eta_ch%d.png" %i)
    #
    #
    #
    #Now we want to see how reconstruction efficiency varies when we vary the cut on the residuals.
    data_propagated1 = t["partialProphitGlobalX"].array( entry_stop = entry_stop1)[cut_chi2]
    chamber = t["partialTrackChamber"].array( entry_stop = entry_stop1)[cut_chi2]
    chamber_rec = t["rechitChamber"].array( entry_stop = entry_stop1)[cut_chi2]
    data_reconstructed1 = t["rechitGlobalX"].array( entry_stop = entry_stop1)[cut_chi2]
    eta_partition_rec = t['rechitEta'].array( entry_stop = entry_stop1)[cut_chi2]
    eta_partition = t['partialProphitEta'].array( entry_stop = entry_stop1)[cut_chi2]
    data = {"chamber": [], "eta": [] , "multiple": [], "efficiency": [], "error": [], "sigma" : []}
    fig, axs = plt.subplots(4, 4, figsize = (30, 30))
    for N in range(40, 0, -1):
        for i in range(0, 4):
            for j in range(1,4):
                cut_ch = (chamber == i)
                cut_rec = chamber_rec == i
                data_reconstructed_ch = data_reconstructed1[cut_rec]
                data_propagated_ch = data_propagated1[cut_ch]
                eta_partition_ch = eta_partition[cut_ch]
                eta_partition_rec_ch = eta_partition_rec[cut_rec]
                cut = eta_partition_ch == j
                cut_rec = eta_partition_rec_ch == j
                pairs = ak.cartesian((data_reconstructed_ch[cut_rec], data_propagated_ch[cut]))
                reconstructed, propagated = ak.unzip(pairs)
                delta_x = reconstructed - propagated
                a = (j-1)//2
                b = (j-1)%2
                fig_eff, ax1= plt.subplots(1, 1, figsize=( 10, 5 ))
                bin_heights, bin_borders, _ = ax1.hist(ak.flatten(delta_x), bins = int(2**((a+b-1)*(a+b))*len(delta_x)**(1/3)), range=(-4., 4.))
                popt, perr = normal_fit(bin_heights, bin_borders, ax1, bounds_array = ([0,-0.5, +0.3],[(3)*(max(bin_heights)), 0.5, 1.5]))
                sigma = popt[2]
                plt.close()
                fig1 = plt.figure( figsize=( 5, 7 ) )
                sigma = popt[2]
                data_propagated_ch = data_propagated1[cut_ch]
                matching = (np.min(np.abs(delta_x), axis = 1) <= N*sigma)
                data_propagated1_matching = data_propagated_ch[matching]
                matching_entries = plt.hist(ak.ravel(data_propagated1_matching), bins = int((len(ak.ravel(data_propagated1_matching)))**(1/3)), ec='blue', fc='none', histtype='step', label = "matching propagated x (%d sigma)" %N)
                entries = plt.hist(ak.ravel(data_propagated_ch[cut]), bins = int(len(ak.ravel(data_propagated1[cut]))), ec='red', fc='none', histtype='step', label = "propagated x")
                plt.close()
                efficiency = (matching_entries[0].sum())/(entries[0].sum())
                eff_error = np.sqrt((efficiency*(1 - efficiency))/entries[0].sum())
                #print("efficiency of layer %d: %.2f" %(i, efficiency) )
                data["multiple"].append(N)
                data["efficiency"].append(efficiency)
                data["error"].append(eff_error)
                data["chamber"].append(i)
                data["eta"].append(j)
                data["sigma"].append(sigma)
#print(data)
    df = pd.DataFrame.from_dict(data)
    #df.groupby(["chamber", "eta"]).apply(plot_efficiency_chamber)
    #fig.tight_layout()
    #fig.savefig(run_number + "/efficiencyVSsigma_layer_eta.png")
    fig, axs = plt.subplots(4, 4, figsize = (30, 30))
    df.groupby(["chamber", "eta"]).apply(plot_efficiency_chamber, fig, axs, run_number)
    fig.tight_layout()

main()
