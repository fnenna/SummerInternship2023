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
#import mplhep as hep
import os
import sys
#hep.style.use("CMS")


# In[2]:


t = uproot.open("/eos/cms/store/group/upgrade/GEM/TestBeams/ME0Stack/digi/00000"+str(309)+".root")["outputtree"]
t.show()


# In[9]:


def linear(x, m, q):
    return m*x + q
def decreasing_expo(x, I):
    return I*np.exp(-x)

def gauss(x, A, mu, sigma): 
    return A/np.sqrt(2*np.pi*(sigma**2))*np.exp(- (x - mu)**2 /(sigma**2))

def gauss_const(x, A, mu, sigma, m, q): 
    return m*x + q + A/np.sqrt(2*np.pi*(sigma**2))*np.exp(- (x - mu)**2 /(sigma**2))

def gauss_expo(x, A, mu, sigma, m, q): 
    return q*np.exp(-m*x) + A/np.sqrt(2*np.pi*(sigma**2))*np.exp(- (x - mu)**2 /(sigma**2))

def rejection_fun(x, k, a, b): 
    return 1/(1+k*(a*x + b))

def double_gauss_const(x, A1, mu1, sigma1, A2, mu2, sigma2, m, q): 
        return m*x + q + A1/np.sqrt(2*np.pi*(sigma1**2))*np.exp(- (x - mu1)**2 /(sigma1**2)) + A2/np.sqrt(2*np.pi*(sigma2**2))*np.exp(- (x - mu2)**2 /(sigma2**2))


#perform the fit of an histogram with a gaussian function.
#input: data_frame, bin heights, bin borders, array of bound (A, mu, sigma) ([lowest values], [largest values])
def normal_fit(bin_heights, bin_borders, histo_name, bounds_array = (-np.inf, +np.inf)): 
    #calculate the bin centers, and then calculate the gauss function in the bin centers. 
    bin_centers = bin_borders[:-1] + np.diff(bin_borders) / 2
    popt, pcov = curve_fit(gauss, bin_centers, bin_heights, bounds= bounds_array)
    perr = np.sqrt(np.diag(pcov))
    bins = np.arange(bin_centers[0], bin_centers[-1], 0.00001)
    histo_name.plot(bins, gauss(bins, *popt), 'r-', label='fit')
    return popt, perr

def linear_fit(bin_heights, bin_borders, histo_name, bounds_array = (-np.inf, +np.inf)): 
    #calculate the bin centers, and then calculate the gauss function in the bin centers. 
    bin_centers = bin_borders[:-1] + np.diff(bin_borders) / 2
    popt, pcov = curve_fit(linear, bin_centers, bin_heights, bounds= bounds_array)
    perr = np.sqrt(np.diag(pcov))
    bins = np.arange(bin_centers[0], bin_centers[-1], 0.00001)
    histo_name.plot(bins, linear(bins, *popt), 'r-', label='fit')
    return popt, perr

def double_fit(bin_heights, bin_borders, histo_name, bounds_array = (-np.inf, +np.inf)): 
    #calculate the bin centers, and then calculate the gauss function in the bin centers. 
    bin_centers = bin_borders[:-1] + np.diff(bin_borders) / 2
    popt, pcov = curve_fit(double_gauss_const, bin_centers, bin_heights, bounds= bounds_array)
    perr = np.sqrt(np.diag(pcov))
    bins = np.arange(bin_centers[0], bin_centers[-1], 0.00001)
    histo_name.plot(bins, double_gauss_const(bins, *popt), 'r-', label='fit')
    histo_name.plot(bins, gauss(bins, popt[0], popt[1], popt[2]), 'b-', label='fit')
    histo_name.plot(bins, gauss(bins, popt[3], popt[4], popt[5]), 'g-', label='fit')
    histo_name.plot(bins, linear(bins, popt[6], popt[7]), 'y-', label='fit')
    return popt, perr

def normal_const_fit(bin_heights, bin_borders, histo_name, bounds_array = (-np.inf, +np.inf)): 
    #calculate the bin centers, and then calculate the gauss function in the bin centers. 
    bin_centers = bin_borders[:-1] + np.diff(bin_borders) / 2
    popt, pcov = curve_fit(gauss_const, bin_centers, bin_heights, bounds= bounds_array)
    perr = np.sqrt(np.diag(pcov))
    bins = np.arange(bin_centers[0], bin_centers[-1], 0.00001)
    histo_name.plot(bins, gauss_const(bins, *popt), 'r-', label='fit')
    return popt, perr

def rejection_fit(x_data, y_data,histo_name, bounds_array = (-np.inf, +np.inf)): 
    #calculate the bin centers, and then calculate the gauss function in the bin centers. 
    popt, pcov = curve_fit(rejection_fun, x_data, y_data, bounds= bounds_array)
    perr = np.sqrt(np.diag(pcov))
    bins = np.arange(np.min(x_data),np.max(x_data), 100)
    histo_name.plot(bins, rejection_fun(bins, *popt), 'r-', label='fit')
    return popt, perr

def normal_expo_fit(bin_heights, bin_borders, histo_name, bounds_array = (-np.inf, +np.inf)): 
    #calculate the bin centers, and then calculate the gauss function in the bin centers. 
    bin_centers = bin_borders[:-1] + np.diff(bin_borders) / 2
    popt, pcov = curve_fit(gauss_expo, bin_centers, bin_heights, bounds= bounds_array)
    perr = np.sqrt(np.diag(pcov))
    bins = np.arange(bin_centers[0], bin_centers[-1], 0.00001)
    histo_name.plot(bins, gauss_expo(bins, *popt), 'r-', label='fit')
    return popt, perr


def saturated_fun(x, B, a, b): 
    #return B*(1-np.exp(-a*x + b))
    return (a*x+b)/(1+B*(a*x+b)) 
def saturated_expo(x, B, a, b): 
    return B*(1-np.exp(-a*x + b))


# In[3]:


run_numbers = {"291":1, "293":22, "294":460, "297":1.5, "299":2.2, "301":3.3, "303":4.6, "305":10, "307":33, "309":46}
#dic = {"oh":[], "vfat":[], "rate":[]}
for run_number in run_numbers.keys():
    root_file = uproot.open("/eos/user/f/fnenna/00000"+(run_number)+"-digi.root")
    t = root_file["outputtree"]
    data_chamber = t["digiChamber"].array(entry_stop = -1)
    data_oh = t["OH"].array(entry_stop = -1)
    data_vfat = t["VFAT"].array(entry_stop = -1)
    data_ch = t["CH"].array(entry_stop = -1)
    n_triggers = ak.num(data_chamber, axis = 0) #gives me the number of events recorded, that should be equal to the number of triggers.
    time_window = 8 * 25 * 10**(-9)
    time = n_triggers * time_window
    #print(f"run {run_number}: {n_triggers}")
    #print(data_oh)
    #print(data_vfat)
    #print(data_ch)
    for oh in np.unique(ak.flatten(data_oh)):
        dic = {"oh":[], "vfat":[], "rate":[]}
        cut_oh = data_oh == oh
        vfat_oh = data_vfat[cut_oh]
        ch_oh = data_ch[cut_oh]
        for vfat in np.unique(ak.flatten(vfat_oh)):
            cut_vfat = vfat_oh == vfat
            ch_vfat = ch_oh[cut_vfat]
            ch_valid = ch_vfat[ak.num(ch_vfat, axis = 1)>=1]  #cutting on the events in which at least one channel is fired
            #print(ch_valid)
            n_events_vfat = ak.num(ch_valid, axis = 0)    #number of events with at least one channel fired
            rate = n_events_vfat / time
            dic["oh"].append(oh)
            dic["vfat"].append(vfat)
            dic["rate"].append(rate)
        fig, ax = plt.subplots(1, 1, figsize = (10, 10))
        #print(dic["vfat"])
        #print(dic["rate"])
        fig.suptitle(f"run {run_number}, ABS = {run_numbers[run_number]}, oh = {oh}")
        ax.scatter(dic["vfat"], dic["rate"])

            
                


# In[4]:


run_numbers = {"291":1, "293":22, "294":460, "297":1.5, "299":2.2, "301":3.3, "303":4.6, "305":10, "307":33, "309":46}
#run_numbers = {"291":1, "293":22, "294":460}
#dic = {"oh":[], "vfat":[], "rate":[]}
dic1 = dic = {"layer":[], "partition":[], "averageRate":[], "ABS":[], "sigmaRate":[]}
for run_number in run_numbers.keys():
    root_file = uproot.open("/eos/user/f/fnenna/00000"+(run_number)+"-digi.root")
    t = root_file["outputtree"]
    data_chamber = t["digiChamber"].array(entry_stop = -1)
    data_eta = t["digiEta"].array(entry_stop = -1)
    data_strip = t["digiStrip"].array(entry_stop = -1)
    n_triggers = ak.num(data_chamber, axis = 0) #gives me the number of events recorded, that should be equal to the number of triggers.
    time_window = 8 * 25 * 10**(-9)
    time = n_triggers * time_window
    print(time)
    #print(f"run {run_number}: {n_triggers}")
    #print(data_oh)
    #print(data_vfat)
    #print(data_ch)
    for chamber in np.unique(ak.flatten(data_chamber)):
        cut_chamber = data_chamber == chamber
        eta_chamber = data_eta[cut_chamber]
        strip_chamber = data_strip[cut_chamber]
        for eta in np.unique(ak.flatten(eta_chamber)):
            dic = {"chamber":[], "eta":[], "strip":[], "rate":[]}
            cut_eta = eta_chamber == eta
            strip_eta = strip_chamber[cut_eta]
            strip_valid = strip_eta[ak.num(strip_eta, axis = 1)>=1]  #cutting on the events in which at least one channel is fired
            for strip in np.unique(ak.flatten(strip_valid)):
                cut_strip = strip_valid == strip
                #print(strip_valid)
                #print(strip)
                #print(strip_valid[ak.count_nonzero(strip_valid[cut_strip], axis = 1)> 0])
                n_events_strip = ak.num(strip_valid[ak.count_nonzero(strip_valid[cut_strip], axis = 1)> 0], axis = 0)
                rate = n_events_strip / time
                #print(n_events_strip)
                dic["chamber"].append(chamber)
                dic["eta"].append(eta)
                dic["strip"].append(strip)
                dic["rate"].append(rate)
            fig, ax = plt.subplots(1, 1, figsize = (10, 10))
            fig.suptitle(f"run {run_number}, ABS = {run_numbers[run_number]}, chamber = {chamber}, eta = {eta}")
            df = pd.DataFrame.from_dict(dic)
            ax.scatter(df["strip"], df["rate"], marker = ".")
            ax.set_xlim([0, ak.num(df["strip"], axis = 0)])
            average = np.mean(df["rate"])
            sigma = np.sqrt(np.sum((df["rate"]-average)**2)/(ak.num(df["rate"],axis = 0)-1))
            sigma = np.std(df["rate"])/np.sqrt(ak.num(df["rate"],axis = 0))
            bins = np.arange(0, ak.num(df["strip"], axis = 0), 0.01)
            ax.plot(bins, average*(bins**0), 'r--', label='average' )
            ax.set_ylabel("rate per strip[Hz]")
            ax.set_xlabel("strip number")
            #plt.savefig()
            dic1["layer"].append(chamber)
            dic1["partition"].append(eta)
            dic1["ABS"].append(run_numbers[run_number])
            dic1["sigmaRate"].append(sigma)
            dic1["averageRate"].append(average)
    
            

            
                


# In[20]:


df2 = pd.DataFrame.from_dict(dic1)
fig, axs = plt.subplots(4, 4, figsize = (30, 30))
for ch in np.unique(df2["layer"]):
    cut_ch = df2["layer"] == ch
    df2_ch = df2[cut_ch]
    for eta in np.unique(df2["partition"]):
        cut_eta = df2_ch["partition"] == eta
        df2_eta = df2_ch[cut_eta]
        a = ch
        b = eta - 1
        axs[a, b].errorbar(np.log(df2_eta["ABS"]), (df2_eta["averageRate"]), yerr = df2_eta["sigmaRate"], marker = ".", linestyle= "none")
        axs[a, b].set_xlabel(r"$log(ABS)$")
        axs[a, b].set_ylabel("average rate per strip[Hz]")
        axs[a, b].set_title("ch.%d, eta%d" %(ch, eta))
        popt, pcov = curve_fit(decreasing_expo, np.log(df2_eta["ABS"]), df2_eta["averageRate"])
        perr = np.sqrt(np.diag(pcov))
        bins = np.arange(0, 7, 0.01)
        axs[a, b].plot(bins, decreasing_expo(bins, *popt), "r-", label = "fit" )
        textstr = (r'$I_0=%d \pm %d$' % (popt[0], perr[0]))
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        # place a text box in upper left in axes coords
        axs[a,b].text(0.60, 0.95, textstr, transform=axs[a,b].transAxes, fontsize=14,
            verticalalignment='top', bbox=props)
fig.tight_layout()
        


# In[21]:


print(df2)

entry_stop1 = 600000
N = 5
run_numbers = [["00000342", "+np.inf", "blue"], ["00000225", "46", "red"],["00000345", "33", "darkred"],["00000254", "22", "darkcyan"], ["00000265", "10", "green"], ["00000267", "6.9", "purple"], ["00000269", "4.6", "orange"]]
#fig, axs = plt.subplots(3, 1, figsize = (10, 30))
#fig1, ax = plt.subplots(1, figsize = (10, 30))
data1 = {"partition": [], "ABS":[], "layer":[], "background":[], "err_background":[]}

for run_number in run_numbers[1:]:
    index = run_numbers.index(run_number)
    root_file = "/eos/user/f/fnenna/"+run_number[0]+ ".root"
    file = uproot.open(root_file)
    #file.classnames()
    t = file["trackTree"]
    #t.show()
    #chi_quad = t["partialTrackChi2"].array(entry_stop = entry_stop1)
    data_propagated1 = t["partialProphitGlobalX"].array( entry_stop = entry_stop1)
    eta_partition = t["partialProphitEta"].array( entry_stop = entry_stop1)
    #partialTrackChamber = t["partialTrackChamber"].array( entry_stop = entry_stop1)
    data_reconstructed1 = t["rechitGlobalX"].array( entry_stop = entry_stop1)
    eta_partition_rec = t["rechitEta"].array( entry_stop = entry_stop1)
    chamber_rec = t["rechitChamber"].array( entry_stop = entry_stop1)
    chamber = t["partialTrackChamber"].array( entry_stop = entry_stop1)
    print(ak.num(ak.flatten(data_propagated1), axis = 0))
    #print(chamber)
    #print(data_propagated1)
    #print(slopeX)
    #print(chi_quad)
    print(data_propagated1)
    for i in range(0, 4):
        fig, axs = plt.subplots(2, 2, figsize=( 20, 20 ))
        fig1, axs1 = plt.subplots(2, 2, figsize = (20, 20))
        cut_layer = (chamber == i)
        cut_layer_rec = (chamber_rec == i)
        fig.suptitle("layer n %d" %i)
        #print(cut_layer)
        data_reconstructed1_1 =data_reconstructed1[cut_layer_rec]
        data_propagated1_1 = data_propagated1[cut_layer]
        #print(data_reconstructed1_1)
        #print(data_propagated1_1)
        eta_partition1 = eta_partition[cut_layer]
        #data = {"partition": [], "sigma": [], "error_sigma": []}
        for j in range(1,4): 
            cut_eta = (eta_partition[cut_layer] == j)
            cut_eta_rec = (eta_partition_rec[cut_layer_rec] == j)
            #delta_x = (data_reconstructed1_1[cut_eta_rec] - data_propagated1_1[cut_eta])
            #print(data_reconstructed1_1[cut_eta_rec])
            #print(data_propagated1_1[cut_eta] )
            pairs = ak.cartesian((data_reconstructed1_1[cut_eta_rec], data_propagated1_1[cut_eta]))
            #print(pairs)
            #print(pairs)
            reconstructed, propagated = ak.unzip(pairs)
            #print(reconstructed)
            #print(propagated)
            delta_x_all = (reconstructed - propagated)
            #delta_x = delta_x_all[np.abs(delta_x_all) == ak.min(np.abs(delta_x_all), axis = 1)]
        #print(delta_x)
        #print(delta_x_all)
        #print(ak.flatten(delta_x))
        #print(delta_x)
            #delta_x_unique = ak.flatten(delta_x)
            a = (j-1) // 2
            b = (j-1) % 2
            #print(delta_x_all)
            #delta_x = delta_x_all[np.abs(delta_x_all) == ak.min(np.abs(delta_x_all), axis = 1)]
            #print(delta_x)
            #print(delta_x_all)
            #print(ak.flatten(delta_x))
            #print(delta_x)
            #delta_x_unique = ak.flatten(delta_x, axis = 0)
            #print(len(ak.flatten(delta_x_unique)))
            #print(ak.flatten(delta_x))
            #print(np.std(delta_x_unique))
            bin_heights, bin_borders, _ = axs[a,b].hist(ak.flatten(delta_x_all), bins = int(8/0.2), range=[-4, 4], ec=run_number[2], fc='none', histtype='step')
            #axs[a,b].set_ylim([0, 1000])
            #axs[a,b].hist(ak.flatten(ak.min(delta_x_all, axis = 1), axis = 0), bins = int(8/0.2) , range=(-4., 4.), ec='blue', fc='none', histtype='step', label = "min sull'x")
            popt, perr = normal_fit(bin_heights, bin_borders, axs[a,b], bounds_array = ([0,-1, 0.],[3*max(bin_heights)/2 + 2000, 1, 1]) )
            #axs[a, b].set_xlabel("x residual[mm]")
            textstr = '\n'.join((
                r'$\mu1=%.2f \pm %.2f$' % (popt[1], perr[1]),
                r'$\sigma1=%.3f \pm %.3f$' % (popt[2], perr[2]),
                r'$A1 = %.1f \pm %.1f$' % (popt[0], perr[0])))
            sigma = popt[2]
            #print(sigma)
            #print(delta_x_all)
            cut_back = ak.count_nonzero(np.abs(delta_x_all) > N*sigma, axis = 1) >= 1 #events with at least one track outside the acceptance region
            #print(ak.flatten(delta_x_all[cut_back][abs(delta_x_all[cut_back])>N*sigma], axis = 1))
            bin_heights, bin_borders, _ = axs1[a,b].hist(ak.flatten(delta_x_all[cut_back][(abs(delta_x_all[cut_back])>2*N*sigma) &(abs(delta_x_all[cut_back])<4*N*sigma)][0::]), bins = int(16/0.2),range = [-30, 30], ec=run_number[2], fc='none', histtype='step')
            #print(cut_back)
            #print(data_propagated1_1[cut_eta][cut_back])
            print(ak.ravel(data_propagated1_1[cut_eta][cut_back]))
            cut_back1 = ak.count_nonzero([(abs(delta_x_all[cut_back])>2*N*sigma) &(abs(delta_x_all[cut_back])<4*N*sigma)]) >= 1 #events with at least one track outside the acceptance region
            #conteggio con cut_back1
            background =(bin_heights.sum())
            print(background)
            #let us compute the total number of events that have one hit in this jth eta partition
            #print(ak.count_nonzero(eta_partition == j, axis = 1))
            print(ak.ravel(data_propagated1_1[cut_eta][ak.count_nonzero(cut_eta, axis = 1)>=1]))
            num_events = (ak.num(ak.ravel(data_propagated1_1[ak.count_nonzero(cut_eta, axis = 1)>=1]), axis = 0))
            print(num_events)
            #print(background)
            #print(ak.num(data_propagated1_1[cut_eta], axis = 0))
            props = dict(boxstyle='round', facecolor='white', alpha=0.5)
            # place a text box in upper left in axes coords
            axs[a, b].text(0.05, 0.95, textstr, transform=axs[a, b].transAxes, fontsize=14,
                verticalalignment='top', bbox=props)
            bin_width = (0.2)*10**3
            axs[a, b].set_ylabel(f"Counts/{bin_width}"+r"$\mu m$")
            axs[a, b].set_title("eta partition n. %d" %j)
            data1["partition"].append(j)
            data1["ABS"].append(float(run_number[1]))
            data1["layer"].append(i)
            data1["background"].append(background/(2*num_events)) #normalized at 10sigma
            data1["err_background"].append((np.sqrt(background)/(2*num_events)))
        plt.close(fig)
        fig1.tight_layout()

df = pd.DataFrame.from_dict(data1)
print(df)
for ch in df["layer"].unique():
    fig1, axs1 = plt.subplots(2, 2, figsize = (20,20))
    cut_ch = df["layer"] == ch
    for eta in df["partition"].unique():
        cut_eta = df["partition"][cut_ch] == eta
        a = (eta-1) //2
        b = (eta-1) %2
        axs1[a, b].scatter((df["ABS"][cut_ch][cut_eta]), df["background"][cut_ch][cut_eta])
        axs1[a, b].set_title("layer %d, eta %d" %(ch, eta))
        axs1[a, b].set_xlabel("ABS")
        axs1[a, b].set_ylabel("background")


# In[ ]:


#N=5
entry_stop1 = 600000
run_numbers = [["00000342", "+np.inf", "blue"], ["00000225", "46", "red"],["00000345", "33", "darkred"],["00000254", "22", "darkcyan"], ["00000265", "10", "green"], ["00000267", "6.9", "purple"], ["00000269", "4.6", "orange"]]
#run_numbers = [["00000265", "10", "green"], ["00000267", "6.9", "purple"], ["00000269", "4.6", "orange"]]
data={"efficiency":[], "error":[], "layer":[], "partition":[], "sigma":[], "ABS": [], "multiple": []}
for run_number in run_numbers[1:]:
    index = run_numbers.index(run_number)
    root_file = "/eos/user/f/fnenna/"+run_number[0]+ ".root"
    file = uproot.open(root_file)
    #file.classnames()
    t = file["trackTree"]
    #t.show()
    #chi_quad = t["partialTrackChi2"].array(entry_stop = entry_stop1)
    data_propagated1 = t["partialProphitGlobalX"].array( entry_stop = entry_stop1)
    eta_partition = t["partialProphitEta"].array( entry_stop = entry_stop1)
    #partialTrackChamber = t["partialTrackChamber"].array( entry_stop = entry_stop1)
    data_reconstructed1 = t["rechitGlobalX"].array( entry_stop = entry_stop1)
    eta_partition_rec = t["rechitEta"].array( entry_stop = entry_stop1)
    chamber_rec = t["rechitChamber"].array( entry_stop = entry_stop1)
    chamber = t["partialTrackChamber"].array( entry_stop = entry_stop1)
    for i in range(0, 4):
        for j in range(1, 4):
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
            #delta_x = pairs["0"] - pairs["1"]
            #delta_x = reconstructed - propagated
            a = (j-1)//2
            b = (j-1)%2
            fig_eff, ax1= plt.subplots(1, 1, figsize=( 10, 5 ))
            bin_heights, bin_borders, _ = ax1.hist(ak.flatten(delta_x), bins = int(2**((a+b-1)*(a+b))*len(delta_x)**(1/3)), range=(-4., 4.))
            popt, perr = normal_fit(bin_heights, bin_borders, ax1, bounds_array = ([0,-1, -1],[2*(max(bin_heights))+2000, 1, 10]))
            sigma = popt[2]
            plt.close()
            #ax1.set_xlabel("x residual")
            #ax1.set_ylabel("dN/dx")
            #print("amplitude: %.2d pm %.2d" %(popt[0], perr[0]))
            #print("mu: %1.2f pm %1.2f" %(popt[1], perr[1]))
            #print("sigma: %1.2f pm %1.2f" %((popt[2]), perr[2]))
            #plt.plot(bin_centers, gauss(bin_centers, *popt), 'r-', label='fit')
            #(mu, sigma) = norm.fit(ak.flatten(delta_x))
            for N in range(40, 0, -1):
                fig1 = plt.figure( figsize=( 5, 7 ) )
                sigma = popt[2]
                data_propagated_ch = data_propagated1[cut_ch]
                matching = (np.min(np.abs(delta_x), axis = 1) <= N*sigma)
                #print(delta_x[matching])
                data_propagated1_matching = data_propagated_ch[matching]
                #print(data_propagated1_matching)
                #print(ak.flatten(data_propagated1_matching))
                #print(ak.min(data_propagated1_matching))
                matching_entries = plt.hist(ak.ravel(data_propagated1_matching), bins = int((len(ak.ravel(data_propagated1_matching)))**(1/3)), ec='blue', fc='none', histtype='step', label = "matching propagated x (%d sigma)" %N)
                #ax2.set_xlabel("propagated x [mm] ( < 5sigma)")
                #ax2.set_ylabel("dN/dx")
                entries = plt.hist(ak.ravel(data_propagated_ch[cut]), bins = int(len(ak.ravel(data_propagated1[cut]))), ec='red', fc='none', histtype='step', label = "propagated x")
                #ax3.set_xlabel("propagated x [mm]")
                #ax3.set_ylabel("dN/dx")
                #ax1.legend((u"\u03bc = %1.2f \u00B1 %1.2f" %(popt[1], perr[1]), u"\u03C3: %1.2f \u00B1 %1.2f" %(popt[2], perr[2])), loc = "upper right" )
                #ax2.legend(loc='upper right', bbox_to_anchor=(1., 1.10))
                #print(matching_entries[0])
                #print(entries[0])
                plt.close()
                efficiency = (matching_entries[0].sum())/(entries[0].sum())
                eff_error = np.sqrt((efficiency*(1 - efficiency))/entries[0].sum())
                data["efficiency"].append(efficiency)
                data["error"].append(eff_error)
                data["layer"].append(i)
                data["partition"].append(j)
                data["sigma"].append(sigma)
                data["ABS"].append(float(run_number[1]))
                data["multiple"].append(N)
df1 = pd.DataFrame.from_dict(data)
print(df1)


# In[ ]:


dataframe_def = df1.merge(df, on = ["ABS", "partition", "layer"]) 
print(dataframe_def)


# In[ ]:


def plot_efficiency_chamber(dataframe): 
    print(dataframe)
    k = dataframe["layer"].iloc[0]
    l = dataframe["partition"].iloc[0]
    a = k
    b = l - 1
    axs[a, b].errorbar(dataframe["multiple"]*dataframe["sigma"], dataframe["efficiency"], yerr = dataframe["error"], marker = ".", linestyle = "none", color = "red", label = "efficiency with background")
    axs[a, b].set_title("Efficiency layer %d, eta %d" %(a, b+1))
    axs[a,b].set_xlabel("residual cut [rad]")
    axs[a, b].set_xlim([0., 15])
    axs[a, b].set_ylim([0., 1.1])
    axs[a, b].set_ylabel("efficiency")
    popt, pcov = curve_fit(saturated_fun, dataframe["multiple"]*dataframe["sigma"], dataframe["efficiency"], bounds = ([1, 0., -10.], [2., 20, 10]))
    #bins = np.arange((dataframe["multiple"]*dataframe["sigma"])[0], (dataframe["multiple"]*dataframe["sigma"])[-1], 0.01)
    bins = np.arange(np.min(dataframe["multiple"]*dataframe["sigma"]), 25, 0.001)
    perr = np.sqrt(np.diag(pcov))
    axs[a, b].plot(bins, saturated_fun(bins, *popt), 'r-', label='fit' )
    bins = np.arange(0., 25, 0.001)
    axs[a, b].plot(bins, popt[0]**(-1)*(bins**0), '--', label='fit' )
    background = dataframe["multiple"]*dataframe["background"]/(5)
    error_measEff= np.sqrt(dataframe["error"]**2/((1-background)**2) + (dataframe["err_background"]**2)*((dataframe["multiple"]/5)*((dataframe["efficiency"])-1)**2)/(1-background)**4)
    print((dataframe["efficiency"]-background)/(1-background))
    axs[a, b].errorbar(dataframe["multiple"]*dataframe["sigma"], (dataframe["efficiency"]-background)/(1-background), yerr = error_measEff, marker = ".", linestyle= "none", color = "blue", label = "efficiency without background")
    axs[a, b].legend()
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
    textstr1 = r"at $5\sigma: %.2f$" %dataframe["efficiency"][dataframe["multiple"]==5]
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
    #fig.savefig(run_number + "/efficiencyVSsigma_layer_eta_ch.png")
    
for ABS in dataframe_def["ABS"].unique(): 
    fig, axs = plt.subplots(4, 4, figsize = (50, 50))
    cut_abs = dataframe_def["ABS"] == ABS
    dataframe_def[cut_abs].groupby(["layer", "partition"]).apply(plot_efficiency_chamber)
    fig.suptitle("ABS = %.1f" %ABS)
    fig.tight_layout()
#fig.savefig(run_number + "/efficiencyVSsigma_layer_eta.png")


# In[ ]:


dataframe_def1 = dataframe_def.merge(df2, on = ["ABS", "partition", "layer"]) 
dataframe_def1.to_csv("efficiencyVSrate.csv")
