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

def rejection_fun(x, k, a): 
    return (a)/(1+k*x)

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

dataframe_def1 = pd.read_csv("efficiencyVSrate.csv")
N = 5
#print(dataframe_def)
for layer in dataframe_def1["layer"].unique(): 
    cut_layer = (dataframe_def1["layer"] == layer) & (dataframe_def1["multiple"] == N)
    data_layer = dataframe_def1[cut_layer]
    fig, axs = plt.subplots(1, 3, figsize = (30, 10))
    for partition in dataframe_def1["partition"].unique():
        cut_part = data_layer["partition"] == partition
        data_part = data_layer[cut_part]
        error_measEff= np.sqrt(data_part["error"]**2/(1-data_part["background"])**2 + (data_part["err_background"]**2)*((data_part["efficiency"])-1)**2/(1-data_part["background"])**4)
        axs[partition-1].errorbar((data_part["averageRate"]), data_part["efficiency"], yerr = data_part["error"], xerr = data_part["sigmaRate"], color = "red", label = "no background rejection", linestyle = "none")
        axs[partition-1].errorbar((data_part["averageRate"]), (data_part["efficiency"]-(data_part["background"]))/(1-(data_part["background"])), yerr = error_measEff, xerr = data_part["sigmaRate"], color = "blue", label = "background rejection", linestyle = "none")
        axs[partition-1].set_title("layer %d, eta%d" %(layer, partition))
        axs[partition-1].set_xlabel("rate per strip [Hz](log scale)")
        axs[partition-1].set_ylabel("efficiency")
        axs[partition-1].set_xscale("log")
        axs[partition-1].legend()
        popt2, perr2 = rejection_fit((data_part["averageRate"]), (data_part["efficiency"]-(data_part["background"]))/(1-(data_part["background"])), axs[partition-1])
        textstr = '\n'.join((  
            r'$\epsilon_0=%.3f \pm %.3f$' % (popt2[1], perr2[1]),
            r'$k = %.7f \pm %.7f$' % (popt2[0], perr2[0])))
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        # place a text box in upper left in axes coords
        axs[partition-1].text(0.05, 0.10, textstr, transform=axs[partition -1].transAxes, fontsize=14,
            verticalalignment='top', bbox=props)
    fig.tight_layout()
    plt.savefig("efficiencyVSrateCH%d"%layer)
