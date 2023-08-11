#Create a detector without consulting a textbook and call it a CFAR
#Is the false alarm rate constant?
#Sean Mason
#July 30, 2023

import numpy as np
import matplotlib.pyplot as plt
import pdb

#Configurable parameters
#Number of samples to observe to set threshold
nsamp_obsv = 1000

#Number of samples to evaluate
nsamp_eval = 10e6

#Number of times that we're going to run this experiment
reps = 1000

#Functions
#Generate receive samples
def generate_rx_samps(nsamp_obsv, noise_pow_dBm):

    #Convert noise_pow_dBm to linear
    noise_pow_mW = 10**(0.1*noise_pow_dBm)

    #Create samples of noise
    rx_samps_re = np.random.normal(0, 1, size = int(nsamp_obsv)) #Real
    rx_samps_im = np.random.normal(0, 1, size = int(nsamp_obsv)) #Imaginary

    #Define the power of what you've received
    rx_pow = rx_samps_re**2 + rx_samps_im**2

    #Normalize
    rx_pow = np.sqrt(noise_pow_mW) * rx_pow/np.std(rx_pow)

    return(rx_pow)

#Preallocate empirical false alarm rate
emp_fa_rate = np.zeros(reps)

#Step through reps
for rep in np.arange(0, reps):

    #Create the samples to observe
    #Since it is difficult to know what the noise power will be before you deploy your detector,
    #measure it in situ. This requires grabbing a sample and usually assuming that there is no
    #signal in your sample, or at least that almost all of what you grab won't have a signal in it
    #We are simulating baseband processing, so define real and imaginary channels
    #Define the noise power in dBm
    noise_pow_dBm = np.random.uniform(low=-130, high=-90, size=1)
    baseline_samples = generate_rx_samps(nsamp_obsv, noise_pow_dBm)

    #Compute a threshold. 

    # Median plus some offset
    # thresh_level = np.median(baseline_samples) + 6 #Three standard deviations of a chi-squared RV with 2 degrees of freedom

    #Taking the median plus three standard deviations often causes false alarms even in this small set of 1000 samples 
    #Make this more robust by taking the maximum and multiplying by 1.1
    thresh_level = 1.1*np.max(baseline_samples)

    #Generate the new samples from the same distribution
    new_rx_samps = generate_rx_samps(nsamp_eval, noise_pow_dBm)

    #Count how many samples are above the threshold
    num_over_thresh = np.sum(new_rx_samps > thresh_level)

    #Empirical false alarm rate
    emp_fa_rate[rep] = num_over_thresh/nsamp_eval


#Plot the empirical false alarm rate
plt.semilogy(emp_fa_rate)
plt.grid()
plt.show()


