import tensortools as tt
import matplotlib.pyplot as plt
import numpy as np

data = np.load('./processed/processed_norm_rmat.npy')

# Fit an ensemble of models, 4 random replicates / optimization runs per model rank
ensemble = tt.Ensemble(fit_method="ncp_hals")
ensemble.fit(data, ranks=range(1, 11), replicates=3)

# Reconstruct tensor - component 10 for test
cp_recon = ensemble.factors(10)[0].full()

# Mean of neuron 10 along all trials - CP est. 
neuron_10 = cp_recon[10,:,:]
print(neuron_10.shape)
mean_neuron_10 = np.mean(neuron_10, axis=0)
print(mean_neuron_10, mean_neuron_10.shape)

# Mean of neuron 10 along all trials - Ground Truth
neuron_10_g = data[10,:,:]
mean_neuron_10_g = np.mean(neuron_10_g, axis=0)

# Plot against
plt.plot(mean_neuron_10, color='blue')
plt.plot(mean_neuron_10_g, color='red')
plt.show()