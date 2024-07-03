#file: informer_OBD_ADMA_ftMS_sl100_ll25_pl5_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_test_0

import numpy as np
import matplotlib.pyplot as plt
import os

file ="informer_OBD_ADMA_ftMS_sl100_ll25_pl5_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_test_0"
data_pred = np.load(f'./{file}/pred.npy')
data_true = np.load(f'./{file}/true.npy')
pred_first_elements = data_pred[:, 0, :]
true_first_elements = data_true[:, 0, :]

abs_diff = np.abs(pred_first_elements - true_first_elements)
bin_edges = np.arange(0, np.max(true_first_elements) + 0.25, 0.25)
bin_indices = np.digitize(true_first_elements, bins=bin_edges)

# Calculate mean absolute error for each bin
mean_abs_error_per_bin = []
bin_centers = []

for i in range(1, len(bin_edges)):
    bin_mask = (bin_indices == i)
    if np.any(bin_mask):
        mean_abs_error = np.mean(abs_diff[bin_mask])
        mean_abs_error_per_bin.append(mean_abs_error)
        bin_centers.append((bin_edges[i-1] + bin_edges[i]) / 2)

# Plot the mean absolute error for each bin
plt.figure(figsize=(10, 6))
plt.plot(bin_centers, mean_abs_error_per_bin, marker='o', linestyle='-')
plt.xlabel('Correvit vehicle slip angle (degrees)')
plt.ylabel('Mean Absolute Error')
plt.title('Mean Absolute Error between prediction vs correvit slip for 0.25 bins')
plt.grid(True)
plt.show()
plt.close()


# Plot histogram with bins of 0.5 degrees
bins = np.arange(0, np.max(abs_diff) + 0.25, 0.25)
hist, bin_edges = np.histogram(abs_diff, bins=bins)
hist_percent = (hist / hist.sum()) * 100

plt.bar(bin_edges[:-1], hist_percent, width=0.25, edgecolor='black', align='center')
plt.xlabel('Absolute Difference vehicle slip angle (degrees)')
plt.ylabel('Frequency (%)')
plt.title('Histogram of Absolute Difference between correvit slip angle vs informer predictions')
plt.grid(True)
for i in range(len(hist)):
    plt.text(bin_edges[i], hist_percent[i] + 0.5, f'{hist_percent[i]:.1f}%', ha='center')
plt.show()
plt.close()


# Plot true vs. predicted values
plt.figure(figsize=(10, 6))
plt.plot(true_first_elements, label='Ground truth')
plt.plot(pred_first_elements, label='Informer prediction')
plt.xlabel('Time stamps')
plt.ylabel('Slip angle in degrees')
plt.title('True vs Informer vehicle slip angle prediction')
plt.legend()
plt.show()
