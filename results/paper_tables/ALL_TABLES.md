# Paper tables

## T1_dataset_statistics

| Polymer | Raw samples | After deduplication | Train | Val | Test | Score mean (train) | Score SD (train) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PET | 441,978 | 381,065 | 304,851 | 38,107 | 38,107 | -27.6782 | 14.1043 |
| PE | 715,508 | 715,508 | 572,406 | 71,551 | 71,551 | -24.9700 | 10.2078 |
| PP | 433,487 | 433,487 | 346,789 | 43,349 | 43,349 | -19.4064 | 10.2462 |
| PS | 405,827 | 405,827 | 324,661 | 40,583 | 40,583 | -15.3918 | 8.8599 |
| PVC | 208,608 | 208,608 | 166,886 | 20,861 | 20,861 | -31.8166 | 10.6002 |
| Nylon | 142,614 | 142,614 | 114,090 | 14,262 | 14,262 | -36.9538 | 11.7548 |

Normalisation statistics are computed from the training split only.


## T2_split_comparison

| Polymer | Split | Exact test-in-train (%) | Mean NN identity (%) | NN <= 2 mutations (%) |
| --- | --- | --- | --- | --- |
| Nylon | clustered | 0.0 | 75.72 | 38.0 |
| Nylon | random | 0.0 | 87.36 | 89.0 |
| PE | clustered | 0.0 | 64.17 | 13.67 |
| PE | random | 0.0 | 72.19 | 37.0 |
| PET | clustered | 0.0 | 68.31 | 21.67 |
| PET | random | 0.0 | 77.53 | 52.67 |
| PP | clustered | 0.0 | 65.83 | 15.67 |
| PP | random | 0.0 | 76.36 | 50.33 |
| PS | clustered | 0.0 | 65.36 | 17.67 |
| PS | random | 0.0 | 74.33 | 44.33 |
| PVC | clustered | 0.0 | 72.92 | 34.0 |
| PVC | random | 0.0 | 83.08 | 73.67 |

## T3_full_metrics_clustered

| Polymer | Architecture | Train R2 | Train RMSE | Train MAE | Validation R2 | Validation RMSE | Validation MAE | Test R2 | Test RMSE | Test MAE | Seeds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Nylon | cnn | 0.9925 ± 0.0002 | 1.021 ± 0.013 | 0.789 ± 0.009 | 0.8840 ± 0.0009 | 4.266 ± 0.017 | 3.134 ± 0.020 | 0.8520 ± 0.0023 | 4.492 ± 0.035 | 3.306 ± 0.028 | 5 |
| Nylon | encdec | 0.8168 ± 0.0064 | 5.031 ± 0.087 | 3.810 ± 0.066 | 0.7972 ± 0.0025 | 5.640 ± 0.034 | 4.287 ± 0.029 | 0.7560 ± 0.0019 | 5.768 ± 0.023 | 4.396 ± 0.016 | 5 |
| Nylon | lstm | 0.7473 ± 0.0140 | 5.908 ± 0.166 | 4.508 ± 0.134 | 0.7570 ± 0.0074 | 6.174 ± 0.095 | 4.726 ± 0.089 | 0.7079 ± 0.0067 | 6.310 ± 0.073 | 4.895 ± 0.077 | 5 |
| Nylon | lstm_vae | 0.7617 ± 0.0148 | 5.736 ± 0.177 | 4.352 ± 0.153 | 0.7696 ± 0.0096 | 6.010 ± 0.125 | 4.567 ± 0.115 | 0.7176 ± 0.0131 | 6.204 ± 0.143 | 4.743 ± 0.143 | 5 |
| PE | cnn | 0.9865 ± 0.0002 | 1.185 ± 0.010 | 0.905 ± 0.007 | 0.8521 ± 0.0003 | 3.888 ± 0.004 | 2.820 ± 0.005 | 0.8556 ± 0.0007 | 3.814 ± 0.009 | 2.768 ± 0.008 | 5 |
| PE | encdec | 0.8561 ± 0.0008 | 3.872 ± 0.011 | 2.915 ± 0.006 | 0.8005 ± 0.0015 | 4.516 ± 0.017 | 3.378 ± 0.013 | 0.8034 ± 0.0013 | 4.450 ± 0.015 | 3.330 ± 0.011 | 5 |
| PE | lstm | 0.8200 ± 0.0017 | 4.331 ± 0.020 | 3.229 ± 0.017 | 0.7847 ± 0.0012 | 4.691 ± 0.013 | 3.495 ± 0.010 | 0.7884 ± 0.0012 | 4.617 ± 0.013 | 3.440 ± 0.013 | 5 |
| PE | lstm_vae | 0.7600 ± 0.0013 | 5.001 ± 0.014 | 3.745 ± 0.007 | 0.7411 ± 0.0016 | 5.145 ± 0.016 | 3.860 ± 0.010 | 0.7424 ± 0.0020 | 5.094 ± 0.020 | 3.829 ± 0.012 | 5 |
| PET | cnn | 0.9920 ± 0.0003 | 1.260 ± 0.022 | 0.964 ± 0.018 | 0.8921 ± 0.0010 | 4.646 ± 0.022 | 3.328 ± 0.024 | 0.8912 ± 0.0008 | 4.665 ± 0.017 | 3.350 ± 0.023 | 5 |
| PET | encdec | 0.9342 ± 0.0038 | 3.618 ± 0.104 | 2.703 ± 0.072 | 0.8530 ± 0.0011 | 5.422 ± 0.020 | 3.965 ± 0.021 | 0.8480 ± 0.0012 | 5.515 ± 0.021 | 4.033 ± 0.020 | 5 |
| PET | lstm | 0.8812 ± 0.0073 | 4.860 ± 0.151 | 3.571 ± 0.110 | 0.8243 ± 0.0022 | 5.927 ± 0.037 | 4.296 ± 0.030 | 0.8243 ± 0.0025 | 5.928 ± 0.042 | 4.302 ± 0.032 | 5 |
| PET | lstm_vae | 0.7984 ± 0.0018 | 6.332 ± 0.028 | 4.779 ± 0.031 | 0.7840 ± 0.0009 | 6.572 ± 0.014 | 4.949 ± 0.027 | 0.7779 ± 0.0012 | 6.665 ± 0.018 | 5.012 ± 0.017 | 5 |
| PP | cnn | 0.9893 ± 0.0003 | 1.061 ± 0.014 | 0.816 ± 0.010 | 0.8677 ± 0.0012 | 3.708 ± 0.017 | 2.714 ± 0.015 | 0.8639 ± 0.0011 | 3.787 ± 0.015 | 2.772 ± 0.016 | 5 |
| PP | encdec | 0.8950 ± 0.0012 | 3.320 ± 0.019 | 2.520 ± 0.013 | 0.8155 ± 0.0006 | 4.379 ± 0.008 | 3.296 ± 0.008 | 0.8155 ± 0.0017 | 4.408 ± 0.021 | 3.330 ± 0.013 | 5 |
| PP | lstm | 0.8433 ± 0.0169 | 4.051 ± 0.226 | 3.041 ± 0.173 | 0.8003 ± 0.0018 | 4.556 ± 0.021 | 3.408 ± 0.021 | 0.8007 ± 0.0029 | 4.582 ± 0.033 | 3.445 ± 0.034 | 5 |
| PP | lstm_vae | 0.7608 ± 0.0038 | 5.011 ± 0.040 | 3.847 ± 0.027 | 0.7490 ± 0.0038 | 5.107 ± 0.038 | 3.918 ± 0.026 | 0.7493 ± 0.0040 | 5.139 ± 0.041 | 3.949 ± 0.034 | 5 |
| PS | cnn | 0.9878 ± 0.0003 | 0.978 ± 0.012 | 0.756 ± 0.010 | 0.8346 ± 0.0014 | 3.630 ± 0.016 | 2.702 ± 0.022 | 0.8373 ± 0.0017 | 3.645 ± 0.019 | 2.699 ± 0.031 | 5 |
| PS | encdec | 0.9079 ± 0.0024 | 2.689 ± 0.035 | 2.070 ± 0.028 | 0.7693 ± 0.0021 | 4.286 ± 0.020 | 3.277 ± 0.016 | 0.7745 ± 0.0004 | 4.291 ± 0.004 | 3.277 ± 0.008 | 5 |
| PS | lstm | 0.8029 ± 0.0032 | 3.934 ± 0.031 | 3.007 ± 0.025 | 0.7467 ± 0.0022 | 4.491 ± 0.019 | 3.424 ± 0.013 | 0.7545 ± 0.0021 | 4.477 ± 0.019 | 3.434 ± 0.013 | 5 |
| PS | lstm_vae | 0.7014 ± 0.0134 | 4.841 ± 0.108 | 3.772 ± 0.091 | 0.6887 ± 0.0095 | 4.979 ± 0.076 | 3.880 ± 0.072 | 0.6940 ± 0.0089 | 4.999 ± 0.073 | 3.912 ± 0.061 | 5 |
| PVC | cnn | 0.9884 ± 0.0002 | 1.141 ± 0.008 | 0.885 ± 0.006 | 0.7864 ± 0.0013 | 4.678 ± 0.014 | 3.479 ± 0.015 | 0.8057 ± 0.0015 | 4.772 ± 0.019 | 3.531 ± 0.015 | 5 |
| PVC | encdec | 0.8604 ± 0.0077 | 3.959 ± 0.107 | 3.019 ± 0.078 | 0.6820 ± 0.0027 | 5.708 ± 0.025 | 4.311 ± 0.014 | 0.7234 ± 0.0036 | 5.694 ± 0.037 | 4.320 ± 0.037 | 5 |
| PVC | lstm | 0.6992 ± 0.0061 | 5.814 ± 0.059 | 4.428 ± 0.053 | 0.6325 ± 0.0059 | 6.137 ± 0.049 | 4.709 ± 0.040 | 0.6770 ± 0.0052 | 6.153 ± 0.049 | 4.736 ± 0.045 | 5 |
| PVC | lstm_vae | 0.6950 ± 0.0135 | 5.853 ± 0.130 | 4.465 ± 0.109 | 0.6320 ± 0.0051 | 6.140 ± 0.042 | 4.709 ± 0.044 | 0.6769 ± 0.0061 | 6.154 ± 0.058 | 4.756 ± 0.046 | 5 |

Mean ± standard deviation over the seeds. RMSE and MAE are in the units of the binding score.


## T3b_full_metrics_random

| Polymer | Architecture | Train R2 | Train RMSE | Train MAE | Validation R2 | Validation RMSE | Validation MAE | Test R2 | Test RMSE | Test MAE | Seeds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Nylon | cnn | 0.9920 ± 0.0002 | 1.057 ± 0.012 | 0.818 ± 0.011 | 0.9549 ± 0.0004 | 2.510 ± 0.011 | 1.861 ± 0.008 | 0.9559 ± 0.0007 | 2.491 ± 0.021 | 1.849 ± 0.018 | 5 |
| Nylon | encdec | 0.8139 ± 0.0068 | 5.100 ± 0.093 | 3.869 ± 0.075 | 0.8008 ± 0.0052 | 5.274 ± 0.069 | 3.996 ± 0.054 | 0.8046 ± 0.0055 | 5.245 ± 0.074 | 3.978 ± 0.070 | 5 |
| Nylon | lstm | 0.7599 ± 0.0199 | 5.790 ± 0.239 | 4.424 ± 0.202 | 0.7521 ± 0.0186 | 5.880 ± 0.220 | 4.488 ± 0.193 | 0.7565 ± 0.0190 | 5.851 ± 0.227 | 4.479 ± 0.201 | 5 |
| Nylon | lstm_vae | 0.7652 ± 0.0095 | 5.729 ± 0.115 | 4.337 ± 0.101 | 0.7593 ± 0.0070 | 5.798 ± 0.084 | 4.386 ± 0.078 | 0.7641 ± 0.0085 | 5.762 ± 0.103 | 4.368 ± 0.094 | 5 |
| PE | cnn | 0.9852 ± 0.0001 | 1.239 ± 0.005 | 0.949 ± 0.002 | 0.9445 ± 0.0003 | 2.399 ± 0.006 | 1.755 ± 0.006 | 0.9446 ± 0.0004 | 2.400 ± 0.009 | 1.760 ± 0.005 | 5 |
| PE | encdec | 0.8501 ± 0.0024 | 3.941 ± 0.032 | 2.963 ± 0.025 | 0.8334 ± 0.0022 | 4.158 ± 0.027 | 3.107 ± 0.022 | 0.8367 ± 0.0023 | 4.119 ± 0.030 | 3.092 ± 0.024 | 5 |
| PE | lstm | 0.8163 ± 0.0015 | 4.363 ± 0.018 | 3.250 ± 0.016 | 0.8078 ± 0.0017 | 4.465 ± 0.019 | 3.313 ± 0.016 | 0.8119 ± 0.0009 | 4.421 ± 0.010 | 3.291 ± 0.008 | 5 |
| PE | lstm_vae | 0.7580 ± 0.0020 | 5.008 ± 0.021 | 3.754 ± 0.023 | 0.7540 ± 0.0021 | 5.051 ± 0.022 | 3.778 ± 0.023 | 0.7578 ± 0.0021 | 5.017 ± 0.022 | 3.763 ± 0.024 | 5 |
| PET | cnn | 0.9915 ± 0.0003 | 1.301 ± 0.021 | 0.997 ± 0.018 | 0.9621 ± 0.0005 | 2.763 ± 0.018 | 1.993 ± 0.024 | 0.9615 ± 0.0008 | 2.773 ± 0.029 | 2.000 ± 0.026 | 5 |
| PET | encdec | 0.9280 ± 0.0014 | 3.784 ± 0.037 | 2.825 ± 0.025 | 0.9037 ± 0.0009 | 4.403 ± 0.020 | 3.245 ± 0.015 | 0.9014 ± 0.0014 | 4.435 ± 0.031 | 3.277 ± 0.019 | 5 |
| PET | lstm | 0.8740 ± 0.0060 | 5.004 ± 0.120 | 3.675 ± 0.084 | 0.8609 ± 0.0045 | 5.290 ± 0.086 | 3.870 ± 0.059 | 0.8581 ± 0.0040 | 5.321 ± 0.076 | 3.893 ± 0.051 | 5 |
| PET | lstm_vae | 0.7947 ± 0.0045 | 6.388 ± 0.070 | 4.828 ± 0.067 | 0.7945 ± 0.0044 | 6.431 ± 0.070 | 4.855 ± 0.070 | 0.7912 ± 0.0040 | 6.454 ± 0.062 | 4.885 ± 0.062 | 5 |
| PP | cnn | 0.9885 ± 0.0002 | 1.099 ± 0.012 | 0.846 ± 0.010 | 0.9503 ± 0.0006 | 2.295 ± 0.014 | 1.688 ± 0.013 | 0.9497 ± 0.0006 | 2.295 ± 0.014 | 1.692 ± 0.015 | 5 |
| PP | encdec | 0.8881 ± 0.0035 | 3.423 ± 0.053 | 2.596 ± 0.040 | 0.8635 ± 0.0027 | 3.805 ± 0.037 | 2.869 ± 0.029 | 0.8630 ± 0.0026 | 3.789 ± 0.036 | 2.863 ± 0.028 | 5 |
| PP | lstm | 0.8419 ± 0.0119 | 4.068 ± 0.152 | 3.055 ± 0.117 | 0.8299 ± 0.0087 | 4.246 ± 0.107 | 3.177 ± 0.084 | 0.8294 ± 0.0082 | 4.227 ± 0.101 | 3.167 ± 0.082 | 5 |
| PP | lstm_vae | 0.7614 ± 0.0023 | 5.000 ± 0.025 | 3.845 ± 0.023 | 0.7584 ± 0.0020 | 5.062 ± 0.021 | 3.875 ± 0.020 | 0.7593 ± 0.0021 | 5.022 ± 0.022 | 3.862 ± 0.021 | 5 |
| PS | cnn | 0.9869 ± 0.0004 | 1.017 ± 0.016 | 0.787 ± 0.012 | 0.9404 ± 0.0013 | 2.183 ± 0.024 | 1.612 ± 0.023 | 0.9391 ± 0.0015 | 2.193 ± 0.027 | 1.620 ± 0.030 | 5 |
| PS | encdec | 0.8980 ± 0.0031 | 2.835 ± 0.043 | 2.183 ± 0.033 | 0.8540 ± 0.0024 | 3.416 ± 0.028 | 2.611 ± 0.022 | 0.8519 ± 0.0023 | 3.420 ± 0.027 | 2.616 ± 0.023 | 5 |
| PS | lstm | 0.7934 ± 0.0015 | 4.035 ± 0.015 | 3.091 ± 0.012 | 0.7825 ± 0.0014 | 4.169 ± 0.014 | 3.181 ± 0.011 | 0.7811 ± 0.0012 | 4.158 ± 0.011 | 3.181 ± 0.009 | 5 |
| PS | lstm_vae | 0.7008 ± 0.0075 | 4.855 ± 0.060 | 3.791 ± 0.061 | 0.7016 ± 0.0066 | 4.883 ± 0.054 | 3.800 ± 0.056 | 0.6996 ± 0.0071 | 4.871 ± 0.057 | 3.799 ± 0.058 | 5 |
| PVC | cnn | 0.9875 ± 0.0003 | 1.184 ± 0.014 | 0.919 ± 0.011 | 0.9328 ± 0.0011 | 2.739 ± 0.023 | 2.025 ± 0.017 | 0.9334 ± 0.0007 | 2.735 ± 0.015 | 2.025 ± 0.018 | 5 |
| PVC | encdec | 0.8453 ± 0.0088 | 4.158 ± 0.120 | 3.171 ± 0.087 | 0.7959 ± 0.0072 | 4.770 ± 0.085 | 3.615 ± 0.065 | 0.7987 ± 0.0060 | 4.754 ± 0.071 | 3.598 ± 0.051 | 5 |
| PVC | lstm | 0.7029 ± 0.0031 | 5.764 ± 0.030 | 4.403 ± 0.028 | 0.6943 ± 0.0033 | 5.839 ± 0.032 | 4.453 ± 0.027 | 0.6979 ± 0.0033 | 5.824 ± 0.032 | 4.450 ± 0.026 | 5 |
| PVC | lstm_vae | 0.6969 ± 0.0103 | 5.822 ± 0.099 | 4.447 ± 0.085 | 0.6879 ± 0.0075 | 5.900 ± 0.070 | 4.501 ± 0.065 | 0.6939 ± 0.0083 | 5.862 ± 0.079 | 4.480 ± 0.071 | 5 |

Mean ± standard deviation over the seeds. RMSE and MAE are in the units of the binding score.


## T4_architecture_significance

Reference architecture: **cnn** (highest mean test R2 across all polymers). Paired across seeds: same split, same data, only the architecture differs.

| Polymer | Comparison | Difference | 95% CI | p (paired t) | Significant |
| --- | --- | --- | --- | --- | --- |
| PET | cnn vs lstm | +0.06688 | [0.064183, 0.069576] | <1e-6 | yes |
| PET | cnn vs lstm_vae | +0.11328 | [0.111866, 0.114702] | <1e-6 | yes |
| PET | cnn vs encdec | +0.04322 | [0.040909, 0.045524] | 1.00e-06 | yes |
| PE | cnn vs lstm | +0.06715 | [0.066254, 0.068042] | <1e-6 | yes |
| PE | cnn vs lstm_vae | +0.11316 | [0.110018, 0.116302] | <1e-6 | yes |
| PE | cnn vs encdec | +0.05216 | [0.050802, 0.053512] | <1e-6 | yes |
| PP | cnn vs lstm | +0.06315 | [0.059099, 0.067206] | 2.00e-06 | yes |
| PP | cnn vs lstm_vae | +0.11453 | [0.109866, 0.119193] | <1e-6 | yes |
| PP | cnn vs encdec | +0.04835 | [0.046372, 0.050321] | <1e-6 | yes |
| PS | cnn vs lstm | +0.08277 | [0.079546, 0.086002] | <1e-6 | yes |
| PS | cnn vs lstm_vae | +0.14331 | [0.13192, 0.154693] | 4.00e-06 | yes |
| PS | cnn vs encdec | +0.06273 | [0.06068, 0.064783] | <1e-6 | yes |
| PVC | cnn vs lstm | +0.12868 | [0.122405, 0.134947] | 1.00e-06 | yes |
| PVC | cnn vs lstm_vae | +0.12877 | [0.120493, 0.137045] | 2.00e-06 | yes |
| PVC | cnn vs encdec | +0.08229 | [0.078683, 0.085897] | <1e-6 | yes |
| Nylon | cnn vs lstm | +0.14407 | [0.137512, 0.15063] | <1e-6 | yes |
| Nylon | cnn vs lstm_vae | +0.13440 | [0.116937, 0.151863] | 2.80e-05 | yes |
| Nylon | cnn vs encdec | +0.09599 | [0.091919, 0.10007] | <1e-6 | yes |

## T5_selectivity

| Optimised for \ Scored by | Nylon | PE | PET | PP | PS | PVC | Selectivity index |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Nylon | **-75.9659** | -37.993 | -39.3065 | -29.842 | -25.2958 | -45.7421 | 40.33 |
| PE | -46.1378 | **-60.5163** | -34.7322 | -26.9635 | -23.3742 | -40.0334 | 26.2681 |
| PET | -45.2051 | -34.1003 | **-64.8443** | -29.3002 | -25.2909 | -37.4989 | 30.5652 |
| PP | -45.2245 | -33.8258 | -36.0896 | **-57.0971** | -22.8474 | -39.315 | 21.6366 |
| PS | -42.7414 | -30.6754 | -33.6296 | -28.5296 | **-43.8936** | -36.2541 | 9.5276 |
| PVC | -42.0434 | -23.5003 | -26.5638 | -21.2328 | -19.0941 | **-69.2302** | 42.7433 |

## T6_novelty

| Polymer | n | Exact matches | NN Hamming (mean) | NN identity (mean) | >=90% identity |
| --- | --- | --- | --- | --- | --- |
| PET | 30 | 0 | 5.333 | %55.56 | %0.0 |
| PE | 30 | 0 | 4.467 | %62.78 | %0.0 |
| PP | 30 | 0 | 5.0 | %58.33 | %0.0 |
| PS | 30 | 0 | 5.033 | %58.06 | %0.0 |
| PVC | 30 | 0 | 5.333 | %55.56 | %0.0 |
| Nylon | 30 | 0 | 4.533 | %62.22 | %0.0 |

## T7_physicochemical

| Polymer | gravy | aromatic_fraction | positive_fraction | negative_fraction | net_charge_ph7_4 | isoelectric_point | tryptophan_count | arginine_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PET | -0.8233 (r=0.301298) | 0.5306 (r=-0.470557) | 0.1583 (r=-0.228247) | 0.0194 (r=0.141792) | 1.6688 (r=-0.280633) | 10.108 (r=-0.167763) | 4.4 (r=-0.571573) | 1.6 (r=-0.279506) |
| PE | -0.8103 (r=-0.413263) | 0.3805 (r=0.751675) | 0.1917 (r=-0.292085) | 0.0361 (r=-0.239336) | 1.8945 (r=-0.116602) | 10.9158 (r=-0.047644) | 3.0333 (r=0.788613) | 1.7667 (r=0.096451) |
| PP | -0.3444 (r=-0.06783) | 0.4556 (r=-0.37163) | 0.1444 (r=0.23681) | 0.0389 (r=0.018882) | 1.2855 (r=0.188427) | 9.9447 (r=0.076699) | 3.6333 (r=-0.483885) | 1.2333 (r=0.092474) |
| PS | -0.6181 (r=-0.134701) | 0.3972 (r=-0.190194) | 0.1917 (r=0.374293) | 0.0361 (r=-0.004426) | 1.8823 (r=0.332983) | 10.8194 (r=0.168838) | 3.3667 (r=-0.188949) | 1.6333 (r=0.324856) |
| PVC | -1.1344 (r=0.433694) | 0.175 (r=0.537767) | 0.3139 (r=-0.701388) | 0.0361 (r=0.252212) | 3.3533 (r=-0.692766) | 11.6779 (r=-0.493718) | 1.6 (r=0.491714) | 3.3 (r=-0.676482) |
| Nylon | -1.5036 (r=-0.080336) | 0.3944 (r=0.061514) | 0.2667 (r=-0.240235) | 0.0306 (r=0.277085) | 2.8656 (r=-0.348753) | 11.826 (r=-0.255585) | 4.1 (r=0.018394) | 2.8 (r=-0.228646) |

## T8_position_importance

| Polymer | P1 | P2 | P3 | P4 | P5 | P6 | P7 | P8 | P9 | P10 | P11 | P12 | Most important | Consensus |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PET | 5.9146 | 5.0585 | 3.663 | 3.7676 | 3.2277 | 3.3675 | 5.1244 | 4.1937 | 3.8463 | 3.1625 | 1.8975 | 2.2506 | [1, 7, 2, 8] | WWWWWWWWWWWW |
| PE | 1.7349 | 1.4265 | 1.9973 | 1.7568 | 1.6434 | 2.8619 | 4.1697 | 2.5703 | 1.0661 | 1.0794 | 4.5463 | 3.7366 | [11, 7, 12, 6] | WWWWWWWWWMWW |
| PP | 3.6318 | 1.6889 | 2.1826 | 1.8716 | 1.832 | 2.3437 | 2.6339 | 2.3662 | 3.2904 | 2.8827 | 1.6401 | 3.1439 | [1, 9, 12, 10] | WWWWWWWWWWRW |
| PS | 1.6152 | 1.693 | 1.8655 | 2.0226 | 2.6702 | 2.3034 | 2.9552 | 2.6866 | 2.5661 | 2.9478 | 2.4272 | 2.1455 | [7, 10, 8, 5] | WWHWWWWWWWWR |
| PVC | 2.5593 | 3.4575 | 2.8841 | 3.5387 | 2.6834 | 2.6565 | 2.841 | 3.125 | 2.5954 | 2.3863 | 2.4207 | 2.2721 | [4, 2, 8, 3] | RWRWRRRRRRRR |
| Nylon | 1.4737 | 5.2898 | 2.2245 | 3.1138 | 1.8653 | 1.4656 | 3.345 | 2.3777 | 2.0536 | 2.5934 | 2.1622 | 2.4864 | [2, 7, 4, 10] | WWWWRRWWWRWW |

## T9_jain_benchmark

### Benchmark against Jain et al. (2025)

Prior architecture: unidirectional LSTM, 2 layers, hidden size 512, one-hot input, as described in that paper. 3 seeds, 60 epochs, cosine annealing.

Within each architecture, the only thing that changes between conditions is how the data is split.

| Polymer | Condition | n | Exact leakage % | Test R² | Test RMSE |
| --- | --- | --- | --- | --- | --- |
| PET | prior / A_ham_rastgele | 441,978 | 20.969 | 0.8890 ± 0.0012 | 4.780 ± 0.026 |
| PET | this work / A_ham_rastgele | 441,978 | 20.969 | 0.9704 ± 0.0004 | 2.470 ± 0.017 |
| PET | prior / B_temiz_rastgele | 381,065 | 0.0 | 0.8722 ± 0.0014 | 5.049 ± 0.028 |
| PET | this work / B_temiz_rastgele | 381,065 | 0.0 | 0.9616 ± 0.0010 | 2.768 ± 0.037 |
| PET | prior / C_temiz_kumelenmis | 381,065 | 0.0 | 0.8396 ± 0.0004 | 5.664 ± 0.007 |
| PET | this work / C_temiz_kumelenmis | 381,065 | 0.0 | 0.8918 ± 0.0014 | 4.652 ± 0.031 |
| PET | _as reported by Jain et al._ | — | — | _0.9755_ | _2.23_ |
| PE | prior / A_ham_rastgele | 715,508 | 0.0 | 0.8183 ± 0.0032 | 4.345 ± 0.038 |
| PE | this work / A_ham_rastgele | 715,508 | 0.0 | 0.9444 ± 0.0004 | 2.404 ± 0.008 |
| PE | prior / B_temiz_rastgele | 715,508 | 0.0 | 0.8183 ± 0.0032 | 4.345 ± 0.038 |
| PE | this work / B_temiz_rastgele | 715,508 | 0.0 | 0.9444 ± 0.0004 | 2.404 ± 0.008 |
| PE | prior / C_temiz_kumelenmis | 715,508 | 0.0 | 0.7930 ± 0.0019 | 4.566 ± 0.021 |
| PE | this work / C_temiz_kumelenmis | 715,508 | 0.0 | 0.8561 ± 0.0003 | 3.807 ± 0.004 |
| PE | _as reported by Jain et al._ | — | — | _0.9517_ | _2.24_ |
| PP | prior / A_ham_rastgele | 433,487 | 0.0 | 0.8323 ± 0.0002 | 4.192 ± 0.002 |
| PP | this work / A_ham_rastgele | 433,487 | 0.0 | 0.9497 ± 0.0009 | 2.297 ± 0.021 |
| PP | prior / B_temiz_rastgele | 433,487 | 0.0 | 0.8323 ± 0.0002 | 4.192 ± 0.002 |
| PP | this work / B_temiz_rastgele | 433,487 | 0.0 | 0.9497 ± 0.0009 | 2.297 ± 0.021 |
| PP | prior / C_temiz_kumelenmis | 433,487 | 0.0 | 0.8077 ± 0.0015 | 4.501 ± 0.017 |
| PP | this work / C_temiz_kumelenmis | 433,487 | 0.0 | 0.8635 ± 0.0005 | 3.791 ± 0.007 |
| PP | _as reported by Jain et al._ | — | — | _0.9640_ | _1.94_ |
| PVC | prior / A_ham_rastgele | 208,608 | 0.0 | 0.7276 ± 0.0031 | 5.530 ± 0.031 |
| PVC | this work / A_ham_rastgele | 208,608 | 0.0 | 0.9329 ± 0.0004 | 2.745 ± 0.009 |
| PVC | prior / B_temiz_rastgele | 208,608 | 0.0 | 0.7276 ± 0.0031 | 5.530 ± 0.031 |
| PVC | this work / B_temiz_rastgele | 208,608 | 0.0 | 0.9329 ± 0.0004 | 2.745 ± 0.009 |
| PVC | prior / C_temiz_kumelenmis | 208,608 | 0.0 | 0.6956 ± 0.0055 | 5.973 ± 0.054 |
| PVC | this work / C_temiz_kumelenmis | 208,608 | 0.0 | 0.8070 ± 0.0013 | 4.757 ± 0.016 |
| PVC | _as reported by Jain et al._ | — | — | _0.9554_ | _2.23_ |
| Nylon | prior / A_ham_rastgele | 142,614 | 0.0 | 0.7719 ± 0.0052 | 5.667 ± 0.064 |
| Nylon | this work / A_ham_rastgele | 142,614 | 0.0 | 0.9557 ± 0.0007 | 2.498 ± 0.020 |
| Nylon | prior / B_temiz_rastgele | 142,614 | 0.0 | 0.7719 ± 0.0052 | 5.667 ± 0.064 |
| Nylon | this work / B_temiz_rastgele | 142,614 | 0.0 | 0.9557 ± 0.0007 | 2.498 ± 0.020 |
| Nylon | prior / C_temiz_kumelenmis | 142,614 | 0.0 | 0.7419 ± 0.0030 | 5.932 ± 0.034 |
| Nylon | this work / C_temiz_kumelenmis | 142,614 | 0.0 | 0.8516 ± 0.0026 | 4.499 ± 0.039 |
| Nylon | _as reported by Jain et al._ | — | — | _0.9774_ | _1.79_ |

#### Reading

- **PET**: 0.8890 under their condition (they report 0.9755), 0.8396 under the identity-aware protocol. The gap of +0.0494 comes from the split alone.
- **PE**: 0.8183 under their condition (they report 0.9517), 0.7930 under the identity-aware protocol. The gap of +0.0253 comes from the split alone.
- **PP**: 0.8323 under their condition (they report 0.9640), 0.8077 under the identity-aware protocol. The gap of +0.0246 comes from the split alone.
- **PVC**: 0.7276 under their condition (they report 0.9554), 0.6956 under the identity-aware protocol. The gap of +0.0320 comes from the split alone.
- **Nylon**: 0.7719 under their condition (they report 0.9774), 0.7419 under the identity-aware protocol. The gap of +0.0300 comes from the split alone.

#### Architecture comparison (cnn vs prior)

Same data, same split, same seeds; only the architecture differs. This isolates the architectural contribution from the evaluation protocol.

| Polymer | Condition | prior | cnn | Difference | p |
| --- | --- | --- | --- | --- | --- |
| PET | A_ham_rastgele | 0.8890 | 0.9704 | +0.0813 | 0.00007 |
| PET | B_temiz_rastgele | 0.8722 | 0.9616 | +0.0894 | 0.00014 |
| PET | C_temiz_kumelenmis | 0.8396 | 0.8918 | +0.0522 | 0.00023 |
| PE | A_ham_rastgele | 0.8183 | 0.9444 | +0.1260 | 0.00019 |
| PE | B_temiz_rastgele | 0.8183 | 0.9444 | +0.1260 | 0.00019 |
| PE | C_temiz_kumelenmis | 0.7930 | 0.8561 | +0.0631 | 0.00022 |
| PP | A_ham_rastgele | 0.8323 | 0.9497 | +0.1173 | 0.00002 |
| PP | B_temiz_rastgele | 0.8323 | 0.9497 | +0.1173 | 0.00002 |
| PP | C_temiz_kumelenmis | 0.8077 | 0.8635 | +0.0558 | 0.00027 |
| PVC | A_ham_rastgele | 0.7276 | 0.9329 | +0.2052 | 0.00010 |
| PVC | B_temiz_rastgele | 0.7276 | 0.9329 | +0.2052 | 0.00010 |
| PVC | C_temiz_kumelenmis | 0.6956 | 0.8070 | +0.1113 | 0.00051 |
| Nylon | A_ham_rastgele | 0.7719 | 0.9557 | +0.1838 | 0.00034 |
| Nylon | B_temiz_rastgele | 0.7719 | 0.9557 | +0.1838 | 0.00034 |
| Nylon | C_temiz_kumelenmis | 0.7419 | 0.8516 | +0.1097 | 0.00065 |

This architecture is better on 5/5 polymers under their condition (A) and 5/5 under the identity-aware split (C); every comparison is statistically significant. The SIZE of the advantage, however, falls from +0.1428 on average in A to +0.0784 in C, a reduction of 45%. The architectural difference is real and holds under both protocols, but a random split inflates it, because part of the measured gap reflects memorisation of near variants. Architecture choice and evaluation protocol therefore have to be reported separately.

#### Significance of the split effect (A vs C, paired)

| Polymer | Difference | 95% CI | p |
| --- | --- | --- | --- |
| PET | +0.0494 | [+0.0466, +0.0523] | 0.00018 |
| PE | +0.0253 | [+0.0211, +0.0296] | 0.00153 |
| PP | +0.0246 | [+0.0208, +0.0284] | 0.00129 |
| PVC | +0.0320 | [+0.0252, +0.0388] | 0.00245 |
| Nylon | +0.0300 | [+0.0190, +0.0409] | 0.00714 |

## T10_generator_comparison

### Surrogate model comparison for sequence generation

The same optimisation algorithms were run against different surrogate models. Selectivity and degeneracy show directly whether the optimiser is exploiting the surrogate rather than the target.

| Surrogate | Train-test R2 gap | Target best | Mean selectivity index | Independent gain | Top AA share | Positional entropy | Degenerate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cnn | 0.1385 | 4/6 | 20.99 | 29.242 | 0.436 | 0.4914 | 1/6 |
| encdec | 0.0916 | 6/6 | 28.51 | 30.9168 | 0.3033 | 0.6119 | 0/6 |

**Selected:** encdec

encdec passes the selectivity test on all 6 polymers, with a mean gain of 30.9168 confirmed by the independent model.

#### cnn - per polymer

| Polymer | Target best | Best-scoring polymer | Selectivity index | Top AA (share) | Distinct AA / peptide | Independent gain |
| --- | --- | --- | --- | --- | --- | --- |
| PET | yes | PET | 27.3865 | W (0.625) | 4.46 | 20.275 |
| PE | yes | PE | 18.817 | W (0.4583) | 5.73 | 30.4406 |
| PP | NO | Nylon | 7.7522 | W (0.4389) | 5.33 | 30.7079 |
| PS | NO | Nylon | 1.3725 | W (0.2917) | 6.47 | 26.2188 |
| PVC | yes | PVC | 35.1438 | R (0.4167) | 5.2 | 29.1049 |
| Nylon | yes | Nylon | 35.4634 | W (0.3851) | 5.69 | 38.705 |

#### encdec - per polymer

| Polymer | Target best | Best-scoring polymer | Selectivity index | Top AA (share) | Distinct AA / peptide | Independent gain |
| --- | --- | --- | --- | --- | --- | --- |
| PET | yes | PET | 30.5652 | W (0.3667) | 6.23 | 28.7852 |
| PE | yes | PE | 26.2681 | W (0.2528) | 7.13 | 36.4016 |
| PP | yes | PP | 21.6366 | W (0.3028) | 7.03 | 29.3614 |
| PS | yes | PS | 9.5276 | W (0.2806) | 6.73 | 23.5911 |
| PVC | yes | PVC | 42.7433 | R (0.275) | 7.17 | 28.2176 |
| Nylon | yes | Nylon | 40.33 | W (0.3417) | 5.97 | 39.1441 |

**Interpretation.** The wider a surrogate's train-test gap, the more the optimiser exploits it: sequences collapse onto a single amino acid and lose specificity for the target polymer. The surrogate that passes this test supplies the peptides reported in this work; the selection criteria and the full results for the rejected candidate are given above so the trade-off is visible.