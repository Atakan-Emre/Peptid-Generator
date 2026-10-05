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