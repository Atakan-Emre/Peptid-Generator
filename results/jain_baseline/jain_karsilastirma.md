# Benchmark against Jain et al. (2025)

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

## Reading

- **PET**: 0.8890 under their condition (they report 0.9755), 0.8396 under the identity-aware protocol. The gap of +0.0494 comes from the split alone.
- **PE**: 0.8183 under their condition (they report 0.9517), 0.7930 under the identity-aware protocol. The gap of +0.0253 comes from the split alone.
- **PP**: 0.8323 under their condition (they report 0.9640), 0.8077 under the identity-aware protocol. The gap of +0.0246 comes from the split alone.
- **PVC**: 0.7276 under their condition (they report 0.9554), 0.6956 under the identity-aware protocol. The gap of +0.0320 comes from the split alone.
- **Nylon**: 0.7719 under their condition (they report 0.9774), 0.7419 under the identity-aware protocol. The gap of +0.0300 comes from the split alone.

## Architecture comparison (cnn vs prior)

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

## Significance of the split effect (A vs C, paired)

| Polymer | Difference | 95% CI | p |
| --- | --- | --- | --- |
| PET | +0.0494 | [+0.0466, +0.0523] | 0.00018 |
| PE | +0.0253 | [+0.0211, +0.0296] | 0.00153 |
| PP | +0.0246 | [+0.0208, +0.0284] | 0.00129 |
| PVC | +0.0320 | [+0.0252, +0.0388] | 0.00245 |
| Nylon | +0.0300 | [+0.0190, +0.0409] | 0.00714 |