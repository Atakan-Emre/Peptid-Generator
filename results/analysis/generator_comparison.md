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
