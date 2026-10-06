# Peptides selected for molecular docking

Surrogate used for the predicted score: **encdec**, identity-aware split.
72 peptides, 12 per polymer.

## Why this composition

Docking is being repeated so that the relationship between the predicted score and the docking score can be reported, as requested in review. That correlation is only interpretable if the docked set spans a wide range of predicted affinity. Docking only the best-scoring designs would restrict the range and make any coefficient meaningless.

| Group | n per polymer | Role |
| --- | --- | --- |
| Designed | 4 | New peptides, spanning the generated score range |
| Training | 5 | Sequences with a **measured** affinity, spanning the measured distribution; these let docking be compared against experiment, not only against the model |
| Random | 3 | Null end of the range; also the baseline control asked for in review |

## Dynamic range achieved

| Polymer | n | Predicted score range | Width |
| --- | --- | --- | --- |
| PET | 12 | -73.75 to -4.28 | 69.5 |
| PE | 12 | -72.41 to -10.27 | 62.1 |
| PP | 12 | -64.77 to -8.94 | 55.8 |
| PS | 12 | -51.56 to -6.16 | 45.4 |
| PVC | 12 | -78.70 to -21.46 | 57.2 |
| Nylon | 12 | -83.05 to -27.04 | 56.0 |

## The list

| Polymer | Sequence | Group | Predicted | Measured | Why |
| --- | --- | --- | --- | --- | --- |
| PET | `WWYWEWRYMRWW` | designed | -73.75 | - | designed, SA |
| PET | `WHWYWNFRWWWR` | designed | -67.05 | - | designed, MOCO-CEM |
| PET | `WWYRFYWRWRWW` | designed | -63.33 | - | designed, ILS |
| PET | `NYWREYTRMWMG` | designed | -53.32 | - | designed, SA |
| PET | `WEWWQMFHHRLR` | training | -58.02 | -62.23 | training, measured p0.01 |
| PET | `HHNWFRMKFWTE` | training | -51.52 | -54.16 | training, measured p1 |
| PET | `DDYQINMRISGY` | training | -36.53 | -39.89 | training, measured p25 |
| PET | `FYSTWNMEHKLD` | training | -26.11 | -26.61 | training, measured p50 |
| PET | `QWSQKNDQNFEM` | training | -13.69 | -9.46 | training, measured p90 |
| PET | `NVKRSGGTGIGA` | random control | -4.28 | - | random control |
| PET | `HGEMAQQDQEII` | random control | -13.46 | - | random control |
| PET | `RHGNNDVDRIRY` | random control | -5.86 | - | random control |
| PE | `TFLMTMKWRMLF` | designed | -72.41 | - | designed, SA |
| PE | `HKYWHELWTMRK` | designed | -63.28 | - | designed, SA |
| PE | `MWRWGMRWLMRW` | designed | -58.96 | - | designed, MOCO-CEM |
| PE | `WWYRFWLWVHFR` | designed | -54.93 | - | designed, MOCO-CEM |
| PE | `AWMWHMKWTMRH` | training | -54.32 | -58.3037 | training, measured p0.01 |
| PE | `TWWGSMWHRLMK` | training | -46.95 | -48.7312 | training, measured p1 |
| PE | `RFTAWQIYGHFS` | training | -32.49 | -32.6778 | training, measured p25 |
| PE | `EDNHNTWKAWFH` | training | -22.76 | -24.5847 | training, measured p50 |
| PE | `TSSTAHVAKQEL` | training | -12.56 | -11.85 | training, measured p90 |
| PE | `ILLGGWFRDHLV` | random control | -22.08 | - | random control |
| PE | `RHGFTDTQSHQW` | random control | -10.27 | - | random control |
| PE | `QHHFYLSMLAHD` | random control | -22.72 | - | random control |
| PP | `FWLWQIFERRLW` | designed | -64.77 | - | designed, MOCO-CEM |
| PP | `WWHWIHFLWRVW` | designed | -60.24 | - | designed, MOCO-CEM |
| PP | `WWRYMRKMWIDW` | designed | -56.52 | - | designed, ILS |
| PP | `LKWMTRMHVIMI` | designed | -45.21 | - | designed, SA |
| PP | `WWQRHMFEFRHW` | training | -49.50 | -51.4766 | training, measured p0.01 |
| PP | `WAWHLFHRRSLW` | training | -45.47 | -43.9751 | training, measured p1 |
| PP | `AHTVTWDWIMQF` | training | -28.68 | -26.848 | training, measured p25 |
| PP | `EYTTTWHRWFWH` | training | -18.14 | -17.9858 | training, measured p50 |
| PP | `VVKWRVQSEQAW` | training | -9.76 | -7.1649 | training, measured p90 |
| PP | `YQTGRSHRKSEV` | random control | -8.94 | - | random control |
| PP | `LRIRDAWNSAME` | random control | -9.74 | - | random control |
| PP | `GNLKQHRHDAEM` | random control | -10.03 | - | random control |
| PS | `IFWRYAVLQHQM` | designed | -51.56 | - | designed, SA |
| PS | `FWSSWILRYWFR` | designed | -45.77 | - | designed, ILS |
| PS | `EEVWRWEIQRYM` | designed | -42.55 | - | designed, SA |
| PS | `NREGFFRMGMVK` | designed | -39.65 | - | designed, SA |
| PS | `WHWQREIWQLMR` | training | -41.01 | -43.8428 | training, measured p0.01 |
| PS | `FRLDHRWWNWLH` | training | -37.76 | -37.3965 | training, measured p1 |
| PS | `EYATKILGHWHL` | training | -19.61 | -21.2229 | training, measured p25 |
| PS | `EKSHFFHNWYFQ` | training | -14.02 | -13.9902 | training, measured p50 |
| PS | `VTNFASLGGMWA` | training | -6.16 | -4.9361 | training, measured p90 |
| PS | `SNAMRDDAYEQH` | random control | -10.28 | - | random control |
| PS | `FTDGKGIHERQS` | random control | -13.65 | - | random control |
| PS | `WWYVYWRQHGKV` | random control | -7.74 | - | random control |
| PVC | `KWTMRVNMRRRR` | designed | -78.70 | - | designed, SA |
| PVC | `WFLERLNERRRR` | designed | -71.05 | - | designed, ILS |
| PVC | `RHAWMRQLRRWR` | designed | -66.27 | - | designed, ILS |
| PVC | `GWWWRRWVWIYW` | designed | -61.78 | - | designed, SA |
| PVC | `WWNWQQNHRFIR` | training | -63.85 | -65.2501 | training, measured p0.01 |
| PVC | `NWQWRRYWTMNQ` | training | -54.94 | -56.0238 | training, measured p1 |
| PVC | `HWVLQSVWNRRA` | training | -40.50 | -38.632 | training, measured p25 |
| PVC | `LRHHAALHWRYI` | training | -36.58 | -31.1613 | training, measured p50 |
| PVC | `MWYGFQWHETTY` | training | -24.55 | -18.8707 | training, measured p90 |
| PVC | `NAVMEWDYKKNW` | random control | -21.46 | - | random control |
| PVC | `KYMWVHSWNHGD` | random control | -30.53 | - | random control |
| PVC | `HSKKRQESAHDF` | random control | -21.63 | - | random control |
| Nylon | `RWMLWHWRLRRW` | designed | -83.05 | - | designed, SA |
| Nylon | `MFYMRNWWKLWW` | designed | -77.95 | - | designed, SA |
| Nylon | `WRHWWMMWMEWR` | designed | -74.64 | - | designed, ILS |
| Nylon | `WRMFHRNKHEWR` | designed | -68.17 | - | designed, ILS |
| Nylon | `HWNHREWYLRQW` | training | -70.94 | -72.6181 | training, measured p0.01 |
| Nylon | `DWWARGFHHRWH` | training | -64.24 | -65.4074 | training, measured p1 |
| Nylon | `WWINHYRDHNFW` | training | -47.29 | -45.3079 | training, measured p25 |
| Nylon | `IGFKQNFVEHQR` | training | -36.65 | -36.0053 | training, measured p50 |
| Nylon | `TTWHNQFVEHYG` | training | -29.50 | -22.2313 | training, measured p90 |
| Nylon | `YHTWLYTRKVDW` | random control | -33.23 | - | random control |
| Nylon | `TRQKYLRHNGKY` | random control | -29.64 | - | random control |
| Nylon | `TYTANGWTWVEK` | random control | -27.04 | - | random control |

## What the laboratory returns

Fill `docking_score`, `oligomer_length`, `n_runs`, `software` and `notes` in `docking_selection.csv`, keeping one row per peptide-polymer pair, then run:

```bash
python scripts/07_run_analyses.py --docking-csv results/analysis/docking_selection.csv
python scripts/10_build_figures.py
python scripts/11_review_compliance.py
```

Report the docking score as the best (most negative) binding energy of the run, in kcal/mol, and keep the oligomer length constant within a polymer so the scores stay comparable.
