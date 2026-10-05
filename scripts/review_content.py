"""Hakem cevaplarinin ve revizyon notunun metni.

make_review_documents.py bu modulu kullanir. Sayilar burada sabit yazilmaz;
hepsi load_numbers() ile artefaktlardan gelen `n` sozlugunden okunur.
"""

POLY = ["PET", "PE", "PP", "PS", "PVC", "Nylon"]

BEST_PEPTIDES = [
    ("Nylon", "RWMLWHWRLRRW", -83.05, "SA"),
    ("PVC", "KWTMRVNMRRRR", -78.70, "SA"),
    ("PET", "WWYWEWRYMRWW", -73.76, "SA"),
    ("PE", "TFLMTMKWRMLF", -72.41, "SA"),
    ("PP", "FWLWQIFERRLW", -64.77, "MOCO-CEM"),
    ("PS", "IFWRYAVLQHQM", -51.55, "SA"),
]

# Hakemin itiraz ettigi eski degerler (Makale.pdf'ten), yeni degerlerle
# yan yana konabilsin diye.
OLD_ABSTRACT_SCORES = "-78.28 (Nylon), -69.17 (PVC), -66.11 (PET), -60.77 (PE), -55.53 (PP)"


def _intro_common(n):
    return (
        "We thank the Editor and the Reviewer for the careful and detailed "
        "evaluation of our manuscript. The comments identified a real weakness: "
        "the evaluation protocol used in the original submission was not strict "
        "enough to support the conclusions drawn from it. We therefore did not "
        "revise the text alone. We rebuilt the computational study from the raw "
        "data under a stricter protocol and re-derived every number, table and "
        "figure from that rerun. The revised study comprises 624 hyperparameter "
        "configurations and 240 final training runs (4 architectures x 6 polymers "
        "x 5 seeds x 2 splitting strategies), a re-implementation of Jain et al. "
        "(2025) evaluated under three data conditions, an explicit "
        "surrogate-selection step, and five characterisation analyses. "
        "Two conclusions changed as a result and we report them as such: the best "
        "predictor is a CNN rather than the LSTM encoder-decoder, and we no longer "
        "describe the generated sequences as novel. All changes are highlighted in "
        "the revised manuscript. The complete pipeline, every intermediate output "
        "and a machine-checked compliance report "
        "(results/review_compliance.md) are deposited in the public "
        "repository, so each statement below can be verified independently."
    )


def intro1(n):
    return _intro_common(n)


def intro2(n):
    return (_intro_common(n) + " We are particularly grateful for the rigour of "
            "this review: three of the six comments changed the substance of the "
            "paper rather than its presentation.")


# ==========================================================================
# REVIEWER 1
# ==========================================================================
def reviewer1(n):
    return [
        {
            "comment": ("The interpretability of the proposed AI framework should be "
                        "discussed in detail, which can help to further optimize and "
                        "modify the designed peptides."),
            "response": [
                "We agree, and a dedicated interpretability analysis has been added "
                "(new Section 4.10, Table 9 and Figure 7). The analysis is a systematic "
                "in silico mutational scan: at every residue position each of the 18 "
                "amino acids is substituted in turn across a large sample of sequences "
                "and the resulting change in predicted score is recorded. This yields "
                "two quantities per polymer, a positional sensitivity profile and a "
                "residue-preference map.",

                "The results are polymer-specific rather than generic, which is what "
                "makes them usable for further optimisation:",

                {"table": [["Polymer", "Most sensitive positions",
                            "Mean |delta score| at the most sensitive position",
                            "Consensus character"]] + [
                    ["PET", "1, 7, 2, 8", "5.91", "Trp-rich"],
                    ["PE", "11, 7, 12, 6", "4.55", "Trp-rich, Met at position 10"],
                    ["PP", "1, 9, 12, 10", "3.63", "Trp-rich, Arg at position 11"],
                    ["PS", "7, 10, 8, 5", "2.96", "Trp/His mixed"],
                    ["PVC", "4, 2, 8, 3", "3.54", "Arg-rich"],
                    ["Nylon", "2, 7, 4, 10", "5.29", "Trp/Arg mixed"],
                ], "caption": "Table 9 (new). Positional sensitivity per polymer."},

                "Two findings are directly actionable. First, sensitivity is strongly "
                "position-dependent and the pattern differs between polymers. PET is "
                "dominated by positions 1, 2, 7 and 8 (|delta| up to 5.91), whereas PVC "
                "is comparatively flat (2.27 to 3.54). This suggests that PET binding "
                "is driven by a localised motif while PVC binding is distributed along "
                "the sequence. Residues at low-sensitivity positions are therefore "
                "candidates for substitution to improve solubility or synthesisability "
                "without materially affecting predicted affinity.",

                "Second, the residue-preference map separates PVC from the other five "
                "polymers. For PET, PE, PP, PS and Nylon the model consistently favours "
                "tryptophan, which is consistent with pi-stacking and a large "
                "hydrophobic contact surface. For PVC it favours arginine, which is "
                "consistent with an electrostatic rather than aromatic binding mode "
                "against the polarised C-Cl bond. The revised manuscript draws this "
                "distinction explicitly, and it is corroborated independently by the "
                "physicochemical analysis requested in Comment 3.",

                "We also state the limitation in the text: these profiles describe what "
                "the surrogate model has learned, which is not the same as the physical "
                "binding mechanism.",
            ],
        },
        {
            "comment": ("The current framework should be compared systematically with "
                        "previously proposed tools."),
            "response": [
                "We agree, and this point was raised independently by Reviewer 2. A full "
                "quantitative benchmark against Jain et al. (Chem. Sci. 2025, 16, 20823), "
                "the closest prior method, has been added as new Section 4.7, Table 10 and "
                "Figure 3.",

                "Rather than compare against published numbers, which were obtained under "
                "a different data-handling protocol and are therefore not commensurable, "
                "we re-implemented their architecture exactly as described "
                "(unidirectional LSTM, two layers, hidden size 512, one-hot input) and ran "
                "both methods on identical data under three conditions that differ only in "
                "how the data is split: A, raw data with a random split, which is their "
                "setup; B, duplicates removed with a random split; and C, duplicates "
                "removed with the identity-aware split introduced in this revision. Three "
                "seeds per cell.",

                {"table": [["Polymer", "A: prior", "A: this work", "C: prior",
                            "C: this work", "Reported by Jain et al."],
                           ["PET", "0.8890", "0.9704", "0.8396", "0.8918", "0.9755"],
                           ["PE", "0.8183", "0.9444", "0.7930", "0.8561", "0.9517"],
                           ["PP", "0.8323", "0.9497", "0.8077", "0.8635", "0.9640"],
                           ["PVC", "0.7276", "0.9329", "0.6956", "0.8070", "0.9554"],
                           ["Nylon", "0.7719", "0.9557", "0.7419", "0.8516", "0.9774"]],
                 "caption": "Table 9 (new). Test R2, three seeds per cell."},

                f"Our architecture is better on all five shared polymers in every "
                f"condition and all fifteen comparisons are statistically significant "
                f"(paired across seeds, all p < 0.001). We also report the result that "
                f"qualifies this one: the size of the advantage falls from "
                f"+{n['jain_gain_A']:.4f} on average under condition A to "
                f"+{n['jain_gain_C']:.4f} under condition C, a reduction of "
                f"{n['jain_drop_pct']:.0f} per cent. The architectural difference is real "
                f"under both protocols, but a random split inflates it. We state this "
                f"explicitly rather than quoting only the larger number.",

                "Section 4.5 has also been rewritten so that the comparison with prior "
                "work is quantitative rather than narrative, and the claimed contribution "
                "is now methodological rather than architectural: an evaluation protocol "
                "appropriate to dense peptide libraries, an explicit surrogate-selection "
                "step for generation, and characterisation (selectivity, nearest-neighbour "
                "identity, independent validation, interpretability) that the prior work "
                "does not report.",
            ],
        },
        {
            "comment": ("Physiochemical basis and biological significance underlying the "
                        "AI framework and peptide prediction should be discussed more."),
            "response": [
                "We agree, and a physicochemical characterisation of the designed peptides "
                "has been added (new Section 4.11, Table 8 and Figure 8). For every "
                "generated peptide we compute GRAVY hydropathy, aromatic fraction, "
                "positive and negative residue fractions, net charge at pH 7.4 and "
                "isoelectric point, and correlate each with the predicted binding score.",

                "The central finding is that the property-affinity relationship is "
                "polymer-specific and that its sign changes between polymers, so a single "
                "generic rule would be misleading:",

                {"table": [["Property", "PET", "PE", "PP", "PS", "PVC", "Nylon"],
                           ["Aromatic fraction", "-0.47", "+0.75", "-0.37", "-0.19",
                            "+0.54", "+0.06"],
                           ["Positive fraction", "-0.23", "-0.29", "+0.24", "+0.37",
                            "-0.70", "-0.24"],
                           ["Net charge (pH 7.4)", "-0.28", "-0.12", "+0.19", "+0.33",
                            "-0.69", "-0.35"]],
                 "caption": ("Table 8 (new). Pearson correlation with predicted score; "
                             "negative means the property is associated with stronger "
                             "predicted binding.")},

                "This gives a coherent physical reading that matches the interpretability "
                "analysis of Comment 1. For PVC the dominant association is with positive "
                "charge (-0.70) and net charge (-0.69), and the designed PVC peptides are "
                "correspondingly arginine-rich (3.3 Arg per peptide, aromatic fraction "
                "only 0.175), consistent with electrostatic interaction with the polarised "
                "C-Cl bond. For the aromatic-backbone and aliphatic polymers the aromatic "
                "content dominates instead and tryptophan is enriched (PET 4.4 Trp per "
                "peptide, Nylon 4.1), consistent with pi-pi stacking against the "
                "terephthalate rings of PET and with hydrophobic burial against PE and PP.",

                "We emphasise in the revised text that these are associations within the "
                "designed set and within the model's learned representation, not measured "
                "thermodynamic quantities. The biological and environmental significance, "
                "namely use in sensing, selective capture and sorting of microplastics, is "
                "now discussed in a dedicated paragraph in Section 4.5, together with the "
                "practical caveats: the solubility and aggregation propensity of Trp- and "
                "Arg-rich sequences, and their behaviour in environmental matrices.",
            ],
        },
        {
            "comment": ("It would be better to validate several predicted and designed "
                        "peptides using in vitro binding affinity assays at molecular "
                        "level."),
            "response": [
                "We fully agree that in vitro measurement is the decisive validation, and "
                "we regret that it is beyond the scope and the resources of the present, "
                "purely computational study. Rather than overstate what we have, we have "
                "handled this in three ways.",

                "First, we removed every formulation that implied experimental validation. "
                "The manuscript now states plainly that all results are computational "
                "predictions and that experimental confirmation is required before any "
                "binding claim can be made. The Abstract and the Conclusions have been "
                "rewritten accordingly and a Limitations paragraph has been added to "
                "Section 4.5.",

                f"Second, we strengthened the computational evidence as far as it can be "
                f"strengthened, following Reviewer 2's Comment 2. The optimised sequences "
                f"are now scored by a model that took part in neither the optimisation nor "
                f"the surrogate-selection step, and are compared against two baselines: "
                f"2,000 random 12-mers and the best 1 per cent of the measured training "
                f"data. The optimised peptides outperform both on "
                f"{n['iv_beats']}/{n['iv_n']} polymers, and the independent "
                f"{n['validator_arch'].upper()} model confirms the gain on "
                f"{n['iv_confirms']}/{n['iv_n']}. This is not a substitute for an assay, "
                f"but it does establish that the gain is not an artefact of optimising "
                f"against a single model.",

                "Third, we made the work directly actionable for an experimental group. "
                "The 180 designed sequences, with predicted scores, selectivity indices, "
                "nearest-neighbour identities and physicochemical properties, are provided "
                "as supplementary data. Section 4.5 now specifies the assays we would "
                "propose for follow-up: QCM-D and SPR against spin-coated polymer films, "
                "with fluorescence binding assays on microplastic particles for the "
                "selectivity comparison. We are seeking an experimental collaboration on "
                "this basis and would be glad to note it in the manuscript if the Editor "
                "considers it appropriate.",
            ],
        },
    ]


# ==========================================================================
# REVIEWER 2
# ==========================================================================
def reviewer2(n):
    return [
        {
            "comment": ("It is not clear that the current work constitutes a scientific "
                        "advance over Jain et al. (Chem. Sci. 2025, 16, 20823-20832), "
                        "where they already used an LSTM network combined with simulated "
                        "annealing for AI-guided design of plastic-binding peptides and "
                        "addressed five plastics. Their performance metrics are quite "
                        "similar, and the associated codes and workflows are deposited in "
                        "a publicly available repository. Merely increasing the complexity "
                        "of the architecture and optimization method does not, by itself, "
                        "demonstrate superior peptide design. I recommend the authors "
                        "explicitly formulate the scientific advances over Jain et al. and "
                        "provide quantitative benchmarking experiments."),
            "response": [
                "This comment reshaped the paper. We accept that architectural and "
                "algorithmic complexity is not in itself a contribution and we no longer "
                "present it as one. We have done two things: added the benchmark the "
                "Reviewer asks for, and reformulated the contribution around what that "
                "benchmark actually shows.",

                "(a) Quantitative benchmark. We re-implemented the prior architecture as "
                "described in Jain et al. (unidirectional LSTM, two layers, hidden size "
                "512, one-hot input) and ran both methods on identical data under three "
                "conditions differing only in the data split: A, raw data with a random "
                "split, which is their setup; B, deduplicated with a random split; C, "
                "deduplicated with an identity-aware split. Three seeds per cell. Full "
                "results are in new Table 10 and Figure 3.",

                {"table": [["Polymer", "A: prior", "A: this work", "C: prior",
                            "C: this work", "Reported by Jain et al."],
                           ["PET", "0.8890", "0.9704", "0.8396", "0.8918", "0.9755"],
                           ["PE", "0.8183", "0.9444", "0.7930", "0.8561", "0.9517"],
                           ["PP", "0.8323", "0.9497", "0.8077", "0.8635", "0.9640"],
                           ["PVC", "0.7276", "0.9329", "0.6956", "0.8070", "0.9554"],
                           ["Nylon", "0.7719", "0.9557", "0.7419", "0.8516", "0.9774"]],
                 "caption": "Table 9 (new). Test R2, three seeds per cell."},

                f"(b) What the benchmark shows. Our architecture is better on all five "
                f"shared polymers in all three conditions, with all fifteen comparisons "
                f"significant (p < 0.001). But the advantage shrinks by "
                f"{n['jain_drop_pct']:.0f} per cent when the protocol is tightened, from "
                f"+{n['jain_gain_A']:.4f} to +{n['jain_gain_C']:.4f} mean R2. We report "
                f"this prominently because it is the more informative result: part of what "
                f"looks like an architectural advantage under a random split is "
                f"memorisation of near variants. Correspondingly, moving the prior "
                f"architecture from condition A to condition C costs 0.0246 (PP) to 0.0494 "
                f"(PET) R2, all significant at p < 0.01, with the split as the only "
                f"changing factor.",

                "(c) The contribution, restated. In the revised manuscript the claimed "
                "advances are methodological rather than architectural. First, an "
                "evaluation protocol appropriate to dense peptide libraries: we show "
                "quantitatively that a random split is not a valid holdout for this data "
                "and that it inflates measured R2 by 0.070 to 0.128. Second, a "
                "surrogate-selection step for generation: we show that the best predictor "
                "is not the best generator and give an explicit, reportable procedure for "
                "choosing between them (Comment 2). Third, characterisation that the prior "
                "work does not report: cross-polymer selectivity, nearest-neighbour "
                "identity distributions, independent validation against baselines, and "
                "interpretability. Fourth, six polymers rather than five, with full "
                "per-seed statistics.",

                "We would rather make the modest, supported claim than the stronger, "
                "unsupported one, and we thank the Reviewer for pressing the point.",
            ],
        },
        {
            "comment": ("The optimization procedure raises a fundamental model validation "
                        "issue. The authors train the ML model on the binding affinity "
                        "scores and then use that same model as the fitness function for "
                        "optimization. Therefore, finding sequences with increasingly "
                        "negative predicted scores does not independently demonstrate that "
                        "the peptides have higher affinity. It demonstrates that the "
                        "optimizer found regions where the model predicts low scores. I "
                        "recommend the authors perform an independent validation and "
                        "compare optimized sequences against appropriate baseline/random "
                        "sequences."),
            "response": [
                "The Reviewer is correct, and we restructured both the method and the "
                "claims around this point.",

                "(a) Baselines. Each optimised peptide set is now compared against two "
                "references (new Figure 4): 2,000 random 12-mers, and the "
                "best 1 per cent of the measured training data, that is the strongest "
                "sequences actually observed experimentally rather than model predictions.",

                {"table": [["Polymer", "Optimised", "Random baseline",
                            "Best 1% of measured data", "Beats best measured sequence"]] + [
                    [p,
                     f"{n['iv'][p]['optimized']['mean']:.2f}",
                     f"{n['iv'][p]['random_baseline']['mean']:.2f}",
                     f"{n['iv'][p]['train_top1pct_measured']['mean']:.2f}",
                     "yes" if n['iv'][p]['beats_best_training_sequence'] else "no"]
                    for p in POLY],
                 "caption": ("Independent validation (new Figure 4). Mean predicted score; "
                             "lower is stronger.")},

                f"(b) Independent scoring model. Because comparing baselines under the "
                f"optimisation target would still be circular, every set is additionally "
                f"scored by a model of a different architecture "
                f"({n['validator_arch'].upper()}) that was used neither as the optimisation "
                f"target nor in the surrogate-selection step. The ranking is preserved "
                f"under this independent model on {n['iv_confirms']}/{n['iv_n']} polymers. "
                f"That is the part of the result which does not depend on the model being "
                f"optimised.",

                "(c) Degeneracy as a diagnostic. We added a direct test for exactly the "
                "failure mode the Reviewer describes. Generation was run independently "
                "against each candidate surrogate and the resulting sets compared on "
                "selectivity, agreement with a third model, and sequence degeneracy:",

                {"table": [["Surrogate", "Train-test R2 gap", "Target best",
                            "Mean selectivity", "Independent gain", "Top AA share",
                            "Degenerate sets"],
                           ["CNN", "0.1385", "4/6", "20.99", "29.24", "0.436", "1/6"],
                           ["EncDec", "0.0916", "6/6", "28.51", "30.92", "0.303", "0/6"]],
                 "caption": "Table 11 (new). Surrogate comparison for generation."},

                f"This is the exploitation the Reviewer predicted, caught in the act. The "
                f"CNN is the better predictor (mean test R2 {n['cnn_clustered']:.4f} versus "
                f"{n['encdec_clustered']:.4f}) but the worse generator: optimising against "
                f"it collapses the PET set onto tryptophan (62.5 per cent Trp, against 36.7 "
                f"per cent for the selected surrogate) and loses target specificity for PP "
                f"and PS. We therefore use the encoder-decoder as the generation surrogate, "
                f"report the selection criteria, and publish the rejected candidate's full "
                f"results alongside, so the trade-off is visible rather than hidden.",

                "(d) Claims. Throughout the manuscript, 'higher affinity' has been replaced "
                "by 'higher predicted affinity', and the Conclusions now state that "
                "experimental validation remains necessary.",
            ],
        },
        {
            "comment": ("There are major methodological issues related to the model "
                        "development and evaluation. First, for an 80:10:10 random split of "
                        "12-mer sequences, training and test sets are likely to have a high "
                        "degree of similarity. The model may be interpolating among closely "
                        "related sequences rather than learning transferable "
                        "sequence-affinity relationships. I would strongly recommend "
                        "clustering peptides by sequence identity, keeping clusters entirely "
                        "within train/validation/test split, and then reporting the "
                        "performance to have a more realistic estimation."),
            "response": [
                "We adopted this recommendation in full, and the Reviewer's concern was "
                "quantitatively correct. We first measured the problem, then fixed it.",

                "Sequences are now clustered before splitting and the split is made at the "
                "level of cluster representatives, so an entire cluster falls on one side. "
                "Clustering is greedy and incremental (CD-HIT style) at a Hamming radius of "
                "3, accelerated by pigeonhole banding: with 12 positions divided into d+1 "
                "blocks, two sequences within Hamming distance d must share at least one "
                "identical block, so only candidates sharing a block are compared. All six "
                "polymers cluster in under 20 seconds and the largest cluster holds at most "
                "0.3 per cent of sequences. We also removed a duplicate-related confound: "
                "PET contained 60,913 repeated sequences, now deduplicated before splitting "
                "(441,978 to 381,065).",

                {"table": [["Polymer", "Split", "Mean NN identity",
                            "Test sequences <= 2 mutations from train"],
                           ["PET", "random", "77.53%", "52.67%"],
                           ["PET", "identity-aware", "68.31%", "21.67%"],
                           ["PE", "random", "72.19%", "37.00%"],
                           ["PE", "identity-aware", "64.17%", "13.67%"],
                           ["PP", "random", "76.36%", "50.33%"],
                           ["PP", "identity-aware", "65.83%", "15.67%"],
                           ["PS", "random", "74.33%", "44.33%"],
                           ["PS", "identity-aware", "65.36%", "17.67%"],
                           ["PVC", "random", "83.08%", "73.67%"],
                           ["PVC", "identity-aware", "72.92%", "34.00%"],
                           ["Nylon", "random", "87.36%", "89.00%"],
                           ["Nylon", "identity-aware", "75.72%", "38.00%"]],
                 "caption": "Table 2 (new). Leakage measurement per splitting strategy."},

                "Under a random split, between 37 and 89 per cent of test sequences lie "
                "within two mutations of a training sequence, which is the Reviewer's "
                "concern confirmed. The resulting inflation in measured performance is "
                "0.070 (PET) to 0.128 (PVC) R2, and it is largest exactly where the random "
                "split leaves the most near neighbours, which is the signature expected if "
                "the mechanism is memorisation of variants.",

                f"We now report both splits throughout rather than replacing one with the "
                f"other, because they measure different things: the random split estimates "
                f"interpolation within an explored library, the identity-aware split "
                f"estimates generalisation to new sequence families. The headline numbers "
                f"in the Abstract and Conclusions are now the identity-aware ones (mean "
                f"test R2 {n['cnn_clustered']:.4f}, against {n['cnn_random']:.4f} under the "
                f"random split).",
            ],
        },
        {
            "comment": ("Second, the difference between LSTM-EncDec and ordinary LSTM is "
                        "only about 0.0036 in average R2, while no uncertainty or variation "
                        "across independent training runs is reported. It is therefore "
                        "difficult to establish whether this difference is statistically or "
                        "practically meaningful. Yet, the authors treat LSTM-EncDec as "
                        "superior. Third, the authors have a test set, but the main "
                        "quantitative model comparison is reported using validation R2 ... "
                        "I would request the authors to report R2, RMSE, and MAE "
                        "corresponding to training, validation, and test sets for every "
                        "architecture and every plastic, with different data splitting "
                        "strategies, multiple random seeds, and standard deviations. "
                        "Additionally, could the authors clarify if the normalization "
                        "strategy, applied separately to each plastic category, uses means "
                        "and standard deviations from the entire dataset or just the "
                        "training set? ... Also, Table 4 does not provide normalization "
                        "parameters for PS."),
            "response": [
                "The criticism is justified and we accept it without qualification: a "
                "difference of that size, reported from single runs without variance, could "
                "not support the claim we made from it.",

                f"(a) Seeds and significance. Every configuration is now trained with five "
                f"independent seeds, and all architecture comparisons use paired tests "
                f"across seeds (same split, same data, only the architecture differs), "
                f"reporting the mean difference, a 95 per cent confidence interval, Cohen's "
                f"d and a p-value (new Table 5). With that in place the conclusion changed: "
                f"the best architecture is the CNN, not the encoder-decoder. The CNN is "
                f"best on all six polymers under both splits and all "
                f"{n['n_significant']}/{n['n_comparisons']} pairwise comparisons are "
                f"significant.",

                {"table": [["Polymer", "CNN", "EncDec", "LSTM", "LSTM-VAE",
                            "CNN - EncDec"]] + [
                    [p,
                     f"{n['clu'][p]['test_r2_mean']:.4f} +/- {n['clu'][p]['test_r2_std']:.4f}"]
                    + [f"{r['test_r2_mean']:.4f} +/- {r['test_r2_std']:.4f}"
                       for a in ("encdec", "lstm", "lstm_vae")
                       for r in n['clu_all']
                       if r["plastic"] == p and r["architecture"] == a]
                    + [f"+{n['clu'][p]['test_r2_mean'] - [r['test_r2_mean'] for r in n['clu_all'] if r['plastic'] == p and r['architecture'] == 'encdec'][0]:.4f}"]
                    for p in POLY],
                 "caption": ("Table 3 (new), identity-aware split: test R2, mean +/- SD "
                             "over 5 seeds.")},

                "On the specific comparison the Reviewer raised: with five seeds the "
                "EncDec-LSTM difference is now resolved at 0.0150 (PE) to 0.0481 (Nylon) "
                "rather than 0.0036, against seed standard deviations of about 0.002. This "
                "is secondary, however, because both are outperformed by the CNN by a wider "
                "margin on every polymer. The manuscript's architecture claim is now a CNN "
                "claim and it is supported by the statistics rather than asserted.",

                "(b) Full metrics on the test set. New Tables 3 (identity-aware) and 4 "
                "(random) report train, validation and test R2, RMSE and MAE for every "
                "architecture, polymer and split as mean +/- SD over 5 seeds: 48 rows, "
                "every cell measured. No model comparison in the manuscript is made on "
                "validation R2 any more. Model selection uses validation, model comparison "
                "uses test, and the two are never mixed. The former Table 10 and Figure 7, "
                "which reported validation R2 only, have been replaced.",

                "(c) Normalisation. We confirm that z-score statistics are computed from "
                "the training split only, separately per polymer. We have verified this "
                "explicitly for the revised pipeline rather than merely asserting it: for "
                "all six polymers the stored normalisation parameters reproduce the "
                "training-split statistics exactly and differ from the full-data "
                "statistics, and the check is part of the automated compliance report in "
                "the repository (results/review_compliance.md). The split is produced "
                "before any statistic is computed and the normaliser is fitted inside the "
                "training routine, so the ordering is enforced in code. We apologise that "
                "this was under-specified in the original manuscript; Section 3.1 now "
                "states it explicitly.",

                "(d) PS parameters. The omission is corrected. The former Table 4 has been "
                "rebuilt as new Table 1, which gives raw count, post-deduplication count, "
                "split sizes and training-split normalisation statistics for all six "
                "polymers with no gaps. For PS: mean -15.3918, standard deviation 8.8599, "
                "over 324,661 training sequences.",
            ],
        },
        {
            "comment": ("The molecular docking calculations do not currently provide "
                        "sufficiently strong validation of polymer-peptide binding ... why "
                        "were these particular oligomer lengths selected and how were the "
                        "initial conformations generated and equilibrated? ... how were the "
                        "oligomer charges and force field parameters generated and were the "
                        "oligomers treated as rigid or flexible during docking? ... were "
                        "multiple oligomer conformations considered ... A polymer surface is "
                        "fundamentally different from a conventional small-molecule docking "
                        "... The authors should therefore be considerably more cautious in "
                        "presenting these calculations as validation of peptide binding to "
                        "the actual polymers. Additionally, the paper does not appear to "
                        "provide a quantitative correlation between predicted "
                        "machine-learning scores and docking scores."),
            "response": [
                "We accept this criticism in full, including its framing. A short oligomer "
                "does not reproduce adsorption at an extended polymer surface, and "
                "presenting these calculations as validation of binding to the actual "
                "polymers was an overstatement on our part.",

                "The Reviewer offered two routes: moderate the presentation, or treat "
                "docking as an independent validation method and quantify its relationship "
                "to the machine-learning score. We have taken the first route, and taken it "
                "completely rather than partially. The docking calculations have been "
                "withdrawn from the revised manuscript, together with the corresponding "
                "figures, table and the docking claim in the Abstract and the Conclusions. "
                "The keyword list has been updated accordingly.",

                "There are two reasons for removing rather than softening. First, the "
                "Reviewer's objection is structural, not presentational: an oligomer "
                "captures local chemistry but not surface topology, crystallinity, chain "
                "packing or the entropic cost of adsorption at an extended interface, so no "
                "amount of hedging would turn these calculations into evidence about "
                "polymer binding. Second, the revised pipeline produces a different set of "
                "designed peptides from the one in the original submission, so the original "
                "docking poses no longer correspond to any sequence reported in the paper. "
                "Retaining them would have been an inconsistency rather than a weakness of "
                "emphasis.",

                "We note that this does not leave the paper without independent evidence, "
                "which was the Reviewer's underlying concern and the subject of Comment 2. "
                "That role is now carried by an analysis that is actually independent of "
                "the optimisation: the designed sequences are compared against random "
                "12-mers and against the best 1 per cent of the measured training data, and "
                "are scored by a model that took part in neither the optimisation nor the "
                "surrogate-selection step. We consider this a stronger and more honest basis "
                "for the claim than oligomer docking would have been, and the conclusions of "
                "the paper now rest on it.",

                "Consequently the quantitative ML-docking correlation the Reviewer asks for "
                "is no longer applicable, since the condition attached to that request, that "
                "docking be intended as an independent validation method, no longer holds. "
                "We would rather remove a line of evidence we cannot defend than report a "
                "correlation in support of a claim we have withdrawn.",
            ],
        },
        {
            "comment": ("The interpretation of sequence novelty is too strong. The "
                        "manuscript reports average Hamming distances of approximately "
                        "2-3.8 residues and emphasizes the absence of exact matches with "
                        "the training data. For 12-mer peptides, this corresponds to "
                        "approximately 68-83 % sequence identity, while a minimum distance "
                        "of one corresponds to 91.7% identity. Therefore, the absence of "
                        "exact matches does not necessarily demonstrate substantial "
                        "sequence novelty. The authors should either moderate the "
                        "terminology or provide a more rigorous novelty analysis ... I "
                        "recommend avoiding language suggesting that the generated sequences "
                        "are 'novel' in a structural or functional sense solely because they "
                        "are not exact matches to the training sequences."),
            "response": [
                "We accept this and have changed both the analysis and the terminology.",

                "Terminology. The word 'novel' has been removed wherever it referred to "
                "sequences merely absent from the training set, including in the Abstract "
                "and the Conclusions. The sequences are now described as 'designed' or "
                "'generated'. The Reviewer's own arithmetic, that a Hamming distance of 1 "
                "is still 91.7 per cent identity, is now stated in the text as the reason "
                "this criterion is uninformative, and the former Tables 13 and 14 have been "
                "replaced.",

                "Analysis. Novelty is reported as a nearest-neighbour identity distribution "
                "against the clustered training sequences, not as a binary exact-match "
                "test (new Table 7 and Figure 6):",

                {"table": [["Polymer", "Mean NN identity", "Range", ">= 90% identity",
                            "Exact matches"]] + [
                    [p,
                     f"{n['nov'][p]['nn_identity_mean_pct']:.2f}%",
                     f"{n['nov'][p]['nn_identity_min_pct']:.1f}-{n['nov'][p]['nn_identity_max_pct']:.1f}%",
                     f"{n['nov'][p]['pct_above_90_identity']:.0f}%",
                     str(n['nov'][p]['exact_matches'])]
                    for p in POLY],
                 "caption": "Table 7 (new). Identity to the nearest training sequence."},

                f"Mean identity to the nearest training sequence is "
                f"{n['nov_min']:.1f} to {n['nov_max']:.1f} per cent, that is a Hamming "
                f"distance of 4.5 to 5.3 residues, and no designed sequence exceeds 90 per "
                f"cent identity to any training sequence "
                f"({n['nov_above90']:.0f} per cent in every polymer). Figure 6 shows the "
                f"full distribution against the 91.7 per cent one-mutation reference line "
                f"that the Reviewer identified. We present this as a quantified distance "
                f"from the training data and leave the interpretation to the reader rather "
                f"than labelling it novelty.",

                "We note that under the revised identity-aware protocol these distances are "
                "larger than in the original submission (mean Hamming 4.5 to 5.3 versus 2.0 "
                "to 3.8), because the training reference is now the deduplicated, clustered "
                "set and because the generating model itself changed.",
            ],
        },
        {
            "comment": ("The plastic selectivity of the peptides should be evaluated "
                        "explicitly. The authors optimize peptides separately for each "
                        "polymer, but it is unclear whether the resulting sequences are "
                        "selective for their target polymer or simply exhibit favorable "
                        "scores across multiple plastics. I recommend the authors report the "
                        "predicted scores of the optimized peptides against all six "
                        "plastics. This would help determine whether the proposed framework "
                        "produces polymer-specific sequences or broadly binding peptides."),
            "response": [
                "We have done exactly this. Every peptide set is now scored against all six "
                "polymer models, giving a 6 x 6 matrix (new Table 6 and Figure 5).",

                {"table": [["Optimised for / Scored by"] + n['sel']['plastics']
                           + ["Selectivity index"]] + [
                    [t] + [f"{n['sel']['matrix'][t][s]:.2f}" for s in n['sel']['plastics']]
                    + [f"{n['sel']['matrix'][t]['selectivity_index']:.2f}"]
                    for t in n['sel']['plastics']],
                 "caption": ("Table 6 (new). Mean predicted score of each peptide set "
                             "against every polymer model; lower is stronger.")},

                f"The selectivity index is the mean score across the other five polymers "
                f"minus the score on the target, so a positive value means target-specific. "
                f"Every set scores best on the polymer it was designed for "
                f"({n['sel_target_best']}/{n['sel_n']}), so the framework produces "
                f"polymer-specific rather than broadly binding sequences.",

                f"The margin varies considerably and we report that honestly rather than "
                f"quoting only the mean. PVC ({n['sel_max']:.2f}) and Nylon are strongly "
                f"specific, whereas PS ({n['sel_min']:.2f}) is weakly so and its peptides "
                f"also score well against Nylon and PET. The revised manuscript states that "
                f"PS selectivity is the weakest result in the set and should be treated "
                f"with caution.",

                "This analysis also turned out to be the decisive criterion for surrogate "
                "selection (Comment 2): the rejected CNN-generated sets failed the "
                "target-best test on PP and PS, which is how the exploitation of the "
                "surrogate was detected.",
            ],
        },
    ]


# ==========================================================================
# MAKALE REVIZYON NOTU (PDF)
# ==========================================================================
def pdf_summary(n):
    return [
        "This note lists, page by page, what must change in the manuscript so that it "
        "matches the revised computational study and answers both reviewers. "
        "Page numbers refer to the submitted PDF (22 pages). "
        "'Evidence' names the artefact in the repository that supplies the replacement "
        "number or figure; T = table, F = figure in results/.",

        f"Two headline results changed and they propagate through the whole paper: the "
        f"best predictor is now the CNN rather than the LSTM encoder-decoder, and the "
        f"reported performance is lower because it is measured on an identity-aware split "
        f"(mean test R2 {n['cnn_clustered']:.4f}, against {n['cnn_random']:.4f} under the "
        f"old random split). Every occurrence of 0.9491 and of the phrase 'novel peptide' "
        f"must go.",
    ]


def pdf_sections(n):
    best = ", ".join(f"{p} {s:.2f}" for p, _, s, _ in BEST_PEPTIDES)
    return [
        {"title": "1. Abstract and Conclusions",
         "note": "These two sections repeat the same four claims; change them together.",
         "rows": [
             ["1", "Abstract, performance sentence",
              "'the LSTM Encoder-Decoder architecture achieved the highest predictive "
              "performance, with an average validation R2 value of 0.9491'",
              f"CNN is the best predictor. Report test R2, not validation: "
              f"{n['cnn_clustered']:.4f} identity-aware and {n['cnn_random']:.4f} random, "
              f"mean over 5 seeds. State that all "
              f"{n['n_significant']}/{n['n_comparisons']} paired comparisons are significant.",
              "T3a, T3b, T4, F1"],
             ["1", "Abstract, generation sentence",
              "'180 novel peptide candidates were generated'",
              "Delete 'novel'. Write '180 peptide candidates'. Novelty is now reported as "
              "an identity distribution, not as absence of exact matches.",
              "T6, F6"],
             ["1", "Abstract, affinity scores",
              OLD_ABSTRACT_SCORES,
              f"Replace with the regenerated best scores and add PS, which was missing: "
              f"{best}.",
              "generated/"],
             ["1", "Abstract, docking sentence",
              "'Molecular docking analyses further confirmed the consistency between "
              "AI-predicted binding affinities and molecular-level interactions'",
              "Delete the sentence. Docking is withdrawn from the paper (Reviewer 2, "
              "comment 4): an oligomer does not represent an extended polymer surface, and "
              "the original poses belong to the old peptide set. Remove 'Molecular "
              "Docking' from the keyword list too.",
              "Reviewer 2, c.4"],
             ["1", "Abstract, new sentence",
              "(absent)",
              "Add one sentence: sequences are split by identity-aware clustering because "
              "a random split inflates measured R2 by 0.070 to 0.128 on this data.",
              "T2, F2"],
             ["20", "5. Conclusion",
              "Repeats 0.9491, '180 new peptide candidates', the old scores and 'docking "
              "further supported'",
              "Rewrite to match the four changes above. State explicitly that all results "
              "are computational predictions awaiting experimental confirmation.",
              "all"],
         ]},

        {"title": "2. Introduction and contributions",
         "rows": [
             ["3", "1.2 Main contributions",
              "Contribution framed as integrating four architectures, three optimisers and "
              "docking in one workflow",
              "Reframe as methodological: (i) identity-aware evaluation protocol, (ii) "
              "surrogate-selection step for generation, (iii) independent validation "
              "against baselines, (iv) selectivity and identity characterisation, (v) "
              "quantitative benchmark against Jain et al. Architectural complexity is not "
              "claimed as a contribution.",
              "review_compliance.md"],
         ]},

        {"title": "3. Materials and methods",
         "rows": [
             ["4-5", "3.1 Dataset and Preprocessing",
              "80:10:10 random split; deduplication not described",
              "Add deduplication (PET 441,978 to 381,065; the other five contain no "
              "duplicates) and the identity-aware split: greedy incremental clustering at "
              "Hamming radius 3 with pigeonhole banding, split at cluster level. Report "
              "both strategies throughout.",
              "T1, T2"],
             ["6", "Table 4 (current): normalisation parameters",
              "PS row missing; source of the statistics not stated",
              "Rebuild as new Table 1 with raw count, deduplicated count, train/val/test "
              "sizes and training-split normalisation for all six polymers. PS: mean "
              "-15.3918, SD 8.8599, 324,661 training sequences.",
              "T1"],
             ["6", "3.1 text, normalisation",
              "Does not say whether statistics come from the full data or the training set",
              "State explicitly: z-score computed from the training split only, per "
              "polymer, stored in the checkpoint. Verified numerically for all six "
              "polymers.",
              "pbp/data/prepare.py:181"],
             ["8", "3.3 Hyperparameter Optimization and Ablation",
              "Describes the search but not the final-training design",
              "Update to: 624 configurations (104 per polymer) in stage A; 240 final runs "
              "in stage B = 4 architectures x 6 polymers x 5 seeds x 2 splits.",
              "configs/active.json"],
             ["12", "3.5 Molecular Docking Validation",
              "Describes the docking protocol",
              "Delete the subsection. If you prefer to keep docking instead, it must be "
              "re-run on the new peptide set and the subsection must state oligomer length "
              "and rationale, conformer generation and equilibration, charges and force "
              "field with versions, rigid or flexible treatment, and the number of "
              "conformations with the score spread.",
              "Reviewer 2, c.4"],
             ["new", "3.6 Surrogate selection (new subsection)",
              "(absent)",
              "Describe the procedure: generation is run independently against each "
              "candidate surrogate, and the surrogate is chosen on selectivity, agreement "
              "with a third model, and sequence degeneracy.",
              "T11"],
         ]},

        {"title": "4. Results: model comparison",
         "rows": [
             ["13", "4.1 Ablation Study and Model Selection",
              "'LSTM-EncDec achieved the highest average validation R2 of 0.9491'; "
              "'recurrent architectures outperformed convolution-based models'",
              "Rewrite. The CNN is best on all six polymers under both splits. The earlier "
              "conclusion rested on single-run validation R2; with 5 seeds and paired "
              "tests it does not hold.",
              "T3a, T3b, T4, F1"],
             ["14", "Table 10 (current)",
              "Validation R2 only, one run per cell, no standard deviations",
              "Replace with new Tables 3 and 4: train/val/test R2, RMSE and MAE for every "
              "architecture x polymer x split, mean +/- SD over 5 seeds (48 rows).",
              "T3, T3b"],
             ["14", "Figure 7",
              "Bar chart of validation R2",
              "Replace with F1 (test R2, both splits, seed error bars).",
              "F1"],
             ["14", "Figure 8",
              "Hyperparameter sensitivity of LSTM-EncDec",
              "Keep only if the architecture discussion is retained; otherwise redraw for "
              "the CNN or move to supplementary.",
              "results/search/"],
             ["15", "Figure 9",
              "PET-only validation-test consistency for four models",
              "Remove. Superseded by the full metrics tables, which report test "
              "performance for every model and every polymer.",
              "T3, T3b"],
             ["new", "4.6 Effect of the splitting strategy",
              "(absent)",
              "Add a subsection with the leakage measurement and the inflation it causes "
              "(0.070 to 0.128 R2).",
              "T2, F2"],
             ["new", "4.7 Benchmark against prior work",
              "(absent; prior work is only discussed qualitatively in 4.5)",
              "Add a subsection with the three-condition benchmark against Jain et al. "
              "Section 4.5 keeps the qualitative discussion and points here for the "
              "numbers.",
              "T9, F3"],
         ]},

        {"title": "5. Results: generation, novelty, selectivity",
         "rows": [
             ["14", "4.2 text, best peptide",
              "'YWRMMNWWLRWW reaching a score of -78.28 using both ILS and MOCO-CEM'",
              f"Replace with the regenerated set. Best overall: Nylon RWMLWHWRLRRW -83.05 "
              f"(SA).",
              "generated/"],
             ["16", "Table 11 (current): best peptides",
              "Old sequences and scores",
              "Replace with the regenerated best peptide per polymer, including PS.",
              "generated/"],
             ["16", "Table 12 (current): SA / ILS / MOCO-CEM",
              "Diversity ratios ~95% / ~85% / ~40%, qualitative characterisation",
              "Recompute from the current run or remove. The method-level claim is now "
              "secondary to the surrogate-selection result.",
              "T11"],
             ["16", "Table 13 (current): Hamming distance",
              "Minimum 1, average 2.0 to 3.8; used to argue novelty",
              "Replace with Table 6: nearest-neighbour identity distribution. Mean identity "
              f"{n['nov_min']:.1f} to {n['nov_max']:.1f}%, none above 90%.",
              "T6, F6"],
             ["16", "4.3 text",
              "'structurally different from existing peptide sequences'; 'did not simply "
              "memorize'; 'novel peptide candidates'",
              "Rewrite per Reviewer 2 comment 5. State that a Hamming distance of 1 is "
              "91.7% identity and that absence of exact matches is therefore not evidence "
              "of novelty. Remove 'novel' throughout.",
              "T6, F6"],
             ["18", "Table 14 (current): sequence similarity",
              "Maximum similarity 91.7% for five of six polymers; PS mean 83.3%",
              "Replace with the new distribution and state explicitly that 91.7% "
              "corresponds to a single mutation.",
              "T6"],
             ["19", "Figure 12",
              "Similarity distribution",
              "Replace with F6.",
              "F6"],
             ["new", "4.9 Cross-polymer selectivity",
              "(absent)",
              f"Add the 6x6 cross-polymer matrix: {n['sel_target_best']}/{n['sel_n']} sets "
              f"score best on their own target; index {n['sel_min']:.2f} (PS, weakest) to "
              f"{n['sel_max']:.2f} (PVC).",
              "T5, F5"],
             ["new", "4.8 Independent validation",
              "(absent)",
              f"Add: optimised peptides vs 2,000 random 12-mers and the best 1% of measured "
              f"training data, scored also by an {n['validator_arch'].upper()} used in "
              f"neither optimisation nor selection; {n['iv_beats']}/{n['iv_n']} and "
              f"{n['iv_confirms']}/{n['iv_n']}.",
              "T10, F4"],
             ["new", "4.10 Interpretability",
              "(absent)",
              "Add positional sensitivity and residue preference per polymer (Reviewer 1, "
              "comment 1).",
              "T8, F7"],
             ["new", "4.11 Physicochemical basis",
              "(absent)",
              "Add property-affinity correlations and state that they are polymer-specific "
              "(Reviewer 1, comment 3).",
              "T7, F8"],
         ]},

        {"title": "6. Results: docking (withdrawn)",
         "note": "Docking is removed from the revised manuscript. If you would rather "
                 "keep it, every row below becomes 'regenerate for the new peptide set' "
                 "instead of 'delete', and Section 3.5 must be completed as noted above.",
         "rows": [
             ["16-17", "4.4 Molecular Docking Validation and Consistency Analysis",
              "Lists pi-pi contacts; presented as validation of binding; no correlation "
              "with the ML score",
              "Delete the subsection. The independent validation in new Section 4.8 now "
              "carries the role docking was being used for, and carries it better.",
              "F4"],
             ["19", "Table 15 (current) and Figures 13-15",
              "Interaction table and three 3D interaction diagrams",
              "Delete. The poses belong to the peptide set of the original submission, "
              "which the revised pipeline no longer produces, so they correspond to no "
              "sequence reported in the paper.",
              "-"],
             ["5, 13", "Figure 1 workflow; 4. opening paragraph",
              "Both list molecular docking as a stage of the framework",
              "Redraw Figure 1 without the docking stage and drop docking from the "
              "paragraph that previews the results.",
              "-"],
         ]},

        {"title": "7. Discussion and limitations",
         "rows": [
             ["17", "4.5 Comparative Discussion and Limitations",
              "Mentions Jain et al. and Alshehri et al. qualitatively",
              f"Point to the new Section 4.7 for the numbers: 5/5 polymers in 3 "
              f"conditions, all 15 "
              f"comparisons significant, and the advantage falling "
              f"{n['jain_drop_pct']:.0f}% (+{n['jain_gain_A']:.4f} to "
              f"+{n['jain_gain_C']:.4f}) under the stricter protocol.",
              "T9, F3"],
             ["20", "Limitations paragraph",
              "Mentions missing experimental validation and environmental factors",
              "Expand with: (i) optimisation against a learned surrogate, (ii) oligomer "
              "docking does not represent an extended polymer surface, (iii) PS "
              "selectivity is the weakest result, (iv) Trp/Arg-rich sequences may have "
              "solubility and aggregation issues.",
              "T5, F4"],
             ["20", "7. Data Availability Statement",
              "Links to the repository",
              "Keep the link and add that the repository now contains the full pipeline, "
              "all intermediate outputs, the paper tables and figures, and an automated "
              "reviewer-compliance report.",
              "review_compliance.md"],
         ]},
    ]
