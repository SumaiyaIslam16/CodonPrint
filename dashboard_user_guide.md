# CodonPrint Dashboard User Guide

**CSE4889 Machine Learning · The Outliers**

## What the dashboard does

The dashboard explores whether codon-usage features can predict an organism's kingdom. It shows an SVM prediction for a selected organism, maps codon profiles with PCA and a Self-Organizing Map (SOM), compares classification models, and examines viral misclassifications and CodonPrint's Host-Mimicry Score (HMS).

The dataset has 13,026 cleaned records across 11 kingdoms. The classification input contains 64 raw codon frequencies, 61 Relative Synonymous Codon Usage (RSCU) values, and one GC3 feature. The dashboard uses generated local model and data artifacts; charts are drawn in the app rather than loaded from PNG files.

## Start the dashboard

From the project folder, install dependencies and generate the required artifacts before launching the dashboard:

```bash
python -m pip install -r requirements.txt
python generate_artifacts.py
python -m streamlit run app.py
```

Artifact generation downloads the UCI dataset and trains models, runs evaluation and ablation experiments, and prepares PCA/SOM data. It needs an internet connection and can take several minutes. Run the generator with its default output directory so that the artifacts are saved beside `app.py`.

## Sidebar: choose an organism

1. Use **Filter by kingdom** to narrow the available organisms, or leave it set to **All**.
2. Use **Select organism** to choose a named organism. The label includes its kingdom code.
3. Explore the tabs; the organism selection is shared across the organism, PCA, and SOM views.

## Tab 1 — Organism Explorer

This tab summarizes the selected organism using the SVM model.

### Summary cards

- **SVM prediction:** the kingdom predicted by the RBF-SVM.
- **Verdict:** whether the prediction matches the dataset label.
- **Model confidence:** the SVM probability for its top predicted kingdom. Confidence is not a guarantee that a prediction is correct.
- **GC3 content:** the fraction of codon usage ending in G or C at the third position.

### RSCU Codon Fingerprint

Each bar represents one of the 61 sense codons:

- **RSCU > 1:** used more often than expected within its synonymous-codon family.
- **RSCU < 1:** used less often than expected.
- **RSCU near 1:** usage is near the family expectation.

This chart describes the organism's relative codon preferences. It is one view of the feature profile, not a full DNA sequence.

### Kingdom Probabilities

The horizontal bars show the SVM's predicted probability across the kingdom labels. Compare the true kingdom with the top prediction and the other probabilities to see whether the model appears decisive or uncertain.

### Top Feature Importances

The chart ranks PCA components by the Random Forest baseline's feature-importance values. These are model importance scores, not direct evidence that a component or an individual codon causes a taxonomic difference.

## Tab 2 — PCA Space

This interactive scatter plot shows organisms in principal-component coordinates. Each dot is an organism, and the color indicates its labeled kingdom. The selected organism is highlighted with a star.

Use the **X axis** and **Y axis** selectors to inspect different components. Nearby points have similar coordinates in the displayed PCA projection; overlap means that this two-dimensional view does not cleanly separate those samples.

PCA is a dimensionality-reduction visualization. It does not prove evolutionary relatedness, and apparent overlap alone does not establish why organisms have similar codon usage.

## Tab 3 — SOM Map

The Self-Organizing Map places codon profiles into a 20 × 20 grid. Similar profiles tend to map to nearby grid locations. The selected organism is marked with a star; cell color indicates the kingdom associated with mapped organisms.

Use the map to explore whether profile neighborhoods look concentrated or mixed. A SOM is an unsupervised visualization: its grid cells are not kingdom predictions, and mixed patterns should be interpreted as exploratory rather than as proof about taxonomy.

## Tab 4 — Classifier Comparison

The results table compares Logistic Regression, the PCA-based Random Forest, RBF-SVM, and CodonPrint. Values are loaded from the generated evaluation CSV, so results can differ from the project report or older screenshots after artifacts are regenerated.

Key metrics:

- **Balanced accuracy:** average recall across kingdoms, giving each class equal weight.
- **Macro-F1:** average of per-class F1 scores, giving small and large kingdoms equal weight.
- **Weighted-F1:** average F1 weighted by each kingdom's number of records.
- **CV:** cross-validation results on the training split.
- **Test:** results on the held-out test split.

Because class sizes are uneven, compare balanced accuracy and macro-F1 as well as weighted-F1. The feature-importance chart shows the highest-ranked PCA components from the baseline RF.

## Tab 5 — Virus / Host Mimicry

This tab lists viruses misclassified by the RBF-SVM and highlights errors predicted as eukaryotic kingdoms. The summary count, top predicted eukaryote classes, confidence, and table contents are generated from the current test split and may change when the dataset or artifacts change.

Codon usage can be influenced by multiple factors, including host-related selection. Therefore, a virus receiving a high probability for a eukaryotic class can motivate biological investigation, but a classifier output alone does not identify the virus's host or prove host mimicry. Treat examples as hypotheses to investigate, not confirmed host relationships.

## Tab 6 — CodonPrint

CodonPrint is the project's biology-weighted Random Forest using all 126 engineered features. This section has five subtabs:

### Design & Weights

Review the feature representation, class-weight rationale, model architecture, and class-weight visualization. The model combines class-frequency balancing with project-defined multipliers for selected classes.

### Performance

Compare baseline RF and CodonPrint test metrics and per-kingdom F1 scores. The figures and summary values come from generated evaluation results; some kingdoms may improve while others do not. Check the per-class values rather than assuming a uniform gain.

### Ablation Study

Compare model results using raw codons, RSCU, GC3 combinations, and the full feature set. The generated table and chart update when the artifact-generation script is rerun. Small differences should be interpreted cautiously because the displayed ablation results are point estimates.

### Confusion Matrix

The app computes and displays CodonPrint's confusion matrix directly from the held-out test labels and model predictions. Rows are true kingdoms; columns are predicted kingdoms; diagonal cells are correct predictions. The table below the charts lists the most frequent off-diagonal confusions for this test run.

The app also generates one-vs-rest ROC curves from the model's test-set probabilities. Each curve evaluates one kingdom against all other kingdoms; its legend includes the corresponding AUC.

### Host-Mimicry Score

For virus samples, HMS is calculated as the CodonPrint probability assigned to the six eukaryotic kingdoms:

```text
HMS = P(inv) + P(mam) + P(pln) + P(pri) + P(rod) + P(vrt)
```

The tab compares the mean HMS for correctly and incorrectly classified viruses and lists the highest-scoring samples. HMS summarizes the model's probability distribution; it is not a measured host relationship. Differences between correct and incorrect groups are associations and do not establish biological causation.

## Suggested exploration

1. Start with **All** in the kingdom filter and select an organism by name.
2. In **Organism Explorer**, compare the SVM prediction, confidence, and codon fingerprint.
3. Find the organism in **PCA Space** and **SOM Map** to inspect its neighborhood.
4. Compare model metrics in **Classifier Comparison**, giving attention to balanced accuracy and macro-F1.
5. In **Virus / Host Mimicry**, review current misclassified-virus examples while keeping in mind that predictions are not host labels.
6. In **CodonPrint**, compare the class-level results, feature ablation, confusion matrix, ROC curves, and HMS.

## Kingdom code reference

| Code | Kingdom |
|---|---|
| `arc` | Archaea |
| `bct` | Bacteria |
| `inv` | Invertebrate |
| `mam` | Mammal |
| `phg` | Bacteriophage |
| `plm` | Plasmid |
| `pln` | Plant |
| `pri` | Primate |
| `rod` | Rodent |
| `vrl` | Virus |
| `vrt` | Vertebrate |

## Troubleshooting

- **Missing file or model error:** run `python generate_artifacts.py` from the project folder, then restart Streamlit.
- **Download failure:** confirm internet access and retry the generator.
- **Different metrics from an older report:** metrics are recalculated from the currently downloaded dataset and generated model run; this is expected.
- **Need to recreate artifacts elsewhere:** `python generate_artifacts.py --output-dir .\generated-artifacts` writes there, but the dashboard expects its files beside `app.py` unless the app's artifact directory is changed.
