# CodonPrint

**Classifying organism kingdoms from codon-usage patterns**

CodonPrint is a machine-learning project that explores whether an organism's codon-usage profile can identify its taxonomic kingdom. It includes a reproducible data and model artifact generator and an interactive Streamlit dashboard for exploring predictions, evaluation results, and biological patterns.

## Project goals

- Compare supervised classifiers for 11-kingdom prediction from codon-use features.
- Explore codon-profile structure with PCA and a Self-Organizing Map (SOM).
- Evaluate a biology-weighted Random Forest model, **CodonPrint**, and compare feature sets.
- Explore whether viruses with greater model probability assigned to eukaryotic kingdoms are more likely to be misclassified.

The Host-Mimicry Score (HMS) is a project-defined model diagnostic. It is not confirmed host metadata and does not by itself establish evolutionary causation.

## Dataset

The project uses the [UCI Codon Usage dataset (ID 577)](https://archive.ics.uci.edu/dataset/577/codon+usage), derived from Codon Usage Tabulated from GenBank (CUTG). It contains 64 codon-frequency measurements and kingdom labels. The generation script cleans the source data, creates RSCU and GC3 features, and uses a stratified 80/20 train/test split.

The engineered model input contains 126 features:

- 64 raw codon frequencies
- 61 Relative Synonymous Codon Usage (RSCU) values
- 1 GC3 value

## Quick start

You need Python and an internet connection for the initial dataset download.

```bash
python -m pip install -r requirements.txt
python generate_artifacts.py
python -m streamlit run app.py
```

Open the local URL printed by Streamlit in your browser. The first run of `generate_artifacts.py` can take several minutes: it downloads the dataset, trains models, runs stratified cross-validation and a feature ablation, and builds the SOM.

Run the generator from the project directory. By default it writes the generated CSV and model artifacts next to `app.py`, where the dashboard expects them. An alternate output directory can be specified for isolated testing:

```bash
python generate_artifacts.py --output-dir ./generated-artifacts
```

The dashboard will still look for artifacts beside `app.py`; use the default output location for a standard run.

## What the generator creates

`generate_artifacts.py` recreates the local project artifacts from the UCI source dataset, including:

- Cleaned source data, engineered features, and train/test splits.
- PCA coordinates, PCA/scaler models, and SOM coordinates/model.
- Logistic Regression, Random Forest, RBF-SVM, and CodonPrint model files.
- Cross-validation, classifier-comparison, feature-importance, ablation, and per-class result CSVs.
- Viral misclassification and Host-Mimicry Score analysis CSVs.

The generated files are intentionally excluded from Git: dataset and result CSVs, model `.pkl` files, and chart PNGs. The dashboard renders its charts at runtime, including the CodonPrint confusion matrix and one-vs-rest ROC curves.

## Dashboard contents

The Streamlit dashboard provides:

1. **Organism Explorer** — predicted kingdom, confidence, probabilities, and codon-profile visualization.
2. **PCA Space** — interactive views of codon-profile positions across kingdoms.
3. **SOM Map** — an unsupervised map of organism codon profiles.
4. **Classifier Comparison** — comparison metrics and Random Forest feature importance.
5. **Virus / Host Mimicry** — SVM viral misclassification examples and model outputs.
6. **CodonPrint** — model design, class weights, performance, feature ablation, confusion matrix, ROC curves, and HMS analysis.

For detailed instructions on using the dashboard, see the [Dashboard User Guide](./dashboard_user_guide.md).

## Reproducibility notes

- The generator uses a fixed random seed and stratified split for reproducible training structure.
- Generated evaluation scores can differ from values in the project report because the script retrains the models and performs its own evaluation.
- Pickled scikit-learn models should be loaded using the version pinned in `requirements.txt`.
- The project dataset and reports are for educational and research use; consult the [UCI dataset page](https://archive.ics.uci.edu/dataset/577/codon+usage) for source and dataset details.

## Project files

| File | Purpose |
|---|---|
| `app.py` | Streamlit dashboard |
| `generate_artifacts.py` | Downloads data and generates local data, evaluation, and model artifacts |
| `requirements.txt` | Python dependencies |
| `dashboard_user_guide.md` | Detailed dashboard instructions |
| `Final_Project_Report.md` | Full project report |
| `Literature Review.pdf` | Supplied literature review |

## Team

**The Outliers** — CSE4889 Machine Learning  
Md. Samiur Rahman Sameer · Sumaiya Islam
