"""Download the UCI codon-usage data and recreate dashboard artifacts."""

from __future__ import annotations

import argparse
import io
import warnings
import urllib.request
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from minisom import MiniSom
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

CODON_GROUPS = (
    "UUU UUC",
    "UUA UUG CUU CUC CUA CUG",
    "AUU AUC AUA",
    "AUG",
    "GUU GUC GUA GUG",
    "UCU UCC UCA UCG AGU AGC",
    "CCU CCC CCA CCG",
    "ACU ACC ACA ACG",
    "GCU GCC GCA GCG",
    "UAU UAC",
    "CAU CAC",
    "CAA CAG",
    "AAU AAC",
    "AAA AAG",
    "GAU GAC",
    "GAA GAG",
    "UGU UGC",
    "UGG",
    "CGU CGC CGA CGG AGA AGG",
    "GGU GGC GGA GGG",
)
BIOLOGY_MULTIPLIERS = {
    "vrl": 2.0,
    "arc": 1.8,
    "plm": 3.0,
    "phg": 1.8,
    "mam": 1.3,
    "rod": 1.3,
    "pri": 1.3,
}
EUKARYOTIC_KINGDOMS = ("inv", "mam", "pln", "pri", "rod", "vrt")


def save_csv(frame: pd.DataFrame, filename: str, *, index: bool = False) -> None:
    frame.to_csv(OUTPUT_DIR / filename, index=index)


def download_dataset() -> pd.DataFrame:
    print("Downloading UCI Codon Usage dataset (dataset ID 577)...")
    url = "https://archive.ics.uci.edu/static/public/577/codon+usage.zip"
    request = urllib.request.Request(
        url, headers={"User-Agent": "CodonPrint-artifact-generator/1.0"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        archive_bytes = response.read()
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        try:
            csv_name = next(
                name for name in archive.namelist()
                if Path(name).name.lower() == "codon_usage.csv"
            )
        except StopIteration as error:
            raise FileNotFoundError("The UCI archive does not contain codon_usage.csv.") from error
        with archive.open(csv_name) as csv_file:
            data = pd.read_csv(csv_file, low_memory=False)
    data.columns = [str(column).strip() for column in data.columns]
    data["Kingdom"] = data["Kingdom"].astype(str).str.strip().str.lower()
    return data


def prepare_features(data: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    codon_columns = [
        column
        for column in data.columns
        if len(column) == 3 and set(column.upper()) <= set("ACGU")
    ]
    # UCI column ordering follows its published codon table rather than a
    # lexical sort; preserve the source order for raw features.
    source_columns = [
        column for column in data.columns
        if len(column) == 3 and set(column.upper()) <= set("ACGU")
    ]
    if len(source_columns) != 64:
        raise ValueError(f"Expected 64 codon columns; found {len(source_columns)}.")
    codon_columns = source_columns

    for column in codon_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=["Kingdom", *codon_columns]).copy()
    data = data[data["Kingdom"].isin(
        {"arc", "bct", "inv", "mam", "phg", "plm", "pln", "pri", "rod", "vrl", "vrt"}
    )].reset_index(drop=True)

    raw = data[codon_columns].astype(float)
    sense_codons = [codon for group in CODON_GROUPS for codon in group.split()]
    if len(sense_codons) != 61 or set(sense_codons) - set(codon_columns):
        raise ValueError("The codon synonym groups do not match the source dataset.")

    rscu = pd.DataFrame(index=data.index)
    for group in CODON_GROUPS:
        synonymous_codons = group.split()
        family_total = raw[synonymous_codons].sum(axis=1)
        values = raw[synonymous_codons].div(family_total.replace(0, np.nan), axis=0)
        values = values.mul(len(synonymous_codons)).fillna(0.0)
        for codon in synonymous_codons:
            rscu[f"RSCU_{codon}"] = values[codon]

    gc3_codons = [codon for codon in codon_columns if codon[-1] in {"G", "C"}]
    gc3 = raw[gc3_codons].sum(axis=1).rename("GC3")
    rscu_columns = list(rscu.columns)
    engineered = pd.concat([raw, gc3, rscu], axis=1)
    return engineered, codon_columns, rscu_columns


def make_models(class_weights: dict[str, float]) -> dict[str, object]:
    return {
        "LogisticRegression": LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=500,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "SVM_RBF": SVC(
            C=10.0,
            kernel="rbf",
            class_weight="balanced",
            probability=True,
            random_state=RANDOM_STATE,
        ),
        "CodonPrint": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight=class_weights,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def main() -> None:
    global OUTPUT_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT,
        help="Directory for generated CSV and model artifacts (default: project folder).",
    )
    args = parser.parse_args()
    OUTPUT_DIR = args.output_dir.resolve()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    warnings.filterwarnings("default", category=UserWarning)
    source = download_dataset()
    save_csv(source, "codon_usage.csv")

    codon_columns = [
        column for column in source.columns
        if len(column) == 3 and set(column.upper()) <= set("ACGU")
    ]
    for column in codon_columns:
        source[column] = pd.to_numeric(source[column], errors="coerce")
    cleaned = source.dropna(subset=["Kingdom", *codon_columns]).copy()
    cleaned = cleaned[cleaned["Kingdom"].isin(
        {"arc", "bct", "inv", "mam", "phg", "plm", "pln", "pri", "rod", "vrl", "vrt"}
    )].reset_index(drop=True)
    save_csv(cleaned, "codon_usage_clean.csv")

    engineered, codon_columns, rscu_columns = prepare_features(cleaned)
    data_with_gc3 = cleaned.copy()
    data_with_gc3["GC3"] = engineered["GC3"]
    save_csv(data_with_gc3, "codon_usage_with_gc3.csv")
    labels = cleaned["Kingdom"].astype(str)
    metadata_columns = [
        column for column in ("SpeciesName", "DNAtype")
        if column in cleaned.columns
    ]
    metadata = cleaned[metadata_columns].copy()
    metadata["Kingdom"] = labels

    train_indices, test_indices = train_test_split(
        np.arange(len(cleaned)),
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels,
    )
    X_train = engineered.iloc[train_indices].reset_index(drop=True)
    X_test = engineered.iloc[test_indices].reset_index(drop=True)
    y_train = labels.iloc[train_indices].reset_index(drop=True)
    y_test = labels.iloc[test_indices].reset_index(drop=True)
    ordered_metadata = pd.concat(
        [metadata.iloc[train_indices], metadata.iloc[test_indices]],
        ignore_index=True,
    )
    save_csv(ordered_metadata[["SpeciesName", "Kingdom"]], "species_names.csv", index=True)
    metadata_test = metadata.iloc[test_indices].reset_index(drop=True)
    save_csv(X_train, "X_train.csv")
    save_csv(X_test, "X_test.csv")
    save_csv(y_train.to_frame("Kingdom"), "y_train.csv")
    save_csv(y_test.to_frame("Kingdom"), "y_test.csv")

    pca_input_columns = rscu_columns + ["GC3"]
    scaler = StandardScaler()
    pca = PCA(n_components=50, random_state=RANDOM_STATE)
    pca_train = pca.fit_transform(scaler.fit_transform(X_train[pca_input_columns]))
    pca_test = pca.transform(scaler.transform(X_test[pca_input_columns]))
    pc_columns = [f"PC{index}" for index in range(1, 51)]
    pca_train_frame = pd.DataFrame(pca_train, columns=pc_columns)
    pca_test_frame = pd.DataFrame(pca_test, columns=pc_columns)
    pca_train_frame["Kingdom"] = y_train
    pca_test_frame["Kingdom"] = y_test
    save_csv(pca_train_frame, "pca_train_coords.csv")
    save_csv(pca_test_frame, "pca_test_coords.csv")
    joblib.dump(scaler, OUTPUT_DIR / "pca_scaler.pkl", compress=3)
    joblib.dump(pca, OUTPUT_DIR / "pca_model.pkl", compress=3)

    classes, class_counts = np.unique(y_train, return_counts=True)
    base_weights = len(y_train) / (len(classes) * class_counts)
    class_weights = {
        str(kingdom): float(weight * BIOLOGY_MULTIPLIERS.get(str(kingdom), 1.0))
        for kingdom, weight in zip(classes, base_weights)
    }
    models = make_models(class_weights)
    pca_train_df = pd.DataFrame(pca_train, columns=pc_columns)
    pca_test_df = pd.DataFrame(pca_test, columns=pc_columns)
    trained = {}
    for name in ("LogisticRegression", "RandomForest", "SVM_RBF"):
        print(f"Training {name}...")
        trained[name] = clone(models[name]).fit(pca_train_df, y_train)
        joblib.dump(trained[name], OUTPUT_DIR / {
            "LogisticRegression": "lr_model.pkl",
            "RandomForest": "rf_model.pkl",
            "SVM_RBF": "svm_model.pkl",
        }[name], compress=3)

    print("Training CodonPrint...")
    trained["CodonPrint"] = clone(models["CodonPrint"]).fit(X_train, y_train)
    joblib.dump(trained["CodonPrint"], OUTPUT_DIR / "custom_rf_model.pkl", compress=3)
    save_csv(
        pd.DataFrame({
            "PC": pc_columns,
            "importance": trained["RandomForest"].feature_importances_,
        }).sort_values("importance", ascending=False),
        "rf_feature_importance.csv",
    )

    # Cross-validation and held-out evaluation for all dashboard classifiers.
    comparison_rows = []
    X_cv = {
        "LogisticRegression": pca_train_df,
        "RandomForest": pca_train_df,
        "SVM_RBF": pca_train_df,
        "CodonPrint": X_train,
    }
    cv_filenames = {
        "LogisticRegression": "cv_results_lr.csv",
        "RandomForest": "cv_results_rf.csv",
        "SVM_RBF": "cv_results_svm.csv",
        "CodonPrint": "cv_results_custom_rf.csv",
    }
    for name, model in models.items():
        print(f"Running {CV_FOLDS}-fold CV for {name}...")
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        scores = cross_validate(
            model, X_cv[name], y_train, cv=cv,
            scoring={
                "balanced_accuracy": "balanced_accuracy",
                "f1_macro": "f1_macro",
                "f1_weighted": "f1_weighted",
            },
            n_jobs=1, error_score="raise",
        )
        cv_frame = pd.DataFrame({
            "model": name,
            "fold": np.arange(1, CV_FOLDS + 1),
            "balanced_accuracy": scores["test_balanced_accuracy"],
            "f1_macro": scores["test_f1_macro"],
            "f1_weighted": scores["test_f1_weighted"],
        })
        save_csv(cv_frame, cv_filenames[name])

        test_features = X_test if name == "CodonPrint" else pca_test_df
        predictions = trained[name].predict(test_features)
        scores_row = {
            "model": name,
            "bal_acc_mean": float(cv_frame["balanced_accuracy"].mean()),
            "bal_acc_std": float(cv_frame["balanced_accuracy"].std(ddof=0)),
            "f1_macro_mean": float(cv_frame["f1_macro"].mean()),
            "f1_macro_std": float(cv_frame["f1_macro"].std(ddof=0)),
            "f1_wtd_mean": float(cv_frame["f1_weighted"].mean()),
            "f1_wtd_std": float(cv_frame["f1_weighted"].std(ddof=0)),
            "test_bal_acc": balanced_accuracy_score(y_test, predictions),
            "test_f1_macro": f1_score(y_test, predictions, average="macro", zero_division=0),
            "test_f1_wtd": f1_score(y_test, predictions, average="weighted", zero_division=0),
        }
        comparison_rows.append(scores_row)

    comparison = pd.DataFrame(comparison_rows)
    save_csv(comparison, "classifier_comparison_table.csv")
    baseline = comparison.set_index("model").loc["RandomForest"]
    custom = comparison.set_index("model").loc["CodonPrint"]
    save_csv(pd.DataFrame({
        "Model": ["Baseline RF", "CodonPrint (ours)"],
        "Balanced_Accuracy": [baseline["test_bal_acc"], custom["test_bal_acc"]],
        "F1_Macro": [baseline["test_f1_macro"], custom["test_f1_macro"]],
    }), "custom_rf_comparison.csv")

    baseline_pred = trained["RandomForest"].predict(pca_test_df)
    custom_pred = trained["CodonPrint"].predict(X_test)
    perclass = pd.DataFrame({
        "Kingdom": classes,
        "Baseline_F1": f1_score(y_test, baseline_pred, labels=classes, average=None, zero_division=0),
        "Custom_F1": f1_score(y_test, custom_pred, labels=classes, average=None, zero_division=0),
    })
    perclass["Change"] = perclass["Custom_F1"] - perclass["Baseline_F1"]
    save_csv(perclass, "custom_rf_perclass_f1.csv")

    # SVM viral errors and CodonPrint host-mimicry probabilities.
    svm_probabilities = trained["SVM_RBF"].predict_proba(pca_test_df)
    svm_classes = np.asarray(trained["SVM_RBF"].classes_)
    svm_pred = svm_classes[np.argmax(svm_probabilities, axis=1)]
    svm_virus_mask = (y_test.to_numpy() == "vrl") & (svm_pred != "vrl")
    svm_errors = pd.DataFrame(svm_probabilities[svm_virus_mask], columns=svm_classes)
    svm_errors["SpeciesName"] = metadata_test.loc[svm_virus_mask, "SpeciesName"].to_numpy()
    svm_errors["y_pred_svm"] = svm_pred[svm_virus_mask]
    if "DNAtype" in metadata_test.columns:
        svm_errors["DNAtype"] = metadata_test.loc[svm_virus_mask, "DNAtype"].to_numpy()
    svm_errors["confidence_wrong"] = np.max(svm_probabilities[svm_virus_mask], axis=1)
    svm_errors["prob_vrl"] = svm_errors["vrl"]
    save_csv(
        svm_errors.sort_values("confidence_wrong", ascending=False),
        "vrl_misclassified_proba.csv",
    )

    custom_probabilities = trained["CodonPrint"].predict_proba(X_test)
    custom_classes = np.asarray(trained["CodonPrint"].classes_)
    class_positions = {str(label): index for index, label in enumerate(custom_classes)}
    euk_positions = [class_positions[label] for label in EUKARYOTIC_KINGDOMS]
    virus_mask = y_test.to_numpy() == "vrl"
    virus_probabilities = custom_probabilities[virus_mask]
    virus_predictions = custom_pred[virus_mask]
    euk_probabilities = virus_probabilities[:, euk_positions]
    top_euk_indices = np.argmax(euk_probabilities, axis=1)
    host_scores = pd.DataFrame({
        "true_label": y_test[virus_mask].to_numpy(),
        "predicted": virus_predictions,
        "host_mimicry_score": euk_probabilities.sum(axis=1),
        "correctly_classified": virus_predictions == "vrl",
        "top_euk_host": np.asarray(EUKARYOTIC_KINGDOMS)[top_euk_indices],
    })
    save_csv(host_scores, "host_mimicry_scores.csv")

    ablation_sets = [
        ("Raw 64 codons only", codon_columns),
        ("Raw + GC3", codon_columns + ["GC3"]),
        ("RSCU only", rscu_columns),
        ("RSCU + GC3", rscu_columns + ["GC3"]),
        ("Raw + RSCU + GC3 (ours)", list(engineered.columns)),
    ]
    ablation_rows = []
    for name, columns in ablation_sets:
        if name == "Raw + RSCU + GC3 (ours)":
            prediction = custom_pred
        else:
            print(f"Running ablation: {name}...")
            model = clone(models["CodonPrint"]).fit(X_train[columns], y_train)
            prediction = model.predict(X_test[columns])
        ablation_rows.append({
            "Feature Set": name,
            "Balanced Acc": balanced_accuracy_score(y_test, prediction),
            "F1 Macro": f1_score(y_test, prediction, average="macro", zero_division=0),
        })
    save_csv(pd.DataFrame(ablation_rows), "ablation_feature_sets.csv")

    save_csv(pd.DataFrame([
        ("vrl", 2.0, "Viruses can adapt codon usage to host translation machinery."),
        ("arc", 1.8, "Archaea and bacteria have overlapping codon-usage profiles."),
        ("plm", 3.0, "Plasmids are the smallest class and need additional attention."),
        ("phg", 1.8, "Phages can adapt codon usage to bacterial hosts."),
        ("mam/rod/pri", 1.3, "Rare, biologically related mammalian subgroups."),
        ("bct/pln/vrt/inv", 1.0, "Default multiplier for other kingdoms."),
    ], columns=["Kingdom", "Multiplier", "Biological Reason"]), "weight_justification.csv")

    # SOM receives PCA coordinates for both partitions in the same order as X_all.
    print("Training 20x20 self-organizing map...")
    som_features = np.vstack([pca_train[:, :20], pca_test[:, :20]])
    som = MiniSom(
        20, 20, 20, sigma=1.0, learning_rate=0.5, random_seed=RANDOM_STATE
    )
    som.random_weights_init(pca_train[:, :20])
    som.train_random(pca_train[:, :20], 10_000, verbose=True)
    joblib.dump(som, OUTPUT_DIR / "som_model.pkl", compress=3)
    bmus = np.asarray([som.winner(row) for row in som_features], dtype=int)
    all_labels = pd.concat([y_train, y_test], ignore_index=True)
    save_csv(pd.DataFrame({
        "BMU_x": bmus[:, 1],
        "BMU_y": bmus[:, 0],
        "Kingdom": all_labels,
    }), "som_bmu_coords.csv")

    print("Artifact generation complete.")
    print(f"Clean records: {len(cleaned):,}; train: {len(X_train):,}; test: {len(X_test):,}")
    print(f"Generated CSV and local model artifacts in {OUTPUT_DIR}.")


if __name__ == "__main__":
    main()
