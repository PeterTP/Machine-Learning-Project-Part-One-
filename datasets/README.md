# Small-data datasets used in the contextual-modelling experiments

This bundle contains the **same dataset versions/populations used in the experiments** discussed for the MSc AI small-data project.

## Main files

All teaching-ready datasets are in `data/`. The final column is called `target`.

| Dataset | File | n | p | classes |
|---|---|---:|---:|---:|
| Iris | `data/iris.csv` | 150 | 4 | 3 |
| Wine | `data/wine.csv` | 178 | 13 | 3 |
| Breast Cancer Wisconsin (Diagnostic) | `data/breast_cancer_wisconsin_diagnostic.csv` | 569 | 30 | 2 |
| Optical Digits (scikit-learn subset) | `data/optical_digits_sklearn.csv` | 1797 | 64 | 10 |
| Glass2 | `data/glass2.csv` | 163 | 9 | 2 |
| Chess KR-vs-KP | `data/chess_kr_vs_kp.csv` | 3196 | 36 | 2 |
| Zoo | `data/zoo.csv` | 101 | 16 | 7 |

## Important reproducibility point

The experimental protocol **did not globally standardise these files in advance**. Features were standardised **within each training split** to avoid leakage. Students should therefore load these files first, split the data, then fit any scaler/other learned preprocessing on the training set only.

## Dataset-specific notes

### Iris, Wine, Breast Cancer, Optical Digits
These are the exact scikit-learn built-in dataset versions used in the experiments (`load_iris`, `load_wine`, `load_breast_cancer`, `load_digits`). Optical Digits is the 1,797-sample scikit-learn copy, not the complete 5,620-case UCI dataset.

### Glass2
`glass2.csv` is the 163-instance, 9-feature, binary **Glass2** benchmark distributed through PMLB/OpenML (OpenML dataset 40675), derived from UCI Glass Identification. This is the version used in our experiment, not the 214-case multiclass UCI Glass data and not the different 214-case KEEL `glass2` variant.

### Chess KR-vs-KP
The authoritative source copy is `source_copies/chess_kr_vs_kp_raw.csv`: 3,196 instances, 36 categorical predictors, plus the `won`/`nowin` target. For student convenience, `data/chess_kr_vs_kp.csv` contains deterministic integer coding of the 36 categorical predictors and `target` (`0=nowin`, `1=won`). The exact coding is recorded in `metadata/chess_encoding.csv`.

### Zoo
The authoritative source copy is `source_copies/zoo_raw.csv`. The teaching/experiment file `data/zoo.csv` removes the animal-name identifier, leaving the 16 predictive attributes. Boolean attributes are represented as 0/1 and the original UCI class numbering is used: 1=mammal, 2=bird, 3=reptile, 4=fish, 5=amphibian, 6=insect, 7=invertebrate.

## Sources / attribution

- Iris — UCI Machine Learning Repository, https://archive.ics.uci.edu/dataset/53/iris
- Wine — UCI Machine Learning Repository, https://archive.ics.uci.edu/dataset/109/wine
- Breast Cancer Wisconsin (Diagnostic) — UCI Machine Learning Repository, https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
- Optical Recognition of Handwritten Digits — scikit-learn `load_digits`; derived from UCI Optical Recognition of Handwritten Digits: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html
- Glass2 — PMLB/OpenML dataset 40675, derived from UCI Glass Identification: https://www.openml.org/d/40675
- Chess (King-Rook vs. King-Pawn) — UCI Machine Learning Repository: https://archive.ics.uci.edu/dataset/22/chess+king+rook+vs+king+pawn
- Zoo — UCI Machine Learning Repository: https://archive.ics.uci.edu/dataset/111/zoo

For formal attribution/licensing details, students should consult the linked source records. Current UCI records for Chess, Zoo and Glass state CC BY 4.0.

## Loading example

```python
import pandas as pd

D = pd.read_csv("data/wine.csv")
X = D.drop(columns="target")
y = D["target"]
```

For the original categorical Chess data, use `source_copies/chess_kr_vs_kp_raw.csv` and apply categorical encoding inside your ML pipeline.
