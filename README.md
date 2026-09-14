# chronic-kidney-disease-classifier
Machine learning classification project using a Chronic Kidney Disease dataset, covering data cleaning, preprocessing, model comparison, cross-validation, and classification performance evaluation.

The `src` scripts use the SVM selected in the notebook (`SVC()`), with its
default decision rule.

- `preprocessing.py`: clean values, impute missing data, scale numeric features, and encode categories.
- `train.py`: train on a stratified 80% split, report test metrics, and save the fitted pipeline.
- `predict.py`: return predictions (`1` = CKD, `0` = not CKD).

Install dependencies and run from the project root:

```sh
pip install pandas "scikit-learn>=1.2" joblib
python -m src.train
python -m src.predict "kidney_disease .csv"
```

Prediction CSVs need the original 24 predictor columns; `id` and `classification`
are optional. The saved pipeline includes cleaning and fitted preprocessing.
