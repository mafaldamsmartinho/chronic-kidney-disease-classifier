# Chronic Kidney Disease Classifier

End-to-end machine learning project for Chronic Kidney Disease classification, covering data preprocessing, model comparison, evaluation, API development, and containerized deployment.

## Tech Stack

Python · scikit-learn · FastAPI · Pydantic · pytest · Docker

## Model

The final model is an SVM (`SVC`) selected after comparing classification models using stratified cross-validation.

The saved pipeline includes:

* Data cleaning
* Missing-value imputation
* Numerical scaling
* Categorical encoding
* Trained classifier

Predictions:

* `1` — CKD
* `0` — Not CKD

## Run with Docker

Clone the repository and build the image:

```bash
docker build -t ckd-classifier .
```

Run the container:

```bash
docker run -p 8000:8000 ckd-classifier
```

Open:

```text
http://localhost:8000
```

## Tests

```bash
pytest
```

