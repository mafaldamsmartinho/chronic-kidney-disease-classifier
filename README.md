# Chronic Kidney Disease Classifier

End-to-end machine learning project for Chronic Kidney Disease classification, covering data preprocessing, model comparison, evaluation, API development, containerized deployment, and cloud deployment.

## Tech Stack

Python · scikit-learn · FastAPI · Pydantic · pytest · Docker · Google Cloud Platform

## Model

The final model is an SVM (`SVC`) selected after comparing classification models using stratified cross-validation.

The saved pipeline includes:

- Data cleaning
- Missing-value imputation
- Numerical scaling
- Categorical encoding
- Trained classifier

Predictions:

- `1` — CKD
- `0` — Not CKD

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
## Google Cloud Deployment

The application was deployed to Google Cloud Run using Cloud Build and Artifact Registry.

The live service was removed after validation to avoid unnecessary cloud costs.

### 1. Authenticate and select the Google Cloud project

```bash
gcloud auth login
```

```bash
gcloud config set project <PROJECT_ID>
```

Example:

```bash
gcloud config set project project-bf8e7dce-6089-4d57-975
```

### 2. Enable the required APIs

```bash
gcloud services enable run.googleapis.com
gcloud services enable artifactregistry.googleapis.com
gcloud services enable cloudbuild.googleapis.com
```

### 3. Create an Artifact Registry repository

```bash
gcloud artifacts repositories create ckd-repo --repository-format=docker --location=europe-west1 --description="Docker repository for CKD classifier"
```

Check that the repository exists:

```bash
gcloud artifacts repositories list
```

### 4. Configure `.gcloudignore`

Keep the files required for deployment:


### 5. Check the Cloud Build service account

```bash
gcloud builds get-default-service-account
```

Grant permission to read uploaded build files:

```bash
gcloud projects add-iam-policy-binding <PROJECT_ID> --member="serviceAccount:<SERVICE_ACCOUNT>" --role="roles/storage.objectViewer"
```

Grant permission to write Cloud Build logs:

```bash
gcloud projects add-iam-policy-binding <PROJECT_ID> --member="serviceAccount:<SERVICE_ACCOUNT>" --role="roles/logging.logWriter"
```

Grant permission to push images to Artifact Registry:

```bash
gcloud projects add-iam-policy-binding <PROJECT_ID> --member="serviceAccount:<SERVICE_ACCOUNT>" --role="roles/artifactregistry.writer"
```

### 6. Build and push the Docker image

Cloud Build uses the project Dockerfile to build the image and push it directly to Artifact Registry.

```bash
gcloud builds submit --tag europe-west1-docker.pkg.dev/project-bf8e7dce-6089-4d57-975/ckd-repo/ckd-app:latest
```

### 7. Deploy the image to Cloud Run

```bash
gcloud run deploy ckd-app --image europe-west1-docker.pkg.dev/<PROJECT_ID>/ckd-repo/ckd-app:latest --region europe-west1 --platform managed --allow-unauthenticated
```

Cloud Run provides a public HTTPS URL where the application can be tested.

### 8. Delete the Cloud Run service

```bash
gcloud run services delete ckd-app --region europe-west1
```

### 9. Delete the entire Artifact Registry repository

```bash
gcloud artifacts repositories delete ckd-repo --location=europe-west1
```



