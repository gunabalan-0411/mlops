# MLOps — Interview Revision Notes (Restructured)

> Goal: quick revision + ability to explain each component in simple terms.

---

## 0) What is MLOps? (1-line answer)

**MLOps = DevOps + ML** → practices to **build, deploy, monitor, and continuously improve ML models reliably**.

### Why companies need MLOps?

* ML models **decay** over time (data drift, concept drift)
* Training/retraining should be **repeatable and traceable**
* Deployment must be **safe + scalable**
* Teams need **collaboration + governance**

---

## 1) Machine Learning Lifecycle (End-to-End)

This is the standard pipeline from idea → production → monitoring.

### 1. Problem Definition

* Define **business objective**, constraints, and measurable success metric.
* Output: problem statement + evaluation metric.

### 2. Data Collection

* Data sources: DBs, APIs, logs, files, web, sensors.
* Output: raw dataset + data dictionary.

### 3. Data Cleaning

* Handle missing values, duplicates, outliers, wrong data types.
* Output: clean dataset.

### 4. Feature Engineering / Transformation

* Scaling/normalization, encoding categorical variables, text vectorization.
* Output: feature pipeline.

### 5. Model Selection

* Choose model family based on constraints.

  * Linear models → speed + interpretability
  * Tree/boosting → strong baseline for tabular
  * Deep learning → images/audio/NLP

### 6. Model Training

* Train on training data, validate with holdout/cross-validation.

### 7. Model Evaluation

* Metrics: accuracy, precision/recall/F1, ROC-AUC, RMSE, etc.
* Output: evaluation report.

### 8. Hyperparameter Tuning

* Grid search / Random search / Bayesian optimization.

### 9. Deploying

* Model served via API, batch scoring, or streaming.

### 10. Monitoring & Maintenance

* Monitor: latency, errors, drift, performance drop.
* Retrain + redeploy safely.

✅ **Interview summary:**

> “Lifecycle is not linear. It’s iterative. Monitoring feeds back into data and retraining.”

---

## 2) Version Control (Code, Data, Models)

### A) Code versioning (Git)

* Tracks code changes and enables collaboration.

### B) Data versioning (DVC)

* Data is too large for Git → **DVC versions large datasets**.
* Git stores **.dvc metadata (checksums)**, actual data stored in **remote storage**.

### C) Model versioning

* Store models as artifacts + track metadata (which data + code created it).
* Usually done via **MLflow model registry** or artifact store.

---

## 3) CI/CD in MLOps (Big Picture)

### What is CI?

* **Continuous Integration** = validate every change (tests, linting, training runs).

### What is CD?

* **Continuous Delivery/Deployment** = push model/app to environments automatically.

### Why CI/CD is harder in ML?

Because changes are not only code changes:

* Data changes
* Features change
* Model behavior changes

---

## 4) CI Example — Training pipeline (GitHub Actions)

### Goal

Automatically train the model and store artifacts whenever code changes.

### Flow explained

* Trigger: push / PR to main
* Runner: Ubuntu
* Matrix build: Python 3.11 + 3.12
* Steps:

  1. Checkout
  2. Setup Python
  3. Install dependencies
  4. Run `train.py`
  5. Save model artifacts
  6. Upload artifacts to GitHub

### Your CI YAML (clean copy)

```yml
name: CI - Train and Save the model

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  matrix-pip:
    name: Train and Save the model
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python: [3.11, 3.12]

    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Setup python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python }}

      - name: Upgrade pip
        run: python -m pip install --upgrade pip setuptools wheel

      - name: Install project dependencies
        run: python -m pip install -r requirements.txt

      - name: Train the model
        run: |
          python train.py
          ls -la artifacts

      - name: Upload artifacts
        uses: actions/upload-artifact@v4
        with:
          name: ml-artifacts-${{ matrix.python }}-${{ github.run_id }}
          path: artifacts
```

✅ Interview line:

> “CI for ML is training + validation. The output artifact is the model file and evaluation report.”

---

## 5) Model Serving — Create API

### Goal

Expose model inference through an endpoint.

### Two common frameworks

* **FastAPI** (recommended): async, automatic docs, production-friendly
* Flask: simple and lightweight

### How inference API works

1. Client sends JSON request
2. API loads model
3. Model predicts
4. API returns JSON response

### Example curl

```bash
curl -X POST "http://127.0.0.1:5001/predict" \
  -H "Content-Type: application/json" \
  -d '{"features": [5.3, 3.5, 1.4, 0.2]}'
```

✅ Interview line:

> “A serving API wraps the model so applications can request predictions without caring about ML internals.”

---

## 6) Deployment Basics — Docker

### Why Docker?

* Same environment everywhere: dev → staging → prod
* Avoids “works on my machine” problems

### Dockerfile explained

* Use base image: `python:3.12-slim`
* Copy requirements
* Install dependencies
* Copy application
* Expose port
* Start app

```dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .

RUN python -m pip install --upgrade pip
RUN python -m pip install -r requirements.txt

COPY . .

EXPOSE 5001

CMD ["python", "app.py"]
```

### Build + Run

```bash
docker build -t hello-mlops:latest .
docker run -d -p 5001:5001 hello-mlops:latest

docker ps
```

### Best practice: `.dockerignore`

Include things like:

* `.git/`
* `__pycache__/`
* `.venv/`
* `data/` (if large)

✅ Interview line:

> “Docker packages model + API + dependencies into a portable unit.”

---

## 7) DVC (Data Version Control)

### What is DVC?

* Git-like workflow for datasets
* Stores **metadata in Git**
* Stores **actual data in remote storage** (S3/Azure/GCP/local)

### Why DVC?

* Reproducibility: “Which exact data version created this model?”
* Collaboration: team sync dataset versions

### Setup

```bash
pip install dvc

git init
dvc init

dvc add data/sample_data.csv
```

This creates:

* `data/sample_data.csv.dvc` (tracked by Git)
* `.dvc/cache` (data content cache)

### Remote storage (S3 example)

```bash
pip install dvc_s3
aws configure

dvc remote add -d dvc_demo s3://dvc_demo
```

### Workflow (step-by-step)

1. Add/update data

```bash
dvc add data/sample_data.csv
```

2. Commit metadata to git

```bash
git add data/sample_data.csv.dvc .gitignore
git commit -m "Track dataset with DVC"
```

3. Push data to remote

```bash
dvc push
```

### Clone scenario

```bash
git clone <repo>
cd <repo>
dvc pull
```

✅ Interview line:

> “DVC gives dataset versioning + remote storage; Git only stores the pointer.”

---

## 8) Experiment Tracking — MLflow

### What problem does MLflow solve?

When training multiple models:

* Which run gave best accuracy?
* What hyperparameters were used?
* Where are the artifacts stored?

MLflow tracks:

* parameters
* metrics
* artifacts
* models

---

### 8.1 Local MLflow setup

Install:

```bash
pip install mlflow
```

Run UI:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 7006
```

* `backend-store-uri` → where metadata (runs, params, metrics) is stored.
* UI helps compare runs.

---

### 8.2 MLflow in code (core commands)

Typical structure:

```python
import mlflow
import mlflow.sklearn

mlflow.set_experiment("iris_rf_experiment")

with mlflow.start_run():
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", 0.97)

    mlflow.sklearn.log_model(model, artifact_path="model")
    mlflow.log_artifact("feature_importance.csv")
```

### Autologging (easy mode)

```python
mlflow.sklearn.autolog()
```

It automatically logs:

* parameters
* metrics
* model artifacts

---

### 8.3 Remote tracking server

```python
mlflow.set_tracking_uri("http://my-mlflow-server:5000")
```

Tracking URI decides **where runs get stored**.

---

### 8.4 Model Registry (production feature)

Purpose: manage model lifecycle states.

Example:

```python
mlflow.register_model(model_uri=model_uri, name="IrisRFModel")
```

Load model:

```python
loaded = mlflow.sklearn.load_model(model_uri)
```

Registry concepts:

* Model name
* Version (v1, v2, ...)
* Stages: Staging / Production / Archived

✅ Interview line:

> “MLflow gives reproducibility: every model run is tied to parameters, metrics, artifacts, and code version.”

---

## 9) What you likely missed (Important topics to add)

These are common MLOps topics beyond what you noted (high interview value).

### A) Testing in MLOps

* Unit tests for feature functions
* Data validation tests
* Model performance regression tests

Tools:

* `pytest`
* Great Expectations / Pandera for data quality

### B) Data validation / schema checks

Why?

* Models break when input distribution/schema changes.

Checks:

* Column type mismatch
* Null % threshold
* Range validation

### C) Feature Store (concept)

* Central store for features used in training + serving.
* Prevents training-serving skew.

Examples: Feast, Tecton.

### D) Model monitoring in production

Monitor:

* latency, throughput
* error rate
* drift
* quality (if labels arrive)

### E) Retraining triggers

* schedule-based retrain (weekly/monthly)
* performance threshold retrain
* drift-based retrain

---

## 10) Interview Quick Revision (1-minute story)

Use this as a ready answer:

> “In ML lifecycle, after defining business problem, we collect and clean data, engineer features, and train models. In MLOps, we ensure reproducibility by versioning code in Git and data using DVC with S3 remote. For experiments, MLflow tracks parameters, metrics, and artifacts across runs and supports model registry for staging/production versions. For deployment, we wrap the model in a FastAPI/Flask service and containerize with Docker so it runs consistently across environments. CI pipelines like GitHub Actions automate training/testing and artifact generation. In real production, we add monitoring, drift detection, and CI/CD to retrain and redeploy safely.”

---

## 11) Mini Glossary (Interview-friendly)

* **Artifact**: output file (model.pkl, metrics.json, plots)
* **Tracking**: logging run history (params + metrics)
* **Registry**: manage model versions and stages
* **Drift**: data distribution changes
* **Training-serving skew**: train features != serving features

---

If you want, I can also add:

* **Deployment + full CI/CD workflow notes** (Kubernetes, Helm, GitHub Actions CD, canary rollout)
* **End-to-end architecture diagram** (simple + interview-ready)


## Model deployment and Serving
1. VM -> artifacts -> server
2. Kubernetes: -> container -> pod -> cluster
3. Managed: -> amazon sagemaker (Easy)
4. K-serve: simple way of using kubernetes

1. VM Approach (but huge costing as lot of vms)
train.py -> .joblib, .pkl -> fastapi (should support concurrency)(uvicorn server interface help to run with multiple worker)-> front end -> Dynamic scaling -> Load balancing (distribute to various vm (eacg vm with code and api))

## User data script
Autoscaling (ASG Autoscalling Group) -> create new vm based on more request
* launch template -> Userdata script (model, config etc) is necessary to create new vm by ASG
* in github usually there will be new branch for VM (kubernets, ec2, gcp etc)
* in this branch user_data.sh will be there. (list of script to set environment, model, etc etc)

## MOdel Deployment of Kubernetes
1. Dockerfile 
2. Model registery
3. Image -> Model
4. Kubernetes cluster (multiple) (we can also use kind, minikube, k3s to create a local kubernets cluster)
5. Prepare Kuberneters manfifest
6. config
7. Deployment
8. ingress controller
9. ingress
10. Inference

in a separate branch in git repo, we will be having a k8s-manifests
  - deployment.yml
  - namespace.yml
  - service.yaml

ingress -> INgress controller -> load balancer (PUblic facing)

## Deployment on Kserve
* 2-3 steps to deployment and serving when compare to vm and kubernetes which take 1-2 days to create things
* it offers frameworks
* autoscaling

* in git hub create a branch for kserve and have this kserver config files

## Amazon sagemaker AI (rebranded version for amazon sagemaker)

* A single platform where ds, mlops, devops can collaborate for ai projects.
* pros: managed ec2 for notebooks, less work on infra, 
* cons: costing ,not opensource

* It has, experiements, jobs, compute, apps like jupyterlab, rstudio, canvas, code editor, mlflow etc. (better than databricks)

1. Create training and inference in jupyter notebook in app->jupyter
2. export model and inference.py by zipping using tar
3. move the tar file to s3 bucket
4. run deploy.py: to import sklearnmodel from sagemaker lib and model.deploy() to sagemaker, get the model from s3 bucket
5. or use deployments option in left panel to deploy the model from s3 bucket to sagemaker

## Kubeflow

## Full CI/ CD Pipeline

* Git repo in two branches
1. Main branch contains normal ML codes
2. CICD branch contains k8s (deployment.yaml, inference.yaml, serviceaccount.yaml), argocd, api.py, generate_data.py, requirements.txt, train.py

In 2. CICD Branch

-- DVC
* install dvc, dvc-s3
* dvc init
* dvc remote add -d s3remote link // before create a folder in s3
* dvc add data/dataset.csv // create data/dataset.csv.dvc
* dvc push // push it s3 buck
* git add . & commit -m "update"
* upload model to s3 // create a folder in s3

-- Kubernetes cluster
* kind get clusters
* kind create cluster --name=model-name // creating kubernetes cluster
* kubectl create namespace kserve // creating kserve, kubectl get crds to check if kserver created
* kubectl create ns ml // to create a namespace created
* paste the manifest file in the namespace // contains apiversion, name, annotations, type, stringdata. 
* kubectl apply -f svcaccount.yaml // this will create a service account
* create inference.yaml,

-- github actions
* create folder .github/workflows
* create mlops-pipeline.yml
* |__ create name, on push, set environments, jobs: steps: setup python, install dependencies, generate dataset, train model, configure aws credentials, push model to s3, update inference.yml, update inference.yml
* add secrets to github

argocd...

 
 
