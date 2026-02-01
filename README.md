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
# MLOps Pending Notes — Full Deployment + CI/CD (AWS + Kubernetes + KServe)

> This section is **ONLY the pending part** after MLflow: full deployment + CI/CD workflow.
> Written for a **Data Scientist**: each new infra concept is explained in 1–2 lines.

---

## 1) Model Deployment & Serving — 4 Common Patterns

### 1. VM-based Deployment (EC2-style)

**Idea:** deploy model API directly on a Virtual Machine.

**Flow:**
`train.py → model.pkl/joblib → FastAPI → Uvicorn/Gunicorn → Load Balancer → Users`

**Why used?**

* Easy to understand
* Works well for low/moderate traffic

**Problems:**

* Scaling is manual or VM-heavy → cost ↑
* Each VM must be configured (dependencies, model download, API setup)

**Key terms (simple):**

* **VM/EC2:** a server you rent in cloud.
* **Load balancer:** distributes requests across multiple servers.
* **Concurrency:** ability to handle many requests at same time.
* **Uvicorn:** ASGI server for FastAPI.
* **Gunicorn:** process manager (can run multiple workers).

✅ Interview line:

> “VM deployment is simplest but scales poorly; infra maintenance is high.”

---

### 2. Kubernetes Deployment (Containers in a Cluster)

**Idea:** run your inference as a container inside a Kubernetes cluster.

**Flow:**
`Docker image → Pod → Service → Ingress → Users`

**Why used?**

* **Auto-scaling** and **self-healing** (if pod dies, it restarts)
* Standard approach for production ML systems

**Problems:**

* More moving pieces → learning curve

**Key terms (simple):**

* **Container (Docker):** package code + dependencies.
* **Kubernetes cluster:** group of machines running containers.
* **Pod:** smallest deployable unit (1+ containers).
* **Deployment:** manages replica pods and rolling updates.
* **Service:** stable networking endpoint to access pods.
* **Ingress:** exposes Service to the internet (public routing).
* **Ingress Controller:** the actual component that implements Ingress (like NGINX).

✅ Interview line:

> “Kubernetes is used to run containers reliably with scaling + rolling updates.”

---

### 3. Managed Deployment (Amazon SageMaker)

**Idea:** AWS manages infra; you focus on model + code.

**Flow:**
`Training job → model.tar.gz → S3 → Endpoint deployment`

**Why used?**

* Less infrastructure work
* Production-grade scaling, monitoring integrations

**Tradeoff:**

* More expensive
* Vendor lock-in

✅ Interview line:

> “SageMaker simplifies production by managing endpoints, scaling, and infra.”

---

### 4. KServe (Kubernetes but ML-friendly)

**Idea:** easiest + most ML-native way to serve models on Kubernetes.

**Why KServe over raw Kubernetes?**

* Plain k8s: you manually write deployment/service/ingress
* KServe: you define one YAML (`InferenceService`) and it handles:

  * autoscaling (even scale-to-zero)
  * traffic routing
  * canary rollouts
  * model server frameworks

✅ Interview line:

> “KServe is Kubernetes optimized for model serving—less YAML, more ML features.”

---

## 2) VM Deployment Detailed (EC2 + Auto Scaling)

### A) VM scaling problem

If traffic increases:

* 1 VM becomes bottleneck
* Solution: add more VMs and load balance

### B) Auto Scaling Group (ASG)

**ASG** automatically creates new EC2 VMs when:

* CPU high, request count high, etc.

### C) Launch Template

Blueprint for creating EC2 instances:

* instance type
* AMI image
* security groups
* userdata script

### D) User Data Script (Critical)

A shell script that runs when VM launches.

Purpose:

* install python + dependencies
* download model from S3
* start API server

Example idea:

```bash
#!/bin/bash
sudo apt-get update
sudo apt-get install -y python3-pip
pip install -r requirements.txt
aws s3 cp s3://<bucket>/model.pkl /app/model.pkl
nohup uvicorn app:app --host 0.0.0.0 --port 5001 --workers 4 &
```

✅ Why it matters?

> “When ASG creates new VMs, userdata ensures model+API setup is automated.”

---

## 3) Kubernetes Model Deployment (Traditional Way)

### Kubernetes deployment checklist

1. Dockerfile
2. Container registry (ECR)
3. Build + push image
4. Kubernetes cluster (EKS / local: kind/minikube/k3s)
5. Apply manifests:

   * namespace
   * deployment
   * service
   * ingress
6. Inference endpoint live

---

### A) Container Registry

**Model registry vs container registry:**

* **Container registry (ECR):** stores Docker images
* **MLflow model registry:** stores model versions

Most real systems use both:

* image = inference code
* artifact/model = trained model file

---

### B) Suggested folder structure in repo (k8s branch)

```
/k8s-manifests
  namespace.yaml
  deployment.yaml
  service.yaml
  ingress.yaml
```

---

### C) Minimal Kubernetes YAML structure (interview-level)

#### 1) Namespace

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: ml
```

#### 2) Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: iris-api
  namespace: ml
spec:
  replicas: 2
  selector:
    matchLabels:
      app: iris-api
  template:
    metadata:
      labels:
        app: iris-api
    spec:
      containers:
        - name: iris-api
          image: <ECR_IMAGE_URI>
          ports:
            - containerPort: 5001
```

#### 3) Service

```yaml
apiVersion: v1
kind: Service
metadata:
  name: iris-api-svc
  namespace: ml
spec:
  selector:
    app: iris-api
  ports:
    - port: 80
      targetPort: 5001
  type: ClusterIP
```

#### 4) Ingress

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: iris-ingress
  namespace: ml
spec:
  rules:
  - host: iris.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: iris-api-svc
            port:
              number: 80
```

✅ Explanation in 1 line each:

* Deployment = run N copies of your app
* Service = stable endpoint inside cluster
* Ingress = public HTTP routing

---

## 4) KServe Deployment (Fastest Production Serving on Kubernetes)

### Why KServe?

Because raw Kubernetes is generic. KServe is ML-serving specialized:

* scale-to-zero
* autoscaling based on requests
* canary rollout
* GPU support
* supports standard predictors

---

### KServe Core Object

✅ **InferenceService**: one YAML defines everything.

Example:

```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: iris-sklearn
  namespace: ml
spec:
  predictor:
    sklearn:
      storageUri: s3://<bucket>/iris-model/
```

### How it works internally

KServe creates:

* predictor pod
* service
* routing
* autoscaling

✅ Interview line:

> “KServe reduces Kubernetes complexity by providing ML-native CRDs like InferenceService.”

---

## 5) Amazon SageMaker — Training + Deployment Notes

### What is SageMaker (simple)

A managed AWS platform where:

* DS trains models
* endpoints are deployed
* infra scaling is handled

### Why use SageMaker vs Kubernetes?

✅ SageMaker advantages:

* easiest managed deployment
* endpoint autoscaling built-in
* integrates with AWS ecosystem

❌ disadvantages:

* higher cost
* AWS-specific

---

### SageMaker deployment workflow (matching your notes)

#### Step 1: Train locally / notebook

* Use JupyterLab in SageMaker Studio

#### Step 2: Save model artifact

Typically create:

* `model.tar.gz` (contains model.pkl + inference code)

#### Step 3: Upload to S3

S3 is AWS object storage.

#### Step 4: Deploy Endpoint

Using SageMaker SDK:

```python
from sagemaker.sklearn.model import SKLearnModel

model = SKLearnModel(
    model_data="s3://bucket/path/model.tar.gz",
    role="<SageMakerExecutionRole>",
    entry_point="inference.py",
    framework_version="1.2-1",
    py_version="py3"
)

predictor = model.deploy(
    instance_type="ml.m5.large",
    initial_instance_count=1
)
```

#### Step 5: Get endpoint + predict

SageMaker gives a hosted endpoint.

✅ Interview line:

> “SageMaker endpoint is managed model serving with scaling and secure API access.”

---

## 6) Kubeflow (very important if asked)

### What is Kubeflow?

ML workflows on Kubernetes.

If Kubernetes is OS for containers,
Kubeflow is the ML toolset on top.

Used for:

* pipeline orchestration (train → validate → deploy)
* experiment tracking
* hyperparameter tuning

✅ Why not everyone uses it?

* powerful but complex
* requires Kubernetes maturity

---

## 7) Full CI/CD Pipeline (GitHub Actions + DVC + AWS + KServe + ArgoCD)

### Big Picture Architecture

**Goal:** Every time code changes, automatically:

* generate/update data
* train model
* push model to S3
* update KServe InferenceService
* deploy to Kubernetes

---

## 7.1 Repo Branching Strategy (matches your notes)

### Branch 1: main

Contains ML code only:

* train.py
* feature engineering
* MLflow tracking

### Branch 2: cicd

Contains infra + deployment:

* k8s manifests / KServe YAML
* GitHub Actions pipelines
* ArgoCD configs

✅ Interview reason:

> “Separating ML code from infra reduces risk and keeps repo clean.”

---

## 7.2 DVC + S3 in CI/CD

### Typical flow

1. DVC tracks dataset metadata
2. dataset stored in S3
3. pipeline pulls data / trains model
4. trained model artifact stored in S3

Commands:

```bash
pip install dvc dvc_s3

dvc init

dvc remote add -d s3remote s3://<bucket>/<folder>

dvc add data/dataset.csv

git add data/dataset.csv.dvc .gitignore
git commit -m "Track dataset"

dvc push
```

✅ Interview line:

> “DVC ensures training is reproducible by linking code changes to specific data versions.”

---

## 7.3 Kubernetes + KServe setup (local dev)

### Local clusters

* **kind**: k8s in Docker (fast for learning)
* **minikube**: local k8s
* **k3s**: lightweight k8s

Example:

```bash
kind create cluster --name=model-name
kubectl create ns ml
```

### Install KServe

After install:

```bash
kubectl get crds | grep kserve
```

---

## 7.4 IAM / Service Account (AWS access from KServe)

### Why needed?

KServe needs to read model from S3.

Kubernetes pods should not embed AWS keys.

Best practice: **IAM role for service account (IRSA)** (EKS).

Simpler learning option:

* use Kubernetes secret with access keys (not best practice)

---

## 7.5 GitHub Actions (CI/CD)

### What GitHub Actions does here

* generate data
* train model
* push artifacts to S3
* update inference YAML
* commit changes OR trigger ArgoCD sync

### Pipeline skeleton (mlops-pipeline.yml)

```yml
name: mlops-pipeline

on:
  push:
    branches: [ cicd ]

jobs:
  train-and-deploy:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.12'

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Generate dataset
        run: python generate_data.py

      - name: Train model
        run: python train.py

      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v4
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: ap-south-1

      - name: Upload model to S3
        run: aws s3 cp artifacts/model.pkl s3://<bucket>/models/model.pkl

      - name: Update KServe InferenceService YAML
        run: |
          sed -i 's|storageUri:.*|storageUri: s3://<bucket>/models/|' kserve/inference.yaml

      - name: Commit updated manifests
        run: |
          git config user.name "github-actions"
          git config user.email "actions@github.com"
          git add kserve/inference.yaml
          git commit -m "Update model storageUri" || echo "No changes"
          git push
```

✅ What DS should say:

> “CI trains and pushes model. CD updates deployment manifests so the cluster serves latest model.”

---

## 7.6 ArgoCD (CD tool)

### What is ArgoCD (simple)

A tool that continuously syncs Kubernetes cluster with Git.

Meaning:

* Git is the source of truth
* If YAML changes in repo → ArgoCD applies it to cluster

Why ArgoCD is popular?

* safer deployments
* audit trail
* rollback is easy

✅ Interview line:

> “ArgoCD enables GitOps: the cluster state matches what is in Git.”

---

## 8) Why Kubernetes + KServe + SageMaker (comparison)

### Kubernetes (generic)

✅ Best when:

* company standard infra is Kubernetes
* multiple apps share same cluster

❌ Harder for DS

### KServe (K8s + ML serving)

✅ Best when:

* you want Kubernetes power with easy ML serving
* need autoscaling, canary, scale-to-zero

### SageMaker (managed)

✅ Best when:

* quickest path to production
* low infra team support

❌ expensive

---

## 9) Final Interview Summary (30 seconds)

> “For deployment, I can use VM-based FastAPI on EC2 with autoscaling groups and userdata scripts for bootstrapping. For scalable production, container-based deployment on Kubernetes is standard using manifests for deployment/service/ingress. For faster ML serving on Kubernetes, KServe simplifies everything with InferenceService CRDs and autoscaling, traffic routing and scale-to-zero. If we want managed deployment, SageMaker hosts the model endpoint directly from S3 artifacts with built-in scaling and AWS integration. In CI/CD, GitHub Actions trains and pushes model to S3, then updates KServe manifests and ArgoCD syncs deployment using GitOps.”

---

## 10) Complete Production Flow on AWS (EKS + ECR + IRSA + ALB Ingress)

> This is the **most interview-relevant missing section**. It completes the full deployment story.

### AWS components (simple explanations)

* **EKS**: AWS-managed Kubernetes cluster (control plane managed by AWS).
* **ECR**: AWS container registry (stores Docker images).
* **ALB**: Application Load Balancer (public HTTP/HTTPS entrypoint).
* **Route53**: AWS DNS service (maps domain → load balancer).
* **ACM**: AWS Certificate Manager (free TLS/SSL certificates).
* **IAM**: AWS identity system (permissions).
* **IRSA**: IAM Roles for Service Accounts → gives pods permission to access AWS without storing keys.

---

### 10.1 End-to-End production deployment (recommended architecture)

#### Step 0: Train model (CI)

* Train via GitHub Actions or SageMaker training job.
* Save artifacts: `model.pkl` / `model.joblib` / `model.tar.gz`.

#### Step 1: Store model in S3

* S3 path example: `s3://mlops-prod/models/iris/2026-02-01/model.pkl`

✅ Why S3?

* cheap + durable
* versionable
* integrates with SageMaker + KServe

#### Step 2: Build inference image and push to ECR

Your inference code (FastAPI / model server wrapper) is packaged into Docker.

Commands (conceptual):

```bash
aws ecr create-repository --repository-name iris-api

# build local
docker build -t iris-api:latest .

# login + push
aws ecr get-login-password | docker login --username AWS --password-stdin <acct>.dkr.ecr.<region>.amazonaws.com

docker tag iris-api:latest <acct>.dkr.ecr.<region>.amazonaws.com/iris-api:latest

docker push <acct>.dkr.ecr.<region>.amazonaws.com/iris-api:latest
```

✅ Why image + model separated?

* image changes rarely (code)
* model changes frequently (retraining)

---

### 10.2 EKS cluster setup (high-level)

Usually done via:

* Terraform / CloudFormation (prod)
* `eksctl` (fast)

Cluster includes:

* Node groups (EC2 workers)
* Add-ons: Metrics server, AWS Load Balancer Controller

---

### 10.3 Install AWS Load Balancer Controller (ALB Ingress)

**Why needed?**
Ingress needs a controller. On AWS, you use **AWS Load Balancer Controller** to create ALB.

This enables:

* `Ingress` → automatically creates ALB
* supports HTTPS, path routing, WAF integration

---

### 10.4 IRSA (no AWS keys inside pods)

**Problem:** KServe predictor needs to download model from S3.

Bad approach:

* store AWS keys as secrets

Best practice:

* create IAM role with S3 read permissions
* bind IAM role to Kubernetes ServiceAccount

✅ Interview line:

> “IRSA allows secure S3 access from pods without embedding credentials.”

---
 
