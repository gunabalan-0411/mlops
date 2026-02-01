# mlops

## Machine Learning Lifecycle
1. Problem Definition
2. Data Collection
3. Data Cleaning
4. Feature Engineering (Transformation)
5. Model Selection
6. Model Training
7. Model Evaluation
8. Hyper Parameter Tuning
9. Deploying
10. Monitoring, Maintaining

Example CI pipeline

* Workflow triggers on:
* Push to main
* Pull request to main
* GitHub starts the job on Ubuntu runner
* Job runs using a Python matrix
* Python 3.11 run
* Python 3.12 run
* Checkout repository code
* Setup selected Python version (from matrix)
* Upgrade pip / setuptools / wheel
* Install dependencies from requirements.txt
* Run training script train.py
* Verify artifacts folder contents (ls -la artifacts)
* Upload artifacts/ as GitHub Actions artifact
* Artifact name includes python version + run id


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

## Create an API to model

refer app.py
either use fastapi or flask

```bash
curl -X POST "http://127.0.0.1:5001/predict" -H "Content-Type: application/json" -d '{"features": [5.3, 3.5, 1.4, 0.2]}'
```

## Deployment

Create a Dockerfile

* Pull base image python:3.12-slim
* Create/enter /app
* Copy requirements.txt
* Upgrade pip
* Install dependencies
* Copy your full project code
* Save final image

build
docker build -t hello-mlops:latest .

run
docker run -d -p 5001:5001 hello-mlops:latest

check
docker ps
``` bash
# Variant: slim (smaller than full Debian image)
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .

RUN python -m pip install --upgrade pip
RUN python -m pip install -r requirements.txt

COPY . . 

EXPOSE 5001

CMD ["python", "app.py"]

# Best practice: create .dockerignore
```

## DVC (Data Version Control)
* We can S3, blob storage, gc and its comes along with data versioning
* pip install dvc
``` bash
git init
dvc init
dvc add data/sample_data.csv # this will create data/sample_data.csv.dvc
```
whenever we change to data, we use dvc add and it will update the check sum
The .dvc will be stored and maintened in git but the data will be stored in s3 bucket

* to set remort s3 bucket for data storage
* use aws configure to store the credentials
* pip install dvc_s3
* use dvc push to upload the data to s3
```bash
dvc remote add -d dvc_demo s3://dvc_demo
```

### Step by step
* dvc add filename
* git add, commit dvc file
* dvc push (will be pushed to s3 and create two diff version of data and checksum details)
for git cloning
* dvc pull will download the data

## Experiment Tracking (ML Flow)
* pip install mlflow
for basic installation
* mlflow ui --backend-store-uri sqlite:///mlfow.db --port 7006
for production
* for mlflow should be in kubernetes cluster and connected to postgresql hosted in aws
  * postgres in aws: RDS -> Database -> postegres -> unique id -> and get endpoint details
  * create database "mlflow" in postegresql and user with full access
  * install mlflow in kubernets cluster with configuration set to this postegresql database, user, port, url etc


* mlflow.set_experiment("iris_rf_experiment") / mlflow.sklearn.autolog()
* with mlflow.start_run():
* mlflow.log_param
* mlflow.log_metric
* mlflow.log_model
* mlflow.log_artifact("feature_importance.csv") # store .pkl, data, confusion matrix etc
* mlflow ui to view
* mlflow.set_tracking_uri("http://my-mlflow-server:5000")
* Tracking URI decides where runs are stored.
* mlflow.register_model(model_uri=model_uri, name="IrisRFModel")
* mlflow.sklearn.load_model(model_uri)
* For every time we run the train.py the logs will be collected
* select all run in mlflow ui to compare (using compare button)

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

 
 