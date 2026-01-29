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

## Experiment Tracking (ML Flow)

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
