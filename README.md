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

## Deployment

Create a Dockerfile

* Pull base image python:3.12-slim
* Create/enter /app
* Copy requirements.txt
* Upgrade pip
* Install dependencies
* Copy your full project code
* Save final image

docker build -t hello-mlops:latest .

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