# Homework 3 - Samuel Braun slb2250

## Table of Contents

For proof of each step, please look at [Proof_Images.pdf](development/Proof_Images.pdf) in this repo. Unless otherwise specified, run all commands in the command line of the GCP Console terminal. Upload this code into the GCP Console code editor.

- [Part 1: Preparation](#part-1-preparation)
- [Part 2: Training Deployment](#part-2-training-deployment)
- [Part 3: Inference Deployment](#part-3-inference-deployment)
- [Acknowledgements](#acknowledgements)


## Part 1: Preparation

### Step 1
In Kubernetes Engine, click "Create" to create a new cluster.

### Step 2
Once it is created, connect to the cluster using this command in the command line: `gcloud container clusters get-credentials hw3-k8s-cluster --region us-central1 --project amlc-hw3`

## Part 2: Training Deployment

### Step 1: Download training operator
Run the command: `kubectl apply -k "github.com/kubeflow/training-operator/manifests/overlays/standalone?ref=v1.7.0"`

### Step 2: Switch namespace
1. Run the command: `kubectl config set-context --current --namespace=kubeflow`
2. Run the command: `kubectl get pods`
3. Wait until the training-operator pod is **Running** before continuing (this might take a few minutes).

### Step 3: Build and push training Docker image
Run these commands:
1. `cd amlc/hw3/training`
2. `docker build -t gcr.io/amlc-hw3/pytorch-mnist:latest .`
3. `docker push gcr.io/amlc-hw3/pytorch-mnist:latest`

### Step 4: Apply PVC
Run these commands:
1. `cd ../persistence`
2. `kubectl apply -f pvc.yaml`

### Step 5: Deploy PyTorch job
Run these commands:
1. `cd ../training`
2. `kubectl create -f train.yaml`
3. `kubectl get pods` (run this command until you see the **pytorch-seelam-master-0** pod has the status **Running**)
4. 'kubectl logs <name of master pod>`

## Part 3: Inference Deployment

### Step 6: Build and push inference Docker image
Run these commands
1. `cd ../inference`
2. `docker build -t gcr.io/amlc-hw3/mnist-inference:latest .`
3. `docker push gcr.io/amlc-hw3/mnist-inference:latest`

### Step 7: Deploy inference job
Run these commands: 
1. `kubectl create -f infer.yaml`
2. `kubectl get pods` (run this command until you see the **mnist-inference** pod has the status **Running**).

### Step 8: Access application
1. Run this command: `kubectl get svc` (run until there is an EXTERNAL-IP for the **mnist-inference** pod).
2. Enter `http://<EXTERNAL-IP>` in web browser.
3. Upload a test image

## Acknowledgements
`train.yaml` is based on the `simple.yaml` file provided by Prof. Seelam in Slack. `mnist.py` is modeled after the `mnist.py` file in [this repo](https://github.com/kubeflow/training-operator/blob/master/examples/pytorch/mnist/mnist.py). `server.py` is modeled after the `server.py` file in [this repo](https://github.com/ml-kubernetes/MNIST/tree/master). The samples used in this repo are also from [this repo](https://github.com/ml-kubernetes/MNIST/tree/master).

