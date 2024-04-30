# Homework 3

## Part 0: Preparation
Unless otherwise specified, run all commands in the command line of the GCP Console terminal. Upload this code into the GCP Console code editor. Follow these steps to set up k8s cluster:

1. In Kubernetes Engine, click "Create" to create a new cluster.
2. Once it is created, connect to the cluster using this command in the command line: `gcloud container clusters get-credentials hw3-k8s-cluster --region us-central1 --project amlc-hw3`

## Part 1: Training

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

## Part 2: Inference

### Step 6: Build and push inference Docker image
Run these commands
1. `cd ../inference`
2. `docker build -t gcr.io/amlc-hw3/mnist-inference:latest .`
3. `docker push gcr.io/amlc-hw3/mnist-inference:latest`

### Step 7: Deploy inference job
Run these commands: 
1. `kubectl create -f infer.yaml`
3. `kubectl get pods` (run this command until you see the **mnist-inference** pod has the status **Running**)