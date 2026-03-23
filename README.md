# UCSF BIOE245 2026 Homework: Pathology Image Classification

My responses are in `hw_responses.md`

In this assignment, we explore the task of using ResNet-18 to perform pathology image classification. We use the PathMNIST dataset from MedMNIST [1], which is included as a submodule in this repository.

## Setup Instructions

### 1. Environment Setup

**Note**: This tutorial assumes you are using conda on Ubuntu. If you are using macOS or another operating system, you may need to modify the commands accordingly.

You can either use an existing conda environment or create a new one. In this tutorial, we name the environment `ResNet`.

#### Create a New Environment (if needed):
```bash
conda create -n ResNet python=3.12.7
```

#### Activate the Environment:
```bash
conda activate ResNet
```

### 2. Install Dependencies

Choose the appropriate installation command depending on whether you are using CPU or GPU:

#### CPU-only:
```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

#### GPU (CUDA support):
```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118
```

**Note**: For GPU installation, specify the CUDA version that matches your system. Common options include:
- `cu118` for CUDA 11.8
- `cu121` for CUDA 12.1

PyTorch includes its own CUDA runtime libraries, so you do not need to install the CUDA toolkit separately. Only compatible NVIDIA drivers are required on your system.

## Running the Experiment

### 1. Configure Parameters

Open `train_and_eval.sh` and modify the following parameters:

- **`DATASET_ROOT`**: Directory where the PathMNIST dataset will be downloaded
- **`OUTPUT_ROOT`**: Directory where model weights and training statistics will be saved
- **`gpu_ids`**: GPU device ID(s) to use (e.g., `0` for first GPU, or leave empty for CPU mode)

### 2. Start Training

Run the training script:
```bash
bash train_and_eval.sh
```

## Assignment Tasks

**Submission**: Complete this assignment as a markdown file (`.md`) and submit it to your GitHub account.

### Task 1: Run Training

Run the training steps outlined in the instructions above. You may use an AI assistant if needed.

### Task 2: Training Configuration Analysis

Answer the following questions about the training setup:

1. What learning rates are used in the training?
2. What is the train/val/test split of the dataset (provide sample counts)?
3. What are the dimensions of the model input per batch?
4. What is the dimension of the model output during training? What does it represent?
5. What type of task is this? What loss function is used?
6. How many files are generated after training? Where are they located and what do they contain?

### Task 3: Training Statistics Visualization

Examine the training statistics and answer the following questions. **Include screenshots of the curves** to support your answers.

1. Where are the training statistics stored? Demonstrate how to visualize them.
2. How many curves are displayed? What do they represent and how are they calculated?
3. How does the learning rate schedule correlate with the behavior of the curves?
4. What observations can you make about the curve trends? Are they monotonically increasing, decreasing, or fluctuating? Explain why.

### Task 4: AUC Metric Analysis

According to this [Google ML Glossary post](https://developers.google.com/machine-learning/glossary#AUC), AUC (Area Under Curve) is a metric between 0.0 and 1.0 that represents a **binary** classification model's ability to separate positive classes from negative classes.

1. Is AUC used in this training script? If so, is it applied directly for binary classification, or are there adaptations for multi-class classification?
2. What curve does "AUC" refer to? Plot this curve for the saved model evaluated on the test set.
3. From the test set, find examples for 5 classes of your choice:
   - One example where the model correctly predicts the class
   - One example where the model incorrectly predicts the class

   Include the images in your report (total: 10 images).

### Task 5: Bonus Challenge

Adapt the code to train a classification model on the **DermaMNIST** dataset instead of PathMNIST.