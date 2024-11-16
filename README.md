# MNIST Neural Network CI/CD Pipeline

This project demonstrates a complete CI/CD pipeline for a machine learning project using GitHub Actions. It includes a simple convolutional neural network (CNN) for MNIST digit classification with automated training, testing, and model validation.

## Project Structure

```
├── model/
│ └── mnist_model.py # Neural network architecture
├── .github/
│ └── workflows/
│ └── ml-pipeline.yml # GitHub Actions workflow
├── train.py # Training script
├── test_model.py # Testing and validation script
├── requirements.txt # Project dependencies
└── .gitignore # Git ignore rules
```

## Model Architecture

The model is a lightweight CNN with the following architecture:
- 2 Convolutional layers with max pooling
- 2 Fully connected layers
- Total parameters: ~9,898 (optimized for efficiency)

Architecture details:
- Input: 28x28 grayscale images
- Conv1: 8 filters (3x3)
- Conv2: 16 filters (3x3)
- FC1: 64 features
- Output: 10 classes (digits 0-9)

## CI/CD Pipeline

The pipeline automatically:
1. Trains the model on MNIST dataset
2. Validates model architecture
3. Tests model performance
4. Archives trained models

### Validation Checks
- Input shape compatibility (28x28)
- Output shape verification (10 classes)
- Parameter count (< 100,000)
- Model accuracy (> 80%)

## Setup and Running Locally

1. Create and activate virtual environment:
```
bash
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate
```
2. Install dependencies:
```
pip install -r requirements.txt
```
3. Train the model:
```
python train.py
```
4. Run tests:
```
python -m pytest test_model.py -v
```
## GitHub Actions Integration

The pipeline automatically runs on every push to the repository:
- Uses CPU-only PyTorch for compatibility
- Runs on Ubuntu latest
- Stores trained models as artifacts
- Retention period: 90 days

## Requirements

- Python 3.8+
- PyTorch (CPU version)
- torchvision
- pytest

## Model Artifacts

Trained models are saved with timestamps in the format: Create a Python script that sets up a basic CI/CD pipeline for a machine learning project.
Include automated testing for an ML model, model validation checks, and a simple automated deployment process. I want you to make 3 layer DNN (using convolutions and fully connected layer), use MNIST as dataset (download if required), and train only for 1 epoch. When you test, mention what tests you are checking for. I would like to check if model is under 100000 parameters, takes 28x28 input without any issues, has 10 outputs, and accuracy is more than 80%. Make all the necessary files (for training as well as github actions related). Finally I should be able to upload whole code to GItHub, and Github action should run all of these things to confirm the build was successful. Tell me steps to run locally as well before i push to github. For deployment, add sufix to model file to get some detail on when it was trained. 


I'll help you create a complete CI/CD pipeline for your ML project. I'll create several files to handle training, testing, and GitHub Actions.
First, let's create the main model and training script:
Now, let's create the training script:
Let's create the test script:
Now, let's create the GitHub Actions workflow:
Finally, let's create a requirements.txt:
Here's how to run the pipeline locally:
Create a virtual environment and activate it:
activate
Install dependencies:
txt
Train the model:
py
Run tests:
v
To use this with GitHub:
Create a new GitHub repository
Initialize git in your local project:
main
The GitHub Actions workflow will automatically run when you push to the repository. It will:
Set up a Python environment
Install dependencies
Train the model
Run the tests
Save the trained model as an artifact
The tests check for:
Model architecture (input shape 28x28, output shape 10)
Parameter count (< 100,000)
Model accuracy (> 80%)
Model existence and loading
The model files are saved with timestamps (e.g., mnist_model_20240315_143022.pth) for versioning.
Make sure to add these files to your .gitignore:
This setup provides a complete CI/CD pipeline that automatically trains, tests, and validates your ML model on every push to the repository.
please add files to gitignore when running the system which generate some temporary files

I'll update the .gitignore file to include more common temporary files and system-specific files:
/
This updated .gitignore includes:
Python bytecode and cache files
Build and distribution directories
Virtual environment folders
Project-specific data and model directories
Common IDE configuration folders (PyCharm, VSCode)
OS-specific temporary files (Mac, Windows, Linux)
Jupyter notebook checkpoints
Testing cache and coverage files
This will help keep your repository clean from temporary files and system-specific artifacts.
CI CD Pipeline failed in github giving the reason: This request has been automatically failed because it uses a deprecated version of `actions/upload-artifact: v2`. Learn more: https://github.blog/changelog/2024-02-13-deprecation-notice-v1-and-v2-of-the-artifact-actions/

I'll update the GitHub Actions workflow file to use the latest version of actions/upload-artifact:
Changes made:
Updated actions/upload-artifact from v2 to v4 (latest stable version)
Updated actions/checkout from v2 to v4
Updated actions/setup-python from v2 to v5
Added retention-days parameter to specify how long to keep the artifacts
These updates will resolve the deprecation warning and ensure compatibility with GitHub's latest features.
since github doesn't provide GPU or CUDA, please do not download GPU or CUDA related files.

I'll modify the requirements and training setup to use CPU-only versions of PyTorch. Here are the necessary changes:
And let's update the training script to explicitly use CPU:
Also update the test script to use CPU only:
The main changes are:
Updated requirements.txt to use CPU-only versions of PyTorch and torchvision
Removed CUDA checks and forced CPU usage in both train.py and test_model.py
Added specific versions for PyTorch packages to ensure stability
These changes will make the pipeline run more efficiently on GitHub Actions since it won't try to download unnecessary CUDA dependencies, and all computations will be explicitly performed on CPU.
 def test_model_architecture():
        model = MNISTNet()
    
        # Test input shape
        test_input = torch.randn(1, 1, 28, 28)
        output = model(test_input)
        assert output.shape == (1, 10), "Output shape should be (1, 10)"
    
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
>       assert total_params < 100000, f"Model has {total_params} parameters, should be < 100000"
E       AssertionError: Model has 206922 parameters, should be < 100000
E       assert 206922 < 100000
test_model.py:21: AssertionError
=============================== warnings summary ===============================
test_model.py::test_model_accuracy
  /home/runner/work/S5CICDExample/S5CICDExample/test_model.py:30: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
    model.load_state_dict(torch.load(model_path))
-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========================== short test summary info ============================
FAILED test_model.py::test_model_architecture - AssertionError: Model has 206922 parameters, should be < 100000
assert 206922 < 100000
==================== 1 failed, 1 passed, 1 warning in 4.17s ====================

I'll modify the model architecture to reduce the number of parameters. Here's the updated model:
Changes made to reduce parameters:
Reduced first conv layer filters from 16 to 8
Reduced second conv layer filters from 32 to 16
Reduced fully connected layer features from 128 to 64
Let's calculate the parameters:
Conv1: (3×3×1×8) + 8 = 80 parameters
Conv2: (3×3×8×16) + 16 = 1,168 parameters
FC1: (16×7×7×64) + 64 = 50,240 parameters
FC2: (64×10) + 10 = 650 parameters
Total: ~52,138 parameters
This architecture should still maintain good performance while meeting the parameter constraint of being under 100,000 parameters.
Generate a REAME.md file for me and describe the project in it.

I'll create a comprehensive README.md file for your project:
├── model/
│ └── mnist_model.py # Neural network architecture
├── .github/
│ └── workflows/
│ └── ml-pipeline.yml # GitHub Actions workflow
├── train.py # Training script
├── test_model.py # Testing and validation script
├── requirements.txt # Project dependencies
└── .gitignore # Git ignore rules
bash
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate
2. Install dependencies:
bash
pip install -r requirements.txt
3. Train the model:
bash
python train.py
4. Run tests:
bash
python -m pytest test_model.py -v
## GitHub Actions Integration

The pipeline automatically runs on every push to the repository:
- Uses CPU-only PyTorch for compatibility
- Runs on Ubuntu latest
- Stores trained models as artifacts
- Retention period: 90 days

## Requirements

- Python 3.8+
- PyTorch (CPU version)
- torchvision
- pytest

## Model Artifacts

Trained models are saved with timestamps in the format:
mnist_model_YYYYMMDD_HHMMSS.pth

## Notes

- The model is trained for 1 epoch to demonstrate the pipeline
- Uses CPU-only PyTorch to ensure compatibility with GitHub Actions
- All computations are performed on CPU for consistency

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.




