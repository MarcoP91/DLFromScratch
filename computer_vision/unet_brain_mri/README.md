# UNet Brain MRI Segmentation

This project implements a UNet model for brain MRI segmentation. The goal is to accurately segment brain tumors from MRI images using deep learning techniques.

## Project Structure

The project is organized as follows:

unet_brain_mri/ ├── data/ # Directory containing the dataset ├── models/ # Directory to save trained models ├── notebooks/ # Jupyter notebooks for experimentation and visualization ├── scripts/ # Python scripts for training and evaluation ├── unet_brain_mri.ipynb # Main Jupyter notebook for training and evaluation └── README.md # Project description and instructions


## Requirements

To run this project, you need the following dependencies:

- Python 3.7 or higher
- PyTorch
- torchvision
- numpy
- pandas
- scikit-learn
- matplotlib
- tqdm
- Jupyter Notebook

You can install the required packages using pip:

```bash
pip install torch torchvision numpy pandas scikit-learn matplotlib tqdm jupyter
```

## Dataset
The dataset used for this project consists of brain MRI images with corresponding segmentation masks. You can download the dataset from Kaggle or any other source. Place the dataset in the data/ directory.

## Training
To train the UNet model, you can use the unet_brain_mri.ipynb Jupyter notebook. The notebook includes code for loading the dataset, defining the model, training, and evaluating the model.

## Hyperparameter Tuning
The notebook also includes a section for hyperparameter tuning using grid search. You can define the hyperparameter grid and iterate over different combinations to find the best hyperparameters.

Example code for hyperparameter tuning:

```code
import itertools
import torch.optim as optim

# Define the hyperparameter grid
learning_rates = [1e-4, 1e-5, 1e-6]
momentums = [0.9, 0.99, 0.999]
weight_decays = [1e-6, 1e-7, 1e-8]
gradient_clippings = [0.5, 1.0, 1.5]

# Create a list of all possible combinations of hyperparameters
hyperparameter_grid = list(itertools.product(learning_rates, momentums, weight_decays, gradient_clippings))

# Iterate over the hyperparameter grid
for lr, momentum, weight_decay, grad_clip in hyperparameter_grid:
    print(f'Training with lr={lr}, momentum={momentum}, weight_decay={weight_decay}, grad_clip={grad_clip}')
    
    # Initialize the model, optimizer, and other components
    model = ...  # Your model initialization
    optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum, foreach=True)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=True)
    
    # Train the model with the current set of hyperparameters
    train(model, train_loader, valid_loader, device, epochs=5, batch_size=64, learning_rate=lr, 
          weight_decay=weight_decay, momentum=momentum, gradient_clipping=grad_clip)
    
    # Evaluate the model and log the results
    val_score = evaluate(model, val_loader, device, amp=True)
    print(f'Validation score: {val_score}')
```

## Evaluation
After training the model, you can evaluate its performance on the validation set using the evaluation code provided in the notebook. The evaluation includes calculating metrics such as Dice coefficient to measure the segmentation accuracy.

## Results
The results of the model, including training and validation metrics, will be displayed in the notebook. You can also visualize the segmentation results using matplotlib.

Conclusion
This project demonstrates the use of a UNet model for brain MRI segmentation. By experimenting with different hyperparameters and training techniques, you can improve the model's performance and achieve accurate segmentation results.

## References

1. **UNet: Convolutional Networks for Biomedical Image Segmentation** - Olaf Ronneberger, Philipp Fischer, Thomas Brox. [Link to paper](https://arxiv.org/abs/1505.04597)
2. **PyTorch Documentation** - Official PyTorch documentation for deep learning framework. [Link to PyTorch](https://pytorch.org/docs/stable/index.html)
3. **Kaggle Brain MRI Segmentation Dataset** - Dataset used for training and evaluation. [Link to dataset](https://www.kaggle.com/datasets)
4. **Scikit-learn Documentation** - Official documentation for machine learning in Python. [Link to Scikit-learn](https://scikit-learn.org/stable/documentation.html)
5. **Matplotlib Documentation** - Official documentation for plotting and visualization in Python. [Link to Matplotlib](https://matplotlib.org/stable/contents.html)
