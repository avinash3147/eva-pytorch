"""
    Train CIFAR10 with pytorch
"""
import argparse

from models import *

from utils.transform import Transforms
from utils.graphs_utility import plot_metrics, plot_misclassified
from utils.misc import get_cuda, run_epochs, get_wrong_predictions, load_classes, \
    get_images_lables_for_grad_cam, grad_cam
from utils.data_utility import train_data_transformation, test_data_transformation, \
    download_train_data, download_test_data, get_data_loader_args, load_test_data, load_train_data


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
# parser.add_argument('--resume', '-r', action='store_true',
#                     help='resume from checkpoint')
parser.add_argument('--model', default='resnet', help='specify which model to run')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
train_transforms = Transforms(train_data_transformation()) # Train Data Transformation
test_transforms = Transforms(test_data_transformation()) # Test Data Transformations

train_data = download_train_data(train_transforms=train_transforms) # Download Train Data
test_data = download_test_data(test_transforms=test_transforms) # Download Test Data

cuda = get_cuda() # Check for cuda
data_loader_args = get_data_loader_args(cuda) # Data Loader Arguments

train_loader = load_train_data(train_data, **data_loader_args) # Load Train Data
test_loader = load_test_data(test_data, **data_loader_args) # Load Test Data

device = torch.device("cuda" if cuda else "cpu")


# Model
print('==> Building model..')
model = ResNet18()
model = model.to(device)

train_accuracy, train_losses, test_accuracy, test_losses = run_epochs(
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        model=model
    )

plot_metrics(train_accuracy, train_losses, test_accuracy, test_losses)

# Wrong Predictions and Plots
wrong_predictions, wrong_images, correct_label = get_wrong_predictions(model, test_loader, device) # Get Wrong Predictions
plot_misclassified(wrong_predictions, load_classes())

# Grad Cam Plots
g_image_list, g_image_label = get_images_lables_for_grad_cam(wrong_images, correct_label)
grad_cam(g_image_list, g_image_label, model)
