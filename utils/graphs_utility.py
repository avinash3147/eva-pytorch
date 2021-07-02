import cv2
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def plot_metrics(train_accuracy, train_losses, test_accuracy, test_losses):
    sns.set(style="whitegrid")
    sns.set(font_scale=1)

    fig, axs = plt.subplots(2, 2, figsize=(25, 15))
    plt.rcParams["figure.figsize"] = (25, 6)

    axs[0, 0].set_title("Training Loss")
    axs[1, 0].set_title("Training Accuracy")
    axs[0, 1].set_title("Test Loss")
    axs[1, 1].set_title("Test Accuracy")

    axs[0, 0].plot(train_losses, label="Training Loss")
    axs[0, 0].set_xlabel('epochs')
    axs[0, 0].set_ylabel('loss')

    axs[1, 0].plot(train_accuracy, label="Training Accuracy")
    axs[1, 0].set_xlabel('epochs')
    axs[1, 0].set_ylabel('accuracy')

    axs[0, 1].plot(test_losses, label="Validation Loss")
    axs[0, 1].set_xlabel('epochs')
    axs[0, 1].set_ylabel('loss')

    axs[1, 1].plot(test_accuracy, label="Validation Accuracy")
    axs[1, 1].set_xlabel('epochs')
    axs[1, 1].set_ylabel('accuracy')


def plot_misclassified(wrong_predictions, classes, mean, std):
    fig = plt.figure(figsize=(10, 12))
    fig.tight_layout()

    for i, (img, pred, correct) in enumerate(wrong_predictions[:20]):
        img, pred, target = img.cpu().numpy().astype(dtype=np.float32), pred.cpu(), correct.cpu()
        for j in range(img.shape[0]):
            img[j] = (img[j] * std[j]) + mean[j]

        img = np.transpose(img, (1, 2, 0))  # / 2 + 0.5
        ax = fig.add_subplot(5, 5, i + 1)
        ax.axis('off')
        ax.set_title(f'\nactual : {classes[target.item()]}\npredicted : {classes[pred.item()]}',
                     fontsize=10)
        ax.imshow(img)

    plt.show()


def visualize_cam(mask, img, alpha=1.0):
    """Make heatmap from mask and synthesize GradCAM result image using heatmap and img.
    Args:
        mask (torch.tensor): mask shape of (1, 1, H, W) and each element has value in range [0, 1]
        img (torch.tensor): img shape of (1, 3, H, W) and each pixel value is in range [0, 1]
    Return:
        heatmap (torch.tensor): heatmap img shape of (3, H, W)
        result (torch.tensor): synthesized GradCAM result of same shape with heatmap.
    """
    heatmap = (255 * mask.squeeze()).type(torch.uint8).cpu().numpy()
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = torch.from_numpy(heatmap).permute(2, 0, 1).float().div(255)
    b, g, r = heatmap.split(1)
    heatmap = torch.cat([r, g, b]) * alpha

    result = heatmap+img.cpu()
    result = result.div(result.max()).squeeze()

    return heatmap, result


def imshow(img,c = "" ):
    #img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    fig = plt.figure(figsize=(10,10))
    plt.imshow(np.transpose(npimg, (1, 2, 0)),interpolation='none')
    plt.title(c)