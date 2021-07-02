"""

"""
import torchvision
from torchvision.utils import make_grid
import torch.cuda
import torch.nn as nn
import train_utility, test_utility


def get_model_summary(model):
    """
    Summary of model will be printed
    Args:
        model: Model

    Returns: summary of model

    """
    from torchsummary import summary
    model_summary = summary(model, input_size=(1, 28, 28))
    return model_summary


def load_optimizer(model):
    """
    define optimizer and scheduler
    Args:
        model: model

    Returns: optimizer and scheduler

    """
    import torch.optim as optim
    optimizer = optim.SGD(model.parameters(), lr=0.015, momentum=0.7)

    return optimizer


def get_cuda():
    seed = 1
    cuda = torch.cuda.is_available()
    torch.manual_seed(seed)

    if cuda:
        torch.cuda.manual_seed(seed)
    return cuda


def run_epochs(train_loader, test_loader, device, model):
    from torch.optim.lr_scheduler import StepLR, OneCycleLR

    model = model.to(device) # Get Model Instance
    summary = get_model_summary(model=model)
    print(summary)

    criterion = nn.CrossEntropyLoss()

    optimizer = load_optimizer(model)
    scheduler = OneCycleLR(optimizer, max_lr=0.015, epochs=20, steps_per_epoch=len(train_loader))

    train_losses = []
    test_losses = []
    train_accuracy = []
    test_accuracy = []

    for epoch in range(1, 21):
        train_utility.train(
            model=model,
            device=device,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            train_acc=train_accuracy,
            train_loss=train_losses
        )

        test_utility.test(
            model=model,
            device=device,
            test_loader=test_loader,
            test_acc=test_accuracy,
            test_losses=test_losses
        )

    return train_accuracy, train_losses, test_accuracy, test_losses


def get_wrong_predictions(model, test_loader, device):
    wrong_images = []
    wrong_label = []
    correct_label = []
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True).squeeze()

            wrong_pred = (pred.eq(target.view_as(pred)) == False)
            wrong_images.append(data[wrong_pred])
            wrong_label.append(pred[wrong_pred])
            correct_label.append(target.view_as(pred)[wrong_pred])

            wrong_predictions = list(zip(torch.cat(wrong_images), torch.cat(wrong_label), torch.cat(correct_label)))
        print(f'Total wrong predictions are {len(wrong_predictions)}')

    return wrong_predictions, wrong_images, correct_label


def load_classes():
    return ['plane', 'car', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck']


def get_images_lables_for_grad_cam(wrong_images, correct_label):
    count = 0
    g_image_list = []
    g_image_label = []
    for images, label in zip(wrong_images, correct_label):
        if [images.size()] == [torch.Size([1, 3, 32, 32])]:
            g_image_list.append(images)
            g_image_label.append(label)
            count += 1
        if count >= 10:
            break
    return g_image_list, g_image_label


def grad_cam(g_image_list, g_image_label, model):
    from gradcam import GradCAM
    from graphs_utility import visualize_cam, imshow
    for image, label in zip(g_image_list, g_image_label):
        torch_img = torchvision.utils.make_grid(torch.unsqueeze(image, 0))
        plot = []
        g1 = GradCAM(model, model.layer1)
        g2 = GradCAM(model, model.layer2)
        g3 = GradCAM(model, model.layer3)
        g4 = GradCAM(model, model.layer4)
        mask1, _ = g1(torch_img)
        mask2, _ = g2(torch_img)
        mask3, _ = g3(torch_img)
        mask4, _ = g4(torch_img)
        heatmap1, result1 = visualize_cam(mask1, torch_img)
        heatmap2, result2 = visualize_cam(mask2, torch_img)
        heatmap3, result3 = visualize_cam(mask3, torch_img)
        heatmap4, result4 = visualize_cam(mask4, torch_img)

        plot.extend([torch_img[0].cpu(), result1, result2, result3, result4])

        grid_image = make_grid(plot, nrow=5)
        imshow(grid_image, label)

