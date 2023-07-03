import torch
from sklearn.metrics import roc_auc_score

from utils import confusion_matrix


def calculate_precision_sensitivity_specificity(confusion_matrix):
    true_positives = torch.diag(confusion_matrix)
    trun_negatives = torch.sum(true_positives) - true_positives
    false_negatives = torch.sum(confusion_matrix, dim=0) - true_positives
    false_positives = torch.sum(confusion_matrix, dim=1) - true_positives

    precision = torch.mean(true_positives / (true_positives + false_positives + 1e-6))
    sensitivity = torch.mean(true_positives / (true_positives + false_negatives + 1e-6))
    specificity = torch.mean(trun_negatives / (trun_negatives + false_positives + 1e-6))

    return precision, sensitivity, specificity


def valid(config, net, val_loader, criterion):
    device = next(net.parameters()).device
    net.eval()

    print("START VALIDING")
    epoch_loss = 0
    y_true, y_score, y_pred = [], [], []

    cm = torch.zeros((config.class_num, config.class_num))
    output = torch.Tensor().to(device)
    for i, pack in enumerate(val_loader):
        images = pack['imgs'].to(device)
        if images.shape[1] == 1:
            images = images.expand((-1, 3, -1, -1))
        names = pack['names']
        labels = pack['labels'].to(device)

        if config.model_name == 'REAF':
            mask = pack['mask'].to(device)
            output = net(images, mask)
        else:
            output = net(images)

        loss = criterion(output, labels)

        pred = output.argmax(dim=1)
        y_true.append(labels.detach().item())
        y_pred.append(pred.item())
        if config.class_num == 2:
            y_score.append(output[0].softmax(0)[1].item())
        else:
            y_score.append(output[0].softmax(0).detach().cpu().tolist())

        cm = confusion_matrix(pred.detach(), labels.detach(), cm)
        epoch_loss += loss.cpu()

    avg_epoch_loss = epoch_loss / len(val_loader)

    spe, sen, pre, auc = 0, 0, 0, 0
    acc = cm.diag().sum() / cm.sum()
    if config.class_num == 2:
        spe, sen = cm.diag() / (cm.sum(dim=0) + 1e-6)
        pre = cm.diag()[1] / (cm.sum(dim=1) + 1e-6)[1]
        auc = roc_auc_score(y_true, y_score)
    else:
        pre, sen, spe = calculate_precision_sensitivity_specificity(cm)
        auc = roc_auc_score(y_true, y_score, multi_class='ovr')

    rec = sen
    f1score = 2 * pre * rec / (pre + rec + 1e-6)

    return [avg_epoch_loss, acc, sen, spe, auc, pre, f1score]
