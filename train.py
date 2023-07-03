import math
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import utils
from config import config
from network.REAF.model.model import Diagnosis as reaf
from utils import confusion_matrix
from valid import valid


def train(config, train_loader, test_loader, fold, test_idx):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = None

    if config.model_name == 'REAF':
        model = reaf(class_num=config.class_num, feature_model_load_path=config.load_model_path)
    model = model.to(device)
    model.train()

    if config.loss_function == 'CE':
        criterion = nn.CrossEntropyLoss().to(device)

    if config.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    elif config.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    elif config.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=config.lr,
                                    momentum=0.9,
                                    weight_decay=5e-4)

    if config.scheduler == 'cosine':
        lr_lambda = lambda epoch:(epoch*(1-config.warmup_decay)/config.warmup_epochs+config.warmup_decay) \
        if epoch < config.warmup_epochs else \
        (1-config.min_lr/config.lr)* 0.5 * (math.cos((epoch-config.warmup_epochs)/(config.epochs-config.warmup_epochs) * math.pi) + 1) + config.min_lr/config.lr
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1)
    elif config.scheduler == 'step':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.step, gamma=0.9)

    writer = SummaryWriter(comment='_' + config.model_name + '_' + config.writer_comment + '_' +
                           str(fold))

    print("START TRAINING")
    best_acc = 0
    ckpt_path = os.path.join(config.model_path, config.model_name, config.writer_comment)
    model_save_path = os.path.join(ckpt_path, str(fold))
    cm = None
    output = torch.Tensor().to(device)
    for epoch in range(config.epochs):
        cm = torch.zeros((config.class_num, config.class_num))
        epoch_loss = 0
        for i, pack in enumerate(train_loader):
            images = pack['imgs'].to(device)
            if images.shape[1] == 1:
                images = images.expand((-1, 3, -1, -1))
            labels = pack['labels'].to(device)

            if config.model_name == 'REAF':
                mask = pack['mask'].to(device)
                output = model(images, mask)
            else:
                output = model(images)

            loss = criterion(output, labels)

            pred = output.argmax(dim=1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            cm = confusion_matrix(pred.detach(), labels.detach(), cm)
        lr_scheduler.step()

        if (epoch + 1) % config.log_step == 0:
            print('[epoch %d]' % epoch)
            with torch.no_grad():
                result = valid(config, model, test_loader, criterion)
            val_loss, val_acc, sen, spe, auc, pre, f1score = result
            print(
                'val_loss: {}, val_acc: {}, sen: {}, spe: {}, auc: {}, pre: {}, f1score: {}'.format(
                    val_loss, val_acc, sen, spe, auc, pre, f1score))
            writer.add_scalar('Val/F1score', f1score, global_step=epoch)
            writer.add_scalar('Val/Pre', pre, global_step=epoch)
            writer.add_scalar('Val/Spe', spe, global_step=epoch)
            writer.add_scalar('Val/Sen', sen, global_step=epoch)
            writer.add_scalar('Val/AUC', auc, global_step=epoch)
            writer.add_scalar('Val/Acc', val_acc, global_step=epoch)
            writer.add_scalar('Val/Val_loss', val_loss, global_step=epoch)

            if epoch > config.epochs // 4:
                if val_acc > best_acc:
                    best_acc = val_acc
                    print("=> saved best model")
                    if not os.path.exists(model_save_path):
                        os.makedirs(model_save_path)
                    if config.save_model:
                        torch.save(model.state_dict(), os.path.join(model_save_path,
                                                                    'bestmodel.pth'))
                    with open(os.path.join(model_save_path, 'result.txt'), 'w') as f:
                        f.write('Best Result:\n')
                        f.write(
                            'AUC: %f, ACC: %f, Specificity: %f, Precision: %f, Recall: %f, F1-score: %f'
                            % (auc, val_acc, spe, pre, sen, f1score))
        if epoch + 1 == config.epochs:
            with torch.no_grad():
                result = valid(config, model, test_loader, criterion)
            val_loss, val_acc, sen, spe, auc, pre, f1score = result
            if config.save_model:
                torch.save(model.state_dict(), os.path.join(model_save_path,
                                                            'last_epoch_model.pth'))
            with open(os.path.join(model_save_path, 'result.txt'), 'a') as f:
                f.write('\nLast Result:\n')
                f.write(
                    'AUC: %f, ACC: %f, Specificity: %f, Precision: %f, Recall: %f, F1-score: %f' %
                    (auc, val_acc, spe, pre, sen, f1score))

        avg_epoch_loss = epoch_loss / len(train_loader)
        writer.add_scalar('Train/LR',
                          optimizer.state_dict()['param_groups'][0]['lr'],
                          global_step=epoch)
        writer.add_scalar('Train/Acc', cm.diag().sum() / cm.sum(), global_step=epoch)
        writer.add_scalar('Train/Avg_epoch_loss', avg_epoch_loss, global_step=epoch)


if __name__ == '__main__':
    args = config()
    if args.model_name == 'REAF':
        for i in range(3):
            train_set = utils.get_REAF_dataset(args.data_path,
                                               args.mask_path,
                                               args.train_csv_path,
                                               args.img_size,
                                               mode='train')
            test_set = utils.get_REAF_dataset(args.data_path,
                                              args.mask_path,
                                              args.test_csv_path,
                                              args.img_size,
                                              mode='test')
            print(args)
            argspath = os.path.join(args.model_path, args.model_name, args.writer_comment)
            if not os.path.exists(argspath):
                os.makedirs(argspath)
            with open(os.path.join(argspath, 'model_info.txt'), 'w') as f:
                f.write(str(args))

            train_loader = DataLoader(train_set,
                                      batch_size=args.batch_size,
                                      shuffle=False,
                                      num_workers=12)
            test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
            train(args, train_loader, test_loader, i, 0)
