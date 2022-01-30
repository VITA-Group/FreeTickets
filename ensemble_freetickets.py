from __future__ import print_function

import collections
import os
import time
import argparse
import logging
import hashlib
import copy
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import sparselearning
from sparselearning.core import Masking, CosineDecay, LinearDecay
from sparselearning.models import AlexNet, VGG16, LeNet_300_100, LeNet_5_Caffe, WideResNet, MLP_CIFAR10
from sparselearning.utils import get_mnist_dataloaders, get_cifar10_dataloaders, get_cifar100_dataloaders
import torchvision.transforms as transforms
import seaborn as sns
sns.set()

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
cudnn.benchmark = True
cudnn.deterministic = True

if not os.path.exists('./models'): os.mkdir('./models')
if not os.path.exists('./logs'): os.mkdir('./logs')
logger = None

models = {}
models['MLPCIFAR10'] = (MLP_CIFAR10,[])
models['lenet5'] = (LeNet_5_Caffe,[])
models['lenet300-100'] = (LeNet_300_100,[])
models['alexnet-s'] = (AlexNet, ['s', 10])
models['alexnet-b'] = (AlexNet, ['b', 10])
models['vgg-c'] = (VGG16, ['C', 10])
models['vgg-d'] = (VGG16, ['D', 10])
models['vgg-like'] = (VGG16, ['like', 10])
models['wrn-28-10'] = (WideResNet, [28, 10, 10, 0.3])
models['wrn-28-2'] = (WideResNet, [28, 2, 10, 0.3])
models['wrn-22-8'] = (WideResNet, [22, 8, 10, 0.3])
models['wrn-16-8'] = (WideResNet, [16, 8, 10, 0.3])
models['wrn-16-10'] = (WideResNet, [16, 10, 10, 0.3])

import re


def sorted_nicely(l):
    """ Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)

def setup_logger(args):
    global logger
    if logger == None:
        logger = logging.getLogger()
    else:  # wish there was a logger.close()
        for handler in logger.handlers[:]:  # make a copy of the list
            logger.removeHandler(handler)

    args_copy = copy.deepcopy(args)
    # copy to get a clean hash
    # use the same log file hash if iterations or verbose are different
    # these flags do not change the results
    args_copy.iters = 1
    args_copy.verbose = False
    args_copy.log_interval = 1
    args_copy.seed = 0

    log_path = './logs/{0}_{1}_{2}.log'.format(args.model, args.density, hashlib.md5(str(args_copy).encode('utf-8')).hexdigest()[:8])

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt='%(asctime)s: %(message)s', datefmt='%H:%M:%S')

    fh = logging.FileHandler(log_path)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

def print_and_log(msg):
    global logger
    print(msg)
    logger.info(msg)


# Given model and test data, return true_labels and predictions.
def evaluate_tsne(args, model, device, test_loader, is_test_set=False):
    model.eval()
    test_loss = 0
    correct = 0
    n = 0
    true_labels = []
    pred_labels = []
    with torch.no_grad():
        for data, target in test_loader:
            n += 1
            if n > 1: break
            data, target = data.to(device), target.to(device)
            if args.fp16: data = data.half()
            output = model(data)

            true_labels.append(target)
            pred_labels.append(output)  ## change here

    pred_labels = torch.cat(pred_labels, dim=0)
    # true_labels = torch.cat(true_labels, dim=0)
    return true_labels, pred_labels


def extract_prediction(val_loader, model, args):
    """
    Run evaluation
    """
    model.eval()
    start = time.time()

    y_pred = []
    y_true = []

    for i, (input, target) in enumerate(val_loader):

        input = input.cuda()
        target = target.cuda()

        # compute output
        with torch.no_grad():
            output = model(input)
            pred = F.softmax(output, dim=1)

            y_true.append(target.cpu().numpy())
            y_pred.append(pred.cpu().numpy())

        if i % args.print_freq == 0:
            end = time.time()
            print('Scores: [{0}/{1}]\t'
                  'Time {2:.2f}'.format(
                i, len(val_loader), end - start))
            start = time.time()

    y_pred = np.concatenate(y_pred, axis=0)
    y_true = np.concatenate(y_true, axis=0)

    print('* prediction shape = ', y_pred.shape)
    print('* ground truth shape = ', y_true.shape)

    return y_pred, y_true

def test_calibration(val_loader, model, args):

    y_pred, y_true = extract_prediction(val_loader, model, args)
    ece = expected_calibration_error(y_true, y_pred)
    nll = F.nll_loss(torch.from_numpy(y_pred).log(), torch.from_numpy(y_true), reduction="mean")
    print('* ECE = {}'.format(ece))
    print('* NLL = {}'.format(nll))

    return ece, nll


def expected_calibration_error(y_true, y_pred, num_bins=15):
    pred_y = np.argmax(y_pred, axis=-1)
    correct = (pred_y == y_true).astype(np.float32)
    prob_y = np.max(y_pred, axis=-1)

    b = np.linspace(start=0, stop=1.0, num=num_bins)
    bins = np.digitize(prob_y, bins=b, right=True)

    o = 0
    for b in range(num_bins):
        mask = bins == b
        if np.any(mask):
            o += np.abs(np.sum(correct[mask] - prob_y[mask]))

    return o / y_pred.shape[0]

def evaluate(args, model, device, test_loader, is_test_set=False):
    model.eval()
    test_loss = 0
    nll_loss = 0
    correct = 0
    n = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            if args.fp16: data = data.half()
            output = model(data)
            softmax_preds = torch.nn.Softmax(dim=1)(input=output)
            nll_loss = F.nll_loss(torch.log(softmax_preds), target, reduction='mean').item()  # NLL
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            n += target.shape[0]

    test_loss /= float(n)

    print_and_log('\n{}: Average loss: {:.4f}, NLL loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        'Test evaluation' if is_test_set else 'Evaluation',
        test_loss, nll_loss,correct, n, 100. * correct / float(n)))
    return correct / float(n), nll_loss

def evaluate_ensemble(args, model, device, test_loader, is_test_set=False):
    model.eval()
    test_loss = 0
    correct = 0
    n = 0
    current_fold_preds = []
    test_data = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            if args.fp16: data = data.half()
            output = model(data)
            softmax_preds = torch.nn.Softmax(dim=1)(input=output)
            current_fold_preds.append(softmax_preds)
            test_data.append(target)
    current_fold_preds = torch.cat(current_fold_preds, dim=0)
    test_data = torch.cat(test_data, dim=0)

    return current_fold_preds, test_data


def evaluate_ensemble_bd(args, model, device, test_loader, is_test_set=False):
    '''
    breakdown version of ensemble
    '''
    model.eval()
    test_loss = 0
    correct = 0
    n = 0
    current_fold_preds = []
    test_data = []
    for i in range(100):
        with torch.no_grad():
            for data, target in test_loader:
                data = data[target==i]
                target = target[target==i]
                if data.nelement() == 0: continue
                data, target = data.to(device), target.to(device)
                if args.fp16: data = data.half()
                output = model(data)
                softmax_preds = torch.nn.Softmax(dim=1)(input=output)
                current_fold_preds.append(softmax_preds)
                test_data.append(target)
    current_fold_preds = torch.cat(current_fold_preds, dim=0)
    test_data = torch.cat(test_data, dim=0)

    return current_fold_preds, test_data

def evaluate_ensemble_KD(args, model, device, test_loader, is_test_set=False):
    model.eval()
    test_loss = 0
    correct = 0
    n = 0
    current_fold_preds = []
    test_data = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            if args.fp16: data = data.half()
            logits = model(data)
            # softmax_preds = torch.nn.functional.log_softmax(input=logits, dim=1)
            current_fold_preds.append(logits)
            test_data.append(target)
    current_fold_preds = torch.cat(current_fold_preds, dim=0)
    test_data = torch.cat(test_data, dim=0)

    return current_fold_preds, test_data

def evaluate_ensemble_KD_intermediate(args, model, device, test_loader, layer_name, is_test_set=False):
    model.eval()

    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    for name, layer in model.named_modules():
        if name == layer_name:
        # if isinstance(layer, nn.Linear):
            layer.register_forward_hook(get_activation(name))

    current_preds = collections.defaultdict(list)
    test_data = []
    with torch.no_grad():
        for index, (data, target) in enumerate(test_loader):
            if index > 0: break
            data, target = data.to(device), target.to(device)
            if args.fp16: data = data.half()
            logits = model(data)
            test_data.append(target)

            for key in activation:
                current_preds[key].append(activation[key])

    for key in current_preds.keys():
        current_preds[key] = torch.cat(current_preds[key], dim=0)
    return current_preds

def loss_fn_kd_models(scores_model, target_scores_model):

    scores = []
    for key in scores_model:
        scores.append(loss_fn_kd(scores_model[key], target_scores_model[key]).cpu().data.numpy())

    return scores

def loss_fn_kd(scores, target_scores):
    """Compute knowledge-distillation (KD) loss given [scores] and [target_scores].
    Both [scores] and [target_scores] should be tensors, although [target_scores] should be repackaged.
    'Hyperparameter': temperature"""
    # if len(scores.size()) != 2:
    #     scores = scores.reshape(scores.size(0), -1)
    #     target_scores = scores.reshape(target_scores.size(0), -1)

    device = scores.device

    log_scores_norm = F.log_softmax(scores)
    targets_norm = F.softmax(target_scores)

    # if [scores] and [target_scores] do not have equal size, append 0's to [targets_norm]
    if not scores.size(1) == target_scores.size(1):
        print('size does not match')

    n = scores.size(1)
    if n>target_scores.size(1):
        n_batch = scores.size(0)
        zeros_to_add = torch.zeros(n_batch, n-target_scores.size(1))
        zeros_to_add = zeros_to_add.to(device)
        targets_norm = torch.cat([targets_norm.detach(), zeros_to_add], dim=1)
    KD_loss_unnorm = F.kl_div(log_scores_norm, targets_norm, reduction="batchmean")

    return KD_loss_unnorm


def test_ensemble_part(model, val_loader):
    """
    Run evaluation
    """
    model.eval()
    current_fold_preds = []
    test_data = []
    for i, (input, target) in enumerate(val_loader):
        input = input.cuda()
        target = target.cuda()

        # compute output
        with torch.no_grad():
            output = model(input)
            softmax_preds = torch.nn.Softmax(dim=1)(input=output)
            current_fold_preds.append(softmax_preds)
            test_data.append(target)

    current_fold_preds = torch.cat(current_fold_preds, dim=0)
    test_data = torch.cat(test_data, dim=0)

    pred = current_fold_preds.argmax(dim=1, keepdim=True)
    correct = pred.eq(test_data.view_as(pred)).sum().item()
    n = test_data.shape[0]
    print('\n{}: Accuracy: {}/{} ({:.3f}%)\n'.format(
        'Test evaluation',
        correct, n, 100. * correct / float(n)))

    return current_fold_preds, test_data

def test_ensemble(val_loader, model, args):
    print_and_log("=> loading checkpoint '{}'".format(args.resume))
    all_folds_preds = []
    model_files = os.listdir(args.resume)
    model_files = sorted_nicely(model_files)

    for file in range(0, len(model_files)):
        print(model_files[file])
        if 'EDST' in args.resume:
            checkpoint = torch.load(args.resume + str(model_files[file]))
        elif 'DST' in args.resume:
            checkpoint = torch.load(args.resume + str(model_files[file] + '/model.pth'))
        model.load_state_dict(checkpoint['state_dict'])
        model.cuda()

        current_fold_preds, target = test_ensemble_part(model, val_loader)
        all_folds_preds.append(current_fold_preds)

    output_mean = torch.mean(torch.stack(all_folds_preds, dim=0), dim=0)
    print(output_mean.size())

    pred = output_mean.argmax(dim=1, keepdim=True)
    correct = pred.eq(target.view_as(pred)).sum().item()
    n = target.shape[0]
    print('\n{}: Accuracy: {}/{} ({:.3f}%)\n'.format(
        'Test evaluation',
            correct, n, 100. * correct / float(n)))

def ensemble_calibration_corruption(model, args, normalize):

    all_folds_preds_c = []
    print_and_log("=> loading checkpoint '{}'".format(args.resume))
    model_files = os.listdir(args.resume)
    model_files = sorted_nicely(model_files)

    for file in range(0, len(model_files)):
        all_preds = []
        all_targets = []

        print(model_files[file])
        if 'EDST' in args.resume:
            checkpoint = torch.load(args.resume + str(model_files[file]))
        elif 'DST' in args.resume:
            checkpoint = torch.load(args.resume + str(model_files[file] + '/model.pth'))
        model.load_state_dict(checkpoint['state_dict'])
        model.cuda()

        cifar_c_path = args.cf_c_path
        file_list = os.listdir(cifar_c_path)
        file_list.sort()

        for file_name in file_list:
            if not file_name == 'labels.npy':
                attack_type = file_name[:-len('.npy')]
                for severity in range(1,6):
                    print('attack_type={}'.format(attack_type), 'severity={}'.format(severity))
                    cifar10c_test_loader = cifar_c_dataloaders(args.batch_size, cifar_c_path, severity, attack_type, normalize)
                    y_pred, y_true = extract_prediction(cifar10c_test_loader, model,args)

                    print('* Acc = {}'.format(np.mean(np.argmax(y_pred, 1)==y_true)))
                    all_preds.append(y_pred)
                    all_targets.append(y_true)

        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        print('* Acc = {}'.format(np.mean(np.argmax(all_preds, 1)==all_targets)))
        all_folds_preds_c.append(all_preds)

        ece = expected_calibration_error(all_targets, all_preds)
        nll = F.nll_loss(torch.from_numpy(all_preds).log(), torch.from_numpy(all_targets), reduction="mean")
        print('* c-ECE = {}'.format(ece))
        print('* c-NLL = {}'.format(nll))

    output_mean = np.mean(np.stack(all_folds_preds_c, 0), 0)
    print(output_mean.shape)

    ece = expected_calibration_error(all_targets, output_mean)
    nll = F.nll_loss(torch.from_numpy(output_mean).log(), torch.from_numpy(all_targets), reduction="mean")
    print('* c-ECE = {}'.format(ece))
    print('* c-NLL = {}'.format(nll))
    print('* Acc = {}'.format(np.mean(np.argmax(output_mean, 1)==all_targets)))

def ensemble_calibration(val_loader, model, args):
    print_and_log("=> loading checkpoint '{}'".format(args.resume))
    all_folds_preds = []
    model_files = os.listdir(args.resume)
    model_files = sorted_nicely(model_files)

    for file in range(0, len(model_files)):
        print(model_files[file])
        checkpoint = torch.load(os.path.join(args.resume, str(model_files[file])))
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.cuda()
        y_pred, y_true = extract_prediction(val_loader, model, args)
        all_folds_preds.append(y_pred)


        ece = expected_calibration_error(y_true, y_pred)
        nll = F.nll_loss(torch.from_numpy(y_pred).log(), torch.from_numpy(y_true), reduction="mean")
        print('* ECE = {}'.format(ece))
        print('* NLL = {}'.format(nll))

    output_mean = np.mean(np.stack(all_folds_preds, 0), 0)
    print(output_mean.shape)

    ece = expected_calibration_error(y_true, output_mean)
    nll = F.nll_loss(torch.from_numpy(output_mean).log(), torch.from_numpy(y_true), reduction="mean")
    print('* ECE = {}'.format(ece))
    print('* NLL = {}'.format(nll))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--grow-switch', default='', type=str,
                           help='flag to switch another grow initialization')
    parser.add_argument('--grow-zero', action='store_true',
                           help='flag to switch another grow initialization')
    parser.add_argument('--grow-max', action='store_true',
                           help='flag to switch another grow initialization')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 100)')
    parser.add_argument('--batch-size-jac', type=int, default=200, metavar='N',
                        help='batch size for jac (default: 1000)')
    parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                        help='input batch size for testing (default: 100)')
    parser.add_argument('--multiplier', type=int, default=1, metavar='N',
                        help='extend training time by multiplier times')
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=17, metavar='S', help='random seed (default: 17)')
    parser.add_argument('--ensemble_size', type=int, default=1, help='ensemble size')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--optimizer', type=str, default='sgd', help='The optimizer to use. Default: sgd. Options: sgd, adam.')
    randomhash = ''.join(str(time.time()).split('.'))
    parser.add_argument('--save', type=str, default=randomhash + '.pt',
                        help='path to save the final model')
    parser.add_argument('--data', type=str, default='mnist')
    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--decay_frequency', type=int, default=25000)
    parser.add_argument('--l1', type=float, default=0.0)
    parser.add_argument('--fp16', action='store_true', help='Run in fp16 mode.')
    parser.add_argument('--valid_split', type=float, default=0.1)
    parser.add_argument('--resume', type=str)
    parser.add_argument('--mode', type=str, help='disagreement model')
    parser.add_argument('--start-epoch', type=int, default=1)
    parser.add_argument('--model', type=str, default='')
    parser.add_argument('--l2', type=float, default=1e-4)
    parser.add_argument('--iters', type=int, default=1, help='How many times the model should be run after each other. Default=1')
    parser.add_argument('--start_iter', type=int, default=1, help='How many times the model should be run before pruning. Default=1')
    parser.add_argument('--save-features', action='store_true', help='Resumes a saved model and saves its feature data to disk for plotting.')
    parser.add_argument('--bench', action='store_true', help='Enables the benchmarking of layers and estimates sparse speedups')
    parser.add_argument('--max-threads', type=int, default=10, help='How many threads to use for data loading.')
    parser.add_argument('--decay-schedule', type=str, default='cosine', help='The decay schedule for the pruning rate. Default: cosine. Choose from: cosine, linear.')
    parser.add_argument('--nolr_scheduler', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--l2_regu', action='store_true', default=False,
                        help='if ture, add a l2 norm')
    parser.add_argument('--no_rewire_extend', action='store_true', default=False,
                        help='if ture, only do rewire for 250 epoochs')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--mgpu', action='store_true', help='Enable snip initialization. Default: True.')
    parser.add_argument('--cf_c_path', type=str, default='', help='path for the CIFAR-10-C')
    parser.add_argument('--output_file', type=str, default='', help='path for output file')
    parser.add_argument('--print_freq', default=50, type=int, help='print frequency')

    sparselearning.core.add_sparse_args(parser)

    args = parser.parse_args()
    setup_logger(args)
    print_and_log(args)
    if args.fp16:
        try:
            from apex.fp16_utils import FP16_Optimizer
        except:
            print('WARNING: apex not installed, ignoring --fp16 option')
            args.fp16 = False

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    print_and_log('\n\n')
    print_and_log('='*80)
    torch.manual_seed(args.seed)
    for i in range(args.iters):
        print_and_log("\nIteration start: {0}/{1}\n".format(i + 1, args.iters))

        if args.dataset == 'mnist':
            train_loader, valid_loader, test_loader = get_mnist_dataloaders(args, validation_split=args.valid_split)
        elif args.dataset == 'cifar10':
            train_loader, valid_loader, test_loader = get_cifar10_dataloaders(args, args.valid_split,
                                                                              max_threads=args.max_threads)
        elif args.data == 'cifar100':
            train_loader, valid_loader, test_loader = get_cifar100_dataloaders(args, args.valid_split,
                                                                               max_threads=args.max_threads)
            outputs = 100

        if args.model not in models:
            print('You need to select an existing model via the --model argument. Available models include: ')
            for key in models:
                print('\t{0}'.format(key))
            raise Exception('You need to select a model')
        else:
            cls, cls_args = models[args.model]
            if args.dataset == 'cifar100':
                cls_args[2] = 100
            model = cls(*(cls_args + [args.save_features, args.bench])).to(device)
            print_and_log(model)
            print_and_log('=' * 60)
            print_and_log(args.model)
            print_and_log('=' * 60)


        if args.resume:
                ###############################################################################
                #                          disagreement                                      #
                ##############################################################################
                if 'disagreement' in args.mode:

                    labels = []
                    val_acc = []
                    nll_loss = []

                    print_and_log("=> loading checkpoint '{}'".format(args.resume))

                    model_files = os.listdir(args.resume)
                    model_files = sorted_nicely(model_files)
                    all_folds_preds = []

                    for file in range(0, len(model_files)):

                        print(model_files[file])
                        checkpoint = torch.load(args.resume + str(model_files[file]))
                        if 'state_dict' in checkpoint:
                            model.load_state_dict(checkpoint['state_dict'])
                        else:
                            model.load_state_dict(checkpoint)

                        current_fold_preds, target = evaluate_ensemble(args, model, device, test_loader)
                        all_folds_preds.append(current_fold_preds)
                        labels.append(target)

                    print(torch.equal(labels[0], labels[1]))

                    predictions = []
                    num_models = len(all_folds_preds)
                    dis_matric = np.zeros(shape=(num_models,num_models))

                    for i in range(num_models):
                        pred_labels = np.argmax(all_folds_preds[i].cpu().numpy(), axis=1)
                        predictions.append(pred_labels)

                    for i in range(num_models):
                        preds1 = predictions[i]
                        for j in range(i, num_models):
                            preds2 = predictions[j]
                            # compute dissimilarity
                            dissimilarity_score = 1 - np.sum(np.equal(preds1, preds2)) / (preds1.shape[0])
                            dis_matric[i][j] = dissimilarity_score
                            if i is not j:
                                dis_matric[j][i] = dissimilarity_score

                    dissimilarity_coeff = dis_matric[::-1]
                    dissimilarity_coeff = dissimilarity_coeff + 0.001 * (dissimilarity_coeff!=0)
                    plt.figure(figsize=(9, 8))
                    ss = np.arange(num_models)[::-1]
                    ax = sns.heatmap(dissimilarity_coeff, cmap='RdBu_r', vmin=0, vmax=0.035)
                    cbar = ax.collections[0].colorbar
                    # here set the labelsize by 20
                    cbar.ax.tick_params(labelsize=22)
                    plt.xticks(list(np.arange(num_models) + 0.5), list(np.arange(num_models) + 1), fontsize=22)
                    plt.yticks(list(np.arange(num_models) + 0.5), list(np.arange(num_models)[::-1] + 1), fontsize=22)
                    print(np.sum(dissimilarity_coeff)/np.sum(dissimilarity_coeff!=0))
                    # plt.savefig("./plots/" + "%s_M=%d_prediction_disagreement_%s.pdf" % (method, num_models, args.dataset))
                    plt.savefig("./plots/" + "D_WRN_CF10_dense.pdf")

                if 'predict' in args.mode:
                    print_and_log("=> loading checkpoint '{}'".format(args.resume))

                    model_files = os.listdir(args.resume)
                    model_files = sorted_nicely(model_files)

                    all_folds_preds = []
                    labels = []
                    val_acc = []
                    nll_loss = []
                    for file in range(0, len(model_files)):

                        print(model_files[file])
                        # if not 'DST' in args.resume:
                        checkpoint = torch.load(args.resume + str(model_files[file]))
                        if 'state_dict' in checkpoint:
                            model.load_state_dict(checkpoint['state_dict'])
                        else:
                            model.load_state_dict(checkpoint)

                        print_and_log('Testing...')
                        indi_acc, indi_loss = evaluate(args, model, device, test_loader)
                        val_acc.append(indi_acc)
                        nll_loss.append(indi_loss)

                        current_fold_preds, target = evaluate_ensemble(args, model, device, test_loader)
                        all_folds_preds.append(current_fold_preds)
                        labels.append(target)

                    print(torch.equal(labels[0], labels[1]))

                    individual_acc_mean = np.array(val_acc).mean(axis=0)
                    individual_acc_std = np.array(val_acc).std(axis=0)
                    individual_nll_mean = np.array(nll_loss).mean(axis=0)
                    individual_nll_std = np.array(nll_loss).std(axis=0)
                    print(f"Averaged individual model: acc is {individual_acc_mean} and std is {individual_acc_std}")
                    print(f"Averaged individual model: NLL is {individual_nll_mean} and std is {individual_nll_std}")

                    output_mean = torch.mean(torch.stack(all_folds_preds, dim=0), dim=0)
                    test_loss = F.nll_loss(torch.log(output_mean), target, reduction='mean').item()  # sum up batch loss
                    print(f"Ensemble NLL is {test_loss}")
                    pred = output_mean.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                    correct = pred.eq(target.view_as(pred)).sum().item()
                    n = target.shape[0]
                    print_and_log('\n{}: Ensemble Accuracy is: {}/{} ({:.3f}%)\n'.format(
                        'Test evaluation',
                         correct, n, 100. * correct / float(n)))
                ###############################################################################
                #                          ensemble calibration                              #
                ##############################################################################
                if 'calibration' in args.mode:
                    cifar_c_path = args.cf_c_path
                    if args.dataset == 'cifar10':
                        cor_path = 'CIFAR-10-C'
                        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                         (0.2023, 0.1994, 0.2010))
                    elif args.dataset == 'cifar100':
                        cor_path = 'CIFAR-100-C'
                        normalize = transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                                                         (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
                    # test_ensemble(test_loader, model, args)
                    ensemble_calibration(test_loader, model, args)
                    # ensemble_calibration_corruption(model, args, normalize)

                ###############################################################################
                #                          ensemble DL                                       #
                ###############################################################################
                if 'KD' in args.mode:
                    print_and_log("=> loading checkpoint '{}'".format(args.resume))
                    all_folds_preds = []
                    model_files = os.listdir(args.resume)
                    model_files = sorted_nicely(model_files)

                    all_folds_preds = []
                    for file in range(0, len(model_files)):
                        print(model_files[file])
                        checkpoint = torch.load(args.resume + str(model_files[file]))
                        if 'state_dict' in checkpoint:
                            model.load_state_dict(checkpoint['state_dict'])
                        else:
                            model.load_state_dict(checkpoint)

                        print_and_log('Testing...')
                        current_fold_preds, data = evaluate_ensemble_KD(args, model, device, test_loader)
                        all_folds_preds.append(current_fold_preds)
                    KL_SCORE = []
                    KL_SCORE.append(loss_fn_kd(all_folds_preds[0], all_folds_preds[1]).cpu().data.numpy())
                    KL_SCORE.append(loss_fn_kd(all_folds_preds[1], all_folds_preds[0]).cpu().data.numpy())
                    KL_SCORE.append(loss_fn_kd(all_folds_preds[0], all_folds_preds[2]).cpu().data.numpy())
                    KL_SCORE.append(loss_fn_kd(all_folds_preds[2], all_folds_preds[0]).cpu().data.numpy())
                    KL_SCORE.append(loss_fn_kd(all_folds_preds[1], all_folds_preds[2]).cpu().data.numpy())
                    KL_SCORE.append(loss_fn_kd(all_folds_preds[2], all_folds_preds[1]).cpu().data.numpy())
                    mean_KD = np.array(KL_SCORE).mean()
                    print('KL is:', mean_KD)
                ###############################################################################
                #                          ensemble tsne                                     #
                ###############################################################################
                if 'tsne' == args.mode:
                    independent_models_tsne = os.listdir(args.resume)
                    predictions_for_tsne = []
                    for i in range(len(independent_models_tsne)):
                        subdir = independent_models_tsne[i]
                        model_files = os.listdir(args.resume + subdir)
                        model_files = sorted_nicely(model_files)
                        predictions = []

                        for model_file in range(0,len(model_files),2):
                            if model_file < 145: continue

                            if model_files[model_file] == 'initial.pt':  # this check is because this model checkpoint didn't serialize properly.
                                continue
                            checkpoint = torch.load(args.resume + str(subdir) + '/'+model_files[model_file])
                            model.load_state_dict(checkpoint)
                            _, preds =  evaluate_tsne(args, model, device, test_loader)
                            predictions.append(preds)
                        # [20, 1000, 10]
                        predictions = torch.stack(predictions, dim=0)
                        print(predictions.size())
                        predictions_for_tsne.append(predictions)

                    predictions_for_tsne = torch.stack(predictions_for_tsne, dim=0)
                    print(predictions_for_tsne.size())

                    predictions_for_tsne = predictions_for_tsne.view(-1, predictions_for_tsne.size()[-2]*predictions_for_tsne.size()[-1])
                    # [60, 10000]
                    print(predictions_for_tsne.size())
                    # torch.save(predictions_for_tsne, args.resume + "/tsne_pred.pt")
                    # predictions_for_tsne = predictions_for_tsne.numpy()
                    NUM_TRAJECTORIES =3
                    fontsize = 20
                    tsne = TSNE(n_components=2)
                    # compute tsne
                    trajectory_embed = []
                    prediction_embed = tsne.fit_transform(predictions_for_tsne.cpu().numpy())
                    # trajectory_embed = prediction_embed.view(NUM_TRAJECTORIES, -1, 2)
                    trajectory_embed = np.reshape(prediction_embed,(NUM_TRAJECTORIES, -1, 2))

                    plt.figure(constrained_layout=True, figsize=(6, 6))
                    Model = ['Subnetwork1', 'Subnetwork2', 'Subnetwork3']
                    colors_list = ['r', 'b', 'g']
                    labels_list = ['traj_{}'.format(i) for i in range(NUM_TRAJECTORIES)]
                    for i in range(NUM_TRAJECTORIES):
                        plt.plot(trajectory_embed[i, :, 0], trajectory_embed[i, :, 1], color=colors_list[i], alpha=0.8,
                                 linestyle="", marker="o",  label=Model[i])
                        plt.plot(trajectory_embed[i, :, 0], trajectory_embed[i, :, 1], color=colors_list[i], alpha=0.3,
                                 linestyle="-", marker="")
                        plt.plot(trajectory_embed[i, 0, 0], trajectory_embed[i, 0, 1], color=colors_list[i], alpha=1.0,
                                 linestyle="", marker="*", markersize=10)

                    plt.xlabel('Dimension 1', fontsize=fontsize)
                    plt.ylabel('Dimension 2', fontsize=fontsize)
                    plt.xticks(fontsize=15)
                    plt.yticks(fontsize=15)
                    plt.legend(fontsize=15)
                    # plt.savefig('cifar100_loss_landscape_2d_lr01.png')
                    plt.savefig('WRN_CF10_DST_1.pdf')
                    # reshape
                    # trajectory_embed = prediction_embed.reshape([NUM_TRAJECTORIES, -1, 2])
                    # print('[INFO] Shape of reshaped tensor: ', trajectory_embed.shape)



if __name__ == '__main__':
   main()
