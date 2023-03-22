# Code reused from https://github.com/arghosh/AKT
import numpy as np
import torch
import math
from math import sqrt
from sklearn import metrics
from utils import model_id_type
import utils as utils
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transpose_data_model = {'man'}


def binaryEntropy(target, pred, mod="avg"):
    loss = target * np.log(np.maximum(1e-10, pred)) + \
        (1.0 - target) * np.log(np.maximum(1e-10, 1.0-pred))
    if mod == 'avg':
        return np.average(loss)*(-1.0)
    elif mod == 'sum':
        return - loss.sum()
    else:
        assert False


def compute_auc(all_target, all_pred):
    #fpr, tpr, thresholds = metrics.roc_curve(all_target, all_pred, pos_label=1.0)
    return metrics.roc_auc_score(all_target, all_pred)

def compute_rmse(prediction, target):
    error = []
    for i in range(len(target)):
        error.append(target[i] - prediction[i])
    
    squaredError = []
    absError = []
    for val in error:
        squaredError.append(val * val)
        absError.append(abs(val))

    rmse =sqrt(sum(squaredError) / len(squaredError))
    return rmse

def compute_acc(all_target, all_pred):
    all_pred[all_pred > 0.5] = 1.0
    all_pred[all_pred <= 0.5] = 0.0
    return metrics.accuracy_score(all_target, all_pred)


def train(net, params, optimizer, data, label):
    net.train()
    s_data = data[0]
    sa_data = data[1]
    e_data = data[2]

    model_type = model_id_type(params.model)

    N = int(math.floor(len(s_data) / params.batch_size))
    s_data = s_data.T
    sa_data = sa_data.T
    shuffled_ind = np.arange(s_data.shape[1])
    np.random.shuffle(shuffled_ind)
    s_data = s_data[:, shuffled_ind]
    sa_data = sa_data[:, shuffled_ind]

    e_data = e_data.T
    e_data = e_data[:, shuffled_ind]

    pred_list = []
    target_list = []
    pred_h_list = []
    target_h_list = []

    element_count = 0
    true_el = 0
    for idx in range(N):
        optimizer.zero_grad()
        s_one_seq = s_data[:, idx*params.batch_size:(idx+1)*params.batch_size]
        e_one_seq = e_data[:, idx * params.batch_size:(idx+1) * params.batch_size]

        sa_one_seq = sa_data[:, idx * params.batch_size:(idx+1) * params.batch_size]
        if model_type in transpose_data_model:
            input_s = np.transpose(s_one_seq[:, :])  # Shape (bs, seqlen)
            input_sa = np.transpose(sa_one_seq[:, :])  # Shape (bs, seqlen)
            target = np.transpose(sa_one_seq[:, :])
            input_e = np.transpose(e_one_seq[:, :])
        else:
            input_s = (s_one_seq[:, :])
            input_sa = (sa_one_seq[:, :])
            target = (sa_one_seq[:, :])
            input_e = (e_one_seq[:, :])
        target = (target - 1) / params.n_skill
        target_1 = np.floor(target)

        el = np.sum(target_1 >= -.9)
        element_count += el

        input_s = torch.from_numpy(input_s).long().to(device)
        input_sa = torch.from_numpy(input_sa).long().to(device)
        target = torch.from_numpy(target_1).float().to(device)

        input_e = torch.from_numpy(input_e).long().to(device)

        loss, pred, true_ct = net(input_s, input_sa, target, input_e)

        pred = pred.detach().cpu().numpy()
        loss.backward()
        true_el += true_ct.cpu().numpy()

        if params.maxgradnorm > 0.:
            torch.nn.utils.clip_grad_norm_(
                net.parameters(), max_norm=params.maxgradnorm)

        optimizer.step()

        # correct: 1.0; wrong 0.0; padding -1.0
        target = target_1.reshape((-1,))
        nopadding_index = np.flatnonzero(target >= -0.9)
        nopadding_index = nopadding_index.tolist()

        pred_nopadding = pred[nopadding_index]
        target_nopadding = target[nopadding_index]
        pred_list.append(pred_nopadding)
        target_list.append(target_nopadding)


    all_pred = np.concatenate(pred_list, axis=0)
    all_target = np.concatenate(target_list, axis=0)

    loss = binaryEntropy(all_target, all_pred)
    auc = compute_auc(all_target, all_pred)
    acc = compute_acc(all_target, all_pred)

    return loss, acc, auc


def test(net, params, optimizer, data, label):
    s_data = data[0]
    sa_data = data[1]
    e_data = data[2]

    model_type = model_id_type(params.model)
    net.eval()

    N = int(math.floor(float(len(s_data)) / float(params.batch_size)))
    s_data = s_data.T
    sa_data = sa_data.T
    e_data = e_data.T

    seq_num = s_data.shape[1]
    pred_list = []
    target_list = []

    count = 0
    true_el = 0
    element_count = 0
    for idx in range(N):
        s_one_seq = s_data[:, idx*params.batch_size:(idx+1)*params.batch_size]
        e_one_seq = e_data[:, idx * params.batch_size:(idx+1) * params.batch_size]

        sa_one_seq = sa_data[:, idx * params.batch_size:(idx+1) * params.batch_size]

        if model_type in transpose_data_model:
            input_s = np.transpose(s_one_seq[:, :])
            input_sa = np.transpose(sa_one_seq[:, :])
            target = np.transpose(sa_one_seq[:, :])
            input_e = np.transpose(e_one_seq[:, :])
        else:
            input_s = (s_one_seq[:, :])
            input_sa = (sa_one_seq[:, :])
            target = (sa_one_seq[:, :])
            input_e = (e_one_seq[:, :])
        target = (target - 1) / params.n_skill
        target_1 = np.floor(target)

        input_s = torch.from_numpy(input_s).long().to(device)
        input_sa = torch.from_numpy(input_sa).long().to(device)
        target = torch.from_numpy(target_1).float().to(device)
        input_e = torch.from_numpy(input_e).long().to(device)

        with torch.no_grad():
            loss, pred, ct = net(input_s, input_sa, target, input_e)
        pred = pred.cpu().numpy()
        true_el += ct.cpu().numpy()

        if (idx + 1) * params.batch_size > seq_num:
            real_batch_size = seq_num - idx * params.batch_size
            count += real_batch_size
        else:
            count += params.batch_size

        target = target_1.reshape((-1,))
        nopadding_index = np.flatnonzero(target >= -0.9)
        nopadding_index = nopadding_index.tolist()
        pred_nopadding = pred[nopadding_index]
        target_nopadding = target[nopadding_index]

        element_count += pred_nopadding.shape[0]
        pred_list.append(pred_nopadding)
        target_list.append(target_nopadding)

    # assert count == seq_num, "Seq not matching"

    all_pred = np.concatenate(pred_list, axis=0)
    all_target = np.concatenate(target_list, axis=0)
    loss = binaryEntropy(all_target, all_pred)
    auc = compute_auc(all_target, all_pred)
    acc = compute_acc(all_target, all_pred)

    if label == 'Test':
        return loss, acc, auc, all_target, all_pred
    else:
        return loss, acc, auc