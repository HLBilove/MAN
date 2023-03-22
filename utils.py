# Code reused from  https://github.com/arghosh/AKT
import os
import torch
from man import MAN
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def try_makedirs(path_):
    if not os.path.isdir(path_):
        try:
            os.makedirs(path_)
        except FileExistsError:
            pass
            
def varible(tensor, gpu):
    if gpu >= 0:
        return torch.autograd.Variable(tensor).cuda()
    else:
        return torch.autograd.Variable(tensor)

def to_scalar(var):
    return var.view(-1).data.tolist()[0]

def get_file_name_identifier(params):
    words = params.model.split('_')
    model_type = words[0]
    if model_type in {'man', 'akt'}:
        file_name = [['_b', params.batch_size], ['_nb', params.n_block], ['_gn', params.maxgradnorm], ['_lr', params.lr],
                     ['_s', params.seed], ['_sl', params.seqlen], ['_do', params.dropout], ['_dm', params.d_model], ['_ts', params.train_set], ['_kq', params.kq_same], ['_l2', params.l2]]
    return file_name


def model_id_type(model_name):
    words = model_name.split('_')
    return words[0]

def load_model(params):
    words = params.model.split('_')
    model_type = words[0]

    if model_type in {'man'}:
        model = MAN(n_skill=params.n_skill, n_exercise=params.n_exercise, n_blocks=params.n_block, d_model=params.d_model,
                        dropout=params.dropout, kq_same=params.kq_same, model_type=model_type, memory_size=params.memory_size,
                        batch_size=params.batch_size, seqlen=params.seqlen, l2=params.l2).to(device)
    else:
        model = None
    return model