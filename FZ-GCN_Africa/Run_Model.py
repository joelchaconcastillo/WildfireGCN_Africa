import os
import sys
file_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(file_dir)
import torch
import torch.nn as nn
import argparse
import configparser
import random
from datetime import datetime
from GCN import GCN
from Trainer import Trainer
from TrainInits import init_seed, print_model_parameters
from dataloader import get_dataloaders

#-----------------------------------------------------------------------------#
# dataset and
Mode = 'train'
DEBUG = 'False'
DATASET = 'Africa'
DEVICE = 'cuda:0'
MODEL = 'fire_GCN'
#get configuration
config_file = '{}_{}.conf'.format(DATASET, MODEL)
config = configparser.ConfigParser()
config.read(config_file)
print("++++++++++++++")
print(config_file) # using PEMSD4_AGCRN.conf as "config"!
print("++++++++++++++")
#-----------------------------------------------------------------------------#


#parser
args = argparse.ArgumentParser(description='arguments')
args.add_argument('--dataset', default=DATASET, type=str)
args.add_argument('--mode', default=Mode, type=str)
args.add_argument('--device', default=DEVICE, type=str, help='indices of GPUs')
args.add_argument('--debug', default=DEBUG, type=eval)
args.add_argument('--model', default=MODEL, type=str)
args.add_argument('--cuda', default=True, type=bool)
# from data, these below information could be found in .conf file
#data
args.add_argument('--lag', default=config['data']['lag'], type=int)
args.add_argument('--horizon', default=config['data']['horizon'], type=int)
args.add_argument('--num_nodes', default=config['data']['num_nodes'], type=int)
args.add_argument('--default_graph', default=config['data']['default_graph'], type=eval)
args.add_argument('--dataset_root', default=config['data']['dataset_root'], type=str)
args.add_argument('--num_workers', default=config['data']['num_workers'], type=int)
args.add_argument('--pin_memory', default=config['data']['pin_memory'], type=bool)
args.add_argument('--nan_fill', default=config['data']['nan_fill'], type=float)
args.add_argument('--prefetch_factor', default=config['data']['prefetch_factor'], type=int)
args.add_argument('--persistent_workers', default=config['data']['persistent_workers'], type=bool)

args.add_argument('--patch_width', default=config['data']['patch_width'], type=int)
args.add_argument('--patch_height', default=config['data']['patch_height'], type=int)
#model
args.add_argument('--input_dim', default=config['model']['input_dim'], type=int)
args.add_argument('--output_dim', default=config['model']['output_dim'], type=int)
args.add_argument('--embed_dim', default=config['model']['embed_dim'], type=int)
args.add_argument('--rnn_units', default=config['model']['rnn_units'], type=int)
args.add_argument('--num_layers', default=config['model']['num_layers'], type=int)
args.add_argument('--link_len', default=config['model']['link_len'], type=int)
args.add_argument('--window_len', default=config['model']['window_len'], type=int)
args.add_argument('--weight_decay', default=config['model']['weight_decay'], type=float)
args.add_argument('--load_model_dir', default=None, type=str)

#train
args.add_argument('--loss_func', default=config['train']['loss_func'], type=str)
args.add_argument('--seed', default=config['train']['seed'], type=int)
args.add_argument('--batch_size', default=config['train']['batch_size'], type=int)
args.add_argument('--minbatch_size', default=config['train']['minbatch_size'], type=int)
args.add_argument('--epochs', default=config['train']['epochs'], type=int)
args.add_argument('--lr_init', default=config['train']['lr_init'], type=float)
args.add_argument('--lr_decay', default=config['train']['lr_decay'], type=eval)
args.add_argument('--lr_decay_rate', default=config['train']['lr_decay_rate'], type=float)
args.add_argument('--lr_decay_step', default=config['train']['lr_decay_step'], type=str)
args.add_argument('--early_stop', default=config['train']['early_stop'], type=eval)
args.add_argument('--early_stop_patience', default=config['train']['early_stop_patience'], type=int)
args.add_argument('--grad_norm', default=config['train']['grad_norm'], type=eval)
args.add_argument('--max_grad_norm', default=config['train']['max_grad_norm'], type=int)
args.add_argument('--teacher_forcing', default=False, type=bool)
args.add_argument('--real_value', default=config['train']['real_value'], type=eval, help = 'use real value for loss calculation')
args.add_argument('--positive_weight', default=config['train']['positive_weight'], type=float)
#log
args.add_argument('--log_dir', default='logs4/', type=str)
args.add_argument('--log_step', default=config['log']['log_step'], type=int)
args.add_argument('--plot', default=config['log']['plot'], type=eval)

args = args.parse_args()
args.start_epoch=0

args.features = [
                 'tp', 
                 'sst', 
                  't2m_mean', 
                  'drought_code_mean', 
                 # 'fwi_mean', 
                  'lst_day', 
                  'ndvi', 
                  'pop_dens', 
                  'ssrd', 
                 # 'fcci_ba', 
                 # 'gfed_ba', 
                 # 'gwis_ba',
                  ]
args.input_dim = len(args.features)

args.scaleParameter =  [0.05]
args.sizeBorder = [3]
args.maxDimHoles = [1]
args.ZPI_dir = 'ZPI/'
init_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.set_device(int(args.device[5]))
else:
    args.device = 'cpu'

#init model
model = GCN(args)
model = model.to(args.device)
for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)
    else:
        nn.init.uniform_(p)

if args.load_model_dir is None:
   print_model_parameters(model, only_num=False)

#loa dataset
train_loader, val_loader, test_loader1, test_loader2 = get_dataloaders(args)
scaler = None
loss = torch.nn.NLLLoss(weight=torch.tensor([1. - args.positive_weight, args.positive_weight]), reduction='sum').to(args.device)

optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr_init, eps=1.0e-8,
                             weight_decay=args.weight_decay, amsgrad=False)
#learning rate decay
lr_scheduler = None
if args.lr_decay:
    print('Applying learning rate decay.')
    lr_decay_steps = [int(i) for i in list(args.lr_decay_step.split(','))]
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                        milestones=lr_decay_steps,
                                                        gamma=args.lr_decay_rate)

#config log path

if args.load_model_dir is None:
   current_time = datetime.now().strftime('%Y%m%d%H%M%S%f')+str(random.randint(0, 100000))
   current_dir = os.path.dirname(os.path.realpath(__file__))
   log_dir = os.path.join(current_dir,'experiments', args.dataset, current_time)
else:
   log_dir = args.load_model_dir

args.log_dir = log_dir
args.best_model = None

###TODO:  load an existing model, this works for an interactive analysis
if args.load_model_dir is not None:
  check_point = torch.load(args.load_model_dir+'/best_model.pth')
  args = check_point['config']
  model.load_state_dict(check_point['state_dict'])
  optimizer.load_state_dict(check_point['optimizer'])
  args.best_model = check_point['state_dict_best']
  args.start_epoch =  check_point['epoch']
  loss = check_point['loss']
  lr_scheduler = check_point['lr_scheduler']
  model.to(args.device)


#start training
trainer = Trainer(model, loss, optimizer, train_loader, val_loader, test_loader1, test_loader2, scaler,
                  args, lr_scheduler=lr_scheduler)
if args.mode == 'train':
    trainer.train()
elif args.mode == 'test':
    model.load_state_dict(torch.load('{}.pth'.format(args.dataset)))
    print("Load saved model")
    trainer.test(model, trainer.args, test_loader, scaler, trainer.logger)
else:
    raise ValueError
