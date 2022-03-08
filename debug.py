
import argparse
import logging
import os
import random
import sys
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.dataset_synapse import Synapse_dataset
from utils import test_single_volume
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser()
parser.add_argument('--volume_path', type=str,
                    default='../data/Synapse/test_vol_h5', help='root dir for validation volume data')  # for acdc volume_path=root_dir
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--num_classes', type=int,
                    default=4, help='output channel of network')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')

parser.add_argument('--max_iterations', type=int,default=20000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int, default=2, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24,
                    help='batch_size per gpu')
parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
parser.add_argument('--is_savenii', action="store_true", help='whether to save results during inference')

parser.add_argument('--n_skip', type=int, default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16', help='select one vit model')

parser.add_argument('--test_save_dir', type=str, default='../predictions', help='saving prediction as nii!')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01, help='segmentation network learning rate')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--vit_patches_size', type=int, default=16, help='vit_patches_size, default is 16')
args = parser.parse_args()
if not args.deterministic:
    cudnn.benchmark = True
    cudnn.deterministic = False
else:
    cudnn.benchmark = False
    cudnn.deterministic = True
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

dataset_config = {
    'Synapse': {
        'Dataset': Synapse_dataset,
        'volume_path': '../data/Synapse/test_vol_h5',
        'list_dir': './lists/lists_Synapse',
        'num_classes': 9,
        'z_spacing': 1,
    },
}
dataset_name = args.dataset
args.num_classes = dataset_config[dataset_name]['num_classes']
args.volume_path = dataset_config[dataset_name]['volume_path']
args.Dataset = dataset_config[dataset_name]['Dataset']
args.list_dir = dataset_config[dataset_name]['list_dir']
args.z_spacing = dataset_config[dataset_name]['z_spacing']
args.is_pretrain = True

# name the same snapshot defined in train script!
args.exp = 'TU_' + dataset_name + str(args.img_size)
snapshot_path = "../model/{}/{}".format(args.exp, 'TU')
snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
snapshot_path += '_' + args.vit_name
snapshot_path = snapshot_path + '_skip' + str(args.n_skip)
snapshot_path = snapshot_path + '_vitpatch' + str(
    args.vit_patches_size) if args.vit_patches_size != 16 else snapshot_path
snapshot_path = snapshot_path + '_epo' + str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
if dataset_name == 'ACDC':  # using max_epoch instead of iteration to control training duration
    snapshot_path = snapshot_path + '_' + str(args.max_iterations)[
                                          0:2] + 'k' if args.max_iterations != 30000 else snapshot_path
snapshot_path = snapshot_path + '_bs' + str(args.batch_size)
snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
snapshot_path = snapshot_path + '_' + str(args.img_size)
snapshot_path = snapshot_path + '_s' + str(args.seed) if args.seed != 1234 else snapshot_path

config_vit = CONFIGS_ViT_seg[args.vit_name]
config_vit.n_classes = args.num_classes
config_vit.n_skip = args.n_skip
config_vit.patches.size = (args.vit_patches_size, args.vit_patches_size)
if args.vit_name.find('R50') != -1:
    config_vit.patches.grid = (int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).to(device)

snapshot = os.path.join(snapshot_path, 'best_model.pth')
if not os.path.exists(snapshot): snapshot = snapshot.replace('best_model', 'epoch_' + str(args.max_epochs - 1))
net.load_state_dict(torch.load(snapshot))
snapshot_name = snapshot_path.split('/')[-1]

log_folder = './test_log/test_log_' + args.exp
os.makedirs(log_folder, exist_ok=True)
logging.basicConfig(filename=log_folder + '/' + snapshot_name + ".txt", level=logging.INFO,
                    format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
logging.info(str(args))
logging.info(snapshot_name)
db_test = args.Dataset(base_dir=args.volume_path, split="test_vol", list_dir=args.list_dir)
print(db_test)
testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
logging.info("{} test iterations per epoch".format(len(testloader)))
model=net
model.eval()
metric_list = 0.0
print(testloader)
#print(len(enumerate(testloader)))
list(enumerate(testloader))
for i_batch, sampled_batch in enumerate(testloader):
    print(i_batch)