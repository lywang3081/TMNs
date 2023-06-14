
from __future__ import print_function
import time,datetime,argparse,os,random
import shutil
import torch.utils.data
from cfg.load_config import opt, cfg_from_file
import inspect
import numpy as np

ts = time.time()

# Arguments
parser=argparse.ArgumentParser(description='xxx')
parser.add_argument('--dataset',default='cifar_10',type=str,required=False, choices=['cifar_10'],help='Dataset name')
parser.add_argument('--method',default='TMNs',type=str,required=False, choices=['TMNs'],help='Method name')
parser.add_argument('--cfg_file',default=None,type=str,required=False, help='Path to the configuration file')
cfg=parser.parse_args()

cfg_file = 'cfg/cifar_10.yml'
cfg_from_file(cfg_file)
print(opt)

#######################################################################################################################
opt.device = torch.device("cuda:" + str(opt.device) if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.set_device(opt.device)
print(opt)


try:
    os.makedirs(opt.outf)
except OSError:
    pass
try:
    os.makedirs(opt.outf_models)
except OSError:
    pass
try:
    os.makedirs(opt.outf + '/mask_histo')
except:
    pass



from dataloaders import cifar_10 as dataloader
from networks import net_TMNs as model
from approaches import TMNs as approach

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
np.random.seed(opt.manualSeed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(opt.manualSeed)


print('Load data...')
data, taskcla, inputsize = dataloader.get(seed=opt.manualSeed,data_root=opt.dataroot)
print('Input size =', inputsize, '\nTask info =', taskcla)
for t in range(10):
    data[t]['train']['y'].data.fill_(t)
    data[t]['test']['y'].data.fill_(t)
    data[t]['valid']['y'].data.fill_(t)

nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
nb_label = 10
if opt.dataset == 'svhn':
    nc = 3
elif opt.dataset == 'cifar':
    nc = 3

#classes are added one by one, we innitialize G with one head output
netG = model.netG(nz, ngf, nc, opt.smax_g, n_classes=1)
print(netG)
netD = model.netD(ndf, nc)
print(netD)
netC = model.netC(ndf, nc)
print(netC)

print("opt.log_dir: ", opt.log_dir)

log_dir = opt.log_dir + datetime.datetime.fromtimestamp(ts).strftime('%Y_%m_%d_%H_%M_%S')
if os.path.exists(log_dir):
    shutil.rmtree(log_dir)
os.makedirs(log_dir)

appr = approach.App(model, netG, netD, netC, log_dir, opt.outf, niter=opt.niter, batchSize=opt.batchSize,
                    imageSize=opt.imageSize, nz=int(opt.nz), nb_label=nb_label, cuda=torch.cuda.is_available(), beta1=opt.beta1,
                    lr_D=opt.lr_D, lr_G=opt.lr_G, lamb_G=opt.lamb_G,
                    reinit_D=opt.reinit_D, old_samples_G_lambda=0, lambda_adv=opt.lambda_adv, lambda_wassersten=opt.lambda_wasserstein, dataset=opt.dataset, store_model = opt.store_models)


lines = inspect.getsource(approach)
lines_model = inspect.getsource(model)
appr.writer.text_summary("appr", str(lines))
appr.writer.text_summary("net", str(lines_model))
appr.writer.text_summary("opt", str(opt))

for t in range(10):
    test_acc_task, conf_matrixes_task, mask_G = appr.train(data, t, smax_g=opt.smax_g, use_aux_G=opt.aux_G)

