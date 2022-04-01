
import torch
import numpy as np
import pickle
import math
import random
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
sns.set(context='notebook', font='simhei', style='whitegrid')
import warnings
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

def statistics(s,covid):
    s = s.flatten()
    zero_dir = np.where(s == 0)
    s = np.delete(s , zero_dir )
    plt.figure(figsize=(8, 6),num=covid)
    sns.distplot(s, bins=14, hist=True, kde=False, norm_hist=False,
                 rug=True, vertical=False, label='Density distribution',rug_kws={'color':'r'},
                 axlabel='Matrix elements sum',hist_kws={'color': 'b', 'edgecolor': 'k'},fit=norm)
    font1 = {'family': 'Times New Roman','weight': 'normal','size': 16,}
    plt.legend(prop=font1)
    plt.tick_params(labelsize=16)
    font2 = {'family': 'Times New Roman','weight': 'normal', 'size': 18,}
    plt.xlabel('Matrix elements sum', font2)
    plt.grid(linestyle='--')
    plt.savefig('./fig/cov{}.png'.format(covid),dpi=300)
    plt.show()
    plt.close()

class mask_vgg_16_bn:
    def __init__(self, model=None, compress_rate=[0.50], job_dir='',device=None):
        self.model = model
        self.compress_rate = compress_rate
        self.cpra = []
        self.mask = {}
        self.job_dir=job_dir
        self.device = device
        self.ws = torch.tensor(0)
        self.bn = torch.tensor(0)
    def layer_mask(self, cov_id, resume=None, param_per_cov=4,  arch="vgg_16_bn", way='A'):
        params = self.model.parameters()
        params = list(params)

        if resume:
            with open(resume, 'rb') as f:
                self.mask = pickle.load(f)
        else:
            resume=self.job_dir+'/mask'

        self.param_per_cov=param_per_cov

        if way == 'A' :
            for index, item in enumerate(params):

                if index == (cov_id - 1) * param_per_cov:
                    f, c, w, h = item.size()
                    self.ws = torch.tensor([torch.sum(item[i, j, :, :]) for i in range(f) for j in range(c)])
                    self.ws = self.ws.view(f, -1).float()
                    self.ws = np.array(self.ws)
                    bs = torch.as_tensor(params[index+1])
                    bs = bs.detach().cpu().numpy()
                    for i in range(len(self.ws)):
                        for j in range(len(self.ws[0])):
                            self.ws[i][j] = self.ws[i][j] + bs[i]
                    statistics(self.ws,cov_id)
                    self.bn = torch.as_tensor(params[index+2])
                    self.bn = self.bn.detach().cpu().numpy()

                    self.ws = np.round( ( 1/(  1+np.exp(-2*(self.ws/np.max(self.ws)))  )-0.5 )*13 )
                    ind = np.argsort(abs(self.bn))[:]
                    pruned_num = int(self.compress_rate[cov_id - 1] * f*c)
                    ones_i = torch.ones(f, c).to(self.device)

                    for i in range(len(ind)):
                        ws_value_list = list(set(self.ws[ind[i]]))
                        for x in range(len(ws_value_list) - 1):
                            for y in range(x + 1, len(ws_value_list)):
                                if np.sum(self.ws == ws_value_list[x]) < np.sum(self.ws == ws_value_list[y]):
                                    ws_value_list[x],ws_value_list[y] = ws_value_list[y],ws_value_list[x]
                        for j in range(len(ws_value_list)) :
                            if pruned_num == 0:
                                break
                            num = np.sum(self.ws[ind[i]] == ws_value_list[j])
                            val_index = np.where(self.ws[ind[i]] == ws_value_list[j] )
                            index_1 = list(range(0,num))
                            index_1 = random.sample(index_1,math.ceil(num*(1-np.max(self.compress_rate)-0.1)))
                            val_index = np.delete(val_index, index_1, axis=1)
                            for m in range(len(val_index[0])) :
                                ones_i[ind[i]][val_index[0][m]] = 0
                                pruned_num -= 1
                                if pruned_num == 0:
                                    break
                        if pruned_num == 0:
                            break
                    cnt_array = np.sum(ones_i.cpu().numpy() == 0)
                    self.cpra.append(format(cnt_array / (f * c), '.2g'))
                    ones = torch.ones(f, c, w, h).to(self.device)
                    for i in range(f):
                        for j in range(c):
                            for k in range(w):
                                for l in range(h):
                                    ones[i, j, k, l] = ones_i[i, j]
                    self.mask[index] = ones
                    item.data = item.data * self.mask[index]

                    with open(resume, "wb") as f:
                        pickle.dump(self.mask, f)
                    break
        elif way == 'B' :
            for index, item in enumerate(params):
                if index == (cov_id - 1) * param_per_cov:
                    f, c, w, h = item.size()
                    item = item.to(self.device)
                    self.ws = torch.tensor([torch.sum(item[i, j, :, :]) for i in range(f) for j in range(c)])
                    self.ws = np.array(self.ws.view(f, -1).float())
                    self.ws = np.round((1 / (1 + np.exp(-2*(self.ws / np.max(self.ws)))) - 0.5) * 13)
                    ws = self.ws[0]
                    for n in range(f - 1) :
                        ws = np.hstack((ws, self.ws[n+1]))
                    pruned_num = int(self.compress_rate[cov_id - 1] * f*c)
                    ones_i = torch.ones(f, c).to(self.device)
                    ws_value_list = list(set(ws))
                    for x in range(len(ws_value_list) - 1):
                        for y in range(x + 1, len(ws_value_list)):
                            if np.sum(self.ws == ws_value_list[x]) < np.sum(self.ws == ws_value_list[y]):
                                ws_value_list[x], ws_value_list[y] = ws_value_list[y], ws_value_list[x]
                    for j in range(len(ws_value_list)) :
                        if pruned_num == 0:
                            break
                        num = np.sum(self.ws == ws_value_list[j])
                        val_index = np.where(self.ws == ws_value_list[j] )
                        index_1 = list(range(0, num))
                        index_1 = random.sample(index_1, math.ceil(num*(1-np.max(self.compress_rate)-0.1)))
                        val_index = np.delete(val_index, index_1, axis=1)
                        for m in range(len(val_index[0])) :
                            ones_i[val_index[0][m]][val_index[1][m]] = 0
                            pruned_num -= 1
                            if pruned_num == 0:
                                break
                        if pruned_num == 0:
                            break
                    cnt_array = np.sum(ones_i.cpu().numpy() == 0)
                    self.cpra.append(format(cnt_array / (f * c), '.2g'))
                    ones = torch.ones(f, c, w, h).to(self.device)
                    for i in range(f):
                        for j in range(c):
                            for k in range(w):
                                for l in range(h):
                                    ones[i, j, k, l] = ones_i[i, j]
                    self.mask[index] = ones
                    item.data = item.data * self.mask[index]

                    with open(resume, "wb") as f:
                        pickle.dump(self.mask, f)
                    break
        else :
            assert 1 == 0

    def grad_mask(self, cov_id,epoch):
        params = self.model.parameters()
        for index, item in enumerate(params):
            if index == cov_id * self.param_per_cov:
                break
            if index in range(0, 48, self.param_per_cov):
                if epoch < 20 :
                    self.mask[index][self.mask[index] != 1] = 1-(epoch+1)*0.05
                item.data = item.data * self.mask[index].to(self.device)

    def pri_comp(self):
        print('compress_rate:{}'.format(self.cpra))

class mask_resnet_56:
    def __init__(self, model=None, compress_rate=[0.50], job_dir='', device=None):
        self.model = model
        self.compress_rate = compress_rate
        self.mask = {}
        self.cpra = []
        self.job_dir = job_dir
        self.device = device
        self.ws = torch.tensor(0)
        self.bn = torch.tensor(0)

    def layer_mask(self, cov_id, resume=None, param_per_cov=3, arch="resnet_56",way='A'):
        params = self.model.parameters()
        params = list(params)

        if resume:
            with open(resume, 'rb') as f:
                self.mask = pickle.load(f)
        else:
            resume = self.job_dir + '/mask'

        self.param_per_cov = param_per_cov

        if way == 'A':
            for index, item in enumerate(params):

                if index == (cov_id - 1) * param_per_cov:
                    f, c, w, h = item.size()
                    self.ws = torch.tensor([torch.sum(item[i, j, :, :]) for i in range(f) for j in range(c)])
                    self.ws = self.ws.view(f, -1).float()
                    self.ws = np.array(self.ws)
                    statistics(self.ws,cov_id)
                    self.bn = torch.as_tensor(params[index + 1])
                    self.bn = self.bn.detach().cpu().numpy()

                    self.ws = np.round((1 / (1 + np.exp(-2*(self.ws / np.max(self.ws)))) - 0.5) * 13)
                    ind = np.argsort(abs(self.bn))[:]
                    pruned_num = int(self.compress_rate[cov_id - 1] * f * c)
                    ones_i = torch.ones(f, c).to(self.device)

                    for i in range(len(ind)):
                        ws_value_list = list(set(self.ws[ind[i]]))
                        for x in range(len(ws_value_list) - 1):
                            for y in range(x + 1, len(ws_value_list)):
                                if np.sum(self.ws == ws_value_list[x]) < np.sum(self.ws == ws_value_list[y]):
                                    ws_value_list[x], ws_value_list[y] = ws_value_list[y], ws_value_list[x]
                        for j in range(len(ws_value_list)):
                            if pruned_num == 0:
                                break
                            num = np.sum(self.ws[ind[i]] == ws_value_list[j])
                            val_index = np.where(self.ws[ind[i]] == ws_value_list[j])
                            index_1 = list(range(0, num))
                            index_1 = random.sample(index_1, math.ceil(num*(1-np.max(self.compress_rate)-0.1)))
                            val_index = np.delete(val_index, index_1, axis=1)
                            for m in range(len(val_index[0])):
                                ones_i[ind[i]][val_index[0][m]] = 0
                                pruned_num -= 1
                                if pruned_num == 0:
                                    break
                        if pruned_num == 0:
                            break

                    cnt_array = np.sum(ones_i.cpu().numpy() == 0)
                    self.cpra.append(format(cnt_array / (f * c), '.2g'))
                    ones = torch.ones(f, c, w, h).to(self.device)
                    for i in range(f):
                        for j in range(c):
                            for k in range(w):
                                for l in range(h):
                                    ones[i, j, k, l] = ones_i[i, j]
                    self.mask[index] = ones
                    item.data = item.data * self.mask[index]

                    with open(resume, "wb") as f:
                        pickle.dump(self.mask, f)
                    break
        elif way == 'B':
            for index, item in enumerate(params):
                if index == (cov_id - 1) * param_per_cov:
                    f, c, w, h = item.size()
                    self.ws = torch.tensor([torch.sum(item[i, j, :, :]) for i in range(f) for j in range(c)])
                    self.ws = self.ws.view(f, -1).float()
                    self.ws = np.array(self.ws)

                    self.ws = np.round((1 / (1 + np.exp(-2*(self.ws / np.max(self.ws)))) - 0.5) * 13)
                    ws = self.ws[0]
                    for n in range(f - 1):
                        ws = np.hstack((ws, self.ws[n + 1]))
                    pruned_num = int(self.compress_rate[cov_id - 1] * f * c)
                    ones_i = torch.ones(f, c).to(self.device)
                    ws_value_list = list(set(ws))
                    for x in range(len(ws_value_list) - 1):
                        for y in range(x + 1, len(ws_value_list)):
                            if np.sum(self.ws == ws_value_list[x]) < np.sum(self.ws == ws_value_list[y]):
                                ws_value_list[x], ws_value_list[y] = ws_value_list[y], ws_value_list[x]
                    for j in range(len(ws_value_list)):
                        if pruned_num == 0:
                            break
                        num = np.sum(self.ws == ws_value_list[j])
                        val_index = np.where(self.ws == ws_value_list[j])
                        index_1 = list(range(0, num))
                        index_1 = random.sample(index_1, math.ceil(num*(1-np.max(self.compress_rate)-0.1)))
                        val_index = np.delete(val_index, index_1, axis=1)
                        for m in range(len(val_index[0])):
                            ones_i[val_index[0][m]][val_index[1][m]] = 0
                            pruned_num -= 1
                            if pruned_num == 0:
                                break
                        if pruned_num == 0:
                            break
                    cnt_array = np.sum(ones_i.cpu().numpy() == 0)
                    self.cpra.append(format(cnt_array / (f * c), '.2g'))
                    ones = torch.ones(f, c, w, h).to(self.device)
                    for i in range(f):
                        for j in range(c):
                            for k in range(w):
                                for l in range(h):
                                    ones[i, j, k, l] = ones_i[i, j]
                    self.mask[index] = ones
                    item.data = item.data * self.mask[index]

                    with open(resume, "wb") as f:
                        pickle.dump(self.mask, f)
                    break
        else:
            assert 1 == 0

    def grad_mask(self, cov_id):
        params = self.model.parameters()
        for index, item in enumerate(params):
            if index == cov_id * self.param_per_cov:
                break
            if index in range(0, 167, self.param_per_cov):
                item.data = item.data * self.mask[index].to(self.device)

    def pri_comp(self):
        print('compress_rate:{}'.format(self.cpra))

class mask_googlenet:
    def __init__(self, model=None, compress_rate=[0.50], job_dir='',device=None):
        self.model = model
        self.compress_rate = compress_rate
        self.cpra = []
        self.mask = {}
        self.job_dir=job_dir
        self.device = device
        self.ws = torch.tensor(0)
        self.bn = torch.tensor(0)

    def layer_mask(self, cov_id, resume=None, param_per_cov=28,  arch="googlenet",way='A'):
        params = self.model.parameters()
        params = list(params)
        if resume:
            with open(resume, 'rb') as f:
                self.mask = pickle.load(f)
        else:
            resume=self.job_dir+'/mask'

        self.param_per_cov=param_per_cov

        if way == 'A':
            for index, item in enumerate(params):
                if index == (cov_id-1) * param_per_cov + 4:
                    break
                if (cov_id == 1 and index == 0) \
                        or index == (cov_id - 1) * param_per_cov - 24 \
                        or index == (cov_id - 1) * param_per_cov - 16 \
                        or index == (cov_id - 1) * param_per_cov - 8 \
                        or index == (cov_id - 1) * param_per_cov - 4 \
                        or index == (cov_id - 1) * param_per_cov:
                    f, c, w, h = item.size()
                self.ws = torch.tensor([torch.sum(item[i, j, :, :]) for i in range(f) for j in range(c)])
                self.ws = self.ws.view(f, -1).float()
                self.ws = np.array(self.ws)
                bs = torch.as_tensor(params[index + 1])
                bs = bs.detach().cpu().numpy()
                for i in range(len(self.ws)):
                    for j in range(len(self.ws[0])):
                        self.ws[i][j] = self.ws[i][j] + bs[i]

                self.bn = torch.as_tensor(params[index + 2])
                self.bn = self.bn.detach().cpu().numpy()
                self.ws = np.round((1 / (1 + np.exp(-2*(self.ws / np.max(self.ws)))) - 0.5) * 13)
                ind = np.argsort(abs(self.bn))[:]
                pruned_num = int(self.compress_rate[cov_id - 1] * f * c)
                ones_i = torch.ones(f, c).to(self.device)

                for i in range(len(ind)):
                    ws_value_list = list(set(self.ws[ind[i]]))
                    for x in range(len(ws_value_list) - 1):
                        for y in range(x + 1, len(ws_value_list)):
                            if np.sum(self.ws == ws_value_list[x]) < np.sum(self.ws == ws_value_list[y]):
                                ws_value_list[x], ws_value_list[y] = ws_value_list[y], ws_value_list[x]
                    for j in range(len(ws_value_list)):
                        if pruned_num == 0:
                            break
                        num = np.sum(self.ws[ind[i]] == ws_value_list[j])
                        val_index = np.where(self.ws[ind[i]] == ws_value_list[j])
                        index_1 = list(range(0, num))
                        index_1 = random.sample(index_1, math.ceil(num*(1-np.max(self.compress_rate)-0.1)))
                        val_index = np.delete(val_index, index_1, axis=1)
                        for m in range(len(val_index[0])):
                            ones_i[ind[i]][val_index[0][m]] = 0
                            pruned_num -= 1
                            if pruned_num == 0:
                                break
                    if pruned_num == 0:
                        break
                cnt_array = np.sum(ones_i.cpu().numpy() == 0)
                self.cpra.append(format(cnt_array / (f * c), '.2g'))
                ones = torch.ones(f, c, w, h).to(self.device)
                for i in range(f):
                    for j in range(c):
                        for k in range(w):
                            for l in range(h):
                                ones[i, j, k, l] = ones_i[i, j]
                self.mask[index] = ones
                item.data = item.data * self.mask[index]

                with open(resume, "wb") as f:
                    pickle.dump(self.mask, f)
                break
        elif way == 'B':
            for index, item in enumerate(params):
                if index == (cov_id - 1) * param_per_cov + 4:
                    break
                if (cov_id == 1 and index == 0) \
                        or index == (cov_id - 1) * param_per_cov - 24 \
                        or index == (cov_id - 1) * param_per_cov - 16 \
                        or index == (cov_id - 1) * param_per_cov - 8 \
                        or index == (cov_id - 1) * param_per_cov - 4 \
                        or index == (cov_id - 1) * param_per_cov:
                    f, c, w, h = item.size()
                    self.ws = torch.tensor([torch.sum(item[i, j, :, :]) for i in range(f) for j in range(c)])
                    self.ws = self.ws.view(f, -1).float()
                    self.ws = np.array(self.ws)

                    self.ws = np.round((1 / (1 + np.exp(-2(self.ws / np.max(self.ws)))) - 0.5) * 13)
                    ws = self.ws[0]
                    for n in range(f - 1):
                        ws = np.hstack((ws, self.ws[n + 1]))
                    pruned_num = int(self.compress_rate[cov_id - 1] * f * c)
                    ones_i = torch.ones(f, c).to(self.device)
                    ws_value_list = list(set(ws))
                    for x in range(len(ws_value_list) - 1):
                        for y in range(x + 1, len(ws_value_list)):
                            if np.sum(self.ws == ws_value_list[x]) < np.sum(self.ws == ws_value_list[y]):
                                ws_value_list[x], ws_value_list[y] = ws_value_list[y], ws_value_list[x]
                    for j in range(len(ws_value_list)):
                        if pruned_num == 0:
                            break
                        num = np.sum(self.ws == ws_value_list[j])
                        val_index = np.where(self.ws == ws_value_list[j])
                        index_1 = list(range(0, num))
                        index_1 = random.sample(index_1, math.ceil(num*(1-np.max(self.compress_rate)-0.1)))
                        val_index = np.delete(val_index, index_1, axis=1)
                        for m in range(len(val_index[0])):
                            ones_i[val_index[0][m]][val_index[1][m]] = 0
                            pruned_num -= 1
                            if pruned_num == 0:
                                break
                        if pruned_num == 0:
                            break
                    cnt_array = np.sum(ones_i.cpu().numpy() == 0)
                    self.cpra.append(format(cnt_array / (f * c), '.2g'))
                    ones = torch.ones(f, c, w, h).to(self.device)
                    for i in range(f):
                        for j in range(c):
                            for k in range(w):
                                for l in range(h):
                                    ones[i, j, k, l] = ones_i[i, j]
                    self.mask[index] = ones
                    item.data = item.data * self.mask[index]

                    with open(resume, "wb") as f:
                        pickle.dump(self.mask, f)
                    break
        else:
            assert 1 == 0

    def grad_mask(self, cov_id):
        params = self.model.parameters()
        for index, item in enumerate(params):
            if index == (cov_id-1) * self.param_per_cov + 4:
                break
            if index not in self.mask:
                continue
            item.data = item.data * self.mask[index].to(self.device)

    def pri_comp(self):
        print('compress_rate:{}'.format(self.cpra))

class mask_resnet_110:
    def __init__(self, model=None, compress_rate=[0.50], job_dir='',device=None):
        self.model = model
        self.compress_rate = compress_rate
        self.cpra = []
        self.mask = {}
        self.job_dir=job_dir
        self.device = device
        self.ws = torch.tensor(0)
        self.bn = torch.tensor(0)
    def layer_mask(self, cov_id, resume=None, param_per_cov=3,  arch="resnet_110_convwise",way='A'):
        params = self.model.parameters()
        params = list(params)
        if resume:
            with open(resume, 'rb') as f:
                self.mask = pickle.load(f)
        else:
            resume=self.job_dir+'/mask'

        self.param_per_cov=param_per_cov

        if way == 'A':
            for index, item in enumerate(params):

                if index == (cov_id - 1) * param_per_cov:
                    f, c, w, h = item.size()
                    self.ws = torch.tensor([torch.sum(item[i, j, :, :]) for i in range(f) for j in range(c)])
                    self.ws = self.ws.view(f, -1).float()
                    self.ws = np.array(self.ws)
                    self.bn = torch.as_tensor(params[index + 1])
                    self.bn = self.bn.detach().cpu().numpy()

                    self.ws = np.round((1 / (1 + np.exp(-2*(self.ws / np.max(self.ws)))) - 0.5) * 13)
                    ind = np.argsort(abs(self.bn))[:]
                    pruned_num = int(self.compress_rate[cov_id - 1] * f * c)
                    ones_i = torch.ones(f, c).to(self.device)

                    for i in range(len(ind)):
                        ws_value_list = list(set(self.ws[ind[i]]))
                        for x in range(len(ws_value_list) - 1):
                            for y in range(x + 1, len(ws_value_list)):
                                if np.sum(self.ws == ws_value_list[x]) < np.sum(self.ws == ws_value_list[y]):
                                    ws_value_list[x], ws_value_list[y] = ws_value_list[y], ws_value_list[x]
                        for j in range(len(ws_value_list)):
                            if pruned_num == 0:
                                break
                            num = np.sum(self.ws[ind[i]] == ws_value_list[j])
                            val_index = np.where(self.ws[ind[i]] == ws_value_list[j])

                            index_1 = list(range(0, num))
                            index_1 = random.sample(index_1, math.ceil(num*(1-np.max(self.compress_rate)-0.1)))
                            val_index = np.delete(val_index, index_1, axis=1)
                            for m in range(len(val_index[0])):
                                ones_i[ind[i]][val_index[0][m]] = 0
                                pruned_num -= 1
                                if pruned_num == 0:
                                    break
                        if pruned_num == 0:
                            break
                    cnt_array = np.sum(ones_i.cpu().numpy() == 0)
                    self.cpra.append(format(cnt_array / (f * c), '.2g'))
                    ones = torch.ones(f, c, w, h).to(self.device)
                    for i in range(f):
                        for j in range(c):
                            for k in range(w):
                                for l in range(h):
                                    ones[i, j, k, l] = ones_i[i, j]
                    self.mask[index] = ones
                    item.data = item.data * self.mask[index]

                    with open(resume, "wb") as f:
                        pickle.dump(self.mask, f)
                    break
        elif way == 'B':
            for index, item in enumerate(params):
                if index == (cov_id - 1) * param_per_cov:
                    f, c, w, h = item.size()
                    self.ws = torch.tensor([torch.sum(item[i, j, :, :]) for i in range(f) for j in range(c)])
                    self.ws = self.ws.view(f, -1).float()
                    self.ws = np.array(self.ws)

                    self.ws = np.round((1 / (1 + np.exp(-2*(self.ws / np.max(self.ws)))) - 0.5) * 13)
                    ws = self.ws[0]
                    for n in range(f - 1):
                        ws = np.hstack((ws, self.ws[n + 1]))
                    pruned_num = int(self.compress_rate[cov_id - 1] * f * c)
                    ones_i = torch.ones(f, c).to(self.device)
                    ws_value_list = list(set(ws))
                    for x in range(len(ws_value_list) - 1):
                        for y in range(x + 1, len(ws_value_list)):
                            if np.sum(self.ws == ws_value_list[x]) < np.sum(self.ws == ws_value_list[y]):
                                ws_value_list[x], ws_value_list[y] = ws_value_list[y], ws_value_list[x]
                    for j in range(len(ws_value_list)):
                        if pruned_num == 0:
                            break
                        num = np.sum(self.ws == ws_value_list[j])
                        val_index = np.where(self.ws == ws_value_list[j])
                        index_1 = list(range(0, num))
                        index_1 = random.sample(index_1, math.ceil(num*(1-np.max(self.compress_rate)-0.1)))
                        val_index = np.delete(val_index, index_1, axis=1)
                        for m in range(len(val_index[0])):
                            ones_i[val_index[0][m]][val_index[1][m]] = 0
                            pruned_num -= 1
                            if pruned_num == 0:
                                break
                        if pruned_num == 0:
                            break
                    cnt_array = np.sum(ones_i.cpu().numpy() == 0)
                    self.cpra.append(format(cnt_array / (f * c), '.2g'))
                    ones = torch.ones(f, c, w, h).to(self.device)
                    for i in range(f):
                        for j in range(c):
                            for k in range(w):
                                for l in range(h):
                                    ones[i, j, k, l] = ones_i[i, j]
                    self.mask[index] = ones
                    item.data = item.data * self.mask[index]

                    with open(resume, "wb") as f:
                        pickle.dump(self.mask, f)
                    break
        else:
            assert 1 == 0

    def grad_mask(self, cov_id):
        params = self.model.parameters()
        for index, item in enumerate(params):
            if index == cov_id*self.param_per_cov:
                break
            if index in range(0, 326, self.param_per_cov):
                item.data = item.data * self.mask[index].to(self.device)

    def pri_comp(self):
        print('compress_rate:{}'.format(self.cpra))

class mask_resnet_50:
    def __init__(self, model=None, compress_rate=[0.50], job_dir='',device=None):
        self.model = model
        self.compress_rate = compress_rate
        self.mask = {}
        self.cpra = []
        self.job_dir = job_dir
        self.device = device
        self.ws = torch.tensor(0)
        self.bn = torch.tensor(0)

    def layer_mask(self, cov_id, resume=None, param_per_cov=3, arch="resnet_56", way='A'):
        params = self.model.parameters()
        params = list(params)

        if resume:
            with open(resume, 'rb') as f:
                self.mask = pickle.load(f)
        else:
            resume = self.job_dir + '/mask'

        self.param_per_cov = param_per_cov

        if way == 'A':
            for index, item in enumerate(params):

                if index == (cov_id - 1) * param_per_cov:
                    f, c, w, h = item.size()
                    self.ws = torch.tensor([torch.sum(item[i, j, :, :]) for i in range(f) for j in range(c)])
                    self.ws = self.ws.view(f, -1).float()
                    self.ws = np.array(self.ws)
                    self.bn = torch.as_tensor(params[index + 1])
                    self.bn = self.bn.detach().cpu().numpy()

                    self.ws = np.round((self.ws / np.max(self.ws)) * (5 - 3 * (cov_id / 12)))
                    self.ws = np.round((1 / (1 + np.exp(-(self.ws / np.max(self.ws)))) - 0.5) * 5)
                    ind = np.argsort(abs(self.bn))[:]
                    pruned_num = int(self.compress_rate[cov_id - 1] * f * c)
                    ones_i = torch.ones(f, c).to(self.device)
                    for i in range(len(ind)):
                        ws_value_list = list(set(self.ws[ind[i]]))
                        for x in range(len(ws_value_list) - 1):
                            for y in range(x + 1, len(ws_value_list)):
                                if np.sum(self.ws == ws_value_list[x]) < np.sum(self.ws == ws_value_list[y]):
                                    ws_value_list[x], ws_value_list[y] = ws_value_list[y], ws_value_list[x]
                        for j in range(len(ws_value_list)):
                            if pruned_num == 0:
                                break
                            num = np.sum(self.ws[ind[i]] == ws_value_list[j])
                            val_index = np.where(self.ws[ind[i]] == ws_value_list[j])

                            index_1 = list(range(0, num))
                            index_1 = random.sample(index_1, math.ceil(num*(1-np.max(self.compress_rate)-0.1)))
                            val_index = np.delete(val_index, index_1, axis=1)
                            for m in range(len(val_index[0])):
                                ones_i[ind[i]][val_index[0][m]] = 0
                                pruned_num -= 1
                                if pruned_num == 0:
                                    break
                        if pruned_num == 0:
                            break
                    cnt_array = np.sum(ones_i.cpu().numpy() == 0)
                    self.cpra.append(format(cnt_array / (f * c), '.2g'))
                    ones = torch.ones(f, c, w, h).to(self.device)
                    for i in range(f):
                        for j in range(c):
                            for k in range(w):
                                for l in range(h):
                                    ones[i, j, k, l] = ones_i[i, j]
                    self.mask[index] = ones
                    item.data = item.data * self.mask[index]

                    with open(resume, "wb") as f:
                        pickle.dump(self.mask, f)
                    break
        elif way == 'B':
            for index, item in enumerate(params):
                if index == (cov_id - 1) * param_per_cov:
                    f, c, w, h = item.size()
                    self.ws = torch.tensor([torch.sum(item[i, j, :, :]) for i in range(f) for j in range(c)])
                    self.ws = self.ws.view(f, -1).float()
                    self.ws = np.array(self.ws)

                    self.ws = np.round((1 / (1 + np.exp(-2*(self.ws / np.max(self.ws)))) - 0.5) * 13)
                    ws = self.ws[0]
                    for n in range(f - 1):
                        ws = np.hstack((ws, self.ws[n + 1]))
                    pruned_num = int(self.compress_rate[cov_id - 1] * f * c)
                    ones_i = torch.ones(f, c).to(self.device)
                    ws_value_list = list(set(ws))
                    for x in range(len(ws_value_list) - 1):
                        for y in range(x + 1, len(ws_value_list)):
                            if np.sum(self.ws == ws_value_list[x]) < np.sum(self.ws == ws_value_list[y]):
                                ws_value_list[x], ws_value_list[y] = ws_value_list[y], ws_value_list[x]
                    for j in range(len(ws_value_list)):
                        if pruned_num == 0:
                            break
                        num = np.sum(self.ws == ws_value_list[j])
                        val_index = np.where(self.ws == ws_value_list[j])
                        index_1 = list(range(0, num))
                        index_1 = random.sample(index_1, math.ceil(num*(1-np.max(self.compress_rate)-0.1)))
                        val_index = np.delete(val_index, index_1, axis=1)
                        for m in range(len(val_index[0])):
                            ones_i[val_index[0][m]][val_index[1][m]] = 0
                            pruned_num -= 1
                            if pruned_num == 0:
                                break
                        if pruned_num == 0:
                            break
                    cnt_array = np.sum(ones_i.cpu().numpy() == 0)
                    self.cpra.append(format(cnt_array / (f * c), '.2g'))
                    ones = torch.ones(f, c, w, h).to(self.device)
                    for i in range(f):
                        for j in range(c):
                            for k in range(w):
                                for l in range(h):
                                    ones[i, j, k, l] = ones_i[i, j]
                    self.mask[index] = ones
                    item.data = item.data * self.mask[index]

                    with open(resume, "wb") as f:
                        pickle.dump(self.mask, f)
                    break
        else:
            assert 1 == 0

    def grad_mask(self, cov_id):
        params = self.model.parameters()
        for index, item in enumerate(params):
            if index == cov_id * self.param_per_cov:
                break
            if index in range(0, 161, self.param_per_cov):
                item.data = item.data * self.mask[index].to(self.device)

    def pri_comp(self):
        print('compress_rate:{}'.format(self.cpra))