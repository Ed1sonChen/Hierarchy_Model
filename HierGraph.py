import math
from typing import Union
from typing import Literal
import numpy as np
import random
import torch
import torchhd
from scipy.special import softmax

class hiergraph(object):
    '''Hyperdimensional graph classification and reasoning algorithm'''

    def __init__(self, 
                 num_classes: int, 
                 channel_mat: Union[np.ndarray, torch.tensor], 
                 adj_mat: Union[np.ndarray, torch.tensor],
                 signal_len: int,
                 dim: int = 1000,
                 VSA:Literal["BSC", "MAP", "HRR","FHRR"]= 'MAP',
                 embedding_type: Literal["sinusoid", "density", "projection","circular"] ='sinusoid'):
        
        self.num_classes = num_classes
        self.channel_mat = channel_mat
        self.adj_mat = adj_mat
        self.signal_len = signal_len
        self.dim = dim
        self.VSA = VSA
        self.embedding_type = embedding_type
        self.class_hv = torchhd.ensure_vsa_tensor(torch.zeros(self.num_classes, self.dim),vsa=self.VSA)
        self.num_parameter = self.channel_mat.shape[0]
        self.parameter_hv = torchhd.ensure_vsa_tensor(torch.zeros([self.num_classes,self.num_parameter,self.dim]),vsa=self.VSA)
        self.channel_hv =  torchhd.ensure_vsa_tensor(torch.zeros([self.num_classes,self.channel_mat.sum(),self.dim]),vsa=self.VSA)
        self.num_channel = np.sum(self.channel_mat)
        self.channel_ID = torchhd.random(self.num_channel,self.dim,vsa=self.VSA)
        self.parameter_ID = torchhd.random(self.num_parameter,self.dim,vsa=self.VSA)

        #different embedding types availiable
        if self.embedding_type == 'sinusoid':
            self.embed = torchhd.embeddings.Sinusoid(self.signal_len, self.dim)

        elif self.embedding_type == 'density':
            self.embed = torchhd.embeddings.Density(self.signal_len, self.dim)

        elif self.embedding_type == 'projection':
            self.embed = torchhd.embeddings.Projection(self.signal_len, self.dim)
        
        elif self.embedding_type == 'circular':
            self.embed = torchhd.embeddings.Circular(self.signal_len,self.dim)

    def __call__(self, x:torch.Tensor, encoded: bool = False):
        x_encoded_params,x_encoded_chan = x if encoded else self.encode(x)
        return self.cos_cdist(x_encoded_params.sum(1),self.class_hv).argmax(1)

    def encode(self,
            x: Union[torch.tensor,np.ndarray],
            batch_size:Union[int, None, float] = 1024):

        xx_HD = torch.from_numpy(x).float() if type(x) == np.ndarray else x
        
        embed_x = torch.empty([x.shape[0],x.shape[1],self.dim])
        #batches to allocate memory usage
        for i in range(0,x.shape[0],batch_size):
            embed_x[i:i+batch_size] = self.embed(xx_HD[i:i+batch_size])

        embed_x = torchhd.ensure_vsa_tensor(torch.sign(embed_x),vsa=self.VSA)
        # embed_x = torchhd.ensure_vsa_tensor(embed_x,vsa=self.VSA)

        #hypervectors of all channels and parameters from x input
        parameter_hvs = torchhd.ensure_vsa_tensor(torch.zeros([embed_x.shape[0],self.num_parameter,self.dim],device = xx_HD.device),vsa=self.VSA)
        channel_hvs = torchhd.ensure_vsa_tensor(torch.zeros([embed_x.shape[0],self.num_channel,self.dim],device = xx_HD.device),vsa=self.VSA)

        for i in range(embed_x.shape[0]):
            #binds all channels with respective hypervector Channel_ID
            channel_hvs[i].add_(embed_x[i].bind(self.channel_ID))

            for j in range(self.num_parameter):
            #indexes the channels from each parameter in sample and bundle together
                pos = sum(self.channel_mat[:j+1])
                #binds the parameter HV representation with the parameter ID HV representation
                parameter_hvs[i,j].add_(torch.sum(channel_hvs[i,pos-self.channel_mat[j]:pos],dim=0).bind(self.parameter_ID[j])) #bundles all respective channels

        return parameter_hvs, channel_hvs
    
    def fit(self,
            x_train:torch.tensor,
            y_train:torch.tensor,
            encoded: bool = False,
            T:float=0.2,
            iter: int = 1,
            lr:float = 0.001,
            alpha: float=0.5,
            epochs: int= 100,
            batch_size:Union[int, None, float] = 1024):
        
        y_train = torch.from_numpy(y_train).long() if type(y_train) == np.ndarray else y_train

        # print('Encoding Started')
        # print('...')
        h_parameters, h_channels = x_train if encoded else self.encode(x_train)
        # print('Encoding Complete!')
        # print('====================')
        # print('Graph Refinement Started')
        # print('...')
        graph_Parameters,graph_Channels = self.refine_graphs(y_train,h_parameters,h_channels,alpha,T,iter)
        # print('Graph Refinement Completed')
        # print('====================')
        # print('Iterative Updated Started')
        # print('...')
        self.iterative_update(h_parameters,h_channels,graph_Channels,graph_Parameters,y_train,lr,epochs,batch_size)
        # print('Iterative Updated Complete')
        return self

    def refine_graphs(self, y, parameter_samples, channel_samples, alpha, T,iter):
    
        num_lbls = y.unique().size(0)
        lbls = y.unique()
        
        mem_nodes = torch.zeros([num_lbls,parameter_samples.shape[1],parameter_samples.shape[2]], device=parameter_samples.device)
        nodes = torch.zeros([num_lbls,self.num_channel,channel_samples.shape[2]], device=parameter_samples.device)

        for lbl in range(num_lbls):
            mem_nodes[lbl].add_(parameter_samples[y == lbl].sum(0))
            nodes[lbl].add_(channel_samples[y == lbl].sum(0))

        mem_node_return = mem_nodes.clone()

        for i in range(iter):
            for lbl in range(num_lbls):
                cur_graph = mem_node_return[lbl]
                other_graph = nodes[lbls[lbls !=lbl]]
                for j in range(self.adj_mat.shape[0]):
                    v_nodes_other = other_graph[:,self.adj_mat[j]]
                    compare = torchhd.cosine_similarity(cur_graph[j],v_nodes_other) > T
                    mem_node_return[lbl,j] += alpha*nodes[lbl,self.adj_mat[j]].sum(0)
                    mem_node_return[lbl,j] -= alpha*v_nodes_other[compare].sum(0)

        return mem_node_return, nodes 


    def iterative_update(self,
                         h_parameters_samples:torch.tensor,
                         h_channel_samples:torch.tensor,
                         refined_channels_nodes: torch.tensor,
                         refined_parameter_mems:torch.tensor,
                         y,
                         lr,
                         epochs,
                         batch_size,
                         ):
        
        h= h_parameters_samples.sum(1)

        self.class_hv.add_(refined_parameter_mems.sum(1))
        self.parameter_hv.add_(refined_parameter_mems)
        self.channel_hv.add_(refined_channels_nodes)
        n = h.size(0)
        
        for epoch in range(epochs):
                for i in range(0, n, batch_size):
                    h_ = h[i:i+batch_size]
                    h_p = h_parameters_samples[i:i+batch_size]
                    h_c = h_channel_samples[i:i+batch_size]
                    y_ = y[i:i+batch_size]
                    scores = self.cos_cdist(h_,self.class_hv)
                    y_pred = scores.argmax(1)
                    wrong = y_ != y_pred

                    # computes alphas to update model
                    # alpha1 = 1 - delta[lbl] -- the true label coefs
                    # alpha2 = delta[max] - 1 -- the prediction coefs
                    aranged = torch.arange(h_.size(0), device=h_.device)
                    alpha1 = (1.0 - scores[aranged,y_]).unsqueeze_(1)
                    alpha2 = (scores[aranged,y_pred] - 1.0).unsqueeze_(1)

                    for lbl in y_.unique():
                        m1 = wrong & (y_ == lbl) # mask of missed true lbl
                        m2 = wrong & (y_pred == lbl) # mask of wrong preds
                        self.class_hv[lbl].add_(lr*(alpha1[m1]*h_[m1]).sum(0))
                        self.class_hv[lbl].add_(lr*(alpha2[m2]*h_[m2]).sum(0))
                        self.parameter_hv[lbl].add_(lr*(alpha1[m1].unsqueeze(2)*h_p[m1]).sum(0))
                        self.parameter_hv[lbl].add_(lr*(alpha2[m2].unsqueeze(2)*h_p[m2]).sum(0))
                        self.channel_hv[lbl].add_(lr*(alpha1[m1].unsqueeze(2)*h_c[m1]).sum(0))
                        self.channel_hv[lbl].add_(lr*(alpha2[m2].unsqueeze(2)*h_c[m2]).sum(0))

    def cos_cdist(self,x1 : torch.Tensor, x2 : torch.Tensor, eps : float = 1e-8):
        '''
        From OnlineHD: https://gitlab.com/biaslab/onlinehd
        Computes pairwise cosine similarity between samples in `x1` and `x2`,
        forcing each point l2-norm to be at least `eps`. This similarity between
        `(n?, f?)` samples described in :math:`x1` and the `(m?, f?)` samples
        described in :math:`x2` with scalar :math:`\varepsilon > 0` is the
        `(n?, m?)` matrix :math:`\delta` given by:

        .. math:: \delta_{ij} = \frac{x1_i \cdot x2_j}{\max\{\|x1_i\|, \varepsilon\} \max\{\|x2_j\|, \varepsilon\}}

        Args:
            x1 (:class:`torch.Tensor`): The `(n?, f?)` sized matrix of datapoints
                to score with `x2`.

            x2 (:class:`torch.Tensor`): The `(m?, f?)` sized matrix of datapoints
                to score with `x1`.

            eps (float, > 0): Scalar to prevent zero-norm vectors.

        Returns:
            :class:`torch.Tensor`: The `(n?, m?)` sized tensor `dist` where
            `dist[i,j] = cos(x1[i], x2[j])` given by the equation above.

        '''
        eps = torch.tensor(eps, device=x1.device)
        norms1 = x1.norm(dim=1).unsqueeze_(1).max(eps)
        norms2 = x2.norm(dim=1).unsqueeze_(0).max(eps)
        cdist = x1 @ x2.T
        cdist.div_(norms1).div_(norms2)
        
        return cdist
    

def seperability(sims:torch.tensor):
    diff_sim = (sims - np.expand_dims(sims.mean(-1),axis=-1))
    diff_class_sum = np.expand_dims(np.abs(diff_sim).sum(-1),axis=-1)
    percentage_class = diff_sim/diff_class_sum
    sep_score = softmax(percentage_class,axis=-1)
    return sep_score


def rel_paraimportance(data,label_val):
    parameter_seperation = data[label_val][:,label_val]
    para_sum = parameter_seperation.sum()
    para_per = (parameter_seperation/para_sum*100).reshape(1,-1)
    # print(para_per)
    parameter_importance = softmax(para_per)
    return parameter_importance


def relative_importance(data):
    parameter_seperation = data
    para_sum = parameter_seperation.sum(1)
    para_per = (parameter_seperation/np.expand_dims(para_sum,1))*100
    # print(para_per.shape)
    parameter_importance = softmax(para_per,1)
    return np.diagonal(parameter_importance,axis1=0,axis2=2)


def shuffle_and_format(xx, yy):
    xx = xx.squeeze()

    labels = np.unique(yy)
    num_labels = len(labels)
    num_samples = min([np.sum(yy == label) for label in labels])

    selected_indices = []
    for label in labels:
        label_indices = np.where(yy == label)[0]
        selected_indices.extend(random.sample(label_indices.tolist(), num_samples))

    random.shuffle(selected_indices)

    xx = xx[selected_indices]
    yy = yy[selected_indices]

    yy = yy.squeeze().astype(np.int)

    return xx, yy
