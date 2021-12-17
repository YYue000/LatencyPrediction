import os
import pickle
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch

_opname_to_index = {
    'none': 0,
    'skip_connect': 1,
    'nor_conv_1x1': 2,
    'nor_conv_3x3': 3,
    'avg_pool_3x3': 4,
    'input': 5,
    'output': 6,
    'global': 7
}

_opindex_to_name = { value: key for key, value in _opname_to_index.items() }

class LatencyDataset(Dataset):
    def __init__(self, meta_file_path, aug_file_path=None):
        super(LatencyDataset, self).__init__()
        self.meat_file_path = meta_file_path
        assert os.path.exists(meta_file_path), f'{meta_file_path} does not exist'
        
        data = pickle.load(open(self.meat_file_path, 'rb')) # {arch: latency}
        aug_data = {}
        
        self.aug_file_path = aug_file_path
        if aug_file_path is not None:
            assert os.path.exists(aug_file_path), f'{aug_file_path} does not exist'
            aug_data = pickle.load(open(self.aug_file_path, 'rb'))
            
        dataset = []
        for  arch_id, (arch, t) in enumerate(data.items()):
            info = {'arch_id': arch_id, 'arch': arch, 'latency': t}
            info['adjacency'], info['features'] = self._get_adjacency_matrix_and_features(arch)
            if self.aug_file_path is not None:
                info['augments'] = np.array([aug_data[arch]])
            dataset.append(info)
            
        self.dataset = dataset

    def _get_adjacency_matrix_and_features(self, arch):
        raise NotImplementedError

    def _get_features(self, arch):
        raise NotImplementedError
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]


class NASBench201Dataset(LatencyDataset):
    """
    Reimplemented version from the original paper
    """    
    def __init__(self, meta_file_path, aug_file_path=None, prune=True, keep_dims=True):
        self.arch_len = 6
        self.node_num = self.arch_len+2+1 # 6 for len(arch), 2 for input-output node, 1 for global node
        self.prune = prune
        self.keep_dims = keep_dims
        
        super(NASBench201Dataset, self).__init__(meta_file_path, aug_file_path)
    
    def _get_adjacency_matrix_and_features(self, arch):
        matrix, alive = self._get_adjacency_matrix(arch)
        features = self._get_features(arch, alive)
        return matrix, features
    
    def _get_adjacency_matrix(self, arch):    
        """
        return adjacency matrix and node status after pruning
        """
        normal_node_num = self.node_num - 1 # no global
        # '|{}~0|+|{}~0|{}~1|+|{}~0|{}~1|{}~2|'
        #        1    2    3      4    5    6
        # groupi->ej: 0.5i^2-0.5i+1+[0, ..., j-1]
        last_group = int((-1+np.sqrt(1+8*(normal_node_num-2)))/2)
        node_group = {i:[int(0.5*i**2-0.5*i+1+j) for j in range(i)] for i in range(1,last_group+1)}
        node_group[0] = [0]
        matrix = np.zeros([normal_node_num, normal_node_num])
        node_link = [[0,None]] # [group, prev_group]
        for g,nl in node_group.items():
            if g == 0: continue
            for i,n in enumerate(nl):
                node_link.append([g,i])
                for prevn in node_group[i]:
                    matrix[prevn][n] = 1        
        for n in node_group[last_group]: # connect to output node
            matrix[n][normal_node_num-1] = 1
        
        for idx, op in enumerate(arch): # remove zero and skip_connect
            n = idx+1
            if op == 0:
                matrix[n,:] = 0
                matrix[:,n] = 0
            elif op == 1: # skip-connection:
                to_del = []
                # triangular matrix; other<n<other2
                for other in range(n):
                    if matrix[other, n]:
                        for other2 in range(n+1, normal_node_num):
                            if matrix[n,other2]:
                                matrix[other, other2] = 1
                                matrix[other, n] = 0
                                to_del.append(other2)
                                
                matrix[n][to_del] = 0
        
        if self.prune:
            matrix, alive = self.prune_matrix(matrix)
        else:
            alive = np.ones(normal_node_num ,dtype=np.bool)

        matrix = self._add_global_node(matrix)
        matrix = self._add_diag(matrix)
        return matrix, alive

    def _get_features(self, arch, alive):
        label = np.array([_opname_to_index['global'], _opname_to_index['input']] + list(arch) + [_opname_to_index['output']])
        label = label - 2 # zero and skip connect are removed
        mask = np.ones(self.node_num, dtype=np.bool)
        mask[1:] = alive
        if self.keep_dims:
            features = np.zeros([self.node_num, self.arch_len])
            features[(mask, label[mask])] = 1
        else:
            n = np.sum(mask)
            features = np.zeros([n, self.arch_len])
            features[(np.arange(n), label[mask])] = 1
        return features

    def prune_matrix(self, matrix):
        """
        input: adjacency matrix
        output: pruned matrix; 
                alive status (with the same length as input matrix width)
        """
        def bfs(mtx): 
            vis = np.zeros(mtx.shape[0], dtype=np.bool)
            q = np.array([0])
            vis[0] = True
            while len(q)>0:
                v = q[0]
                nx = np.where((mtx[v]>0) & (~ vis))[0]
                vis[nx]=True
                q = np.hstack([q[1:], nx])
                
            return vis
                
        visited_fw = bfs(matrix)        
        visited_bw = bfs(np.transpose(matrix)[::-1,::-1]) [::-1]
        alive = visited_bw & visited_fw
        if self.keep_dims:
            matrix[~alive] = 0
            matrix[:, ~alive] = 0
        else:
            matrix = matrix[alive][:, alive]
        return matrix, alive

    def _add_global_node(self, matrix):
        _to_glb = np.zeros(matrix.shape[0]).reshape(-1,1)
        gmatrix = np.hstack([_to_glb, matrix])
        _from_glb = np.ones(gmatrix.shape[1]).reshape(1,-1)
        gmatrix = np.vstack([_from_glb, gmatrix])
        return gmatrix

    def _add_diag(self, matrix):
        idx = np.arange(matrix.shape[0])
        matrix[(idx,idx)] = 1
        return matrix


class NASBench201Dataset2(LatencyDataset):
    """
    Refactored version
    """    
    def __init__(self, meta_file_path, aug_file_path=None):
        self.node_num = 4+1+1 # 4 for |V| including the input node, 1 for output node, 1 for global node
        
        super(NASBench201Dataset2, self).__init__(meta_file_path, aug_file_path)
    
    def _get_adjacency_matrix(self, arch):    
        normal_node_num = self.node_num - 1 # no global
        # '|{}~0|+|{}~0|{}~1|+|{}~0|{}~1|{}~2|'
        matrix = np.zeros(normal_node_num, normal_node_num)
        
        k = 0
        for i in range(1, normal_node_num-1): # no input, output node.
            for j in range(i):
                matrix[j][i] = arch[k]
                k += 1
        assert k == 6, f'{k}'

        adjacency_matrix = (matrix>0)

        ## todo: add global node
        
        return adjacency_matrix


class NASBench201AblationDataset(NASBench201Dataset):
    def __init__(self, meta_file_path, aug_file_path=None, prune=True, keep_dims=True):
        super(NASBench201AblationDataset, self).__init__(meta_file_path, aug_file_path=aug_file_path, prune=prune, keep_dims=keep_dims)

    def _get_adjacency_matrix(self, arch):
        matrix = np.zeros([self.node_num, self.node_num])
        matrix = self._add_diag(matrix)
        return matrix

class NASBench201AblationFeatureDataset(NASBench201Dataset):
    def __init__(self, meta_file_path, aug_file_path=None, prune=True, keep_dims=True):
        super(NASBench201AblationFeatureDataset, self).__init__(meta_file_path, aug_file_path=aug_file_path, prune=prune, keep_dims=keep_dims)

    def _get_features(self, arch, alive):
        return np.random.random([self.node_num, len(_opname_to_index)-2])

def _collate_fn(batch):
    ipt = {k:[_[k] for _ in batch] for k in ['arch_id', 'arch']}
    for k in ['adjacency', 'features', 'latency']:
        ipt[k] = torch.as_tensor(np.array([_[k] for _ in batch])).double()
    k = 'augments'
    if k in batch[0].keys():
        ipt[k] = torch.as_tensor(np.array([_[k] for _ in batch])).double()
    return ipt


def build_dataloader(cfg_data, mode):
    cfg_data = cfg_data.copy()
    cfg_data.update(cfg_data[mode])

    search_space = cfg_data.get('search_space', 'nasbench201')
    if search_space == 'nasbench201':
        dataset = NASBench201Dataset(cfg_data['meta_file_path'], cfg_data.get('aug_file_path', None))
    elif search_space == 'nasbench201_ablation_matrix':
        dataset = NASBench201AblationDataset(cfg_data['meta_file_path'])
    elif search_space == 'nasbench201_ablation_feature':
        dataset = NASBench201AblationFeatureDataset(cfg_data['meta_file_path'])
    else:
        raise NotImplementedError 

    dataloader = DataLoader(dataset, batch_size=cfg_data['batch_size'], 
                            shuffle=cfg_data.get('shuffle', True),
                            collate_fn=_collate_fn)
    return dataloader
