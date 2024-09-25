import sys
sys.path.append('/nfs_beijing_ai/jinxian/rama-scoring1.3.0')
import argparse
import os
import random
from scipy.spatial.transform import Rotation as R

import numpy as np
import pandas as pd
# from dataset.equiformerv2_dataset import data_list_collater
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Batch
import torch.optim as optim

from np.protein import from_pdb_string

from dataset.feature.featurizer import process
from utils.data_encoding import encode_structure, encode_features, extract_topology
from dataset.screener import UpperboundScreener, ProbabilityScreener1d
from utils.logger import Logger
from utils.constants.atom_constants import *
from utils.constants.residue_constants import *
import np.residue_constants as rc

logger = Logger.logger
"""Protein data type."""
import dataclasses
import io
import re
import collections
from typing import Any, Mapping, Optional, Sequence
import numpy as np
from np import residue_constants
from Bio.PDB import PDBParser
from anarci import anarci
from utils.opt_utils import superimpose_single
from utils.logger import Logger
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from utils import dist_utils

logger = Logger.logger

FeatureDict = Mapping[str, np.ndarray]
ModelOutput = Mapping[str, Any]
PICO_TO_ANGSTROM = 0.01
PDB_CHAIN_IDS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
PDB_MAX_CHAINS = len(PDB_CHAIN_IDS)

from model.equiformer_v2_model import EquiformerV2

from torch_geometric.data import Dataset
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
torch.autograd.set_detect_anomaly(True)
loss_mse = torch.nn.MSELoss()
import wandb
import logging

os.environ["WANDB_MODE"] = "run"

# start a new wandb run to track this script
# wandb.init(
#     # set the wandb project where this run will be logged
#     project="jinxianwang",

#     # track hyperparameters and run metadata
#     config={
#     "architecture": "equiformer2",
#     "dataset": "AF2",
#     "epochs": 10,
#     }
# )


def set_random_seed(seed):
    """Set random seed for reproducability."""
    if seed is not None:
        assert seed > 0
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True  # False
        torch.backends.cuda.matmul.allow_tf32 = (
            False  # if set it to True will be much faster but not accurate
        )
        # if (
        #     not deepspeed_version_ge_0_10 and deepspeed.checkpointing.is_configured()
        # ):  # This version is a only a rough number
        #     mpu.model_parallel_cuda_manual_seed(seed)

def load_data(csv, chain_type=None, filter=None):
    df = pd.read_csv(csv)
    logger.info(f"rows: {len(df)}")
    logger.info(f"filter={filter}")
    if filter == "ca_only":
        for pdb in ab_filter.ca_only_list:
            df = df.drop(df[df["pdb"] == pdb].index)
    elif filter == "backbone_only":
        for pdb in ab_filter.backbone_only_list:
            df = df.drop(df[df["pdb"] == pdb].index)
    elif filter == "long":
        for pdb in ab_filter.long_list:
            df = df.drop(df[df["pdb"] == pdb].index)
    elif filter == "missing_res":
        for pdb in ab_filter.missing_res_list:
            df = df.drop(df[df["pdb"] == pdb].index)
    elif filter == "missing_atom":
        for pdb in ab_filter.missing_atom_list:
            df = df.drop(df[df["pdb"] == pdb].index)
    elif filter == "corner_case":
        for pdb in ab_filter.corner_case_list:
            df = df.drop(df[df["pdb"] == pdb].index) 
    elif filter == "region_le2":
        for pdb in ab_filter.region_le2_list:
            df = df.drop(df[df["pdb"] == pdb].index)
    elif filter == "many_x":
        for pdb in ab_filter.many_x_res_list:
            df = df.drop(df[df["pdb"] == pdb].index)
    elif filter == "all":
        for pdb in ab_filter.ca_only_list:
            df = df.drop(df[df["pdb"] == pdb].index)
        for case in ab_filter.bad_ab_list:
            pdb, chain_id = case.split("-")
            df = df.drop(df[(df["pdb"] == pdb) & (df["chain"] == chain_id)].index)
    else:
        pass
    logger.info(f"rows after drop bad pdbs: {len(df)}")
    if chain_type == "h":
        df = df[df["chain_type"] == chain_type]
        logger.info(f"rows for vh : {len(df)}")
    elif chain_type == "l":
        df = df[(df["chain_type"] == "l") | (df["chain_type"] == "k")]
        logger.info(f"rows for vl : {len(df)}")
    else:
        pass

    return df
    
def data_list_collater(data_list):
    """Run data_list_collater method."""
    # code.
    data_list = [data for data in data_list if data is not None]
    if len(data_list) == 0:
        return None
    if len(data_list[0]) == 2:
        data_list1 = [data[0] for data in data_list]
        data_list2 = [data[1] for data in data_list]
        batch1 = Batch.from_data_list(data_list1)
        batch2 = Batch.from_data_list(data_list2)
        return [batch1, batch2]
    elif len(data_list[0]) == 1:
        data_list1 = [data[0] for data in data_list]
        batch1 = Batch.from_data_list(data_list1)
        return [batch1]
    else:
        raise RuntimeError(f"Unsupported data list collater! data_list: {data_list}")

            
def prepare_features(gt_pdb_filepath, af_pdb_filepath, chain, full_seq_AMR):
        """Run prepare_features method.
        gt_pdb_filepath: a file path of gt_pdb_filepath
        af_pdb_filepath: a file path of af_pdb_filepath
        chain: the chain of pdb

        output: a torch_geometric data of the pdb
        """
        
        # code.
        with open(af_pdb_filepath, 'r') as f:
            pdb_str = f.read()
        
        with open(gt_pdb_filepath, 'r') as f:
            gt_pdb_str = f.read()
        po = from_pdb_string(pdb_str=pdb_str, gt_pdb_str=gt_pdb_str, chain_id=chain, ca_only=False, full_seq_AMR=full_seq_AMR)
        if po == False:
            return False

        atom_mask = po.atom_mask.reshape(-1).astype(bool)

        po2struc = {
            'atom_positions': 'xyz',
            'atom_names': 'name',
            # 'atom2cgids': 'atom2cgid',
            'elements': 'element',
            'residue_names': 'resname',
            'resids': 'resid',
            'hetfields': 'het_flag',
            'icodes': 'icode'
        }
        structure = {}
        for k in [
            'atom_positions',
            'atom_names',
            'residue_names',
            # 'atom2cgids',
            'elements',
            'hetfields',
            'resids',
            'icodes',
        ]:
            item = eval(f"po.{k}")
            if k == 'atom2cgids':
                item = item.reshape(-1, item.shape[-1])
            elif k == 'atom_positions':
                item = item.reshape(-1, item.shape[-1])
            else:
                item = item.reshape(-1, )
            item = item[atom_mask] # mask the 0
            structure[po2struc[k]] = item


        X = torch.from_numpy(structure['xyz'].astype(np.float32))

        # atom2cgids = torch.from_numpy(structure['atom2cgid'].astype(np.float32))

        atom_names = structure['name'].tolist()
        atom_numbers = torch.Tensor([atom2id[atname] for atname in atom_names])     # atom number of atoms in residues. max number is 37.

        elements = structure['element'].tolist()
        atomic_numbers = [element2atomic_numbers[element]for element in elements]
        atomic_numbers = torch.Tensor(atomic_numbers)

        res_names = structure['resname'].tolist()
        res1 = [restype_3to1[resname] for resname in res_names]
        resid = [res_type12id[res1[i]] for i in range(len(res1))]
        resid = torch.Tensor(resid)

        n_nodes = X.size(0)

        # extend element 
        gt_atom_mask = torch.from_numpy(eval(f"po.{'gt_atom_mask'}").astype(np.float32))
        atom_mask = torch.from_numpy(atom_mask)

        

        atom2cgids = torch.from_numpy(eval(f"po.{'atom2cgids'}").astype(np.float32))
        gt_atom_positions = torch.from_numpy(eval(f"po.{'gt_atom_positions'}").astype(np.float32))
        all_atom_positions = torch.from_numpy(eval(f"po.{'atom_positions'}").astype(np.float32))
        gt_res_mask = torch.from_numpy(eval(f"po.{'gt_res_mask'}"))
        rmsd = torch.from_numpy(eval(f"po.{'rmsd'}").astype(np.float32))

        

        if all_atom_positions.shape[0] != gt_res_mask.shape[0]:
            return False

        trans_target = torch.from_numpy(eval(f"po.{'trans_target'}").astype(np.float32))
        rot_m4_target = torch.from_numpy(eval(f"po.{'rot_m4_target'}").astype(np.float32))

        pdb_name = af_pdb_filepath.split('/')[-1]

        data = Data(
            pos=X, 
            atom_numbers=atom_numbers, 
            atomic_numbers=atomic_numbers, 
            atom_mask=atom_mask,
            resid=resid, 
            n_nodes=n_nodes,
            gt_atom_positions=gt_atom_positions,
            gt_atom_mask=gt_atom_mask,
            gt_res_mask=gt_res_mask,
            all_atom_positions=all_atom_positions,
            atom2cgids= atom2cgids,
            pdb_name = pdb_name,
            trans_target= trans_target,
            rot_m4_target = rot_m4_target,
            rmsd = rmsd)
        return data

def rmsd_comp(init_model, gt_model, gt_res_mask, gt_atom_mask, rot, tran):
    """
    iter add the r and t to each atom temp. update through add R and T to all atom one step. 
    Args:
        init_model:
            [batch*residue, 37, 3], tensor
        gt_model:
            [batch*residue, 37, 3], tensor
        rot:
            [batch*residue, 37, 3, 3], rotation tensor
        tran:
            [batch*residue, 37, 3], translation tensor
        gt_res_mask:
            [batch*residue], tensor
        gt_atom_mask:
            [batch*gt_residue, 37], tensor
    Return:
        rmsd
            
    """
    # 采用einsum
    # rot = torch.rand(2,5,3,3)
    # coords = torch.rand(2,5,3)
    # tran = torch.rand(2,5, 3)

    refine_model = torch.einsum("brij, brj -> bri", rot, init_model) + tran
    refine_model_after_res_mask = refine_model[gt_res_mask==1]
    refine_model_after_res_mask = refine_model_after_res_mask.view(-1,refine_model.shape[-1])

    trans_gt_model = gt_model.view(-1, gt_model.shape[-1])
    # trans_refine_model = refine_model.view(-1,refine_model.shape[-1])
    trans_gt_mask = gt_atom_mask.view(-1)
    rot_, tran_, rmsd, superimpose = superimpose_single(
            trans_gt_model, refine_model_after_res_mask, differentiable=True, mask=trans_gt_mask # mask中为1则是存在，不用mask
        )

    trans_init_model = init_model[gt_res_mask==1].view(-1,init_model.shape[-1])
    _, _, rmsd_start, _ = superimpose_single(
            trans_gt_model, trans_init_model,differentiable=True, mask=trans_gt_mask
        )

    return rmsd, rmsd_start


import torch.nn.functional as F
from scipy.linalg import logm  

# 示例旋转矩阵（真实值和预测值），这两个是3x3的正交矩阵
# R_true = torch.tensor([[0.866, -0.500, 0.0], 
#                        [0.500, 0.866, 0.0], 
#                        [0.0, 0.0, 1.0]], requires_grad=True, dtype=torch.float32)

# R_pred = torch.tensor([[0.707, -0.707, 0.0], 
#                        [0.707, 0.707, 0.0], 
#                        [0.0, 0.0, 1.0]], requires_grad=True, dtype=torch.float32)


# Quaternion
def quaternion_loss(q_true, q_pred):
    # 计算四元数的内积差异
    dot_products = torch.abs(torch.sum(q_true * q_pred, dim=-1))
    # Loss = 1 - 四元数内积
    loss = 1.0 - dot_products
    return loss.sum()  # 损失 sum() or mean()

# t_true = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
# t_pred = torch.tensor([1.5, 2.5, 3.5], requires_grad=True)
import torch.nn as nn
# 使用 L2 Loss 计算平移向量之间的差异
def l2_loss(t_true, t_pred):
    return torch.norm(t_true - t_pred, p=2)


#  使用 Huber Loss 计算平移向量差异
def huber_loss(t_true, t_pred, delta=1.0):
    loss = nn.HuberLoss(delta=delta)
    return loss(t_true, t_pred)

def loss_function_R_T(gt_m4_rot, gt_tran, gt_res_mask, rot_m4, tran):
    """
    iter add the r and t to each atom temp. update through add R and T to all atom one step. 
    Args:
        gt_m4_rot:
            [gt_residue, 37, 4], tensor
        gt_tran:
            [gt_residue, 37, 3], tensor
        rot_m4:
            [af_residue, 37, 4], rotation tensor
        tran:
            [af_residue, 37, 3], translation tensor
        gt_res_mask:
            [gt_residue], tensor
      
        
    Return:
        rmsd
            
    """
    # 采用einsum
    # rot = torch.rand(2,5,3,3)
    # coords = torch.rand(2,5,3)
    # tran = torch.rand(2,5, 3)
    refine_rot_m4 = rot_m4[gt_res_mask==1]
    refine_tran = tran[gt_res_mask==1]
    
    # rot_quaternion_loss = quaternion_loss(gt_m4_rot, refine_rot_m4)
    rot_mse_loss = loss_mse(gt_m4_rot, refine_rot_m4)

    tran_mse_loss = loss_mse(refine_tran,gt_tran)

    return rot_mse_loss + tran_mse_loss


def loss_function(init_model, gt_model, gt_res_mask, gt_atom_mask, rot, tran):
    """
    iter add the r and t to each atom temp. update through add R and T to all atom one step. 
    Args:
        init_model:
            [batch*residue, 37, 3], tensor
        gt_model:
            [batch*residue, 37, 3], tensor
        rot:
            [batch*residue, 37, 3, 3], rotation tensor
        tran:
            [batch*residue, 37, 3], translation tensor
        gt_res_mask:
            [batch*residue], tensor
        gt_atom_mask:
            [batch*gt_residue, 37], tensor
        
    Return:
        rmsd
            
    """
    # 采用einsum
    # rot = torch.rand(2,5,3,3)
    # coords = torch.rand(2,5,3)
    # tran = torch.rand(2,5, 3)
    refine_model = torch.einsum("brij, brj -> bri", rot, init_model) + tran
    refine_model_after_res_mask = refine_model[gt_res_mask==1]
    refine_model_after_atom_mask = refine_model_after_res_mask.view(-1,refine_model.shape[-1])

    trans_gt_model = gt_model.view(-1, gt_model.shape[-1])
    # trans_refine_model = refine_model.view(-1,refine_model.shape[-1])
    trans_gt_mask = gt_atom_mask.view(-1)
    rot_, tran_, rmsd, superimpose = superimpose_single(
            trans_gt_model, refine_model_after_atom_mask,differentiable=True, mask=trans_gt_mask # mask中为1则是存在，不用mask
        )


    return rmsd

def computer_rmsd4test(init_model, gt_model, gt_res_mask, gt_atom_mask, rot_m4, tran, atom2cgids):
    """
    iter add the r and t to each atom temp. update through add R and T to all atom one step. 
    Args:
        init_model:
            [batch*residue, 37, 3], tensor
        gt_model:
            [batch*residue, 37, 3], tensor
        rot:
            [batch*residue, 37, 3, 3], rotation tensor
        tran:
            [batch*residue, 37, 3], translation tensor
        gt_res_mask:
            [batch*residue], tensor
        gt_atom_mask:
            [batch*gt_residue, 37], tensor
        
    Return:
        rmsd
            
    """
    # 采用einsum
    # rot = torch.rand(2,5,3,3)
    # coords = torch.rand(2,5,3)
    # tran = torch.rand(2,5, 3)

    # first get the R [3,3] from m4 [4]
    # 找到所有行都为 0 的行
    zero_rows = (rot_m4 == 0).all(dim=1)

    # 将这些行的第一个元素设置为 1
    rot_m4[zero_rows, 0] = 1.0

    cg_rot = torch.from_numpy(R.from_quat(rot_m4.cpu().numpy()).as_matrix()).to(tran.device)

    each_res_atom_times = torch.sum(atom2cgids, dim=-1)
    # 转为形状 [batch_size, residue，atom=37, 1] 以进行广播
    t_last = each_res_atom_times.unsqueeze(-1)
    # 防止除以 0，将 t_last 中的 0 替换为一个很小的值（例如1e-6），避免计算错误
    t_last_safe = torch.where(t_last == 0, torch.tensor(1e-6).to(tran.device), t_last)
    # x 的最后一个维度除以 t 的最后一个值
    mean_cg_trans_atom = atom2cgids / t_last_safe
    
    # 然后乘以输出，获得每个原子的平均值,R and T
    cg_rot = cg_rot.view(-1,4,3,3).to(torch.float32)
    atom_rot = torch.matmul(mean_cg_trans_atom, cg_rot.view(-1,4,9)).view(mean_cg_trans_atom.shape[0],mean_cg_trans_atom.shape[1],3,3)
    atom_tran = torch.matmul(mean_cg_trans_atom, tran.view(-1,4,3))



    refine_model = torch.einsum("brij, brj -> bri", atom_rot, init_model) + atom_tran
    refine_model_after_res_mask = refine_model[gt_res_mask==1]
    refine_model_after_atom_mask = refine_model_after_res_mask.view(-1,refine_model.shape[-1])

    trans_gt_model = gt_model.view(-1, gt_model.shape[-1])
    # trans_refine_model = refine_model.view(-1,refine_model.shape[-1])
    trans_gt_mask = gt_atom_mask.view(-1)
    rot_, tran_, rmsd, superimpose = superimpose_single(
            trans_gt_model, refine_model_after_atom_mask, differentiable=True,mask=trans_gt_mask # mask中为1则是存在，不用mask
        )


    return rmsd

def get_dataset_filepath(csv, sample_num = -1):
    if sample_num == -1:
        df = load_data(csv)
    else:
        df = load_data(csv).head(sample_num)
  
    # pdb_ids = [list(df["pdb"]) for df in df_list]
    chain_ids = list(df["chain"])
    pdb_files = list(df["pdb_fpath"])
    pdb_files_gt = list(df["pdb_fpath_gt"]) 
    fv_seq_amr = list(df["full_seq_AMR"]) 
    
    data_list = []
    total_data = 0
    for af_pdb, gt_pdb, chain, gt_seq in zip(pdb_files,pdb_files_gt,chain_ids,fv_seq_amr):
        if pd.isna(chain):
            chain = None
        if os.path.exists(af_pdb) and os.path.exists(gt_pdb):
            tem =prepare_features(af_pdb_filepath=af_pdb, gt_pdb_filepath=gt_pdb, chain=chain, full_seq_AMR=gt_seq) 
            if tem:
                total_data += 1
                print('total data:', total_data)
                data_list.append(tem)
            else:
                continue
        else:
            continue
        
    # af_data = prepare_features(af_pdb_filepath=af_pdb, gt_pdb_filepath=gt_pdb, chain=chain, full_seq_AMR=gt_seq) # chain id needed
    print('get data:', len(data_list))
    return data_list

class SimpleDataset(Dataset):
    def __init__(self, csv):
        self.csv = csv
        self.df = self.load_data(self.csv)
        self.chain_ids = list(self.df["chain"])
        self.pdb_files = list(self.df["pdb_fpath"])
        self.pdb_files_gt = list(self.df["pdb_fpath_gt"]) 
        self.fv_seq_amr = list(self.df["full_seq_AMR"]) 


    def __len__(self):
        return len(self.pdb_files_gt)
    def __getitem__(self, idx):
        af_pdb, gt_pdb, chain, gt_seq = self.pdb_files[idx],self.pdb_files_gt[idx],self.chain_ids[idx],self.fv_seq_amr[idx]
        if pd.isna(chain):
            chain = None
        return prepare_features(af_pdb_filepath=af_pdb, gt_pdb_filepath=gt_pdb, chain=chain, full_seq_AMR=gt_seq) 

    
    def load_data(csv, chain_type=None, filter=None):
        df = pd.read_csv(csv)
        logger.info(f"rows: {len(df)}")
        logger.info(f"filter={filter}")
        if filter == "ca_only":
            for pdb in ab_filter.ca_only_list:
                df = df.drop(df[df["pdb"] == pdb].index)
        elif filter == "backbone_only":
            for pdb in ab_filter.backbone_only_list:
                df = df.drop(df[df["pdb"] == pdb].index)
        elif filter == "long":
            for pdb in ab_filter.long_list:
                df = df.drop(df[df["pdb"] == pdb].index)
        elif filter == "missing_res":
            for pdb in ab_filter.missing_res_list:
                df = df.drop(df[df["pdb"] == pdb].index)
        elif filter == "missing_atom":
            for pdb in ab_filter.missing_atom_list:
                df = df.drop(df[df["pdb"] == pdb].index)
        elif filter == "corner_case":
            for pdb in ab_filter.corner_case_list:
                df = df.drop(df[df["pdb"] == pdb].index) 
        elif filter == "region_le2":
            for pdb in ab_filter.region_le2_list:
                df = df.drop(df[df["pdb"] == pdb].index)
        elif filter == "many_x":
            for pdb in ab_filter.many_x_res_list:
                df = df.drop(df[df["pdb"] == pdb].index)
        elif filter == "all":
            for pdb in ab_filter.ca_only_list:
                df = df.drop(df[df["pdb"] == pdb].index)
            for case in ab_filter.bad_ab_list:
                pdb, chain_id = case.split("-")
                df = df.drop(df[(df["pdb"] == pdb) & (df["chain"] == chain_id)].index)
        else:
            pass
        logger.info(f"rows after drop bad pdbs: {len(df)}")
        if chain_type == "h":
            df = df[df["chain_type"] == chain_type]
            logger.info(f"rows for vh : {len(df)}")
        elif chain_type == "l":
            df = df[(df["chain_type"] == "l") | (df["chain_type"] == "k")]
            logger.info(f"rows for vl : {len(df)}")
        else:
            pass

        return df


def main_worker(rank,nprocs,args):
    ## DDP：DDP backend初始化
    local_rank = rank
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl',init_method=args.dist_url,
                        world_size=args.nprocs,
                        rank=rank)

    # af_pdb = '/nfs_beijing_ai/ttt/model_eval/general_pro/pred_af3/casp14/fold_t1057/fold_t1057_model_0.pdb'
    # gt_pdb = '/pfs_beijing/share/data/ground_truth_pdb/casp14/T1057.pdb'
    # af_seq = 'MLKNVLRYPGGKSKALKYILPNLPVGFREYREPMVGGGAVALAVKQLYTNVKIKINDLNYDLICFWKQLRDNPVQLIEEVSKIKENYKDGRKLYEFLTSQNGGGEFERAVRFYILNRITFSGTVDSGGYSQQSFENRFTWSAINKLKQAAEIIKDFEISHGDYEKLLFEPGNEVFIFLDPPYYSTTESRLYGKNGDLHLSFDHERFAFNIKKCPHLWMITYDDSPEVRKLFKFANIYEWELQYGMNNYKQSKAEKGKELFITNYKLEELRQKEKYALGL'
    # gt_seq = 'MLKNVLRYPGGKSKALKYILPNLPVGFREYREPMVGGGAVALAVKQLYTNVKIKINDLNYDLICFWKQLRDNPVQLIEEVSKIKENYKDGRKLYEFLTSQNGGGEFERAVRFYILNRITFSG[T][V][D][S][G]GYSQQSFENRFTWSAINKLKQAAEIIKDFEISHGDYEKLLFEPGNEVFIFLDPPYYS[T][T][E][S][R][L][Y][G][K][N][G][D][L][H][L]SFDHERFAFNIKKCPHLWMITYDDSPEVRKLFKFANIYEWEL[Q][Y][G][M][N][N][Y][K][Q][S][K][A][E]KGKELFITNYKLEELRQKEKYALGL'
    # af_data = prepare_features(af_pdb_filepath=af_pdb, gt_pdb_filepath=gt_pdb, chain=None, full_seq_AMR=gt_seq) # chain id needed
    # # gt_data = prepare_features(pdb=gt_pdb, chain=None)
    # if af_data == False:
    #     print('Residue length not match!')

    # train_data = [af_data, af_data]
    # trainfp = '/nfs_beijing_ai/jinxian/rama-scoring/6944580/NB_train_dataset.csv'
    trainfp =  '/nfs_beijing_ai/jinxian/rama-scoring1.3.0/dataset/datasets/NB_train_dataset.csv'
    testfp = '/nfs_beijing_ai/jinxian/rama-scoring/6944580/casp14_test_file.csv'
    train_dataset = get_dataset_filepath(trainfp, 10000)
    test_dataset = train_dataset[:10]
    # test_dataset = get_dataset_filepath(testfp)

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    train_sampler = None
    test_sampler = None
    if dist_utils.get_world_size() > 1:
        train_sampler = DistributedSampler(
            train_dataset,
            rank=dist_utils.get_rank(),
            num_replicas=dist_utils.get_world_size(),
            shuffle=False,
        )

    
        test_sampler = DistributedSampler(
            test_dataset,
            rank=dist_utils.get_rank(),
            num_replicas=dist_utils.get_world_size(),
            shuffle=False,
        )

    test_dataloader = DataLoader(
            test_dataset,
            batch_size=1,
            # num_workers=self.config_data['num_workers'],
            shuffle= False,
            collate_fn=data_list_collater,
            pin_memory=True,
            # prefetch_factor=self.config_data['prefetch_factor'],
            sampler=test_sampler,
        )
    
    model = EquiformerV2().to(local_rank)

    
    

    # # 注册钩子
    # for name, param in model.named_parameters():
    #     param.register_hook(lambda grad, name=name: print_grad(grad, name))

    loss_file = "/nfs_beijing_ai/jinxian/rama-scoring1.3.0/rmsd_values.txt"
    with open(loss_file, "w") as file:
        file.write("Epoch, refine rmsd, start rmsd \n")  

    
    only_test = False

    if only_test:
        ckpt_path = "/nfs_beijing_ai/jinxian/rama-scoring/model/ckpt/temp.ckpt"
        if dist.get_rank() == 0 and ckpt_path is not None:
            model.load_state_dict(torch.load(ckpt_path))
        model.eval()
        test_rmsd = 0.0
        test_rmsd_start = 0.0
        with torch.no_grad():
            print('to test!')
            for data in test_dataloader:
                cg_embedding = model(data.to(local_rank))  # 输出维度为 [batch_size, residues, atoms, output_size]
                cg_embedding_all_R_m4, cg_embedding_all_T = torch.split(cg_embedding, [4, 3], dim=-1)
                
                # 计算rmsd
                rmsd = computer_rmsd4test(data.all_atom_positions.to(local_rank), data.gt_atom_positions.to(local_rank), data.gt_res_mask.to(local_rank),  data.gt_atom_mask.to(local_rank), cg_embedding_all_R_m4.to(local_rank), cg_embedding_all_T.to(local_rank), data.atom2cgids.to(local_rank))
                # computer_rmsd4test(init_model, gt_model, gt_res_mask, gt_atom_mask, rot_m4, tran, data.atom2cgids)
                rmsd_start = torch.sum(data.rmsd.to(local_rank))

                test_rmsd += rmsd.item()
                test_rmsd_start += rmsd_start.item()
                
                print('start rmsd:', rmsd_start.item())
                print('refine rmsd:', rmsd.item())
                print('---------')
                pdb_name = data.pdb_name
                
                with open(loss_file, "a") as file:
                    file.write(f"{epoch}, {pdb_name}, {rmsd:.4f}, {rmsd_start:.4f}\n")
        # log metrics to wandb
                
        test_rmsd = test_rmsd / (len(test_dataloader.dataset))
        
        test_rmsd_start = test_rmsd_start / (len(test_dataloader.dataset))
        print(' rmsd: ', test_rmsd)
        print(' test_rmsd_start: ', test_rmsd_start)


    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    # model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    optimizer = optim.Adam(model.parameters(), lr=0.001)


    train_dataloader = DataLoader(
            train_dataset,
            batch_size=8,
            # num_workers=self.config_data['num_workers'],
            shuffle=True if train_sampler is None else False,
            collate_fn=data_list_collater,
            drop_last=True,
            pin_memory=True,
            # prefetch_factor=self.config_data['prefetch_factor'],
            sampler=train_sampler,
        )

    comput_start_tag = True
    # start_rmsd_all = 0.0
    epochs = 200
    min_rmsd = 10000000
    for epoch in range(epochs):
        model.train()
        print('Model train')
        
        train_dataloader.sampler.set_epoch(epoch)
        for data in train_dataloader:
            # print(data.ptr)
            optimizer.zero_grad()
            cg_embedding = model(data.to(local_rank))
            # all_R, all_T = torch.split(outputs, [9, 3], dim=-1)
            # re_all_R = all_R.reshape(*all_R.shape[:-1], 3, 3) 
            # #[all residue, 37, 3, 3]
            
            cg_embedding_all_R_m4, cg_embedding_all_T = torch.split(cg_embedding, [4, 3], dim=-1)
            
            try:
                loss = loss_function_R_T(data.rot_m4_target.to(local_rank), data.trans_target.to(local_rank), data.gt_res_mask.to(local_rank), cg_embedding_all_R_m4, cg_embedding_all_T)
                # loss, rmsd_start = rmsd_comp(data.all_atom_positions.to(local_rank), data.gt_atom_positions.to(local_rank), data.gt_res_mask.to(local_rank),  data.gt_atom_mask.to(local_rank), re_all_R, all_T)
            
            except ValueError as e: 
                print("遇到了一个问题：")
                print('the bad pdb name is :', data.pdb_name)

            # start_rmsd_all += rmsd_start.item()
            # else:
            #     loss = loss_function(data.all_atom_positions.to(local_rank), data.gt_atom_positions.to(local_rank), data.gt_res_mask.to(local_rank),  data.gt_atom_mask.to(local_rank), re_all_R, all_T)
            print('batch loss: ', loss)
            # train_loss += loss.item()
            
            loss.backward()
            optimizer.step()
        
        # comput_start_tag = False
        # wandb.log({"Training loss": train_loss})
        # print('epoch sum loss:', train_loss)
        # print('start_rmsd_all:', start_rmsd_all)

        test_rmsd = 0.0
        test_rmsd_start = 0.0   
        model.eval()
        with torch.no_grad():
            print('to test!')
            for data in test_dataloader:
                cg_embedding = model(data.to(local_rank))  # 输出维度为 [batch_size, residues, atoms, output_size]
                cg_embedding_all_R_m4, cg_embedding_all_T = torch.split(cg_embedding.view(-1, cg_embedding.shape[-1]), [4, 3], dim=-1)
                
                # 计算损失
                rmsd = computer_rmsd4test(data.all_atom_positions.to(local_rank), data.gt_atom_positions.to(local_rank), data.gt_res_mask.to(local_rank),  data.gt_atom_mask.to(local_rank), cg_embedding_all_R_m4.to(local_rank), cg_embedding_all_T.to(local_rank), data.atom2cgids.to(local_rank))
                # computer_rmsd4test(init_model, gt_model, gt_res_mask, gt_atom_mask, rot_m4, tran, data.atom2cgids)
                rmsd_start = torch.sum(data.rmsd.to(local_rank))

                test_rmsd += rmsd.item()
                test_rmsd_start += rmsd_start.item()
                
                print('start rmsd:', rmsd_start.item())
                print('refine rmsd:', rmsd.item())
                print('---------')
                pdb_name = data.pdb_name
                
                with open(loss_file, "a") as file:
                    file.write(f"{epoch}, {pdb_name}, {rmsd:.4f}, {rmsd_start:.4f}\n")
        # log metrics to wandb
                
        test_rmsd = test_rmsd / (len(test_dataloader.dataset))
        
        test_rmsd_start = test_rmsd_start / (len(test_dataloader.dataset))
        print(' rmsd: ', test_rmsd)
        print(' test_rmsd_start: ', test_rmsd_start)
        with open(loss_file, "a") as file:
            file.write(f"{epoch}, , , {test_rmsd:.4f}, {test_rmsd_start:.4f}\n")

        if dist.get_rank() == 0 and test_rmsd < min_rmsd:
            min_rmsd = test_rmsd
            torch.save(model.module.state_dict(), "/nfs_beijing_ai/jinxian/rama-scoring/model/ckpt/temp.ckpt")
            print('svae new model!')
def print_grad(grad, name):
    if grad is None:
        print(f'No gradient for {name}')
    else:
        print(f'Gradient for {name}: {grad.shape}')

def main():
    set_random_seed(13)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'

    ## DDP：从外部得到local_rank参数。从外面得到local_rank参数，在调用DDP的时候，其会自动给出这个参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default=-1, type=int)
    args = parser.parse_args()

    port_id = 20000 + np.random.randint(0, 10000)
    args.dist_url = 'tcp://127.0.0.1:' + '8003' #str(port_id)
    args.nprocs = torch.cuda.device_count()
    mp.spawn(main_worker, nprocs=args.nprocs, args=(args.nprocs, args))
    

if __name__ == '__main__':
    main()
    
    