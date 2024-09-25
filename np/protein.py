"""Code."""
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
from utils.protein import get_seq_info
from utils.constants import cg_constants
from utils.logger import Logger
from utils.opt_utils import superimpose_single, masked_differentiable_rmsd
import torch
import math
from scipy.spatial.transform import Rotation as R
logger = Logger.logger

FeatureDict = Mapping[str, np.ndarray]
ModelOutput = Mapping[str, Any]
PICO_TO_ANGSTROM = 0.01
PDB_CHAIN_IDS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
PDB_MAX_CHAINS = len(PDB_CHAIN_IDS)


@dataclasses.dataclass(frozen=True)
class Protein:
    """Define Class Protein."""

    """Protein structure representation."""
    rmsd: np.ndarray
    atom_positions: np.ndarray
    trans_target: np.ndarray
    rot_m4_target: np.ndarray
    gt_res_mask: np.ndarray
    gt_atom_positions: np.ndarray
    atom2cgids: np.ndarray
    aatype: np.ndarray
    atom_mask: np.ndarray
    gt_atom_mask: np.ndarray
    residue_index: np.ndarray
    b_factors: np.ndarray
    chain_index: np.ndarray
    atom_names: np.ndarray
    residue_names: np.ndarray
    elements: np.ndarray
    ele2nums: np.ndarray
    hetfields: np.ndarray
    resids: np.ndarray
    icodes: np.ndarray
    chains: list
    remark: Optional[str] = None
    parents: Optional[Sequence[str]] = None
    parents_chain_index: Optional[Sequence[int]] = None
    resolution: any = None

    def __post_init__(self):
        """Run __post_init__ method."""
        # code.
        """
    __post_init__:
    Args:
        self : self
    Returns:
    """
        if len(np.unique(self.chain_index)) > PDB_MAX_CHAINS:
            raise ValueError(
                f'Cannot build an instance with more than {PDB_MAX_CHAINS} chains because these cannot be written to PDB format'
            )

def read_pdb(input_chain):
    res_list = list()  # a list of residue instance
    res1_list = list()  # a list of type1 amino acid
    for i, res in enumerate(input_chain):
        if res.id[0] != " ":  # skip HETATM
            continue
        # res_idx = res.id[1]
        # icode = res.id[2].strip()
        # idx = f"{res_idx}{icode}"
        res1 = residue_constants.restype_3to1.get(res.resname.strip(), "X")
        res_list.append(res)
        res1_list.append(res1)

    pdb_seq = "".join(res1_list)
    # res_list 包含链中的所有残基实例，pdb_seq 是由单字母代码组成的该链的氨基酸序列。
    return res_list, pdb_seq



def get_pdb_seq_by_CA(model, chain_id):
    """Run get_pdb_seq_by_CA method."""
    # code.
    """
    get_pdb_seq_by_CA:
    Args:
        model : model
    Returns:
    """
    seq = []
    for chain in model:
        if (chain_id is not None) and (chain.get_id() != chain_id):
            continue
        for res in chain:
            has_CA_flag = False
            for atom in res:
                if atom.name == 'CA':
                    has_CA_flag = True
                    break
            if has_CA_flag:
                # seq.append(residue_constants.restype_3to1.get(res.resname, 'X'))
                # residue with X name is considered to be missing in full_seq_AMR, so we have to skip it here.
                # todo: check if this will cause any problem.
                resname = residue_constants.restype_3to1.get(res.resname, 'X')
                if resname != 'X':
                    seq.append(resname)
                else:
                    continue
    seq = "".join(seq)
    return seq


def get_model_from_str(pdb_str):
    pdb_fh = io.StringIO(pdb_str)
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('none', pdb_fh)
    resolution = structure.header['resolution']
    if resolution is None:
        resolution = 0.0
    models = list(structure.get_models())
    model = models[0]
    return model, resolution

def from_pdb_string(
    pdb_str: str,
    gt_pdb_str: str,
    chain_id: Optional[str] = None,
    return_id2seq: bool = False,
    ca_only=False,
    full_seq_AMR=None,
) -> Protein:
    """Run from_pdb_string method."""
    # code.
    """Takes a PDB string and constructs a Protein object.

    WARNING: All non-standard residue types will be converted into UNK. All
      non-standard atoms will be ignored.

    Args:
      pdb_str: The contents of the pdb file
      chain_id: If chain_id is specified (e.g. A), then only that chain is
      parsed. Else, all chains are parsed.

    Returns:
      A new `Protein` parsed from the pdb contents.
    """
    # pdb_fh = io.StringIO(pdb_str)
    # parser = PDBParser(QUIET=True)
    # structure = parser.get_structure('none', pdb_fh)
    # resolution = structure.header['resolution']
    # if resolution is None:
    #     resolution = 0.0
    # models = list(structure.get_models())
    # model = models[0]

    model, resolution = get_model_from_str(pdb_str)
    gt_model, _ = get_model_from_str(gt_pdb_str)

    chain_id_set = set()
    if chain_id is not None:
        if ',' in chain_id:
            chain_id = chain_id.split(',')
        chain_id_set = set(chain_id)

    # update for checking the full_seq_AMR to solve the cut_fv issue.
    if full_seq_AMR is not None:
        # full_seq_AMR is given, need to check the AMR and cut_fv issue.
        pdb_seq = get_pdb_seq_by_CA(model, chain_id)
        
        cleaned_seq, is_add_list = get_seq_info(full_seq_AMR)
        # print('af_length:', len(pdb_seq))
        # print('gt_AMR_full length:', len(cleaned_seq))
        if pdb_seq != cleaned_seq: # af length != full AMR length
            return False
        assert len(cleaned_seq) == len(is_add_list), "The length of cleaned_seq and is_add_list should be the same"
        
        gt_fv_seq = [e for e, c in zip(cleaned_seq, is_add_list) if c == 0]
        pdb_seq_gt = get_pdb_seq_by_CA(gt_model, chain_id)
        if len(pdb_seq_gt) != len(gt_fv_seq):
            return False

        gt_fv_seq = "".join(gt_fv_seq)
        # pdb_seq, the aa sequence from the given pdb file
        # cleaned_seq, the fv sequence, full_seq_AMR, without brackets.
        # gt_fv_seq, the subsequence of the full_seq_AMR, which the ground truth pdb contains.
        # considering the missing residue issue, the gt_fv_seq may not be a contiguous subsequence of the full_seq_AMR
        # but it must be a contiguous subsequence of pdb_seq! Otherwise the input data is not correct.

        if pdb_seq.strip() == cleaned_seq.strip():
            # sequence in pdb is totally the same with full_seq_AMR, i.e. the fv sequence.
            # indicates no missing residue, no cut fv issue.
            # Probably decoys predicted by upstream single models.
            # start from 0, end by len(pdb_seq)
            # and as the pdb is not gt, so the gt_fv_seq may not match the pdb_seq.
            start_idx = 0
            end_idx = len(pdb_seq)
        else:
            # with missing residue, or cut fv issue.
            start_idx = pdb_seq.find(gt_fv_seq)      # gt_fv_seq should be a contiguous subsequence of pdb_seq! get the start idx.
            assert start_idx != -1, "gt_fv_seq should be a contiguous subsequence of pdb_seq!"   # -1 indicates not found.
            end_idx = start_idx + len(gt_fv_seq)
            assert end_idx <= len(pdb_seq), "end_idx should not exceed the length of pdb_seq."
            
            
    else:
        # No full_seq_AMR given. Ignore the missing residue and cut fv issue
        # All residues in the PDB file are used.
        start_idx = 0
        end_idx = -1

        pdb_seq = get_pdb_seq_by_CA(model, chain_id)
        gt_pdb_seq = get_pdb_seq_by_CA(gt_model, chain_id)
        if pdb_seq != gt_pdb_seq:
            return False
        is_add_list = [0] * len(pdb_seq)

    gt_res_mask = [1- isad for isad in is_add_list]
    # update for checking number of chains of the input pdb.
    chain_cnt = 0
    for _ in model:
        chain_cnt += 1

    if chain_id is None and chain_cnt > 1:
        logger.warning(f"Multiple chains detected in the PDB file. Please specify chain_id.")
        logger.warning(f"Read multiple chains as default.")

    
    # get the gt model pos and mask
    gt_atom_positions = []
    gt_atom_mask = []
    gt_cg_mask = []
    atom_type_num = residue_constants.atom_type_num
    accept_res_cnt = 0
    for chain in gt_model:
        cur_chain_aatype = []
        if chain_id is not None and chain.id not in chain_id_set:
            continue
        cur_idx = -1
        for res in chain:
            has_ca_flag = False
            for atom in res:
                if atom.name == 'CA':
                    has_ca_flag = True
                    break

            if has_ca_flag:
                # only when the residue has ca, this residue will be counted to cur_idx.
                cur_idx += 1

            # if a residue has no CA, this residue should not be count into cur_idx.
            # but the information of other heavy atoms of this residue can be used.
            # so we don't skip the case that has_ca_flag == False

            # if the first residue has no CA, then cur_idx = -1, if start_idx = 0, then it will skip this first residue.
            # and if the residue has CA but this residue is not in the range of start_idx and end_idx (not a fv res), then it will skip this residue.
            if cur_idx < start_idx:
                continue
            # if this residue exceed the range of end_idx, then skip this residue
            # if end_idx < 0, which means full_seq_AMR is not given, so we won't skip.
            elif (cur_idx >= end_idx) and (end_idx > 0):
                continue
            # if cur_idx is in the range of [start_idx, end_idx], but this residue has no CA, it is actually should be treat as missing residue.
            # We can preserve the information of this residue, but it should not take this residue into account when counting accepted residues.
            elif has_ca_flag:
                accept_res_cnt += 1


            het_flag, resseq, ic = res.id
            if ic == ' ':
                ic = ''
            if het_flag == ' ':
                het_flag = 'A'
            if res.id[0] != ' ':
                continue
            
            pos = np.zeros((atom_type_num, 3))
            mask = np.zeros((atom_type_num,))
            cg_mask = np.zeros((4,))

            # cg_id_type = [0] * 4
                
            # cg_list = cg_constants.cg_dict[res.resname]
            # for i in range(len(cg_list)):
            #     if atom.name in cg_list[i]:
            #         res_id = residue_constants.restype_order[residue_constants.restype_3to1[res.resname]]
            #         cg_id = cg_constants.cg2id[(res_id, i)] # (残积id, 该残积上的第i个cg): cg id
            #         cg_id_type[i] = 1 

            for atom in res:
                if atom.name not in residue_constants.atom_types:
                    continue
                if ca_only:
                    if atom.name != 'CA':
                        continue

                # residue_constants.atom_order[atom.name] is atom_id
                
                
                pos[residue_constants.atom_order[atom.name]] = atom.coord
                mask[residue_constants.atom_order[atom.name]] = 1.0
                

            if np.sum(mask) < 0.5:
                continue
            
            gt_atom_positions.append(pos)
            gt_atom_mask.append(mask)


    # get af model information
    atom_positions = []
    atom2cgids = []
    aatype = []
    atom_mask = []
    residue_index = []
    chain_ids = []
    b_factors = []
    id2seq = {}
    atom_names = []
    residue_names = []
    elements = []
    ele2nums = []
    hetfields = []
    resids = []
    icodes = []
    chains = []

    
    # considering the residue missing CA, we should ignore these residues since they are not included in gt_fv_seq.
    accept_res_cnt = 0
    for chain in model:
        cur_chain_aatype = []
        if chain_id is not None and chain.id not in chain_id_set:
            continue
        chains.append(chain.id)
        cur_idx = -1
        for res in chain:
            has_ca_flag = False
            for atom in res:
                if atom.name == 'CA':
                    has_ca_flag = True
                    break

            if has_ca_flag:
                # only when the residue has ca, this residue will be counted to cur_idx.
                cur_idx += 1

            # if a residue has no CA, this residue should not be count into cur_idx.
            # but the information of other heavy atoms of this residue can be used.
            # so we don't skip the case that has_ca_flag == False

            # if the first residue has no CA, then cur_idx = -1, if start_idx = 0, then it will skip this first residue.
            # and if the residue has CA but this residue is not in the range of start_idx and end_idx (not a fv res), then it will skip this residue.
            if cur_idx < start_idx:
                continue
            # if this residue exceed the range of end_idx, then skip this residue
            # if end_idx < 0, which means full_seq_AMR is not given, so we won't skip.
            elif (cur_idx >= end_idx) and (end_idx > 0):
                continue
            # if cur_idx is in the range of [start_idx, end_idx], but this residue has no CA, it is actually should be treat as missing residue.
            # We can preserve the information of this residue, but it should not take this residue into account when counting accepted residues.
            elif has_ca_flag:
                accept_res_cnt += 1


            het_flag, resseq, ic = res.id
            if ic == ' ':
                ic = ''
            if het_flag == ' ':
                het_flag = 'A'
            if res.id[0] != ' ':
                continue
            res_shortname = residue_constants.restype_3to1.get(res.resname, 'X'
                                                               )
            restype_idx = residue_constants.restype_order.get(res_shortname,
                                                              residue_constants.restype_num)
            atom_type_num = residue_constants.atom_type_num
            pos = np.zeros((atom_type_num, 3))
            atom2cgid = np.zeros((atom_type_num, 4)) # max cg num is 4 in a res 
            mask = np.zeros((atom_type_num,))
            res_b_factors = np.zeros((atom_type_num,))

            atom_name = np.empty((atom_type_num,), dtype=object)
            residue_name = np.empty((atom_type_num,), dtype=object)
            element = np.empty((atom_type_num,), dtype=object)
            ele2num = np.empty((atom_type_num,), dtype=object)
            hetfield = np.empty((atom_type_num,), dtype=object)
            resid = np.zeros((atom_type_num,))
            icode = np.empty((atom_type_num,), dtype=object)

            for atom in res:
                if atom.name not in residue_constants.atom_types:
                    continue
                if ca_only:
                    if atom.name != 'CA':
                        continue

                # get cg id from residue and atom
                cg_id_type = [0] * 4
                
                cg_list = cg_constants.cg_dict[res.resname]
                for i in range(len(cg_list)):
                    if atom.name in cg_list[i]:
                        res_id = residue_constants.restype_order[residue_constants.restype_3to1[res.resname]]
                        cg_id = cg_constants.cg2id[(res_id, i)] # (残积id, 该残积上的第i个cg): cg id
                        cg_id_type[i] = 1 

                # residue_constants.atom_order[atom.name] is atom_id
                atom2cgid[residue_constants.atom_order[atom.name]] = cg_id_type
                pos[residue_constants.atom_order[atom.name]] = atom.coord
                mask[residue_constants.atom_order[atom.name]] = 1.0
                res_b_factors[residue_constants.atom_order[atom.name]] = atom.bfactor
                atom_name[residue_constants.atom_order[atom.name]] = atom.name
                residue_name[residue_constants.atom_order[atom.name]] = res.resname
                element[residue_constants.atom_order[atom.name]] = atom.element
                ele2num[residue_constants.atom_order[atom.name]] = residue_constants.ele2num[atom.element]
                hetfield[residue_constants.atom_order[atom.name]] = het_flag
                resid[residue_constants.atom_order[atom.name]] = len(resids) + 1
                icode[residue_constants.atom_order[atom.name]] = ic

            if np.sum(mask) < 0.5:
                continue
            aatype.append(restype_idx)
            cur_chain_aatype.append(restype_idx)
            atom_positions.append(pos)
            atom2cgids.append(atom2cgid)
            atom_mask.append(mask)
            residue_index.append(res.id[1])
            chain_ids.append(chain.id)
            b_factors.append(res_b_factors)
            atom_names.append(atom_name)
            residue_names.append(residue_name)
            elements.append(element)
            ele2nums.append(ele2num)
            hetfields.append(hetfield)
            resids.append(resid)
            icodes.append(icode)

        id2seq[chain.id] = cur_chain_aatype
    unique_chain_ids = np.unique(chain_ids)
    chain_id_mapping = {cid: n for n, cid in enumerate(unique_chain_ids)}
    chain_index = np.array([chain_id_mapping[cid] for cid in chain_ids])

    # if we only use CA atoms, then the accept_res_cnt should be the same as the number of nodes.
    if ca_only:
        assert accept_res_cnt == len(residue_names), "accept_res_cnt should be the same as the number of nodes. Check what's wrong."


    # here we would frist computer the R and T of each cg between start model and gt model
    # first obtein the superimpose, 
    re =torch.from_numpy(np.array(gt_atom_positions))
    
    atommaks = torch.from_numpy(np.array(gt_atom_mask))
    resmaks = torch.from_numpy(np.array(gt_res_mask))
    cor = torch.from_numpy(np.array(atom_positions))
    cor = cor[resmaks==1]
    if cor.shape[0] == re.shape[0]:
        rot_, tran_, rmsd, superimpose = superimpose_single(
                re.view(-1,re.shape[-1]), cor.view(-1,cor.shape[-1]), differentiable=False , mask=atommaks.view(-1) # 1 replace true in mask, else is miss.
            )
    else:
        print(re.shape)
        print(cor.shape)
        print(atommaks.shape)
        return False


    # then iter the cg from superimpose and gt model one by one.
    # superimpose shape [R, 37, 3] like GT model
    reshape_superimpose = superimpose.view(-1,re.shape[-2],re.shape[-1])
    reshape_atom2cgids = torch.from_numpy(np.array(atom2cgids))[resmaks==1]

    trans_target = []
    rot_m4_target = []
    for i in range(reshape_superimpose.shape[0]): # iter res
        res_all_pos_re = re[i] # gt model one_res_atom_pos
        res_all_pos_reshape_superimpose = reshape_superimpose[i] # aligned from start model, one_res_atom_pos
        res_all_pos_reshape_atom2cgids = reshape_atom2cgids[i].transpose(0, 1)
        res_all_pos_atommaks = atommaks[i]

        tran_tem = np.zeros((4, 3))

        rot_m4_tem = np.zeros((4, 4))

        for j in range(res_all_pos_reshape_atom2cgids.shape[0]): # iter atom
            cg_atom_index = res_all_pos_reshape_atom2cgids[j]
            if torch.sum(cg_atom_index) < 1: # if this cg is None
                continue
            cg_atom_re_pos = res_all_pos_re[cg_atom_index==1] # get the cg atom in gt res
            cg_atom_superimpose_pos = res_all_pos_reshape_superimpose[cg_atom_index==1] # get the cg atom in gt res
            cg_atommaks = res_all_pos_atommaks[cg_atom_index==1]
            # computer the r and t from cg_atom_superimpose_pos to cg_atom_re_pos
            tran, rot = masked_differentiable_rmsd(cg_atom_re_pos.unsqueeze(0), cg_atom_superimpose_pos.unsqueeze(0), cg_atommaks.unsqueeze(0))
            tran_tem[j] = tran.view(-1)
            r3 = R.from_matrix(rot.squeeze(0))
            rot_m4_tem[j] = r3.as_quat()


        trans_target.append(tran_tem)
        rot_m4_target.append(rot_m4_tem)




    if return_id2seq:
        return Protein(
            trans_target=np.array(trans_target),
            rot_m4_target=np.array(rot_m4_target),
            gt_res_mask=np.array(gt_res_mask),
            atom_positions=np.array(atom_positions),
            gt_atom_positions=np.array(gt_atom_positions),
            atom2cgids = np.array(atom2cgids),
            atom_mask=np.array(atom_mask),
            gt_atom_mask=np.array(gt_atom_mask),
            aatype=np.array(aatype),
            chain_index=chain_index,
            residue_index=np.array(residue_index),
            b_factors=np.array(b_factors),
            resolution=resolution
        ), id2seq
    else:
        return Protein(
            rmsd = np.array(rmsd),
            trans_target=np.array(trans_target),
            rot_m4_target=np.array(rot_m4_target),
            gt_res_mask=np.array(gt_res_mask),
            atom_positions=np.array(atom_positions),
            gt_atom_positions=np.array(gt_atom_positions),
            atom2cgids = np.array(atom2cgids),
            atom_mask=np.array(atom_mask),
            gt_atom_mask=np.array(gt_atom_mask),
            aatype=np.array(aatype),
            chain_index=chain_index,
            residue_index=np.array(residue_index),
            b_factors=np.array(b_factors),
            resolution=resolution,
            atom_names=np.array(atom_names),
            residue_names=np.array(residue_names),
            elements=np.array(elements),
            ele2nums=np.array(ele2nums),
            hetfields=np.array(hetfields),
            resids=np.array(resids).astype(int),
            icodes=np.array(icodes),
            chains=chains,
        )


def is_antibody(seq, scheme='imgt', ncpu=4):
    """Run is_antibody method."""
    # code.
    """
    is_antibody:
    Args:
        seq : seq
        scheme : scheme
        ncpu : ncpu
    Returns:
    """
    seqs = [('0', seq)]
    numbering, alignment_details, hit_tables = anarci(seqs, scheme=scheme,
        output=False, ncpu=ncpu)
    if numbering[0] is None:
        return False, None
    if numbering[0] is not None and len(numbering[0]) > 1:
        logger.warning('There are %d domains in %s' % (len(numbering[0]), seq))
    chain_type = alignment_details[0][0]['chain_type'].lower()
    if chain_type is None:
        return False, None
    else:
        return True, chain_type
