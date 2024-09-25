import logging
import time
import math
import numpy as np
import torch
import torch.nn as nn
from pyexpat.model import XML_CQUANT_OPT

from ocpmodels.common.utils import conditional_grad
from ocpmodels.models.base import BaseModel
from ocpmodels.models.scn.sampling import CalcSpherePoints
from ocpmodels.models.scn.smearing import (
    GaussianSmearing,
    LinearSigmoidSmearing,
    SigmoidSmearing,
    SiLUSmearing,
)

try:
    from e3nn import o3
except ImportError:
    pass

from model.equiformerv2.gaussian_rbf import GaussianRadialBasisLayer
from torch.nn import Linear
from model.equiformerv2.edge_rot_mat import init_edge_rot_mat
from model.equiformerv2.so3 import (
    CoefficientMappingModule,
    SO3_Embedding,
    SO3_Grid,
    SO3_Rotation,
    SO3_LinearV2
)
from model.equiformerv2.module_list import ModuleListInfo
from model.equiformerv2.so2_ops import SO2_Convolution
from model.equiformerv2.radial_function import RadialFunction
from model.equiformerv2.layer_norm import (
    EquivariantLayerNormArray,
    EquivariantLayerNormArraySphericalHarmonics,
    EquivariantRMSNormArraySphericalHarmonics,
    EquivariantRMSNormArraySphericalHarmonicsV2,
    get_normalization_layer
)
from model.equiformerv2.transformer_block import (
    SO2EquivariantGraphAttention,
    FeedForwardNetwork,
    TransBlockV2,
)
from model.equiformerv2.input_block import EdgeDegreeEmbedding

# Statistics of IS2RE 100K
_AVG_NUM_NODES = 77.81317
_AVG_DEGREE = 23.395238876342773  # IS2RE: 100k, max_radius = 5, max_neighbors = 100


class EquiformerV2(BaseModel):
    """
    Equiformer with graph attention built upon SO(2) convolution and feedforward network built upon S2 activation

    Args:
        use_pbc (bool):         Use periodic boundary conditions
        regress_forces (bool):  Compute forces
        otf_graph (bool):       Compute graph On The Fly (OTF)
        max_neighbors (int):    Maximum number of neighbors per atom
        max_radius (float):     Maximum distance between nieghboring atoms in Angstroms
        max_num_elements (int): Maximum atomic number

        num_layers (int):             Number of layers in the GNN
        sphere_channels (int):        Number of spherical channels (one set per resolution)
        attn_hidden_channels (int): Number of hidden channels used during SO(2) graph attention
        num_heads (int):            Number of attention heads
        attn_alpha_head (int):      Number of channels for alpha vector in each attention head
        attn_value_head (int):      Number of channels for value vector in each attention head
        ffn_hidden_channels (int):  Number of hidden channels used during feedforward network
        norm_type (str):            Type of normalization layer (['layer_norm', 'layer_norm_sh', 'rms_norm_sh'])

        lmax_list (int):              List of maximum degree of the spherical harmonics (1 to 10)
        mmax_list (int):              List of maximum order of the spherical harmonics (0 to lmax)
        grid_resolution (int):        Resolution of SO3_Grid

        num_sphere_samples (int):     Number of samples used to approximate the integration of the sphere in the output blocks

        edge_channels (int):                Number of channels for the edge invariant features
        use_atom_edge_embedding (bool):     Whether to use atomic embedding along with relative distance for edge scalar features
        share_atom_edge_embedding (bool):   Whether to share `atom_edge_embedding` across all blocks
        use_m_share_rad (bool):             Whether all m components within a type-L vector of one channel share radial function weights
        distance_function ("gaussian", "sigmoid", "linearsigmoid", "silu"):  Basis function used for distances

        attn_activation (str):      Type of activation function for SO(2) graph attention
        use_s2_act_attn (bool):     Whether to use attention after S2 activation. Otherwise, use the same attention as Equiformer
        use_attn_renorm (bool):     Whether to re-normalize attention weights
        ffn_activation (str):       Type of activation function for feedforward network
        use_gate_act (bool):        If `True`, use gate activation. Otherwise, use S2 activation
        use_grid_mlp (bool):        If `True`, use projecting to grids and performing MLPs for FFNs.
        use_sep_s2_act (bool):      If `True`, use separable S2 activation when `use_gate_act` is False.

        alpha_drop (float):         Dropout rate for attention weights
        drop_path_rate (float):     Drop path rate
        proj_drop (float):          Dropout rate for outputs of attention and FFN in Transformer blocks

        weight_init (str):          ['normal', 'uniform'] initialization of weights of linear layers except those in radial functions
    """

    def __init__(
            self,
            use_pbc=False,
            regress_forces=False,
            otf_graph=True,
            max_neighbors=10,
            max_radius=5.0,
            max_num_elements=90,
            max_num_atom_names=37,
            max_num_residues=21,

            num_layers=4,
            sphere_channels=16,
            attn_hidden_channels=8,
            num_heads=4,
            attn_alpha_channels=16,
            attn_value_channels=8,
            ffn_hidden_channels=8,

            norm_type='layer_norm_sh',

            lmax_list=[4],
            mmax_list=[2],
            grid_resolution=18,

            num_sphere_samples=16,

            edge_channels=8,
            use_atom_edge_embedding=True,
            share_atom_edge_embedding=False,
            use_m_share_rad=False,
            distance_function="gaussian",
            num_distance_basis=512,

            attn_activation='silu',
            use_s2_act_attn=False,
            use_attn_renorm=True,
            ffn_activation='silu',
            use_gate_act=False,
            use_grid_mlp=True,
            use_sep_s2_act=True,

            alpha_drop=0.1,
            drop_path_rate=0.05,
            proj_drop=0.0,

            weight_init='uniform',
            **kwargs
    ):
        super().__init__()

        self.use_pbc = use_pbc
        self.regress_forces = regress_forces
        self.otf_graph = otf_graph
        self.max_neighbors = max_neighbors
        self.max_radius = max_radius
        self.cutoff = max_radius
        self.max_num_elements = max_num_elements
        self.max_num_atom_names = max_num_atom_names
        self.max_num_residues = max_num_residues

        self.num_layers = num_layers
        self.sphere_channels = sphere_channels
        self.attn_hidden_channels = attn_hidden_channels
        self.num_heads = num_heads
        self.attn_alpha_channels = attn_alpha_channels
        self.attn_value_channels = attn_value_channels
        self.ffn_hidden_channels = ffn_hidden_channels
        self.norm_type = norm_type

        self.lmax_list = lmax_list
        self.mmax_list = mmax_list
        self.grid_resolution = grid_resolution

        self.num_sphere_samples = num_sphere_samples

        self.edge_channels = edge_channels
        self.use_atom_edge_embedding = use_atom_edge_embedding
        self.share_atom_edge_embedding = share_atom_edge_embedding
        if self.share_atom_edge_embedding:
            assert self.use_atom_edge_embedding
            self.block_use_atom_edge_embedding = False
        else:
            self.block_use_atom_edge_embedding = self.use_atom_edge_embedding
        self.use_m_share_rad = use_m_share_rad
        self.distance_function = distance_function
        self.num_distance_basis = num_distance_basis

        self.attn_activation = attn_activation
        self.use_s2_act_attn = use_s2_act_attn
        self.use_attn_renorm = use_attn_renorm
        self.ffn_activation = ffn_activation
        self.use_gate_act = use_gate_act
        self.use_grid_mlp = use_grid_mlp
        self.use_sep_s2_act = use_sep_s2_act

        self.alpha_drop = alpha_drop
        self.drop_path_rate = drop_path_rate
        self.proj_drop = proj_drop

        self.weight_init = weight_init
        assert self.weight_init in ['normal', 'uniform']

        self.device = 'cpu'  # torch.cuda.current_device()

        self.grad_forces = False
        self.num_resolutions = len(self.lmax_list)
        self.sphere_channels_all = self.num_resolutions * self.sphere_channels

        # Weights for message initialization
        self.sphere_embedding = nn.Embedding(self.max_num_elements, self.sphere_channels_all)
        self.element_sphere_embedding = nn.Embedding(self.max_num_elements, self.sphere_channels_all)
        self.atom_name_sphere_embedding = nn.Embedding(self.max_num_atom_names, self.sphere_channels_all)
        self.resid_sphere_embedding = nn.Embedding(self.max_num_residues, self.sphere_channels_all)

        # Initialize the function used to measure the distances between atoms
        assert self.distance_function in [
            'gaussian',
        ]
        if self.distance_function == 'gaussian':
            self.distance_expansion = GaussianSmearing(
                0.0,
                self.cutoff,
                600,
                2.0,
            )
            # self.distance_expansion = GaussianRadialBasisLayer(num_basis=self.num_distance_basis, cutoff=self.max_radius)
        else:
            raise ValueError

        # Initialize the sizes of radial functions (input channels and 2 hidden channels)
        self.edge_channels_list = [int(self.distance_expansion.num_output)] + [self.edge_channels] * 2

        # Initialize atom edge embedding
        if self.share_atom_edge_embedding and self.use_atom_edge_embedding:
            self.source_embedding = nn.Embedding(self.max_num_elements, self.edge_channels_list[-1])
            self.target_embedding = nn.Embedding(self.max_num_elements, self.edge_channels_list[-1])
            self.edge_channels_list[0] = self.edge_channels_list[0] + 2 * self.edge_channels_list[-1]
        else:
            self.source_embedding, self.target_embedding = None, None

        # Initialize the module that compute WignerD matrices and other values for spherical harmonic calculations
        self.SO3_rotation = nn.ModuleList()
        for i in range(self.num_resolutions):
            self.SO3_rotation.append(SO3_Rotation(self.lmax_list[i]))

        # Initialize conversion between degree l and order m layouts
        self.mappingReduced = CoefficientMappingModule(self.lmax_list, self.mmax_list)

        # Initialize the transformations between spherical and grid representations
        self.SO3_grid = ModuleListInfo('({}, {})'.format(max(self.lmax_list), max(self.lmax_list)))
        for l in range(max(self.lmax_list) + 1):
            SO3_m_grid = nn.ModuleList()
            for m in range(max(self.lmax_list) + 1):
                SO3_m_grid.append(
                    SO3_Grid(
                        l,
                        m,
                        resolution=self.grid_resolution,
                        normalization='component'
                    )
                )
            self.SO3_grid.append(SO3_m_grid)

        # Edge-degree embedding
        self.edge_degree_embedding = EdgeDegreeEmbedding(
            3*self.sphere_channels,
            self.lmax_list,
            self.mmax_list,
            self.SO3_rotation,
            self.mappingReduced,
            self.max_num_elements,
            self.edge_channels_list,
            self.block_use_atom_edge_embedding,
            rescale_factor=_AVG_DEGREE
        )

        # Initialize the blocks for each layer of EquiformerV2
        self.blocks = nn.ModuleList()
        for i in range(self.num_layers):
            block = TransBlockV2(
                3*self.sphere_channels,
                self.attn_hidden_channels,
                self.num_heads,
                self.attn_alpha_channels,
                self.attn_value_channels,
                self.ffn_hidden_channels,
                3*self.sphere_channels,
                self.lmax_list,
                self.mmax_list,
                self.SO3_rotation,
                self.mappingReduced,
                self.SO3_grid,
                self.max_num_elements,
                self.edge_channels_list,
                self.block_use_atom_edge_embedding,
                self.use_m_share_rad,
                self.attn_activation,
                self.use_s2_act_attn,
                self.use_attn_renorm,
                self.ffn_activation,
                self.use_gate_act,
                self.use_grid_mlp,
                self.use_sep_s2_act,
                self.norm_type,
                self.alpha_drop,
                self.drop_path_rate,
                self.proj_drop
            )
            self.blocks.append(block)

        # Output blocks for energy and forces
        self.norm = get_normalization_layer(self.norm_type, lmax=max(self.lmax_list), num_channels=3*self.sphere_channels)
        self.energy_block = FeedForwardNetwork(
            3*self.sphere_channels,
            self.ffn_hidden_channels,
            1,   # can be modified
            self.lmax_list,
            self.mmax_list,
            self.SO3_grid,
            self.ffn_activation,
            self.use_gate_act,
            self.use_grid_mlp,
            self.use_sep_s2_act
        )
        self.All_atom_embedding = FeedForwardNetwork(
            3*self.sphere_channels,
            self.ffn_hidden_channels,
            128,   # can be modified
            self.lmax_list,
            self.mmax_list,
            self.SO3_grid,
            self.ffn_activation,
            self.use_gate_act,
            self.use_grid_mlp,
            self.use_sep_s2_act
        )

        self.output = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 7)  # R_m4[1,4] and T[1,3]
        )

        if self.regress_forces:
            self.force_block = SO2EquivariantGraphAttention(
                3*self.sphere_channels,
                self.attn_hidden_channels,
                self.num_heads,
                self.attn_alpha_channels,
                self.attn_value_channels,
                1,
                self.lmax_list,
                self.mmax_list,
                self.SO3_rotation,
                self.mappingReduced,
                self.SO3_grid,
                self.max_num_elements,
                self.edge_channels_list,
                self.block_use_atom_edge_embedding,
                self.use_m_share_rad,
                self.attn_activation,
                self.use_s2_act_attn,
                self.use_attn_renorm,
                self.use_gate_act,
                self.use_sep_s2_act,
                alpha_drop=0.0
            )

        self.apply(self._init_weights)
        self.apply(self._uniform_init_rad_func_linear_weights)

    @conditional_grad(torch.enable_grad())
    def forward(self, data):
        # print(data)
        self.batch_size = len(data.n_nodes)
        self.dtype = data.pos.dtype
        self.device = data.pos.device
        # print(self.device)

        atomic_numbers = data.atomic_numbers.long()
        atom_numbers = data.atom_numbers.long()
        resid = data.resid.long()
        num_atoms = len(atomic_numbers)
        # pos = data.pos

        # to adapt with ocpmodel BaseModel, we should modify some attributes
        data.natoms = data.n_nodes

        (
            edge_index,
            edge_distance,
            edge_distance_vec,
            cell_offsets,
            _,  # cell offset distances
            neighbors,
        ) = self.generate_graph(data)

        ###############################################################
        # Initialize data structures
        ###############################################################

        # Compute 3x3 rotation matrix per edge
        edge_rot_mat = self._init_edge_rot_mat(
            data, edge_index, edge_distance_vec
        )

        # Initialize the WignerD matrices and other values for spherical harmonic calculations
        for i in range(self.num_resolutions):
            self.SO3_rotation[i].set_wigner(edge_rot_mat)

        ###############################################################
        # Initialize node embeddings
        ###############################################################

        # Init per node representations using an atomic number based embedding
        offset = 0
        x = SO3_Embedding( # there x dim is [atom, 25, 96], 25 is from (lmax_list[0] + 1) ** 2
            num_atoms,
            self.lmax_list,
            3*self.sphere_channels,
            self.device,
            self.dtype,
        )

        offset_res = 0
        offset = 0
        # Initialize the l = 0, m = 0 coefficients for each resolution
        for i in range(self.num_resolutions):
            if self.num_resolutions == 1:
                x.embedding[:, offset_res, :] = torch.cat([self.element_sphere_embedding(atomic_numbers),
                                                           self.atom_name_sphere_embedding(atom_numbers),
                                                           self.resid_sphere_embedding(resid)], dim=-1)
                # x.embedding[:, offset_res, :] = self.sphere_embedding(atomic_numbers)
            else:
                x.embedding[:, offset_res, :] = torch.cat([self.element_sphere_embedding(atomic_numbers)[:, offset: offset + self.sphere_channels],
                                                           self.atom_name_sphere_embedding(atom_numbers)[:, offset: offset + self.sphere_channels],
                                                           self.resid_sphere_embedding(resid)[:, offset: offset + self.sphere_channels]], dim=-1)
                # x.embedding[:, offset_res, :] = self.sphere_embedding(
                #     atomic_numbers
                # )[:, offset: offset + self.sphere_channels]
            offset = offset + self.sphere_channels
            offset_res = offset_res + int((self.lmax_list[i] + 1) ** 2)

        # Edge encoding (distance and atom edge)
        edge_distance = self.distance_expansion(edge_distance)
        if self.share_atom_edge_embedding and self.use_atom_edge_embedding:
            source_element = atomic_numbers[edge_index[0]]  # Source atom atomic number
            target_element = atomic_numbers[edge_index[1]]  # Target atom atomic number
            source_embedding = self.source_embedding(source_element)
            target_embedding = self.target_embedding(target_element)
            edge_distance = torch.cat((edge_distance, source_embedding, target_embedding), dim=1)

        # Edge-degree embedding
        edge_degree = self.edge_degree_embedding(
            atomic_numbers,
            edge_distance,
            edge_index)
        x.embedding = x.embedding + edge_degree.embedding

        ###############################################################
        # Update spherical node embeddings
        ###############################################################

        for i in range(self.num_layers):
            x = self.blocks[i](
                x,  # SO3_Embedding
                atomic_numbers,
                edge_distance,
                edge_index,
                batch=data.batch  # for GraphDropPath
            )

        # Final layer norm
        x.embedding = self.norm(x.embedding)

        ###############################################################
        # score estimation
        ###############################################################
        all_atom_embedding = self.All_atom_embedding(x)
        all_atom_embedding = all_atom_embedding.embedding.narrow(1, 0, 1)

        # all_atom_embedding = self.output(all_atom_embedding)

        # create a tensor which shape is same as original atom pos, and make sure its grad is ok.
        # x_restored = torch.zeros((data.atom_mask.shape[0], all_atom_embedding.shape[-1]), device=self.device, requires_grad=True).detach().clone()
        
        # x_restored = torch.zeros((data.atom_mask.shape[0], all_atom_embedding.shape[-1]), device=self.device, requires_grad=True)
        # x_restored[data.atom_mask] = all_atom_embedding.squeeze(1)

        # 找到 mask 为 True 的索引
        indices = torch.nonzero(data.atom_mask, as_tuple=False).squeeze(1)  # shape: [5]

        # 创建一个与 X 形状相同的零张量，用于存放替换后的值
        Y_new = torch.zeros((data.atom_mask.shape[0], all_atom_embedding.shape[-1]), device=self.device)
        
        Y_new[indices] = all_atom_embedding.squeeze(1)

        # 重塑张量形状
        x_restored = Y_new.reshape(-1, 37, Y_new.shape[-1])

        # use the mask pos of x_restored to have the update embedding
        
        cg_embedding = torch.matmul(data.atom2cgids.transpose(1, 2), x_restored)
        cg_embedding = self.output(cg_embedding)

        # the next is to 
        # 按列相加，主要是为了计算每个氨基酸残基中每个原子在cg中出现的总次数[batch_size, residue，atom=37]
        # each_res_atom_times = torch.sum(data.atom2cgids, dim=-1)
        # # 转为形状 [batch_size, residue，atom=37, 1] 以进行广播
        # t_last = each_res_atom_times.unsqueeze(-1)
        # # 防止除以 0，将 t_last 中的 0 替换为一个很小的值（例如1e-6），避免计算错误
        # t_last_safe = torch.where(t_last == 0, torch.tensor(1e-6).to(self.device), t_last)
        # # x 的最后一个维度除以 t 的最后一个值
        # mean_cg_trans_atom = data.atom2cgids / t_last_safe
        
        # # 然后乘以输出，获得每个原子的平均值
        # output = torch.matmul(mean_cg_trans_atom, cg_embedding)


        #then, reshape the data to [R, 37, emb] for transfer the atom embedding to cg embedding 
        #DataBatch(pos=[4634, 3], 
        # atom_numbers=[4634], 
        # atomic_numbers=[4634], 
        # resid=[4634], 
        # n_nodes=[2], 
        # gt_atom_positions=[492, 37, 3], 
        # gt_atom_mask=[492, 37], 
        # gt_res_mask=[558], 
        # batch=[4634], 
        # ptr=[3], 
        # natoms=[2])

        return cg_embedding

        # ###############################################################
        # # Force estimation
        # ###############################################################
        # if self.regress_forces:
        #     forces = self.force_block(x,
        #                               atomic_numbers,
        #                               edge_distance,
        #                               edge_index)
        #     forces = forces.embedding.narrow(1, 1, 3)
        #     forces = forces.view(-1, 3)
        #
        # if not self.regress_forces:
        #     return energy
        # else:
        #     return energy, forces

    # Initialize the edge rotation matrics
    def _init_edge_rot_mat(self, data, edge_index, edge_distance_vec):
        return init_edge_rot_mat(edge_distance_vec)

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def _init_weights(self, m):
        if (isinstance(m, torch.nn.Linear)
                or isinstance(m, SO3_LinearV2)
        ):
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
            if self.weight_init == 'normal':
                std = 1 / math.sqrt(m.in_features)
                torch.nn.init.normal_(m.weight, 0, std)

        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)

    def _uniform_init_rad_func_linear_weights(self, m):
        if (isinstance(m, RadialFunction)):
            m.apply(self._uniform_init_linear_weights)

    def _uniform_init_linear_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
            std = 1 / math.sqrt(m.in_features)
            torch.nn.init.uniform_(m.weight, -std, std)

    @torch.jit.ignore
    def no_weight_decay(self):
        no_wd_list = []
        named_parameters_list = [name for name, _ in self.named_parameters()]
        for module_name, module in self.named_modules():
            if (isinstance(module, torch.nn.Linear)
                    or isinstance(module, SO3_LinearV2)
                    or isinstance(module, torch.nn.LayerNorm)
                    or isinstance(module, EquivariantLayerNormArray)
                    or isinstance(module, EquivariantLayerNormArraySphericalHarmonics)
                    or isinstance(module, EquivariantRMSNormArraySphericalHarmonics)
                    or isinstance(module, EquivariantRMSNormArraySphericalHarmonicsV2)
                    or isinstance(module, GaussianRadialBasisLayer)):
                for parameter_name, _ in module.named_parameters():
                    if (isinstance(module, torch.nn.Linear)
                            or isinstance(module, SO3_LinearV2)
                    ):
                        if 'weight' in parameter_name:
                            continue
                    global_parameter_name = module_name + '.' + parameter_name
                    assert global_parameter_name in named_parameters_list
                    no_wd_list.append(global_parameter_name)
        return set(no_wd_list)