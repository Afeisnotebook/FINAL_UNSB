"""Official CUT baseline adapter.

Source: taesungp/contrastive-unpaired-translation, commit
``b3ac297708dfb6f7589d04662277e53c0d579c27``, models/cut_model.py.
The upstream BSD license is retained in ``src/THIRD_PARTY_CUT_LICENSE``.
Only formatting and the repository's recoverable BaseModel hooks differ.
"""

import numpy as np
import torch

from .base_model import BaseModel
from . import networks
from .patchnce import PatchNCELoss
import util.util as util


class CUTModel(BaseModel):
    """Contrastive Learning for Unpaired Image-to-Image Translation."""

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument(
            "--CUT_mode", type=str, default="CUT",
            choices=("CUT", "cut", "FastCUT", "fastcut"),
        )
        parser.add_argument("--lambda_GAN", type=float, default=1.0)
        parser.add_argument("--lambda_NCE", type=float, default=1.0)
        parser.add_argument(
            "--nce_idt", type=util.str2bool, nargs="?", const=True, default=False,
        )
        parser.add_argument("--nce_layers", type=str, default="0,4,8,12,16")
        parser.add_argument(
            "--nce_includes_all_negatives_from_minibatch",
            type=util.str2bool, nargs="?", const=True, default=False,
        )
        parser.add_argument(
            "--netF", type=str, default="mlp_sample",
            choices=("sample", "reshape", "mlp_sample"),
        )
        parser.add_argument("--netF_nc", type=int, default=256)
        parser.add_argument("--nce_T", type=float, default=0.07)
        parser.add_argument("--num_patches", type=int, default=256)
        parser.add_argument(
            "--flip_equivariance", type=util.str2bool, nargs="?", const=True,
            default=False,
        )
        parser.set_defaults(pool_size=0)
        opt, _ = parser.parse_known_args()
        if opt.CUT_mode.lower() == "cut":
            parser.set_defaults(nce_idt=True, lambda_NCE=1.0)
        elif opt.CUT_mode.lower() == "fastcut":
            parser.set_defaults(
                nce_idt=False, lambda_NCE=10.0, flip_equivariance=True,
                n_epochs=150, n_epochs_decay=50,
            )
        else:
            raise ValueError(opt.CUT_mode)
        return parser

    def __init__(self, opt):
        super().__init__(opt)
        self.loss_names = ["G_GAN", "D_real", "D_fake", "G", "NCE"]
        self.visual_names = ["real_A", "fake_B", "real_B"]
        self.nce_layers = [int(value) for value in self.opt.nce_layers.split(",")]
        if opt.nce_idt and self.isTrain:
            self.loss_names += ["NCE_Y"]
            self.visual_names += ["idt_B"]
        self.model_names = ["G", "F", "D"] if self.isTrain else ["G"]
        self.netG = networks.define_G(
            opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG,
            not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias,
            opt.no_antialias_up, self.gpu_ids, opt,
        )
        self.netF = networks.define_F(
            opt.input_nc, opt.netF, opt.normG, not opt.no_dropout,
            opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt,
        )
        if self.isTrain:
            self.netD = networks.define_D(
                opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD,
                opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt,
            )
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionNCE = [PatchNCELoss(opt).to(self.device) for _ in self.nce_layers]
            self.criterionIdt = torch.nn.L1Loss().to(self.device)
            self.optimizer_G = torch.optim.Adam(
                self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2),
            )
            self.optimizer_D = torch.optim.Adam(
                self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2),
            )
            self.optimizers += [self.optimizer_G, self.optimizer_D]

    def data_dependent_initialize(self, data):
        bs_per_gpu = data["A"].size(0) // max(len(self.opt.gpu_ids), 1)
        self.set_input(data)
        self.real_A = self.real_A[:bs_per_gpu]
        self.real_B = self.real_B[:bs_per_gpu]
        self.forward()
        if self.opt.isTrain:
            self.compute_D_loss().backward()
            self.compute_G_loss().backward()
            if self.opt.lambda_NCE > 0.0:
                self.optimizer_F = torch.optim.Adam(
                    self.netF.parameters(), lr=self.opt.lr,
                    betas=(self.opt.beta1, self.opt.beta2),
                )
                self.optimizers.append(self.optimizer_F)

    def optimize_parameters(self):
        self.forward()
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.loss_D = self.compute_D_loss()
        self.loss_D.backward()
        self.optimizer_D.step()
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        if self.opt.netF == "mlp_sample":
            self.optimizer_F.zero_grad()
        self.loss_G = self.compute_G_loss()
        self.loss_G.backward()
        self.optimizer_G.step()
        if self.opt.netF == "mlp_sample":
            self.optimizer_F.step()

    def set_input(self, input):
        a_to_b = self.opt.direction == "AtoB"
        self.real_A = input["A" if a_to_b else "B"].to(self.device)
        self.real_B = input["B" if a_to_b else "A"].to(self.device)
        self.image_paths = input["A_paths" if a_to_b else "B_paths"]

    def forward(self):
        self.real = (
            torch.cat((self.real_A, self.real_B), dim=0)
            if self.opt.nce_idt and self.opt.isTrain else self.real_A
        )
        if self.opt.flip_equivariance:
            self.flipped_for_equivariance = self.opt.isTrain and np.random.random() < 0.5
            if self.flipped_for_equivariance:
                self.real = torch.flip(self.real, [3])
        self.fake = self.netG(self.real)
        self.fake_B = self.fake[: self.real_A.size(0)]
        if self.opt.nce_idt:
            self.idt_B = self.fake[self.real_A.size(0):]

    def compute_D_loss(self):
        pred_fake = self.netD(self.fake_B.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False).mean()
        self.pred_real = self.netD(self.real_B)
        self.loss_D_real = self.criterionGAN(self.pred_real, True).mean()
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        return self.loss_D

    def compute_G_loss(self):
        if self.opt.lambda_GAN > 0.0:
            self.loss_G_GAN = (
                self.criterionGAN(self.netD(self.fake_B), True).mean()
                * self.opt.lambda_GAN
            )
        else:
            self.loss_G_GAN = 0.0
        self.loss_NCE = (
            self.calculate_NCE_loss(self.real_A, self.fake_B)
            if self.opt.lambda_NCE > 0.0 else 0.0
        )
        if self.opt.nce_idt and self.opt.lambda_NCE > 0.0:
            self.loss_NCE_Y = self.calculate_NCE_loss(self.real_B, self.idt_B)
            loss_nce = (self.loss_NCE + self.loss_NCE_Y) * 0.5
        else:
            loss_nce = self.loss_NCE
        self.loss_G = self.loss_G_GAN + loss_nce
        return self.loss_G

    def calculate_NCE_loss(self, src, tgt):
        feat_q = self.netG(tgt, self.nce_layers, encode_only=True)
        if self.opt.flip_equivariance and self.flipped_for_equivariance:
            feat_q = [torch.flip(value, [3]) for value in feat_q]
        feat_k = self.netG(src, self.nce_layers, encode_only=True)
        feat_k_pool, sample_ids = self.netF(feat_k, self.opt.num_patches, None)
        feat_q_pool, _ = self.netF(feat_q, self.opt.num_patches, sample_ids)
        total = 0.0
        for f_q, f_k, criterion in zip(feat_q_pool, feat_k_pool, self.criterionNCE):
            total += (criterion(f_q, f_k) * self.opt.lambda_NCE).mean()
        return total / len(self.nce_layers)
