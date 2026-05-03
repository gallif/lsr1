from itertools import chain
import numpy as np
import io
from PIL import Image
import os
import plotly.express as px
import pandas as pd
import wandb
from copy import copy

import time
import torch
import torch.optim as optim
import pytorch_lightning as pl
from smplx import SMPL

from ..utils.loss import weighted_L1_loss
from ..utils.camera import PinholeCamera
from ..utils.pose_util import compute_MPJPE, normalize_p2d
from ..utils.network import MLPBlock
from ..optimizer import LSR1Optimizer
from ..utils.smpl import SMPLJointMapper

class LOPTModel(pl.LightningModule, pl.core.hooks.CheckpointHooks):
    def __init__(self, opt):
        super().__init__()
        self.automatic_optimization = False

        # Constants
        NUM_JOINT = 18
        NUM_BETA = 10
        NUM_THETA = 24 * 3
        NUM_HIDDEN = 1024
        self.NUM_STAGES = opt.inner_num_steps
        self.opt = opt
        
        # model
        self.training_modules = [
            'shape_init_net', 'pose_init_net' 
        ]
        self.pose_init_net = MLPBlock(NUM_JOINT * 2, NUM_THETA, NUM_HIDDEN)
        self.shape_init_net = MLPBlock(NUM_JOINT * 2, NUM_BETA, NUM_HIDDEN)
        
        self.pose_upd_net = LSR1Optimizer(opt.lsr1)
        self.shape_upd_net = LSR1Optimizer(opt.lsr1)
        self.training_modules.extend(['shape_upd_net', 'pose_upd_net'])

        # other submodules
        self.camera = PinholeCamera(
            fx=256, fy=256, cx=0, cy=0, R=np.eye(3), t=(0, 0, -6))
        self.body_model = SMPL(model_path=os.path.join(opt.path,'smplx_models/smpl'), gender="male")
        self.joint_mapper = SMPLJointMapper(opt.path)
        self.criterionLoss = torch.nn.L1Loss()

        self.confidence_threshold = 0.01

        self.confidence_dist = torch.distributions.Bernoulli(0.8)
        self.joint_weights = torch.ones(1, 18, 3).to("cuda")
        self.joint_weights[:, [4, 7, 10, 13]] = 8
        self.joint_weights[:, [3, 6, 9, 12]] = 4
        self.joint_weights[:, [2, 5, 8, 11]] = 2


    def forward(self, batch, chidx=0, thetas0=None, betas0=None):
        torch.set_grad_enabled(True)

        if self.training:
            self._set_grads(True)
        else:
            self._set_grads(False)

        x = batch["joint2d"].flatten(start_dim=1)
        confidence = batch["confidence"]

        thetas_history = []
        betas_history = []
        joint2d_history = []
        joint3d_history = []
        thetas_sec_history = []
        betas_sec_history = []
        thetas_lr_history = []
        betas_lr_history = []
        reprojections_history = []
        
        # clear buffers            
        self.pose_upd_net.reset()
        self.shape_upd_net.reset()
        
        if chidx == 0:
            thetas0 = self.pose_init_net(x)
            betas0 = self.shape_init_net(x)
            joint3d0 = self.get_joint3d(thetas0, betas0)
            joint2d0 = self.camera(joint3d0)
            joint2d0 = normalize_p2d(joint2d0,
                                    confidence,
                                    self.confidence_threshold)
            thetas_history.append(thetas0)
            betas_history.append(betas0)
            joint2d_history.append(joint2d0)
            joint3d_history.append(joint3d0)
            thetas_sec_history.append((torch.zeros_like(thetas0),torch.zeros_like(thetas0)))
            betas_sec_history.append((torch.zeros_like(betas0),torch.zeros_like(betas0)))

        thetas = thetas0.detach().requires_grad_(True)
        betas = betas0.detach().requires_grad_(True)
        
        end = time.time()
        
        for i in range(self.NUM_STAGES):
            # inner loss computation
            joint3d = self.get_joint3d(thetas, betas)
            joint2d = self.camera(joint3d)
            joint2d = normalize_p2d(joint2d,
                                    confidence,
                                    self.confidence_threshold)
            reprojection_error = self.opt.inner_loss_weight * weighted_L1_loss(
                joint2d, x,
                confidence.unsqueeze(-1).expand(-1, -1, 2))
            reprojection_error.backward(retain_graph=True)

            # update step
            dthetas, th_sec, th_lr = self.pose_upd_net(thetas.detach(), thetas.grad.detach())
            dbetas, be_sec, be_lr = self.shape_upd_net(betas.detach(), betas.grad.detach())
            thetas = thetas + dthetas
            betas = betas + dbetas
            
            # for logging and meta-loss
            thetas_lr_history.append(th_lr)
            betas_lr_history.append(be_lr)
            reprojections_history.append(reprojection_error.detach())
            thetas_history.append(thetas)
            betas_history.append(betas)
            joint2d_history.append(joint2d)
            joint3d_history.append(joint3d)
            thetas_sec_history.append(th_sec)
            betas_sec_history.append(be_sec)
            thetas = thetas.detach().requires_grad_(True)
            betas = betas.detach().requires_grad_(True)

            if self.opt.verbose:
                i_str = f'step = {i+1}/{self.NUM_STAGES}: Time = {time.time()-end}, Mem = {torch.cuda.memory_allocated()}'
                print(i_str)
            
            end = time.time()

        return {
            "betas_history": betas_history,
            "thetas_history": thetas_history,
            "joint2d_history": joint2d_history,
            "joint3d_history": joint3d_history,
            "thetas_sec_history" : thetas_sec_history,
            "betas_sec_history": betas_sec_history,
            "thetas_lr_history": thetas_lr_history,
            "betas_lr_history": betas_lr_history,
            "reprojections_history": reprojections_history
        }


    def configure_optimizers(self):
        params = chain(*[
            getattr(self, module).parameters()
            for module in self.training_modules
        ])
        meta_lr = self.opt.meta_optim.meta_lr
        schd_step_size = self.opt.meta_optim.schd_step_size
        schd_gamma = self.opt.meta_optim.schd_gamma

        self.optimizer = optim.AdamW(params, lr=meta_lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer,
                                                   step_size=schd_step_size,
                                                   gamma=schd_gamma)
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": self.scheduler,
        }


    def training_step(self, batch, batch_idx, **kwargs):
        # data augmentation
        joint3d = self.get_joint3d(batch["thetas"], betas=batch["betas"])
        joint2d = self.camera(joint3d)
        confidence = self.confidence_dist.sample(
            joint2d.shape[:2]).to(self.device).float()
        joint2d = normalize_p2d(joint2d,
                                confidence,
                                threshold=self.confidence_threshold)
        joint2d[confidence < self.confidence_threshold, :] = 0.5
        batch["joint2d"] = joint2d
        batch["joint3d"] = joint3d
        batch["confidence"] = confidence

        # meta-training
        losses_all = []
        thetas0 = None
        betas0 = None

        for ch in range(self.opt.num_chunks):
            predicts = self.forward(batch, ch, thetas0, betas0)

            # meta-loss
            losses = self.compute_loss(predicts, batch)

            # meta-optimization
            self.optimizer.zero_grad()
            losses["loss_total"].backward()

            # clip gradients
            params = chain(*[
                getattr(self, module).parameters()
                for module in self.training_modules
                ])
            torch.nn.utils.clip_grad_norm_(params, max_norm=1)

            self.optimizer.step()

            thetas0 = predicts['thetas_history'][-1].detach()
            betas0 = predicts['betas_history'][-1].detach()

            losses_all.append(losses)

        # logging
        for k, v in losses.items():
            if "loss_total" in k:
                prog_bar = True
            else:
                prog_bar = False
            self.log(f"train/{k}", v, on_step=True, on_epoch=False, prog_bar=prog_bar)
        
        for opt_ in ['betas','thetas']:
            self.log(f"train/mean_lr_{opt_}", torch.stack(predicts[f'{opt_}_lr_history'],dim=0).mean().item(), on_step=True, on_epoch=False)
        lr = self.optimizer.param_groups[0]['lr']
        self.log("trainer/meta_lr", lr)
        
        return {
            k: v.detach() for (k, v) in losses.items()
        }


    def training_epoch_end(self, *args, **kwargs):
        self.scheduler.step()


    def validation_step(self, batch, batch_idx):
        batch["joint2d"] = normalize_p2d(
            batch["joint2d"],
            batch["confidence"],
            threshold=self.confidence_threshold
        )
        batch["confidence"] = (batch["confidence"] >=
                               self.confidence_threshold).float()
        # forward pass
        outputs = self.forward(batch)
        
        # evaluate
        losses_vs_iteratios = []
        for betas_, thetas_ in zip(outputs["betas_history"],outputs["thetas_history"]):
            predicts = {
                "betas": betas_,
                "thetas": thetas_,
            }
            losses = self.compute_valid_MPJPE(predicts, batch)
            losses_vs_iteratios.append(losses)
        
        # log errors
        losses_out = copy(losses_vs_iteratios[-1])

        for i,loss_i in enumerate(losses_vs_iteratios):
            if i == 0:
                losses_out.update({f'{k}_iterations' : [v] for k,v in loss_i.items()})
            else:
                for k,v in loss_i.items():
                    losses_out[f'{k}_iterations'] += [v]
        
        for k,v in losses_out.items():
            if '_iterations' in k:
                losses_out[k] = torch.stack(v,dim=1)
        
        # log losses
        losses_out.update({'reprojections_iterations': torch.stack(outputs['reprojections_history']).unsqueeze(0)})

        return {
            k: v.detach() for (k, v) in losses_out.items()
        }


    def validation_epoch_end(self, outputs):
        total_loss_val = {}
        for output in outputs:
            for k, v in output.items():
                if k not in total_loss_val:
                    total_loss_val[k] = [v]
                else:
                    total_loss_val[k].append(v)
        
        for k, v in total_loss_val.items():
            if '_iterations' in k:
                metric = torch.cat(v,dim=0).mean(dim=0).cpu().tolist()
                df = pd.DataFrame({"Iteration": range(len(metric)), k : metric})
                fig = px.line(df, x = "Iteration", y = k, title = k)
                self.logger.experiment.log({f"val/{k}": wandb.Image(Image.open(io.BytesIO(fig.to_image(format="png"))))})

            else:
                self.log(f"val/{k}", torch.cat(v).mean())


    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)


    def test_epoch_end(self, outputs):
        total_loss_val = {}
        for output in outputs:
            for k, v in output.items():
                if k not in total_loss_val:
                    total_loss_val[k] = [v]
                else:
                    total_loss_val[k].append(v)
        
        for k, v in total_loss_val.items():
            if '_iterations' in k:
                metric = torch.cat(v,dim=0).mean(dim=0).cpu().tolist()
                df = pd.DataFrame({"Iteration": range(len(metric)), k : metric})
                fig = px.line(df, x = "Iteration", y = k, title = k)
                self.logger.experiment.log({f"test/{k}": wandb.Image(Image.open(io.BytesIO(fig.to_image(format="png"))))})
                
                metrics_log = open(os.path.join(self.trainer.default_root_dir, f'{k}.txt'), 'w')
                [metrics_log.write(f"{value}\n") for value in metric]
                metrics_log.close()

            else:
                self.log(f"test/{k}", torch.cat(v).mean())


    def compute_loss(self, predicts, targets):
        steps = len(predicts["betas_history"])

        loss_joint2d_total = 0
        loss_joint3d_total = 0
        loss_thetas_total = 0
        loss_betas_total = 0
        thetas_sec_total = 0
        betas_sec_total = 0

        for step in range(steps):
            thetas_pred = predicts["thetas_history"][step]
            betas_pred = predicts["betas_history"][step]
            joint2d_pred = predicts["joint2d_history"][step]
            joint3d_pred = predicts["joint3d_history"][step]

            loss_joint2d_total = loss_joint2d_total + weighted_L1_loss(joint2d_pred, targets["joint2d"], self.joint_weights[:, :, :2])
            loss_joint3d_total = loss_joint3d_total + weighted_L1_loss(joint3d_pred, targets["joint3d"], self.joint_weights)
            loss_thetas_total = loss_thetas_total + self.criterionLoss(thetas_pred, targets["thetas"])
            loss_betas_total = loss_betas_total + self.criterionLoss(betas_pred, targets["betas"])
            
            loss_total = (loss_joint2d_total * 0.3 + loss_joint3d_total * 0.3 +
                      loss_thetas_total * 1.0 + loss_betas_total * 1.0) / steps

            sec_th = predicts["thetas_sec_history"][step]
            sec_be = predicts["thetas_sec_history"][step]
            bkdg_th, dx_th = sec_th
            bkdg_be, dx_be = sec_be
            thetas_sec_total = thetas_sec_total + ((bkdg_th - dx_th)**2).mean()
            betas_sec_total = betas_sec_total + ((bkdg_be - dx_be)**2).mean()
            sec_total = (thetas_sec_total + betas_sec_total) / steps 
            loss_total = loss_total + sec_total    

        return {
            "loss_betas": loss_betas_total / steps,
            "loss_thetas": loss_thetas_total / steps,
            "loss_joint2d": loss_joint2d_total / steps,
            "loss_joint3d": loss_joint3d_total / steps,
            "sec_total": sec_total,
            "loss_total": loss_total,
        }

    def compute_valid_MPJPE(self, x, y):
        # SPIN
        joint3d_pr = self.get_joint3d(x["thetas"], x["betas"], protocol='SPIN')
        joint3d_gt = self.get_joint3d(y["thetas"], y["betas"], protocol='SPIN')
        err_spin = compute_MPJPE(joint3d_pr, joint3d_gt * 1000)
        
        return {
            "MPJPE_aligned": torch.from_numpy(np.asarray(err_spin)),
        }


    def get_joint3d(self, theta, betas, protocol="COCO18"):
        smpl_output = self.body_model.forward(betas=betas,
                                              body_pose=theta[:, 3:],
                                              global_orient=theta[:, :3])
        joint3d = self.joint_mapper(smpl_output.joints,
                                    smpl_output.vertices,
                                    output_format=protocol)
        return joint3d
    

    def _set_grads(self, mode=True):
        params = chain(*[
            getattr(self, module).parameters()
            for module in self.training_modules
        ])
        for param in params:
            param.requires_grad = mode

