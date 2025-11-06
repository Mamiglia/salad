import re
import torch
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from os.path import join as pjoin
from diffusers import DDIMScheduler

from models.vae.model import VAE
from models.denoiser.model import Denoiser
from models.denoiser.trainer import DenoiserTrainer
from options.denoiser_option import arg_parse

from utils.get_opt import get_opt
from utils.fixseed import fixseed

from motion_loaders.dataset_motion_loader import get_dataset_motion_loader
from models.t2m_eval_wrapper import EvaluatorModelWrapper


def load_vae(vae_opt):
    print(f'Loading VAE Model {vae_opt.name}')

    model = VAE(vae_opt)
    ckpt = torch.load(pjoin(vae_opt.checkpoints_dir, vae_opt.dataset_name, vae_opt.name, 'model', 'net_best_fid.tar'),
                            map_location='cpu')
    model.load_state_dict(ckpt["vae"])
    model.freeze()
    return model


def load_denoiser(opt, vae_dim):
    print(f'Loading Denoiser Model {opt.name}')
    denoiser = Denoiser(opt, vae_dim)
    ckpt = torch.load(pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name, 'model', 'net_best_fid.tar'),
                            map_location='cpu')
    missing_keys, unexpected_keys = denoiser.load_state_dict(ckpt["denoiser"], strict=False)
    assert len(unexpected_keys) == 0
    assert all([k.startswith('clip_model.') for k in missing_keys])
    return denoiser



def reduce_sum(tensor, dim=-1):
    # compute sum over all dimensions except specified dim
    dims = list(range(tensor.dim()))
    dims.remove(dim)
    n_elements = np.prod([tensor.size(d) for d in dims])
    return n_elements, tensor.sum(dim=dims)

class ActivationRecorder:
    def __init__(self, model, layer_names):
        self.model = model
        self.layer_names = layer_names
        self.activations = {}
        self.num_samples = defaultdict(int)
        self.hooks = []

    def register_hooks(self):
        for name, module in self.model.named_modules():
            if name in self.layer_names:
                hook = module.register_forward_hook(self.save_activation(name))
                self.hooks.append(hook)

    def save_activation(self, name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
                
            # print(f"Recording activation for layer: {name}, output shape: {output.shape}")
            if not name in self.activations:
                self.activations[name] = torch.zeros(output.shape[-1], device='cpu')
            n_elements, output = reduce_sum(output, dim=2)
            self.activations[name] += output.detach().cpu()
            self.num_samples[name] += n_elements
        return hook

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

    def __enter__(self):
        self.register_hooks()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove_hooks()


if __name__ == '__main__':
    opt = arg_parse(False)
    #######
    pos_split = 'kw_splits/test-w-kick'
    neg_split = 'kw_splits/test-wo-kick'
    
    #######
    vae_name = get_opt(pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name, 'opt.txt'), opt.device).vae_name
    vae_opt = get_opt(pjoin(opt.checkpoints_dir, opt.dataset_name, vae_name, 'opt.txt'), opt.device)

    cond_scale = opt.cond_scale
    num_inference_timesteps = opt.num_inference_timesteps
    opt = get_opt(pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name, 'opt.txt'), opt.device)
    opt.cond_scale = cond_scale
    opt.num_inference_timesteps = num_inference_timesteps
    fixseed(opt.seed)
    
    # evaluation setup
    dataset_opt_path = f"checkpoints/{opt.dataset_name}/Comp_v6_KLD005/opt.txt"
    wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'))
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)
    pos_loader, _ = get_dataset_motion_loader(dataset_opt_path, opt.batch_size, pos_split, device=opt.device)
    neg_loader, _ = get_dataset_motion_loader(dataset_opt_path, opt.batch_size, neg_split, device=opt.device)

    # models & noise scheduler
    vae_model = load_vae(vae_opt).to(opt.device)
    denoiser = load_denoiser(opt, vae_opt.latent_dim).to(opt.device)
    scheduler = DDIMScheduler(
        num_train_timesteps=opt.num_train_timesteps,
        beta_start=opt.beta_start,
        beta_end=opt.beta_end,
        beta_schedule=opt.beta_schedule,
        prediction_type=opt.prediction_type,
        clip_sample=False,
    )
    
    trainer = DenoiserTrainer(opt, denoiser, vae_model, scheduler)

    
    ###
 ### Activation Recording Setup  ###
    layer_regex = r'cross_attn.Wo'
    modules = dict(denoiser.named_modules())
    layer_names = [layer for layer in modules.keys() if re.search(layer_regex, layer)]
    assert len(layer_names) > 0, f"No layers matched the regex: {layer_regex}"
    pos_recorder = ActivationRecorder(denoiser, layer_names=layer_names)
    neg_recorder = ActivationRecorder(denoiser, layer_names=layer_names)
    
    

    # Forward through datasets
    with pos_recorder:
        trainer.test(eval_wrapper, pos_loader, 1,
                save_dir=pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name, 'eval'), cal_mm=False, save_motion=False)
    with neg_recorder:
        trainer.test(eval_wrapper, neg_loader, 1,
                save_dir=pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name, 'eval'), cal_mm=False, save_motion=False)

    # Compute average activations
    pos_avg_activations = {name: pos_recorder.activations[name] / pos_recorder.num_samples[name] for name in layer_names}
    neg_avg_activations = {name: neg_recorder.activations[name] / neg_recorder.num_samples[name] for name in layer_names}
    
    # Print results
    for name in layer_names:
        pos_act = pos_avg_activations[name]
        neg_act = neg_avg_activations[name]
        diff = pos_act - neg_act
        print(f"Layer: {name}")
        print(f"  Pos Avg Activation: {pos_act}")
        print(f"  Neg Avg Activation: {neg_act}")
        print(f"  Difference: {diff}")
        
        with torch.no_grad():
            modules[name].bias.copy_(modules[name].bias + 5 * diff.to(opt.device))

    trainer.optim = None
    trainer.lr_scheduler = None
    trainer.save(
        pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name, 'model', 'steered.tar'),
        -1, -1        
    )