import re
import torch
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from os.path import join as pjoin
from copy import deepcopy
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

clip_version = "ViT-B/32"

# debugpy.listen(("localhost", 5678))
# print("Waiting for debugger attach")
# debugpy.wait_for_client()


@torch.no_grad()
def close_form_emb_regzero(
    proj_layers,
    concept,
    device="cpu",
    with_to_k=True,
    save_path=None,
    save_name=None,
    regeular_scale=1e-3,
    seed=123,
):
    """Close form solution for the adversarial embedding.

    Args:
        proj_matrices: List of projection matrices
        concept: Concept embedding
    """
    proj_layers = [deepcopy(l) for l in proj_layers]

    # Eq. 8 in the paper
    mat1 = torch.eye(proj_layers[0].weight.shape[1]).to(device) * regeular_scale
    mat2 = torch.zeros(
        (proj_layers[0].weight.shape[1], proj_layers[0].weight.shape[1])
    ).to(device)

    for idx_, l in enumerate(proj_layers):
        mat1 = mat1 + torch.matmul(l.weight.T, l.weight)
        mat2 = mat2 + torch.matmul(l.weight.T, proj_layers[idx_].weight)
    coefficent = torch.matmul(torch.inverse(mat1), mat2)
    adv_embedding = torch.matmul(concept, coefficent.T).unsqueeze(0)

    return concept.unsqueeze(0), adv_embedding


@torch.no_grad()
def edit_model_adversarial(
    proj_layers,
    forget_emb,
    target_emb,
    retain_emb,
    layers_to_edit=None,
    lamb=0.1,
    erase_scale=0.1,
    preserve_scale=0.1,
    with_to_k=True,
    technique="tensor",
):
    """Edit the model adversarially.

    Args:
        proj_layers: List of projection matrices
        forget_emb: List of embeddings to forget
        target_emb: List of target embeddings
        retain_emb: List of embeddings to retain
    """
    ######################## START ERASING ###################################
    for layer_num in range(len(proj_layers)):
        if (layers_to_edit is not None) and (layer_num not in layers_to_edit):
            continue

        #### prepare input k* and v*
        mat1 = lamb * proj_layers[layer_num].weight
        mat2 = lamb * torch.eye(
            proj_layers[layer_num].weight.shape[1],
            device=proj_layers[layer_num].weight.device,
        )

        for cnt, (old_emb, new_emb) in enumerate(zip(forget_emb, target_emb)):
            context = old_emb.unsqueeze(0).detach()
            new_emb = new_emb.unsqueeze(0).detach()

            values = []
            for layer in proj_layers:
                if technique == "tensor":
                    o_embs = layer(old_emb).detach()
                    u = o_embs
                    u = u / u.norm()

                    new_embs = layer(new_emb).detach()
                    new_emb_proj = (u * new_embs).sum()

                    target = new_embs - (new_emb_proj) * u
                    values.append(target.detach())
                elif technique == "replace":
                    values.append(layer(new_emb).detach())
                else:
                    values.append(layer(new_emb).detach())

            context_vector = context.reshape(context.shape[0], context.shape[1], 1)
            context_vector_T = context.reshape(context.shape[0], 1, context.shape[1])
            value_vector = values[layer_num].reshape(
                values[layer_num].shape[0], values[layer_num].shape[1], 1
            )
            for_mat1 = (value_vector @ context_vector_T).sum(dim=0)
            for_mat2 = (context_vector @ context_vector_T).sum(dim=0)
            mat1 += erase_scale * for_mat1
            mat2 += erase_scale * for_mat2

        for old_emb, new_emb in zip(retain_emb, deepcopy(retain_emb)):
            context = old_emb.unsqueeze(0).detach()
            new_emb = new_emb.unsqueeze(0).detach()

            with torch.no_grad():
                values = [layer(new_emb[:]).detach() for layer in proj_layers]

            context_vector = context.reshape(context.shape[0], context.shape[1], 1)
            context_vector_T = context.reshape(context.shape[0], 1, context.shape[1])
            value_vector = values[layer_num].reshape(
                values[layer_num].shape[0], values[layer_num].shape[1], 1
            )
            for_mat1 = (value_vector @ context_vector_T).sum(dim=0)
            for_mat2 = (context_vector @ context_vector_T).sum(dim=0)
            mat1 += preserve_scale * for_mat1
            mat2 += preserve_scale * for_mat2

        # Update the projection matrix directly
        # with side effect on the original model
        proj_layers[layer_num].weight = torch.nn.Parameter(mat1 @ torch.inverse(mat2))

    return proj_layers



def main(forget_text, retain_text, target, epochs, opt, preserve_scale=0.1):
    #######
    vae_name = get_opt(pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name, 'opt.txt'), opt.device).vae_name
    vae_opt = get_opt(pjoin(opt.checkpoints_dir, opt.dataset_name, vae_name, 'opt.txt'), opt.device)

    cond_scale = opt.cond_scale
    num_inference_timesteps = opt.num_inference_timesteps
    opt = get_opt(pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name, 'opt.txt'), opt.device)
    opt.cond_scale = cond_scale
    opt.num_inference_timesteps = num_inference_timesteps
    fixseed(opt.seed)
    
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
    trainer.optim=None
    trainer.lr_scheduler=None


    ##### END COPY #####
    proj_layers = [denoiser.word_emb]

    concept_embedding, forget_attn_mask, _ = denoiser.clip_model.encode_text(forget_text)
    retain_embedding, retain_attn_mask, _ = denoiser.clip_model.encode_text(retain_text)
    target_embedding, target_attn_mask, _ = denoiser.clip_model.encode_text([target for _ in forget_text])

    def get_eos_token(emb, mask):
        return emb.reshape(emb.size(0)*emb.size(1), -1)
        # emb: [B, T, C], mask: [B, T] with 1s for valid tokens (incl. EOS)
        eos_idx = mask.long().sum(dim=1) - 1  # [B]
        eos_idx = eos_idx.clamp(min=0)
        batch_idx = torch.arange(emb.size(0), device=emb.device)
        return emb[batch_idx, eos_idx]  # [B, C]

    concept_embedding = get_eos_token(concept_embedding, forget_attn_mask)
    retain_embedding = get_eos_token(retain_embedding, retain_attn_mask)
    target_embedding = get_eos_token(target_embedding, target_attn_mask)
    print(concept_embedding.shape, target_embedding.shape, retain_embedding.shape)
    assert target_embedding.shape == concept_embedding.shape

    # UCE
    model = edit_model_adversarial(  # implements Eq. 3 in the paper (UCE)
        proj_layers, concept_embedding, target_embedding, retain_embedding, preserve_scale=preserve_scale
    )

    trainer.save(
        f"{opt.checkpoints_dir}/{opt.dataset_name}/{opt.name}/model/UCE_{preserve_scale}.tar", 
        epoch=0, total_iter=0
    )
    

    # RECE
    for _ in range(epochs):  # 10 iterations
        adv_embedding = [
            close_form_emb_regzero(
                proj_layers, concept, device=opt.device
            )[1].squeeze()
            for concept in concept_embedding
        ]

        adv_embedding = torch.stack(adv_embedding, dim=0)
        print(adv_embedding.shape, retain_embedding.shape)

        model = edit_model_adversarial(  # implements Eq. 3 in the paper (UCE)
            proj_layers, adv_embedding, target_embedding, retain_embedding, preserve_scale=preserve_scale
        )

    trainer.save(
        f"{opt.checkpoints_dir}/{opt.dataset_name}/{opt.name}/model/RECE{epochs}_{preserve_scale}.tar", 
        epoch=0, total_iter=0
    )

    return 0

if __name__ == "__main__":
    # parser = EvalT2MOptions()
    # parser.parser.add_argument('--preserve_scale', type=float, default=0.1)
    # parser.parser.add_argument('--forget_text', type=str, nargs='+', required=True)
    # parser.parser.add_argument('--retain_text', type=str, nargs='+', default=["walk", "run", "jump", "sit", "stand"])
    # parser.parser.add_argument('--target_text', type=str, default="")
    # parser.parser.add_argument('--epochs', type=int, default=3)
    # parser.parser.add_argument('--ckpt', type=str, default="base.tar")
    # opt = parser.parse()
    # fixseed(opt.seed)
    opt = arg_parse(False)
    preserve_scale = 0.5
    forget_text = ["kick", "punch", "hit", "beat","boxe", "boxing", "jab", "uppercut","hook"]
    retain_text = ["a person is walking", "a person is running", "a person is jumping", "a person is sitting", "a person is standing", 'dance', 'walk', 'move', 'run', 'jump']
    target_text = ""
    epochs = 1
    


    main(
        forget_text=forget_text,
        retain_text=retain_text,
        target=target_text,
        epochs=epochs,
        opt=opt,
        preserve_scale=preserve_scale,
    )
