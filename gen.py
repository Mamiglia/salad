# %%
from t2m import Text2Motion
from utils.get_opt import get_opt
from utils.fixseed import fixseed

import torch
import numpy as np
from os.path import join as pjoin

# %%
denoiser_name = "t2m_denoiser_vpred_vaegelu"
# denoiser_name = "denoiser_aaai"
dataset_name = "t2m"
generator = Text2Motion(denoiser_name, dataset_name)

opt = generator.opt
wrapper_opt = get_opt(opt.dataset_opt_path, torch.device("cuda"))
mean = np.load(pjoin(wrapper_opt.meta_dir, "mean.npy"))
std = np.load(pjoin(wrapper_opt.meta_dir, "std.npy"))

# %% [markdown]
# ### Original

# %%
fixseed(42)
# src_text = "a man is running, jumps and then crouches"
# src_text = 'a man kicks, jumps and finally punches to the front'
src_text = "a person is doing martial arts with punches and kicks"
m_lens = 64
cfg_scale = 7.5
num_inference_timesteps = 50

init_noise, src_motion, (sa, ta, ca) = generator.generate(src_text,
                                                          m_lens,
                                                          cfg_scale,
                                                          num_inference_timesteps)

# %% [markdown]
# ### Visualize

# %%
# %load_ext autoreload
# %autoreload 2
import os
from os.path import join as pjoin
import torch
import numpy as np
from utils.motion_process import recover_from_ric
from utils.plot_script import plot_3d_motion
from utils.get_opt import get_opt

def plot_t2m(data, text, filename):
    os.makedirs("edit_result", exist_ok=True)
    #data = data[:m_lens[0].item()]
    data = data[:m_lens]
    joint = recover_from_ric(torch.from_numpy(data).float(), opt.joints_num).numpy()
    save_path = pjoin("edit_result", f"{filename}.mp4")
    plot_3d_motion(save_path, opt.kinematic_chain, joint, title=text, fps=20)

    np.save(pjoin("edit_result", f"{filename}_pos.npy"), joint)
    np.save(pjoin("edit_result", f"{filename}_feats.npy"), data)
    
# mean and std for de-normalization
wrapper_opt = get_opt(opt.dataset_opt_path, torch.device('cuda'))
mean = np.load(pjoin(wrapper_opt.meta_dir, 'mean.npy'))
std = np.load(pjoin(wrapper_opt.meta_dir, 'std.npy'))

# %% [markdown]
# ### Plot Motions

# %%
src_motion = src_motion.detach().cpu().numpy() * std + mean
plot_t2m(src_motion[0], src_text, "src")
# %% [markdown]
# ### Video Visualization

