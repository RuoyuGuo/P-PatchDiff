import math
import torch
from torch import device, nn, einsum
import torch.nn.functional as F
from inspect import isfunction
from functools import partial
import numpy as np
from tqdm import tqdm
import random
from basicsr.utils.registry import ARCH_REGISTRY
from scripts.utils import pad_tensor, pad_tensor_back

from torchvision.transforms import functional as AugF
from torchvision.transforms.functional import crop
import torchvision

def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(
        linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas



def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3, stretch=False):
    if schedule == 'quad':
        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5,
                            n_timestep, dtype=np.float64) ** 2
    elif schedule == 'linear':
        betas = np.linspace(linear_start, linear_end,
                            n_timestep, dtype=np.float64)
        if stretch:
            from scripts.utils import stretch_linear
            betas = stretch_linear(betas)
    elif schedule == 'warmup10':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.1)
    elif schedule == 'warmup50':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.5)
    elif schedule == 'const':
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(n_timestep,
                                 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) /
            n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(schedule)
    return betas


# gaussian diffusion trainer class

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


@ARCH_REGISTRY.register()
class PGCGaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        image_size,
        channels=3,
        loss_type='l1',
        conditional=True,
        schedule_opt=None,
        color_fn=None,
        ppredictor_fn=None,
        pencoder_fn=None,
        colorp_fn=None,
        color_limit=None,
        resize_all=False,
        resize_res=-1,
        progressive_list=[1],
        stride_list=[1],
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.denoise_fn = denoise_fn
        self.loss_type = loss_type
        self.conditional = conditional
        self.color_fn = color_fn
        
        self.pencoder_fn   = pencoder_fn
        self.colorp_fn= colorp_fn
        
        if schedule_opt is not None:
            pass
        if color_limit:
            self.color_limit = color_limit
        else:
            self.color_limit = -1
        self.resize_all = resize_all
        self.resize_res = resize_res
        self.progressive_list = progressive_list
        self.stride_list = stride_list

    def set_loss(self, device):
        if self.loss_type == 'l1':
            self.loss_func = nn.L1Loss(reduction='sum').to(device)
        elif self.loss_type == 'l2':
            self.loss_func = nn.MSELoss(reduction='sum').to(device)
        else:
            raise NotImplementedError()

    def set_new_noise_schedule(self, schedule_opt, device):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)

        betas = make_beta_schedule(
            schedule=schedule_opt['schedule'],
            n_timestep=schedule_opt['n_timestep'],
            linear_start=schedule_opt['linear_start'],
            linear_end=schedule_opt['linear_end'],
            stretch=schedule_opt.get('stretch', False))
        
        # make patch schedule
        assert schedule_opt['n_timestep'] % len(self.progressive_list) == 0
        segment_length = schedule_opt['n_timestep'] // len(self.progressive_list)
        patch_schedule = []
        for patch_scale in self.progressive_list:
            patch_schedule += [patch_scale] * segment_length
        self.patch_schedule = np.array(patch_schedule)
        # patch_schedule = torch.from_numpy(patch_schedule).to(device)
        # self.register_buffer('patch_schedule', patch_schedule)
        
#         # make stride schedule for sampling
#         assert schedule_opt['n_timestep'] % len(self.stride_list) == 0
#         segment_length = schedule_opt['n_timestep'] // len(self.stride_list)
#         stride_schedule = []
#         for stride_scale in self.stride_list:
#             stride_schedule += [stride_scale] * segment_length
#         stride_schedule = np.array(stride_schedule)
#         # stride_schedule = torch.from_numpy(stride_schedule).to(device)
#         self.register_buffer('stride_schedule', stride_schedule)        
        
        
        betas = betas.detach().cpu().numpy() if isinstance(
            betas, torch.Tensor) else betas
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)

        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        self.sqrt_alphas_cumprod_prev = np.sqrt(
            np.append(1., alphas_cumprod))

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev',
                             to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod',
                             to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod',
                             to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * \
            (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance',
                             to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(
            np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))



    def _extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = a.to(t.device).gather(0, t).float()
        out = out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
        return out
    
    def _overlapping_grid_indices(self, xt, p_size, r):
        _, c, h, w = xt.shape
        h_list = [i for i in range(0, h - p_size + 1, r)]
        w_list = [i for i in range(0, w - p_size + 1, r)]

        if h_list[-1] != h-p_size:
            h_list.append(h-p_size)
        if w_list[-1] != w-p_size:
            w_list.append(w-p_size)
        
        corners = [(i, j) for i in h_list for j in w_list]
        
        return corners
    
    def _patchify_pred(self, x_in, sample_img, noise_level, p_size, stride):
        # print(f'Processing patch size {p_size}.')
        with torch.no_grad():
            #generate patches coord
            corners = self._overlapping_grid_indices(x_in, p_size, stride)
            
            #generate empty tensor,  weight
            et_output =torch.zeros((x_in.shape[0], 3, x_in.shape[2], x_in.shape[3]), dtype=x_in.dtype, layout=x_in.layout, device=x_in.device)
            x_grid_mask = torch.zeros_like(et_output)
            for (hi, wi) in corners:
                x_grid_mask[:, :, hi:hi + p_size, wi:wi + p_size] += 1
            
            #concate patches along the batch channel
            x_in_patch = torch.cat([crop(x_in, hi, wi, p_size, p_size) for (hi, wi) in corners], dim=0)
            sample_img_patch = torch.cat([crop(sample_img, hi, wi, p_size, p_size) for (hi, wi) in corners], dim=0)
            
            #define batch_size 64 for 64, 32 for 96, 16 for 128, 8 for 160,  8 for 192,
            if p_size >= 160:
                manual_batching_size = 8
            elif p_size >= 128:
                manual_batching_size = 16
            elif p_size >= 96:
                manual_batching_size = 32
            elif p_size >= 64:
                manual_batching_size = 64
            
            resized_lq = AugF.resize(x_in[:,:3], (p_size, p_size))
            colorp_out = self.colorp_fn(resized_lq)
            encoder_input = torch.cat([torch.mean(resized_lq, dim=1, keepdim=True), 
                                       torch.max(resized_lq, dim=1, keepdim=True)[0]], dim=1)
            
            pmap_feats_origin = self.pencoder_fn(torch.cat([colorp_out, encoder_input], dim=1), noise_level)
            
           
            expand_resized_lq = resized_lq.repeat(manual_batching_size, 1, 1, 1)
            expand_noise_level = noise_level.repeat(manual_batching_size, 1)#.to(x_in.device)
            
            for i in range(0, len(corners), manual_batching_size):
                #if not the last batch
                if x_in_patch[i:i+manual_batching_size].shape[0] == expand_resized_lq.shape[0]:
                     
                    pmap_feats = [e.repeat(manual_batching_size, 1, 1, 1) for e in pmap_feats_origin]

                    outputs = self.denoise_fn(torch.cat([x_in_patch[i:i+manual_batching_size], expand_resized_lq, 
                                                         sample_img_patch[i:i+manual_batching_size]], dim=1), expand_noise_level, pmap_feats)
                else:
                    last_batch_size = x_in_patch[i:i+manual_batching_size].shape[0]
                    last_expand_resized_lq = resized_lq.repeat(last_batch_size, 1, 1, 1)
                    last_expand_noise_level = noise_level.repeat(last_batch_size, 1)#.to(x_in.device)
                    
                    
                    pmap_feats = [e.repeat(last_batch_size, 1, 1, 1) for e in pmap_feats_origin]
                    
                    outputs = self.denoise_fn(torch.cat([x_in_patch[i:i+manual_batching_size], last_expand_resized_lq, 
                                                         sample_img_patch[i:i+manual_batching_size]], dim=1), last_expand_noise_level, pmap_feats)
                    
                for idx, (hi, wi) in enumerate(corners[i:i+manual_batching_size]):
                    et_output[0, :, hi:hi + p_size, wi:wi + p_size] += outputs[idx]
            
            et = torch.div(et_output, x_grid_mask)
            pred_noise = et
            
        return pred_noise
        
    # use ddim to sample
    @torch.no_grad()
    def ddim_pyramid_sample(
        self,
        x_in,
        progressive_list,
        stride_list,
        ddim_timesteps=50,
        ddim_discr_method="uniform",
        ddim_eta=0.0,
        clip_denoised=True,
        continous=False,
        return_x_recon=False,
        return_pred_noise=False,
        return_all=False,
        pred_type='noise',
        clip_noise=False,
        save_noise=False,
        color_gamma=None,
        color_times=1,
        fine_diffV2=False,
        fine_diffV2_st=200,
        fine_diffV2_num_timesteps=20,
        do_some_global_deg=False,
        use_up_v2=False):

        #assert len(pyramid_list) == ddim_timesteps, f'len(pyramid_list):{len(pyramid_list)} != ddim_timesteps{ddim_timesteps}'
        
        #generate progressive list
        segment_length = ddim_timesteps // len(progressive_list)
        patch_schedule = []
        for patch_scale in progressive_list:
            patch_schedule += [patch_scale] * segment_length
        
        stride_schedule = []
        for stride_scale in stride_list:
            stride_schedule += [stride_scale] * segment_length 
        
        if return_all:
            assert not (return_x_recon or return_pred_noise), "[return_x_recon, return_pred_noise, return_all], choose one or not!"
        assert not (return_x_recon and return_pred_noise), "[return_x_recon, return_pred_noise, return_all], choose one or not!"
        
        # make ddim timestep sequence
        if ddim_discr_method == 'uniform':
            c = self.num_timesteps // ddim_timesteps
            ddim_timestep_seq = list(reversed(range(self.num_timesteps - 1, -1, -c)))
            ddim_timestep_seq = np.asarray(ddim_timestep_seq)
        elif ddim_discr_method == 'quad':
            ddim_timestep_seq = (
                (np.linspace(0, np.sqrt(self.num_timesteps * .8), ddim_timesteps)) ** 2
            ).astype(int)
        else:
            raise NotImplementedError(f'There is no ddim discretization method called "{ddim_discr_method}"')
        # previous sequence
        ddim_timestep_prev_seq = np.append(np.array([-1]), ddim_timestep_seq[:-1])
        
        device = x_in.device
        b, c, h, w = x_in[:, :3, :, :].shape
        
        
        # init_h = h // pyramid_list[-1]
        # init_w = w // pyramid_list[-1]
        # start from pure noise (for each example in the batch)
        
        # print(x_in.shape)
        
        sample_img = torch.randn((b, c, h, w), device=device)
        sample_inter = (1 | (ddim_timesteps//10))
        ret_img = x_in[:, :3, :, :]
        for i in tqdm(reversed(range(0, ddim_timesteps)), desc='sampling loop time step', total=ddim_timesteps):
            if return_all and i % sample_inter == 0:
                all_process = [sample_img]
            t = torch.full((b,), ddim_timestep_seq[i], device=device, dtype=torch.long)
            
            noise_level = torch.FloatTensor([self.sqrt_alphas_cumprod_prev[t + 1]]).to(x_in.device)
            # noise_level = torch.FloatTensor([self.sqrt_alphas_cumprod_prev[t + 1]]).repeat(b, 1).to(x_in.device)
            # noise_level = [self.sqrt_alphas_cumprod_prev[t + 1]]
            
            prev_t = torch.full((b,), ddim_timestep_prev_seq[i], device=device, dtype=torch.long)
            
            # get current and previous alpha_cumprod
            alpha_cumprod_t = self._extract(self.alphas_cumprod, t, sample_img.shape)
            if i == 0:
                alpha_cumprod_t_prev = torch.ones_like(alpha_cumprod_t)
            else:
                alpha_cumprod_t_prev = self._extract(self.alphas_cumprod, prev_t, sample_img.shape)
            
            #patchify predict
            if pred_type == 'noise':
                # 2. predict noise using model
                # pred_noise = self.denoise_fn(torch.cat([x_in, x_in[:,:3], sample_img], dim=1), noise_level)
                pred_noise = self._patchify_pred(x_in, sample_img, noise_level, patch_schedule[i], stride_schedule[i])
                
                if clip_noise:
                    pred_noise = torch.clamp(pred_noise, -1, 1)
                
                # 3. get the predicted x_0
                pred_x0 = (sample_img - torch.sqrt((1. - alpha_cumprod_t)) * pred_noise) / torch.sqrt(alpha_cumprod_t)
                
            else:
                assert False, "only pred noise"

            if return_all and i % sample_inter == 0:
                all_process.append(F.interpolate(pred_x0, (h, w)))

            if clip_denoised:
                pred_x0 = torch.clamp(pred_x0, min=-1., max=1.)
            
            sample_already = False

            # if i != 0 and pyramid_list[i] != pyramid_list[i - 1]:
            #     upsample_scale = pyramid_list[i] // pyramid_list[i - 1]
            #     pred_x0 = F.interpolate(pred_x0, scale_factor=upsample_scale)
            #     pred_noise = F.interpolate(pred_noise, scale_factor=upsample_scale)
            #     if use_up_v2:
            #         noise = torch.randn_like(pred_x0)
            #         sample_img = self.sqrt_alphas_cumprod[prev_t] * pred_x0 +  \
            #                     self.sqrt_one_minus_alphas_cumprod[prev_t] * noise
            #         sample_already = True
            if i != 0 and patch_schedule[i] != patch_schedule[i-1] and use_up_v2:
                noise = torch.randn_like(pred_x0)
                sample_img = self.sqrt_alphas_cumprod[prev_t] * pred_x0 +  \
                            self.sqrt_one_minus_alphas_cumprod[prev_t] * noise
                sample_already = True
                
            if not sample_already:
                # compute variance: "sigma_t(η)" -> see DDIM formula (16)
                # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
                sigmas_t = ddim_eta * torch.sqrt(
                    (1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t) * (1 - alpha_cumprod_t / alpha_cumprod_t_prev))
                
                # compute "direction pointing to x_t" of DDIM formula (12)
                pred_dir_xt = torch.sqrt(1 - alpha_cumprod_t_prev - sigmas_t**2) * pred_noise
                
                # compute x_{t-1} of DDIM formula (12)
                x_prev = torch.sqrt(alpha_cumprod_t_prev) * pred_x0 + pred_dir_xt + sigmas_t * torch.randn_like(pred_x0)

                sample_img = x_prev
            
            if return_all and i % sample_inter == 0:
                all_process.append(F.interpolate(sample_img, (h, w)))

            if i % sample_inter == 0:
                if return_x_recon:
                    ret_img = torch.cat([ret_img, pred_x0], dim=0)
                elif return_pred_noise:
                    ret_img = torch.cat([ret_img, pred_noise], dim=0)
                elif return_all:
                    ret_img = torch.cat([ret_img, torch.cat(all_process, dim=0)], dim=0)
                else:
                    ret_img = torch.cat([ret_img, sample_img], dim=0)
        if continous:
            return ret_img
        else:
            return sample_img
    


    def q_sample(self, x_start, continuous_sqrt_alpha_cumprod, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        # random gama
        return (
            continuous_sqrt_alpha_cumprod * x_start +
            (1 - continuous_sqrt_alpha_cumprod**2).sqrt() * noise
        )

    def p_losses_patch(self, x_HR, x_SR, noise=None, different_t_in_one_batch=False, clip_noise=False, t_range=None, cs_on_shift=False,
                    cs_shift_range=None, t_border=1000, input_mode=None, crop_size=None, divide=None,
                    shift_x_recon_detach=False, shift_x_recon_detach_range=0.47, frozen_denoise=False, down_uniform=False, down_hw_split=False, pad_after_crop=False):
        assert input_mode is not None, "must indicate input_mode, [crop, pad]!!"
        
        if not t_range:
            t_range = [1, self.num_timesteps]
        [b, c, h, w] = x_HR.shape
        if different_t_in_one_batch:
            t = torch.randint(0, self.num_timesteps, (b,)).long() + 1
            t = t.to(x_HR.device)
            continuous_sqrt_alpha_cumprod = self._extract(torch.from_numpy(self.sqrt_alphas_cumprod_prev), t, x_start.shape)
            continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(b, -1)
        else:
            t = np.random.randint(t_range[0], t_range[1] + 1) # [1, 2000] [1, 2001)
            continuous_sqrt_alpha_cumprod = torch.FloatTensor(
                np.random.uniform(
                    self.sqrt_alphas_cumprod_prev[t-1],
                    self.sqrt_alphas_cumprod_prev[t],
                    size=b
                )
            ).to(x_HR.device)
            # continuous_sqrt_alpha_cumprod 是 sqrt(γ)
            continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(
                b, -1)
        

        # r = self.downsampling_schedule[t - 1]
        _, _, H, W = x_HR.shape
        crop_size = [self.patch_schedule[t - 1], self.patch_schedule[t - 1]]
        h = np.random.randint(0, H - crop_size[0] + 1)
        w = np.random.randint(0, W - crop_size[1] + 1)
        
        #generate input and target before crop
        resized_lq = AugF.resize(x_SR[:,:3], crop_size)
        
        colorp_target = torch.cat([torch.mean(x_HR[:,:3], dim=1, keepdim=True), 
                                   torch.max(x_HR[:,:3], dim=1, keepdim=True)[0]], dim=1)
        colorp_target = AugF.resize(colorp_target, crop_size)
        
        encoder_input = torch.cat([torch.mean(resized_lq, dim=1, keepdim=True), 
                                   torch.max(resized_lq, dim=1, keepdim=True)[0]], dim=1)
        
        #crop SR and HR
        x_HR = x_HR[:, :, h: h + crop_size[0], w: w + crop_size[1]]
        # x_HR = x_HR[:,:3]
        

        #split att map and SR
        x_SR_predict_input = x_SR[:, :, h: h + crop_size[0], w: w + crop_size[1]]
        x_SR = torch.cat([x_SR_predict_input, resized_lq], dim=1)
        # x_SR = x_SR_predict_input
        
        #Training process
        x_start = x_HR
        [b, c, h, w] = x_start.shape
        
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(
            x_start=x_start, continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1), noise=noise)
        
        #predictor illumination map
        colorp_out = self.colorp_fn(resized_lq)
        pmap_feats = self.pencoder_fn(torch.cat([colorp_out, encoder_input], dim=1), continuous_sqrt_alpha_cumprod)

        if frozen_denoise:
            with torch.no_grad():
                if not self.conditional:
                    model_output = self.denoise_fn(x_noisy, continuous_sqrt_alpha_cumprod, pmap_feats)
                else:
                    model_output = self.denoise_fn(
                        torch.cat([x_SR, x_noisy], dim=1), continuous_sqrt_alpha_cumprod, pmap_feats)
        else:
            if not self.conditional:
                model_output = self.denoise_fn(x_noisy, continuous_sqrt_alpha_cumprod, pmap_feats)
            else:
                model_output = self.denoise_fn(
                    torch.cat([x_SR, x_noisy], dim=1), continuous_sqrt_alpha_cumprod, pmap_feats)
        if clip_noise:
            model_output = torch.clamp(model_output, -1, 1)
            
        self.noise = noise
        self.pred_noise = model_output
        self.pred_noise_detach = self.pred_noise.detach()
        self.x_start = x_start
        self.x_noisy = x_noisy
        # self.predict_map = predict_map
        # self.x_map_gt    = x_map_gt
        # self.pmap_feats = pmap_feats

        self.x_recon = self.sqrt_recip_alphas_cumprod[t - 1] * x_noisy - self.sqrt_recipm1_alphas_cumprod[t - 1] * model_output
        self.x_recon = torch.clamp(self.x_recon, -1, 1)
        self.t = t - 1
        
        # self.x_recon_detach = self.x_recon.detach()
        
        return self.pred_noise, self.noise, self.x_recon, self.x_start, self.t, colorp_out, colorp_target
        
    def forward(self, x_HR, x_SR, train_type='ddpm', *args, **kwargs):
        kwargs_cp = kwargs.copy()
        for k in kwargs_cp:
            if kwargs[k] is None:
                kwargs.pop(k)
        # if train_type == 'ddpm':
        #     return self.p_losses(x_HR, x_SR, *args, **kwargs)
        # # elif train_type == 'ddpm_cs':
        #     return self.p_losses_cs(x_HR, x_SR, *args, **kwargs)
        # elif train_type == 'ddpm_cs_pyramid':
        #     return self.p_losses_cs_pyramid(x_HR, x_SR, *args, **kwargs)
        if train_type == 'ddpm_patch':
            return self.p_losses_patch(x_HR, x_SR, *args, **kwargs)
        else:
            assert False, f"Wrong train_type={train_type}"
