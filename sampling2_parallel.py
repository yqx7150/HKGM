# coding=utf-8
"""Various sampling methods."""
import functools

import torch
import numpy as np
import abc

from models.utils import from_flattened_numpy, to_flattened_numpy, get_score_fn
from scipy import integrate
import sde_lib
from models import utils as mutils
try:
  from skimage.measure import compare_psnr,compare_ssim
except:
  from skimage.metrics import peak_signal_noise_ratio as compare_psnr
  from skimage.metrics import structural_similarity as compare_ssim
import cv2
import os.path as osp
import matplotlib.pyplot as plt
import scipy.io as io
from SAKE import fft2c, ifft2c, im2row, row2im, sake
import math
def write_kdata(Kdata,name):
    temp = np.log(1+abs(Kdata))    
    plt.axis('off')
    plt.imshow(abs(temp),cmap='gray')
    plt.savefig(osp.join('./result/parallel_12ch/',name),transparent=True, dpi=128, pad_inches = 0,bbox_inches = 'tight')

def write_Data(model_num,psnr,ssim):
    filedir="result.txt"
    with open(osp.join('./result/parallel_12ch/',filedir),"w+") as f:#a+
        f.writelines(str(model_num)+' '+'['+str(round(psnr, 2))+' '+str(round(ssim, 4))+']')
        f.write('\n')
        
def write_Data2(psnr,ssim):
    filedir="PC.txt"
    with open(osp.join('./result/parallel_12ch/',filedir),"a+") as f:#a+
        f.writelines('['+str(round(psnr, 2))+' '+str(round(ssim, 4))+']')
        f.write('\n')
        
def write_images(x,image_save_path):
    x = np.clip(x * 255, 0, 255).astype(np.uint8)
    cv2.imwrite(image_save_path, x)
  
def k2wgt(X,W):
    Y = np.multiply(X,W) 
    return Y

def wgt2k(X,W,DC):
    Y = np.multiply(X,1./W)
    Y[W==0] = DC[W==0] 
    return Y
def im2row(im,winSize):
    size = (im).shape
    out = np.zeros(((size[0]-winSize[0]+1)*(size[1]-winSize[1]+1),winSize[0]*winSize[1],size[2]),dtype=np.complex64)
    count = -1
    for y in range(winSize[1]):
        for x in range(winSize[0]):
            count = count + 1                 
            temp1 = im[x:(size[0]-winSize[0]+x+1),y:(size[1]-winSize[1]+y+1),:]
            temp2 = np.reshape(temp1,[(size[0]-winSize[0]+1)*(size[1]-winSize[1]+1),1,size[2]],order = 'F')
            out[:,count,:] = np.squeeze(temp2)          
            
    return out
def row2im(mtx,size_data,winSize):
    size_mtx = mtx.shape 
    sx = size_data[0]
    sy = size_data[1] 
    sz = size_mtx[2] 
    
    res = np.zeros((sx,sy,sz),dtype=np.complex64)
    W = np.zeros((sx,sy,sz),dtype=np.complex64)
    out = np.zeros((sx,sy,sz),dtype=np.complex64)
    count = -1
    
    for y in range(winSize[1]):
        for x in range(winSize[0]):
            count = count + 1
            res[x : sx-winSize[0]+x+1 ,y : sy-winSize[1]+y+1 ,:] = res[x : sx-winSize[0]+x+1 ,y : sy-winSize[1]+y+1 ,:] + np.reshape(np.squeeze(mtx[:,count,:]),[sx-winSize[0]+1,sy-winSize[1]+1,sz],order = 'F')  
            W[x : sx-winSize[0]+x+1 ,y : sy-winSize[1]+y+1 ,:] = W[x : sx-winSize[0]+x+1 ,y : sy-winSize[1]+y+1 ,:] + 1
            

    out = np.multiply(res,1./W)
    return out
    
_CORRECTORS = {}
_PREDICTORS = {}


def register_predictor(cls=None, *, name=None):
  """A decorator for registering predictor classes."""

  def _register(cls):
    if name is None:
      local_name = cls.__name__
    else:
      local_name = name
    if local_name in _PREDICTORS:
      raise ValueError(f'Already registered model with name: {local_name}')
    _PREDICTORS[local_name] = cls
    return cls

  if cls is None:
    return _register
  else:
    return _register(cls)


def register_corrector(cls=None, *, name=None):
  """A decorator for registering corrector classes."""

  def _register(cls):
    if name is None:
      local_name = cls.__name__
    else:
      local_name = name
    if local_name in _CORRECTORS:
      raise ValueError(f'Already registered model with name: {local_name}')
    _CORRECTORS[local_name] = cls
    return cls

  if cls is None:
    return _register
  else:
    return _register(cls)


def get_predictor(name):
  return _PREDICTORS[name]


def get_corrector(name):
  return _CORRECTORS[name]


def get_sampling_fn(config, sde, shape, inverse_scaler, eps):
  """Create a sampling function.

  Args:
    config: A `ml_collections.ConfigDict` object that contains all configuration information.
    sde: A `sde_lib.SDE` object that represents the forward SDE.
    shape: A sequence of integers representing the expected shape of a single sample.
    inverse_scaler: The inverse data normalizer function.
    eps: A `float` number. The reverse-time SDE is only integrated to `eps` for numerical stability.

  Returns:
    A function that takes random states and a replicated training state and outputs samples with the
      trailing dimensions matching `shape`.
  """

  sampler_name = config.sampling.method
  # Probability flow ODE sampling with black-box ODE solvers
  if sampler_name.lower() == 'ode':
    sampling_fn = get_ode_sampler(sde=sde,
                                  shape=shape,
                                  inverse_scaler=inverse_scaler,
                                  denoise=config.sampling.noise_removal,
                                  eps=eps,
                                  device=config.device)
  # Predictor-Corrector sampling. Predictor-only and Corrector-only samplers are special cases.
  elif sampler_name.lower() == 'pc':
    predictor = get_predictor(config.sampling.predictor.lower())
    corrector = get_corrector(config.sampling.corrector.lower())
    sampling_fn = get_pc_sampler(sde=sde,
                                 shape=shape,
                                 predictor=predictor,
                                 corrector=corrector,
                                 inverse_scaler=inverse_scaler,
                                 snr=config.sampling.snr,
                                 n_steps=config.sampling.n_steps_each,
                                 probability_flow=config.sampling.probability_flow,
                                 continuous=config.training.continuous,
                                 denoise=config.sampling.noise_removal,
                                 eps=eps,
                                 device=config.device)
  else:
    raise ValueError(f"Sampler name {sampler_name} unknown.")

  return sampling_fn


class Predictor(abc.ABC):
  """The abstract class for a predictor algorithm."""

  def __init__(self, sde, score_fn, probability_flow=False):
    super().__init__()
    self.sde = sde
    # Compute the reverse SDE/ODE
    self.rsde = sde.reverse(score_fn, probability_flow)
    self.score_fn = score_fn

  @abc.abstractmethod
  def update_fn(self, x, t):
    """One update of the predictor.

    Args:
      x: A PyTorch tensor representing the current state
      t: A Pytorch tensor representing the current time step.

    Returns:
      x: A PyTorch tensor of the next state.
      x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
    """
    pass


class Corrector(abc.ABC):
  """The abstract class for a corrector algorithm."""

  def __init__(self, sde, score_fn, snr, n_steps):
    super().__init__()
    self.sde = sde
    self.score_fn = score_fn
    self.snr = snr
    self.n_steps = n_steps

  @abc.abstractmethod
  def update_fn(self, x, t):
    """One update of the corrector.

    Args:
      x: A PyTorch tensor representing the current state
      t: A PyTorch tensor representing the current time step.

    Returns:
      x: A PyTorch tensor of the next state.
      x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
    """
    pass



@register_predictor(name='reverse_diffusion')
class ReverseDiffusionPredictor(Predictor):
  def __init__(self, sde, score_fn, probability_flow=False):
    super().__init__(sde, score_fn, probability_flow)
  
  def update_fn(self, x, t):
    f, G = self.rsde.discretize(x, t) 
    z = torch.randn_like(x) 
    x_mean = x - f 
    x = x_mean + G[:, None, None, None] * z 
    
    return x, x_mean

@register_predictor(name='none')
class NonePredictor(Predictor):
  """An empty predictor that does nothing."""

  def __init__(self, sde, score_fn, probability_flow=False):
    pass

  def update_fn(self, x, t):
    return x, x


@register_corrector(name='langevin')
class LangevinCorrector(Corrector):
  def __init__(self, sde, score_fn, snr, n_steps):
    super().__init__(sde, score_fn, snr, n_steps)
    if not isinstance(sde, sde_lib.VPSDE) \
        and not isinstance(sde, sde_lib.VESDE) \
        and not isinstance(sde, sde_lib.subVPSDE):
      raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

  def update_fn(self, x1,x2,x3,x_mean,t):
    sde = self.sde
    score_fn = self.score_fn
    n_steps = self.n_steps
    target_snr = self.snr
    if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):
      timestep = (t * (sde.N - 1) / sde.T).long()
      alpha = sde.alphas.to(t.device)[timestep]
    else:
      alpha = torch.ones_like(t)
    
    for i in range(n_steps):
   
      grad1 = score_fn(x1, t) 
      grad2 = score_fn(x2, t)  
      grad3 = score_fn(x3, t)
      noise1 = torch.randn_like(x1) 
      noise2 = torch.randn_like(x2)
      noise3 = torch.randn_like(x3) 

      
      grad_norm1 = torch.norm(grad1.reshape(grad1.shape[0], -1), dim=-1).mean()
      noise_norm1 = torch.norm(noise1.reshape(noise1.shape[0], -1), dim=-1).mean()
      grad_norm2 = torch.norm(grad2.reshape(grad2.shape[0], -1), dim=-1).mean()
      noise_norm2 = torch.norm(noise2.reshape(noise2.shape[0], -1), dim=-1).mean()      
      grad_norm3 = torch.norm(grad3.reshape(grad3.shape[0], -1), dim=-1).mean()
      noise_norm3 = torch.norm(noise3.reshape(noise3.shape[0], -1), dim=-1).mean()            
      
      grad_norm =(grad_norm1+grad_norm2+grad_norm3)/3.0
      noise_norm = (noise_norm1+noise_norm2+noise_norm3)/3.0
      
      step_size =  (2 * alpha)*((target_snr * noise_norm / grad_norm) ** 2 )  
   
      x_mean = x_mean + step_size[:, None, None, None] * (grad1+grad2+grad3)/3.0 
      
      x1 = x_mean + torch.sqrt(step_size * 2)[:, None, None, None] * noise1 
      x2 = x_mean + torch.sqrt(step_size * 2)[:, None, None, None] * noise2 
      x3 = x_mean + torch.sqrt(step_size * 2)[:, None, None, None] * noise3 
      
    return x1,x2,x3,x_mean

@register_corrector(name='none')
class NoneCorrector(Corrector):
  """An empty corrector that does nothing."""

  def __init__(self, sde, score_fn, snr, n_steps):
    pass

  def update_fn(self, x, t):
    return x, x


def shared_predictor_update_fn(x, t, sde, model, predictor, probability_flow, continuous):
  """A wrapper that configures and returns the update function of predictors."""
  score_fn = mutils.get_score_fn(sde, model, train=False, continuous=continuous)
  if predictor is None:
    # Corrector-only sampler
    predictor_obj = NonePredictor(sde, score_fn, probability_flow)
  else:
    predictor_obj = predictor(sde, score_fn, probability_flow)
  return predictor_obj.update_fn(x, t)


def shared_corrector_update_fn(x1,x2,x3,x_mean,t,sde, model, corrector, continuous, snr, n_steps):
  """A wrapper tha configures and returns the update function of correctors."""
  score_fn = mutils.get_score_fn(sde, model, train=False, continuous=continuous)
  if corrector is None:
    # Predictor-only sampler
    corrector_obj = NoneCorrector(sde, score_fn, snr, n_steps)
  else:
    corrector_obj = corrector(sde, score_fn, snr, n_steps)
  return corrector_obj.update_fn(x1,x2,x3,x_mean,t)


def get_pc_sampler(sde, shape, predictor, corrector, inverse_scaler, snr,
                   n_steps=1, probability_flow=False, continuous=False,
                   denoise=True, eps=1e-3, device='cuda'):
  """Create a Predictor-Corrector (PC) sampler.

  Args:
    sde: An `sde_lib.SDE` object representing the forward SDE.
    shape: A sequence of integers. The expected shape of a single sample.
    predictor: A subclass of `sampling.Predictor` representing the predictor algorithm.
    corrector: A subclass of `sampling.Corrector` representing the corrector algorithm.
    inverse_scaler: The inverse data normalizer.
    snr: A `float` number. The signal-to-noise ratio for configuring correctors.
    n_steps: An integer. The number of corrector steps per predictor update.
    probability_flow: If `True`, solve the reverse-time probability flow ODE when running the predictor.
    continuous: `True` indicates that the score model was continuously trained.
    denoise: If `True`, add one-step denoising to the final samples.
    eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
    device: PyTorch device.

  Returns:
    A sampling function that returns samples and the number of function evaluations during sampling.
  """
  # Create predictor & corrector update functions
  predictor_update_fn = functools.partial(shared_predictor_update_fn,
                                          sde=sde,
                                          predictor=predictor,
                                          probability_flow=probability_flow,
                                          continuous=continuous)
  corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                          sde=sde,
                                          corrector=corrector,
                                          continuous=continuous,
                                          snr=snr,
                                          n_steps=n_steps)

  def pc_sampler(model):
    """ The PC sampler funciton.
    Args:
      model: A score model.
    Returns:
      Samples, number of function evaluations.
    """
    with torch.no_grad():
    
      #Initial sample
      timesteps = torch.linspace(sde.T, eps, sde.N, device=device)

      #load data
      coil = 12
      file_path='./Brain_test_12ch/1.mat'
      ori_data = np.zeros([256,256,coil],dtype=np.complex64)
      ori_data = io.loadmat(file_path)['Img']
      ori_data = ori_data/np.max(abs(ori_data))
      ori = np.copy(ori_data)
      ori_data = np.swapaxes(ori_data,0,2)
      ori_data = np.swapaxes(ori_data,1,2) 
      ori_data_sos = np.sqrt(np.sum(np.square(np.abs(ori_data)),axis=0)) 
      write_images(abs(ori_data_sos),osp.join('./result/parallel_12ch/'+'ori'+'.png'))
      mask = np.zeros((coil,256,256))
      mask_item = io.loadmat('./contract_mask/poisson/r4.mat')['mask']   
      
      for i in range(coil):
        mask[i,:,:] = mask_item
      print(np.sum(mask_item)/65536)
      write_images(abs(mask_item),osp.join('./result/parallel_12ch/'+'mask'+'.png'))
      
      ww = io.loadmat('./weight.mat')['weight'] 
      weight = np.zeros((coil,256,256))       
      for i in range(coil):
        weight[i,:,:] = ww

      Kdata = np.zeros((coil,256,256),dtype=np.complex64)
      Ksample = np.zeros((coil,256,256),dtype=np.complex64)
      zeorfilled_data = np.zeros((coil,256,256),dtype=np.complex64)
      k_w = np.zeros((coil,256,256),dtype=np.complex64)
      for i in range(coil):
        Kdata[i,:,:] = np.fft.fftshift(np.fft.fft2(ori_data[i,:,:]))
        Ksample[i,:,:] = np.multiply(mask[i,:,:],Kdata[i,:,:])
        k_w[i,:,:] = k2wgt(Ksample[i,:,:],weight[i,:,:])           
        zeorfilled_data[i,:,:] = np.fft.ifft2(Ksample[i,:,:])  
     
      zeorfilled_data_sos = np.sqrt(np.sum(np.square(np.abs(zeorfilled_data)),axis=0))

      ori_data_sos = ori_data_sos/np.max(np.abs(ori_data_sos))
      zeorfilled_data_sos = zeorfilled_data_sos/np.max(np.abs(zeorfilled_data_sos))  
      print(abs(k_w).max())
      psnr_zero=compare_psnr(255*abs(zeorfilled_data_sos),255*abs(ori_data_sos),data_range=255)
      ssim_zero=compare_ssim(abs(zeorfilled_data_sos),abs(ori_data_sos),data_range=1)
      print('psnr_zero: ',psnr_zero,'ssim_zero: ',ssim_zero)
      write_images(abs(zeorfilled_data_sos),osp.join('./result/parallel_12ch/'+'Zeorfilled_'+str(round(psnr_zero, 2))+str(round(ssim_zero, 4))+'.png'))
      io.savemat(osp.join('./result/parallel_12ch/'+'zeorfilled.mat'),{'zeorfilled':zeorfilled_data})
      
      #run
      ksize=[8,8]
      wnthresh = 1.8
      size_data = [256,256,12]
      x_input=np.stack((np.real(k_w),np.imag(k_w)),1)
      x_mean = torch.tensor(x_input,dtype=torch.float32).cuda()
      x1 = x_mean
      x2 = x_mean
      x3 = x_mean    
      max_psnr = 0
      max_ssim = 0
      for i in range(sde.N):
        print('======== ',i)
        t = timesteps[i]
        vec_t = torch.ones(shape[0], device=t.device) * t
        
        ##======================================================= Predictor
        x, x_mean = predictor_update_fn(x_mean, vec_t, model=model)
        
        #h
        x_mean = x_mean.cpu().numpy() # (8,2,256,256)    
        x_mean = np.array(x_mean,dtype=np.float32) 
        A_new_complex=x_mean[:,0,:,:]+1j*x_mean[:,1,:,:] 
        A_new_complex=A_new_complex.transpose(1,2,0)
        hankel=im2row(A_new_complex,ksize)
        size_temp = hankel.shape
        A = np.reshape(hankel,[size_temp[0],size_temp[1]*size_temp[2]],order = 'F')

        #SVD
        svd_input  = torch.tensor( A,dtype=torch.complex64)
        U,S,V = torch.svd(svd_input)
        S = torch.diag(S)
        U = np.array(U,dtype=np.complex64)
        S = np.array(S,dtype=np.complex64)
        V = np.array(V,dtype=np.complex64)          
        uu = U[:,0:math.floor(wnthresh*ksize[0]*ksize[1])]
        ss = S[0:math.floor(wnthresh*ksize[0]*ksize[1]),0:math.floor(wnthresh*ksize[0]*ksize[1])]
        vv = V[:,0:math.floor(wnthresh*ksize[0]*ksize[1])] 
        A_svd = np.dot(np.dot(uu,ss),vv.T) 
        A_svd = np.reshape(A_svd,[size_temp[0],size_temp[1],size_temp[2]],order = 'F')

        #h+
        kcomplex_h = row2im(A_svd,size_data,ksize)

        #fidelity
        k_complex = np.zeros((coil,256,256),dtype=np.complex64)
        k_complex2 = np.zeros((coil,256,256),dtype=np.complex64)  
        for ii in range(coil):
            k_complex[ii,:,:] = wgt2k(kcomplex_h[:,:,ii],weight[ii,:,:],Ksample[ii,:,:])
            k_complex2[ii,:,:] = Ksample[ii,:,:] + k_complex[ii,:,:]*(1-mask[ii,:,:])
        x_input = np.zeros((coil,2,256,256),dtype=np.float32)
        for i in range(coil): 
          k_w[i,:,:] = k2wgt(k_complex2[i,:,:],weight[i,:,:])
          x_input[i,0,:,:] = np.real(k_w[i,:,:])
          x_input[i,1,:,:] = np.imag(k_w[i,:,:])
        x_mean = torch.tensor(x_input,dtype=torch.float32).cuda()
        
        ##======================================================= Corrector
        x1,x2,x3,x_mean = corrector_update_fn(x1,x2,x3,x_mean, vec_t, model=model) 
      
        #h
        x_mean = x_mean.cpu().numpy()    
        x_mean = np.array(x_mean,dtype=np.float32) 
        A_new_complex=x_mean[:,0,:,:]+1j*x_mean[:,1,:,:]
        A_new_complex=A_new_complex.transpose(1,2,0)
        hankel=im2row(A_new_complex,ksize)
        size_temp = hankel.shape
        A = np.reshape(hankel,[size_temp[0],size_temp[1]*size_temp[2]],order = 'F')

        #SVD
        svd_input  = torch.tensor(A,dtype=torch.complex64)
        U,S,V = torch.svd(svd_input)
        S = torch.diag(S)
        U = np.array(U,dtype=np.complex64)
        S = np.array(S,dtype=np.complex64)
        V = np.array(V,dtype=np.complex64)          
        uu = U[:,0:math.floor(wnthresh*ksize[0]*ksize[1])]
        ss = S[0:math.floor(wnthresh*ksize[0]*ksize[1]),0:math.floor(wnthresh*ksize[0]*ksize[1])]
        vv = V[:,0:math.floor(wnthresh*ksize[0]*ksize[1])]
        A_svd = np.dot(np.dot(uu,ss),vv.T) 
        A_svd = np.reshape(A_svd,[size_temp[0],size_temp[1],size_temp[2]],order = 'F')

        #h+
        kcomplex_h = row2im(A_svd,size_data,ksize)

        #fidelity
        k_complex = np.zeros((coil,256,256),dtype=np.complex64)
        k_complex2 = np.zeros((coil,256,256),dtype=np.complex64)
        rec_Image = np.zeros((coil,256,256),dtype=np.complex64)
  
        for ii in range(coil):
            k_complex[ii,:,:] = wgt2k(kcomplex_h[:,:,ii],weight[ii,:,:],Ksample[ii,:,:])
            k_complex2[ii,:,:] = Ksample[ii,:,:] + k_complex[ii,:,:]*(1-mask[ii,:,:])
            rec_Image[ii,:,:] = np.fft.ifft2(k_complex2[ii,:,:])
        x_input = np.zeros((coil,2,256,256),dtype=np.float32)
        for i in range(coil): 
          k_w[i,:,:] = k2wgt(k_complex2[i,:,:],weight[i,:,:])
          x_input[i,0,:,:] = np.real(k_w[i,:,:])
          x_input[i,1,:,:] = np.imag(k_w[i,:,:])
        x_mean = torch.tensor(x_input,dtype=torch.float32).cuda()    
        x_mean = x_mean.to(device) 
  
        #save result
        rec_Image_sos = np.sqrt(np.sum(np.square(np.abs(rec_Image)),axis=0))
        rec_Image_sos = rec_Image_sos/np.max(np.abs(rec_Image_sos))
        psnr = compare_psnr(255*abs(rec_Image_sos),255*abs(ori_data_sos),data_range=255)
        ssim = compare_ssim(abs(rec_Image_sos),abs(ori_data_sos),data_range=1)
        print(' PSNR:', psnr,' SSIM:', ssim)  
        write_Data2(psnr,ssim) 
        
        if max_ssim<=ssim:
          max_ssim = ssim
        if max_psnr<=psnr:
          max_psnr = psnr
          write_Data('checkpoint',max_psnr,ssim) 
          write_images(abs(rec_Image_sos),osp.join('./result/parallel_12ch/'+'Rec'+'.png'))
          io.savemat(osp.join('./result/parallel_12ch/'+'HKGM.mat'),{'HKGM':rec_Image})
      return x_mean 
  return pc_sampler
