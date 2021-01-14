""" Main hider model implementation. 
Final submitted model:
    transform timeseries into trajectory, do SVD, discard some components, reconstruct, transform approximate trajectory into timeseries
"""

import logging
import numpy as np
import scipy as sp
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FactorAnalysis as FA
from sklearn.decomposition import PCA
from sklearn.utils.extmath import randomized_svd
import sys
import torch

MAKE_PLOTS = False
if MAKE_PLOTS:
    import matplotlib.pyplot as plt

##################################################################
########### NOTE: CURRENTLY NONE OF THESE IS USED ###############
def from_U_to_U_i_page(U, N, L, n_comps):
    U_i = np.zeros((N,L,n_comps))
    print(U.shape, U_i.shape)
    for i_obs in range(N):
        for l in range(L):
            U_i[i_obs,l,:] = U[i_obs*L+l,:n_comps]
    return U_i

def from_U_i_to_U_page(U_i,N,L, n_comps):
    U = np.zeros((N*L,n_comps))
    for i_obs in range(N):
        for l in range(L):
            U[i_obs*L+l,:] = U_i[i_obs,l,:n_comps]
    return U

def page_from_svd(U,s,V,n_comps):
    return U[:,:n_comps] @ (np.diag(s[:n_comps]) @ V[:n_comps,:])

def page_from_ts_matrix(data, L):
    N,T,D = data.shape
    assert np.mod(T,L) == 0, 'L needs to divide T for page matrix, T/L={}'.format(np.mod(T,L))
    page = np.zeros((N*L,T//L*D))
    for i_obs in range(N):
        for i_dim in range(D):
            page[i_obs*L:(i_obs+1)*L, (i_dim*T//L):((i_dim+1)*T//L)] = data[i_obs,:,i_dim].reshape(L,T//L,order='F')
    return page

def ts_from_page_matrix(page, D, L):
    NL,TperLD = page.shape
    N = NL//L
    T = TperLD*L//D
    ts = np.zeros((N,T,D))
    for i_obs in range(N):
        for i_dim in range(D):
            ts[i_obs,:,i_dim] = np.concatenate(np.split(page[i_obs*L:(i_obs+1)*L, i_dim*T//L:(i_dim+1)*T//L],T//L,axis=1)).reshape(T)
    return ts

########### NOTE: CURRENTLY NONE OF THE ABOVE IS USED ###########
##################################################################



# some functions for working with trajectory matrices, fixed to (N*L,D*K) arrays
def traj_from_ts_matrix(data, L):
    '''Function for constructing (stacked) trajectory matrix from time series
    Args:
        data : (N, T, D) matrix, where N is observations, T is sequence length, D dimensions
        L : window length
    Returns:
        traj : (N*L, D*K) trajectory matrix, K = T-L+1
    '''
    N, T, D = data.shape
    assert L >= 1 and L <= T//2, 'Must have 1 <= L <= T//2'
    K = T - L + 1
    traj = np.zeros((N*L,D*K))
    for i_obs in range(N):
        for i_dim in range(D):
            for k in range(K):
                traj[i_obs*L:(i_obs+1)*L,(i_dim*K+k):(k+1+i_dim*K)] = data[i_obs,k:(L+k),i_dim].reshape((-1,1))
    return traj

def ts_from_traj_matrix(traj, N, D):
    """Function for reconstructing time series from trajectory matrix
    Args:
        traj : (N*L, D*K) trajectory matrix, N=number of samples, D = dimensionality, L window size
        N : number of samples
        D : number of features
    returns : (N, T, D) time series matrix
    """
    L, K = traj.shape[0]//N, traj.shape[1]//D
    T = K + L - 1
    ts = np.zeros((N,T,D))
    for i_obs in range(N):
        for i_dim in range(D):
            ts[i_obs,:,i_dim] = np.concatenate(( traj[i_obs*L:((i_obs+1)*L),i_dim*K:i_dim*K+1].reshape(L,),
                                                         traj[((i_obs+1)*L-1),(i_dim*K+1):((i_dim+1)*K)]))
    return ts

def traj_from_svd(U,s,V, n_comps,L,K):
    """Function for reconstructing trajetory matrix from SVD components. Assume traj is (N*L x D*K) array
    Args:
        U : left singular vectors (col vecs)
        s : singular values
        V : right singular vectors (row vecs)
        L : window size
    """
    traj = U[:,:n_comps] @ (np.diag(s[:n_comps]) @ V[:n_comps,:])

    NL, DK = traj.shape
    N = NL//L
    D = DK//K
    for i_obs in range(N):
        for i_dim in range(D):
            apu = np.fliplr(traj[(i_obs)*L:((i_obs+1)*L),i_dim*K:(i_dim+1)*K])
            all_means = np.zeros((L+K-1))
            for k in range(K):
                all_means[k] = np.diag(apu,k=k).mean()
            for k in range(1,L):
                all_means[K+k-1] = np.diag(apu,k=-k).mean()
            inds = np.triu_indices(apu.shape[0], k=0, m=apu.shape[1])
            for i,ii in zip(inds[0],inds[1]):
                apu[i,ii] = all_means[ii-i]
            inds = np.tril_indices(apu.shape[0], k=-1, m=apu.shape[1])
            for i,ii in zip(inds[0],inds[1]):
                apu[i,ii] = all_means[K+i-ii-1]
            traj[(i_obs)*L:((i_obs+1)*L),i_dim*K:(i_dim+1)*K] = np.fliplr(apu)
    return traj




##################################################################
########### NOTE: CURRENTLY NONE OF THESE IS USED ###############
def from_U_to_U_i(U, N, n_comps):
    """
    NOTE: CURRENTLY NOT USED
    """
    L = U.shape[0]//N
    U_i = U[:,:n_comps].reshape((N,L,n_comps))
    return U_i

def from_U_i_to_U(U_i, n_comps):
    """
    NOTE: CURRENTLY NOT USED
    """
    N,L = U_i.shape[:2]
    U = U_i[:,:,:n_comps].reshape((N*L,n_comps))
    return U

def fit_FA_on_U(U, N, L, D, n_comps, model=1, use_scaling=True, stack_method='traj'):
    """
    NOTE: CURRENTLY NOT USED
    """
    if stack_method == 'traj':
        U_i = from_U_to_U_i(U,N,n_comps)
    elif stack_method == 'page':
        U_i = from_U_to_U_i_page(U, N, L, n_comps)

    if model == 0:
        sys.exit('Model 0 not implemented!')
    elif model == 1:
        U_i_ = U_i
    else:
        sys.exit('Unknown model: {}'.format(model))
    all_models = []
    if use_scaling:
        scalers = []
    else:
        scalers = None
    for i_comp in range(n_comps):
        if use_scaling:
            scalers.append(StandardScaler())
            all_models.append( FA(n_components = U_i_.shape[1]).fit(scalers[-1].\
                                        fit_transform(U_i_[:,:,i_comp])) )
        # non-transformed version:
        else:
            all_models.append( FA(n_components = U_i_.shape[1]).fit(U_i_[:,:,i_comp]) )
    return all_models, scalers

def generate_vecs(all_models, n_samples, scalers=None, noise_scale=1.0):
    """
    NOTE: CURRENTLY NOT USED
    """
    Dims = all_models[0].get_covariance().shape[0]
    n_comps = len(all_models)
    U_gen_i = np.zeros((n_samples, Dims, n_comps))
    for i_comp in range(n_comps):
        U_gen_i[:,:,i_comp] = np.random.multivariate_normal(np.zeros(Dims), noise_scale*all_models[i_comp].get_covariance(),size=(n_samples))
        if scalers is not None:
            U_gen_i[:,:,i_comp] = scalers[i_comp].inverse_transform(U_gen_i[:,:,i_comp])

    return U_gen_i

def get_symm_noise(dim,noise_std):
    """
    NOTE: CURRENTLY NOT USED
    """
    symm_noise = torch.zeros(dim, dim)
    symm_noise[torch.triu(torch.ones(symm_noise.shape,dtype=bool))] = \
        noise_std*torch.randn( dim*(dim+1)//2 )
    symm_noise.T[torch.triu(torch.ones(symm_noise.shape,dtype=bool),diagonal=1)] = \
        symm_noise[torch.triu(torch.ones(symm_noise.shape,dtype=bool),diagonal=1)]
    return symm_noise

########### NOTE: CURRENTLY NONE OF THE ABOVE IS USED ###########
##################################################################



def do_exact_svd(stacked_data, n_comps, L,noise_scale, use_scaling, stack_method, use_random_comps, comp_inds,N,T,D,K,mask_val, V_noise_std):
    logging.info( 'Starting exact SVD decomposition using {} matrices, with L={}, n_comps={}, V_noise_std={}'.format(stack_method, L, n_comps, V_noise_std) )
    """
    Function for doing SVD. For the final version, only the following are used:
        stacked_data : trajectory matrix
        n_comps : number of compoenents to keep from SVD
    """

    if V_noise_std > 0:
        # not used in the final version
        logging.debug('Calculating noisy s,V...')
        stacked_data = torch.from_numpy(stacked_data)
        apu = stacked_data.shape[1]
        symm_noise = get_symm_noise(apu, V_noise_std)
        matmul_data = torch.matmul(stacked_data.T, stacked_data)
        stacked_data = stacked_data.numpy()
        pca = PCA(random_state=2303, n_components=n_comps).fit( (matmul_data + symm_noise).numpy())
        s, V = np.sqrt(pca.singular_values_), pca.components_
        logging.debug('V shape: {}, s: {}'.format(V.shape,s.shape))
        symm_noise = get_symm_noise(apu, V_noise_std)
        pca = PCA(random_state=2303, n_components=n_comps).fit( (matmul_data + symm_noise).numpy())
        del matmul_data, symm_noise

        logging.debug('Calculating noisy U...')
        s2, V2 = np.sqrt(pca.singular_values_), pca.components_
        logging.debug('Using s instead s2 in U construction..')
        U = stacked_data @ (V2.T @ np.diag(1/s) )
        del pca, s2, V2

    else:
        # do this in the final version
        logging.debug('Calculating exact U,s,V...')
        U,s,V = randomized_svd(stacked_data, n_components=n_comps)

    logging.debug('Actual SVD shapes, U:{}, s:{}, V:{}'.format(U.shape, s.shape,V.shape))
    del stacked_data

    return U,s,V



##################################################################
########### NOTE: CURRENTLY NONE OF THESE IS USED ###############

def clip_grads(g1,g2, norm_max, grad_noise_sigma):
    """
    NOTE: CURRENTLY NOT USED
    """
    g_norm = torch.norm(g2, p=2)**2
    g_norm += torch.norm(g1, p=2)**2
    g_norm = torch.sqrt(g_norm)
    g1 = g1/torch.clamp(g_norm/norm_max, min=1) + \
        (grad_noise_sigma*norm_max)*torch.normal(mean=torch.zeros_like(g1), std=1.0)
    g2 = g2/torch.clamp(g_norm/norm_max, min=1) + \
        (grad_noise_sigma*norm_max)*torch.normal(mean=torch.zeros_like(g2), std=1.0)
    return  g1,g2

def loss_fun(C_l,C_r, X, soft_ortho_w=1.0, pad_mask=None,C_l_pad=None,C_r_pad=None):
    """
    NOTE: CURRENTLY NOT USED
    """
    apu = X-torch.matmul(C_l,C_r)
    if pad_mask is not None:
        apu[pad_mask] = 0
    loss_X = torch.norm(apu)
    loss_ortho = 0
    if soft_ortho_w != 0:
        if C_l_pad is not None and C_r_pad is not None:
            apu = torch.matmul(C_l.T,C_l) - torch.eye(C_l.shape[1])
            apu[torch.matmul(C_l_pad.T,C_l_pad).bool()] = 0
            loss_ortho += soft_ortho_w*torch.norm(apu)
            apu = torch.matmul(C_r,C_r.T) - torch.eye(C_r.shape[0])
            apu[C_r_pad] = 0
            loss_ortho += soft_ortho_w*torch.norm(apu)
        else:
            loss_ortho += soft_ortho_w*torch.norm(torch.matmul(C_l.T,C_l) - torch.eye(C_l.shape[1])) \
                        + soft_ortho_w*torch.norm(torch.matmul(C_r,C_r.T) - torch.eye(C_r.shape[0]))
    return loss_X + loss_ortho


def do_full_optimisation(stacked_data,use_masks,n_comps,L,noise_scale, FA_model, use_scaling, stack_method, use_random_comps, comp_inds,N,T,D,K, grad_norm_max, dp_noise, batch_size, soft_ortho_w,lr,max_iters,weight_decay, mask_val):
    """
    NOTE: CURRENTLY NOT USED
    """
    logging.info( 'Starting optimisation for generative SVD using {} matrices, with L={}, n_comps={}, noise_scale={}, max_iters={}, lr={}, grad_norm_max={}, dp_noise={}, weight_decay={}'.format(stack_method, L, n_comps, noise_scale,max_iters,lr,grad_norm_max,dp_noise,weight_decay) )
    stacked_data = torch.from_numpy(stacked_data)
    C_l = torch.from_numpy(np.random.normal(0,.1,size=(stacked_data.shape[0],n_comps)))
    C_r = torch.from_numpy(np.random.normal(0,.1,size=(n_comps, stacked_data.shape[1])))
    logging.debug('Shapes, left matrix:{}, right matrix:{}'.format(C_l.shape,C_r.shape))
    C_l.requires_grad = True
    C_r.requires_grad = True
    if not use_masks:
        logging.debug('Optimisation SVD filling pads/missing vals with 0s')
        stacked_data[np.isnan(stacked_data)] = 0
        if not np.isnan(mask_val):
            stacked_data[stacked_data == mask_val] = 0
        pad_traj, C_l_pad, C_r_pad = None, None, None
    else:
        if not np.isnan(mask_val):
            stacked_data[stacked_data == mask_val] = float('nan')
        pad_traj = stacked_data.isnan()
        C_l_pad = torch.zeros_like(C_l)
        C_r_pad = torch.zeros_like(C_r)
        for ii in range(stacked_data.shape[1]):
            for i in range(stacked_data.shape[0]):
                if pad_traj[i,ii]:
                    C_l_pad[i,:] = 1
                    C_r_pad[:,ii] = 1
        if batch_size < N:
            pass
        else:
            C_l_pad = torch.matmul(C_l_pad.T,C_l_pad).bool()
        C_r_pad = torch.matmul(C_r_pad,C_r_pad.T).bool()
    if torch.cuda.is_available():
        stacked_data.cuda() # probably won't fit to GPU memory
        C_l.cuda()
        C_r.cuda()
        if use_masks:
            pad_traj.cuda()
            C_l_pad.cuda()
            C_r_pad.cuda()
    sampler_ = torch.utils.data.RandomSampler(torch.linspace(0,N-1,N), replacement=False)
    sampler = torch.utils.data.BatchSampler(sampler_, batch_size=batch_size, drop_last=True)
    optimizer = torch.optim.Adam([C_l, C_r], lr=lr,weight_decay=weight_decay)
    loss_trace = np.zeros(max_iters)
    for i in range(max_iters):
        if i > 0 and i % 10 == 0:
            logging.debug('Starting iteration {} at loss={}'.format(i, loss_trace[i-1]))
        optimizer.zero_grad()
        for batch_idx, inds in enumerate(sampler):
            if torch.cuda.is_available():
                inds.cuda()
            loss_acc = 0
            acc_C_r = torch.zeros_like(C_r)
            if torch.cuda.is_available():
                acc_C_r.cuda()
            for i_obs in inds:
                if use_masks:
                    loss = loss_fun(C_l[i_obs*L:(i_obs+1)*L,:],C_r, stacked_data[i_obs*L:(i_obs+1)*L,:],soft_ortho_w,
                                pad_traj[i_obs*L:(i_obs+1)*L,:],C_l_pad[i_obs*L:(i_obs+1)*L,:],C_r_pad)
                else:
                    loss = loss_fun(C_l[i_obs*L:(i_obs+1)*L,:],C_r, stacked_data[i_obs*L:(i_obs+1)*L,:],soft_ortho_w,
                                pad_traj,C_l_pad,C_r_pad)
                if torch.cuda.is_available():
                    loss_acc += loss.cpu().detach().numpy()
                else:
                    loss_acc += loss.detach().numpy()
                loss.backward()
                if grad_norm_max is not None:
                    apu = clip_grads(C_l.grad[i_obs*L:(i_obs+1)*L,:],C_r.grad, grad_norm_max, dp_noise)
                    C_l.grad[i_obs*L:(i_obs+1)*L,:] = apu[0]
                    acc_C_r += apu[1]
                else:
                    acc_C_r += C_r.grad
                C_r.grad = torch.zeros_like(C_r.grad)
            acc_C_r /= batch_size
            C_r.grad = acc_C_r
        optimizer.step()
        loss_trace[i] = loss_acc
    if MAKE_PLOTS:
        logging.debug('Full loss trace:\n{}'.format(loss_trace))
        plt.plot(loss_trace)
        plt.title('Loss trace')
        plt.savefig('loss_trace.pdf', format='pdf')
        plt.close()
        fig, axs = plt.subplots(1,3,figsize=(15,5))
        axs[0].imshow((C_l.detach().numpy() @ C_r.detach().numpy())[:100,:100])
        axs[0].set_title('Optimised prod matrix')
        if stack_method == 'traj':
            axs[1].imshow((traj_from_svd(C_l.detach().numpy(),np.ones(C_r.shape[0]), C_r.detach().numpy(), n_comps,L,K))[:100,:100])
        else:
            axs[1].imshow((page_from_svd(C_l.detach().numpy(),np.ones(C_r.shape[0]), C_r.detach().numpy(), n_comps))[:100,:100])
        axs[1].set_title('Rehankelized optimised prod matrix')
        axs[2].imshow(stacked_data.detach().numpy()[:100,:100])
        axs[2].set_title('True trajectory matrix')
        plt.savefig('reconstructions.pdf', format='pdf')
        plt.close()
    logging.info('Optimisation done, final loss={}'.format(loss_trace[-1]))
    del stacked_data, optimizer, pad_traj, C_l_pad, C_r_pad
    if torch.cuda.is_available:
        C_l, C_r = C_l.cpu(), C_r.cpu()
    return C_l.detach().numpy(), np.ones(C_r.shape[0]) , C_r.detach().numpy()

########### NOTE: CURRENTLY NONE OF THE ABOVE IS USED ###########
##################################################################



def generative_svd(data, use_masks=False, mask_val=-1.0, n_comps = 1000, L = 50, noise_scale=1.0, FA_model=None, use_scaling=True, stack_method='traj', use_random_comps=False, decomp_method='exact', V_noise_std=0.0, lr=1e-4, max_iters=1,weight_decay=0.0, grad_norm_max=1.0, dp_noise=0.0, batch_size=10, soft_ortho_constr=.0, zero_pads=False, flipped_pads=True):
    '''
    Main function, most of the parameters are not used for the final submitted model
    Args:
        data : array of size (N,T,D), assume no missing values
        use_masks : (bool) Only for optimisation, exact can't handle missing vals or padding (will replace with 0s/true vals).
        mask_val : mask value used for padding and missing vals
        L : window size for trajectory/page matrix, in [1,T/2], set to None for max values
        n_comps : number of components to use in SVD, in [1,K], set to None for max values
        noise_scale : scaling factor for noise when generating data, 1.0 to use model Cov as learned
        FA_model : (int) {0,1}, which model to use. None to do only reconstruction
            If 0, do FA on U assuming samples, window steps, and components are cond.ind. (MVN has D dims)
                NOTE: 0 not currently checked, and shouldn't run that crap anyways
            If 1, do FA assuming samples and components are cond.ind. (MNV has Dims*L dims)
        use_scaling : normalize components when fitting models
        stack_method : 'traj' to use trajectory matrices, 'page' to use page matrices
        use_random_comps : (bool) use randomly chosen components or first n_comps
        decomp_method : 'exact' or 'opt'.  Use 'exact' for (noisy) SVD, 'opt' to optimise a loss function
        V_noise_std : when using 'exact', noise std to add to right singular vectors (as in Analyze Gauss paper)
        lr : lr for Adam optimiser
        n_epochs : (int)
        weight_decay : regulariser weight for Adam
        grad_norm_max : per example grad clipping norm bound
        dp_noise : Gaussian noise std for DP SGD
        batch_size : (int), currently loops over if > 1
        soft_ortho_constr : soft constraint weight for opt to encourage learning orthogonal left, right matrices
        zero_pads : (bool)(only when decomp_method is 'exact' or not 'use_masks') If True replace pads by 0s, otherwise use first actual value
        flipped_pads : (bool) if True use flipped padding (pads in the end of series). NOTE: this needs to match with the pipeline!
    '''
    N,T,D = data.shape

    # fill missing vals with 0s, replace pads
    if decomp_method == 'exact' or not use_masks:
        logging.debug('Filling nans with 0s')
        data[np.isnan(data)] = 0
        if not np.isnan(mask_val):
            if zero_pads:
                logging.debug('Replacing pads with 0s')
                data[data == mask_val] = 0
            else:
                # impute pads with medians/first actual value
                logging.debug('Replacing pads with first/last actual obs')
                for i_obs in range(N):
                    for i_dim in range(D):
                        apu = np.nonzero(data[i_obs,:,i_dim] == mask_val)[0]
                        if len(apu) > 0 and len(apu) < T:
                            if not flipped_pads:
                                data[i_obs,apu,i_dim] = data[i_obs, apu[-1]+1, i_dim]
                            else:
                                data[i_obs,apu,i_dim] = data[i_obs, apu[0]-1, i_dim]

    # stack data, uses 'traj' in the final version
    if L is None:
        L = T//2
    if stack_method == 'traj':
        K = T - L + 1
        stacked_data = traj_from_ts_matrix(data, L)
    elif stack_method == 'page':
        K = T//L
        stacked_data = page_from_ts_matrix(data, L)
    del data
    logging.debug('full stacked data shape: {}'.format(stacked_data.shape))
    if n_comps is None or n_comps > stacked_data.shape[0]:
        n_comps = np.minimum(D*K, stacked_data.shape[0])
    if use_random_comps:
        comp_inds = np.random.choice(np.linspace(0,D*K-1,D*K,dtype='int'),size=n_comps, replace=False)
    else:
        comp_inds = np.linspace(0,n_comps-1,n_comps,dtype='int')

    if decomp_method == 'exact':
        # do this in the final version
        U,s,V = do_exact_svd(stacked_data=stacked_data, n_comps=n_comps, L=L,noise_scale=noise_scale, use_scaling=use_scaling, stack_method=stack_method, use_random_comps=use_random_comps, comp_inds=comp_inds,N=N,T=T,D=D,K=K, mask_val=mask_val, V_noise_std=V_noise_std)

    elif decomp_method == 'opt':
        # not currently used
        U,s,V = do_full_optimisation(stacked_data,use_masks,n_comps,L,noise_scale, FA_model, use_scaling, stack_method, use_random_comps, comp_inds,N,T,D,K, grad_norm_max, dp_noise, batch_size, soft_ortho_constr, lr,max_iters,weight_decay,mask_val)
    else:
        sys.exit('Unknown decomp_method: {}'.format(decomp_method))

    if FA_model is not None:
        # not currently used
        # learn generative model for U
        logging.debug('Starting to fit generative models...')
        all_models, scalers = fit_FA_on_U(U, N, L, D, n_comps=n_comps, model=FA_model, use_scaling=use_scaling, stack_method=stack_method)
        del U

        logging.debug('Generating new components...')
        U_gen_i = generate_vecs(all_models, n_samples=N, scalers=scalers, noise_scale=1.0)
        if stack_method == 'traj':
            U_gen = from_U_i_to_U(U_gen_i,n_comps)
        elif stack_method == 'page':
            U_gen = from_U_i_to_U_page(U_gen_i,N,L,n_comps)
        del all_models, scalers, U_gen_i
    else:
        # do this in the final version
        # pure reconstruction
        U_gen = U

    logging.debug('Reconstructing (generated) data...')
    if stack_method == 'traj':
        generated_data = ts_from_traj_matrix( traj_from_svd(U_gen,s,V,n_comps,L,K),N,D)
    elif stack_method == 'page':
        generated_data = ts_from_page_matrix( page_from_svd(U_gen,s,V,n_comps),D,L)

    del U_gen,s,V
    logging.debug('All done!')

    return generated_data
