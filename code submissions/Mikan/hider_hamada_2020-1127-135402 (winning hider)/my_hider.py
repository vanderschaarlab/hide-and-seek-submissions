## #3, 1
import numpy as np

def ano_noise(data, mask, x=0.1):
    """Add noise to data using add_noise.add_noise
    Args:
        data: time-series data
        mask: mask for data
        x: amplitude of the added noise
    Returns:
        data: generated synthetic data
        mask: mask for data
    """
    #print("ano_noise", x)
    from examples.hider.add_noise import add_noise
    data = add_noise.add_noise(data, noise_size=x)
    return data, mask

def ano_shuffle_over_all_ids(data, mask):
    """Shuffle data along the 0th and 1st axes, then move padding to the beginning in each sequence
    Args:
        data: time-series data
        mask: mask for the data
    Returns:
        d: shuffled data
        m: mask for the data
    """
    #print("ano_shuffle_over_all_ids")
    #print(mask, 'mask')
    d = np.copy(data)
    m = np.copy(mask)
    n_ex, n_seq, n_fea = d.shape
    #print(d, 'd')
    if mask is not None:
        d = np.where(mask, np.nan, d)
    #print(d, 'd')
    d = d.reshape(-1, n_fea)
    np.random.shuffle(d)
    d = d.reshape(n_ex, n_seq, n_fea)
    for i in range(n_ex):
        t = np.copy(d[i,:,:])
        ct_pad = 0
        ct_val = 0
        for j in range(n_seq):
            if np.isnan(d[i,j,0]):
                t[ct_pad] = d[i,j]
                ct_pad += 1
            else:
                ct_val += 1
                t[-ct_val] = d[i,j]
        while ct_pad >= n_seq:
            ri = np.random.randint(0, n_ex)
            rj = np.random.randint(0, n_seq)
            if np.isnan(d[ri,rj,0]):
                continue
            t[ct_pad - 1] = np.copy(d[ri, rj])
            ct_pad -= 1
        d[i] = t
        for j in range(n_seq):
            m[i,j] = (j < ct_pad)
    return d, m

def set_min_time_to_zero(gd, gm):
    """Set the minimum time of each sequence to 0
    Args:
        gd: time-series data
        gm: mask for gd
    """
    n_ex, n_seq, n_fea = gd.shape
    for i in range(n_ex):
        t = gd[i,:,0]
        x0 = np.nanmin(t)
        #print(i, x0)
        t = np.where(t == x0, 0, t)
        gd[i,:,0] = t

def clip(gd, gm, gd2, gm2):
    """Clip the value of each feature in gd2 to the minimum and maximum values of the corresponding feature in gd
    Args:
        gd2: time-series data to be clipped
        gm2: mask for gd2
        gd: referenced time-series data
        gm: mask for gd
    """
    n_ex, n_seq, n_fea = gd.shape
    n_ex2, n_seq2, n_fea2 = gd2.shape
    gd = np.where(gm, np.nan, gd).reshape(-1, n_fea)
    gd2 = np.where(gm2, np.nan, gd2).reshape(-1, n_fea)
    for k in range(n_fea):
        t = gd[:,k]
        x0 = np.nanmin(t)
        x1 = np.nanmax(t)
        if np.isnan(x0) or np.isnan(x1) or x1 - x0 < 0.1:
            #continue
            gd2[:,k] = gd[:,k]
        else:
            gd2[:,k] = np.clip(gd2[:,k], x0, x1)
    return gd2.reshape(n_ex2, n_seq2, n_fea2), gm2

def hider_shuffle_noise_clip(gd, gm, p = 0.3):
    """Add noise, shuffle, clip, and correct time
    Args:
        gd: time-series data
        gm: mask for gd
        p: amplitude of the added noise
    """
    gd2, gm2 = ano_noise(gd, gm, p)
    gd2, gm2 = ano_shuffle_over_all_ids(gd2, gm2)
    gd, gm = clip(gd, gm, gd2, gm2)
    set_min_time_to_zero(gd, gm)
    return gd, gm

def myhider(gd, gm):
    """ Call hider_shuffle_noise_clip wth p = 0.3
    """
    gd, gm = hider_shuffle_noise_clip(gd, gm, 0.3)
    return gd, gm
