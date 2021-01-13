## was my_seeker_2020-1122-125008.py
import numpy as np

def nonzero(mask):
    """Count the number of non-zero elements
    Args:
        mask: ndarray
    Returns:
        the number of non-zero elements in mask
    """
    #return np.count_nonzero(mask)
    return np.sum(mask)

def count_pad(mask_i):
    """Count the number of padding elements
    Args:
        mask_i: ndarray
    Returns:
        the number of padding elements in mask_i
    """
    n_seq, n_fea = mask_i.shape
    n_pad = 0
    while n_pad < n_seq:
        if np.sum(mask_i[n_pad]) < n_fea:
            break
        n_pad += 1
    return n_pad

def to_real_features(data, mask):
    """Concat non-padding elements
    Args:
        data: time-series data
        mask: mask for data
    Returns:
        an ndarray with shape (the number of non-padding elements, n_seq, n_fea)
    """
    n_ex, n_seq, n_fea = data.shape
    res = []
    for i in range(n_ex):
        n_pad = count_pad(mask[i])
        #print(mask[i,:,0])
        #exit()
        res.append(data[i,n_pad:])
    return np.vstack(res)

def avg_min_dist(x, y):
    """Calculate the mean for i of the minimum distances between x[i] and y[j] for j
    Args:
        x: non-padding sequence
        y: non-padding sequences in the generated data
    Returns:
        the mean for i of the minimum distances between x[i] and y[j] for j
    """
    if len(y) <= 0:
        return 1e10
    xx = np.sum(x ** 2, axis=1)
    yy = np.sum(y ** 2, axis=1)
    xy = np.tensordot(x, y, (1,1))
    s = np.amin(xy * (-2) + yy, axis=1) + xx
    res = np.mean(s)
    #res = np.median(s)
    return res

def shuffle_real(data, mask):
    """Shuffle non-padding elements
    Args:
        data: time-series data
        mask: mask for data
    Returns:
        data: shuffled data
    """
    data = np.copy(data)
    n_ex, n_seq, n_fea = data.shape
    for i in range(n_ex):
        n_pad = nonzero(mask[i,:,0])
        np.random.shuffle(data[i,n_pad:])
    return data

def seeker_avg_min_dist(g, g_m, e, e_m):
    """Retruns the indices for top half of avg_min_dist(x[i], f) for i, where f is non-padding elements in e and x[k] is the non-padding elements in e[i]
    Args:
        g: generated time-series data
        g_m: mask for g
        e: extended real time-series data
        e_m: mask for e
    Returns:
        an ndarray z, where z[i] != 0 iff i is selected
    """
    #print('seeker: avg_min_dist')
    #g = g[:,:,1:]
    #g_m = g_m[:,:,1:]
    #e = e[:,:,1:]
    #e_m = e_m[:,:,1:]
    g[:,:,0] = 0
    e[:,:,0] = 0
    
    #s = 10
    s = 0 ## all
    if s > 0:
        g = shuffle_real(g, g_m)
        e = shuffle_real(e, e_m)
        g = g[:,-s:]
        g_m = g_m[:,-s:]
        e = e[:,-s:]
        e_m = e_m[:,-s:]

    n_ex, n_seq, n_fea = g.shape
    f = to_real_features(g, g_m)
    n_ex2, n_seq2, n_fea2 = e.shape
    #print(g.shape)
    #print(e.shape)
    #print(f.shape)
    a = []
    for i in range(n_ex2):
        #print(i)
        n_pad = nonzero(e_m[i,:,0])
        ei = e[i,n_pad:]
        dist = avg_min_dist(ei, f)
        a.append([dist, i])
    a = sorted(a)
    #print(a)
    res = [0] * n_ex2
    for i in range(n_ex):
        _, j = a[i]
        res[j] = 1
    return np.array(res)

def myseeker(gd, gm, ed, em):
    """Call seeker_avg_min_dist. If a mask of None exists, replace it with mask of all False.
    Args:
        gd: generated time-series data
        gm: mask for g
        ed: extended real time-series data
        em: mask for e
    """
    if gm is None:
        gm = np.zeros_like(gd, dtype=bool)
    if em is None:
        em = np.zeros_like(ed, dtype=bool)
    return seeker_avg_min_dist(gd, gm, ed, em)
