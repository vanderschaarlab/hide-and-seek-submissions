## synthetic0
import numpy as np
from sklearn.preprocessing import PowerTransformer 

#### begin synthetic.py
class MultipleSyntheticDataGenerator():
    def __init__(self, data: np.ndarray, intcol:list, floatcol:list):
        """ Initialize.
        Args:
            data: data with shape (n_seq, n_time ,n_fea) 
            intcol: Integer column number
            floatcol: Floating column number
        """
    
        if len(data.shape) != 3:
            raise ValueError("A shape of data must be 3d array.") 
        self.data = data
        self.intcol = intcol
        self.floatcol = floatcol
        
    def gen(self):
        """ Generate a synthetic data.
        Returns:
            data: synthetic data.
        """
        syn = np.zeros(self.data.shape)
        T = self.data.shape[1] 
        for t in range(T):
            gen = SyntheticDataGenerator(self.data[:,t,:],self.intcol,self.floatcol)
            syn[:,t,:] = gen.inv(gen.correct(gen.generator(self.data.shape[0])))
        return syn    

class SyntheticDataGenerator():
    def __init__(self, data: np.ndarray, intcol:list, floatcol:list):
        """ Initialize.
        Args:
            data: data with shape (n_seq, n_fea) 
            intcol: Integer column number
            floatcol: Floating column number
        """
        if len(data.shape) != 2:
            raise ValueError("A shape of data must be 2d array.") 
        self.intcol = intcol
        self.floatcol = floatcol

        """ Power transform """
        self.pt = PowerTransformer()
        self.pt.fit(data)
        ddata = self.pt.transform(data)

        self.min = np.min(ddata, axis=0)
        self.max = np.max(ddata, axis=0)
        self.data = ddata
        self.dim = ddata.shape[1]

    def inv(self, data:np.ndarray):
        """ Inverse power transform.
        Args:
            data: data with shape (n_seq, n_fea)
        Returns:
            data: inverse power transform
        """
        return self.pt.inverse_transform(data)

    def gaussgen(self, vector:np.ndarray, num: int):
        """ Random data generation for gaussian distribution.
        Args:
            vector: target vector 
            num: number of samples
        Returns:
            data: random data for gaussian distribution
        """
        std = np.std(vector)
        mean = np.mean(vector)
        return np.random.normal(loc=mean,scale=std,size=num)
        
    def generator(self, num:int):
        """ Generates synthetic data with the specified number of records.
        Args:
            num: number of records
        Returns:
            syn: data with shape (n_seq, n_fea)
        """
        syn = np.zeros((num,self.dim))

        for j in self.intcol:
            syn[:,j] = self.gaussgen(self.data[:,j], num)
        
        for j in self.floatcol:
            syn[:,j] = self.gaussgen(self.data[:,j], num)

        """ Whitening transformation """
        mu_tmp = np.mean(syn, axis=0)
        sigma_tmp = np.cov(syn.T)
        u_tmp, s_tmp, v_tmp = np.linalg.svd(sigma_tmp)
        u_tmp = np.linalg.inv(u_tmp.dot(np.diag(np.sqrt(s_tmp))))

        for i in range(0, num):
            syn[i] = u_tmp.dot(syn[i] - mu_tmp)

        """ Transform data so that it is the mean and variance of the power transformed original data. """
        mu = np.mean(self.data, axis=0)
        sigma = np.cov(self.data.T)

        u, s, v = np.linalg.svd(sigma)
        u = u.dot(np.diag(np.sqrt(s)))
        for i in range(0, num):
            syn[i] = u.dot(syn[i]) + mu
        
        return syn

    def correct(self, syn: np.ndarray):
        """ Adjust a min and value max of a synthetic data.
        Args:
            syn: data with shape (n_seq, n_fea)
        Returns:
            syn: Adjust a min and value max of a synthetic data
        """
        for j in self.intcol:
            syn[:,j] = syn[:,j].clip(self.min[j],self.max[j])
            syn[:,j] = np.round(syn[:,j])
        for j in self.floatcol:
            syn[:,j] = syn[:,j].clip(self.min[j],self.max[j])
        
        return syn

#### end synthetic.py

def resize_2d(data, mask, n_seq2):
    """Extract n_seq2 non-padding elements from data
    Args:
        data: time-series data with shape (n_seq, n_fea)
        mask: mask for data
        n_seq2: the number of elements to be extracted
    Returns:
        d_real[:n_seq2]: first n_seq2 non-padding elements in data. If the number of real elements is less than n_seq2, non-padding elements are duplicated.
    """
    n_seq, n_fea = data.shape
    n_pad = np.count_nonzero(mask[:,0])
    d_real = data[n_pad:]
    while d_real.shape[0] < n_seq2:
        d_real = np.vstack([d_real, d_real])
    return d_real[:n_seq2]

def remove_pad(data, mask):
    """Copy non-padding elements to padding elements
    Args:
        data: time-series data
        mask: mask for data
    """
    n_ex, n_seq, n_fea = data.shape
    data2 = np.copy(data)
    for i in range(n_ex):
        data2[i] = resize_2d(data[i], mask[i], n_seq)
    mask2 = np.zeros_like(mask)
    return data2, mask2

def mean_pad_len(mask):
    """Calculate the average padding length of mask
    Args:
        mask: mask
    """
    n_ex, n_seq, n_fea = mask.shape
    return np.count_nonzero(mask) // n_fea // n_ex

def add_pad(data, mask, pad_size):
    """Set the first pad_size elements of data to padding.
    Args:
        data: time-series data to be modified
        mask: mask for data
        pad_size: size of padding
    """
    mask2 = np.zeros_like(mask)
    mask2[:,:pad_size] = True
    data2 = np.where(mask2, np.nan, data) ## if use nan
    return data2, mask2

def ano_syn0_2(data, mask):
    """ Remove padding, apply MultipleSyntheticDataGenerator, and restore padding
    """
    #print("ano_syn0_2")
    pad_len = mean_pad_len(mask)
    data, mask = remove_pad(data, mask)
    n_ex, n_seq, n_fea = data.shape
    #import MultipleSyntheticDataGenerator from synthetic
    intcol = []
    floatcol = np.arange(n_fea)
    gen = MultipleSyntheticDataGenerator(data, intcol, floatcol)
    data = gen.gen()
    data, mask = add_pad(data, mask, pad_len)
    return data, mask

def conv_minmax(gd, gm, gd2, gm2):
    """Modify the value of each feature in gd2 to the minimum and maximum values of the corresponding feature in gd
    Args:
        gd2: time-series data to be modified
        gm2: mask for gd2
        gd: referenced time-series data
        gm: mask for gd
    
    """
    #print('conv_minmax')
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
            t2 = np.clip(gd2[:,k], x0, x1)
            y0 = np.nanmin(t2)
            y1 = np.nanmax(t2)
            t2 = np.where(t2 == y0, x0, t2)
            t2 = np.where(t2 == y1, x1, t2)
            gd2[:,k] = t2
    return gd2.reshape(n_ex2, n_seq2, n_fea2), gm2

def myhider(gd, gm):
    """ Call ano_syn0_2 and conv_minmax
    """
    gd2, gm2 = ano_syn0_2(gd, gm)
    gd, gm = conv_minmax(gd, gm, gd2, gm2)
    return gd, gm
