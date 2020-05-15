import mrcfile
import numpy as np
import numpy.random
from enum import Enum

# just a little function that makes sure no incorrect types are specified (e.g. '3d' or '1d' or '1D')
class project(Enum):  
    type3D = '3D'
    type2D = '2D'


def compute_mean_std(data):
    """
    Compute mean and standard deviation of a given image file.
    
    Parameters
    ----------
    data : array(float)
        The data.
        
    Returns
    -------
    float
        mean
    float
        standard deviation
    """
    mean, std = np.mean(data), np.std(data)
        
    return mean, std


# manager function that passes data off to sample_coordinates_2D or sample_coordinates_3D
def sample_coordinates(mask, num_train_samples, num_val_samples, net_dim, sample_length=96):
    """
    Sample random coordinates for train and validation volumes. The train and validation 
    volumes will not overlap. The volumes are only sampled from foreground regions in the mask.
    
    Parameters
    ----------
    mask : array(int)
        Binary image indicating foreground/background regions. Sampling only from foreground regions.
    num_train_samples : int
        Number of train-sample coordinates.
    num_val_samples : int
        Number of validation-sample coordinates.
    net_dim : String
        Dimension of requested training samples. Reflects choice of denoiser (3D or 2D)
    sample_length : int
        Dimensionality of the extracted volumes. Default: 96
    """
    
    if net_dim == '3D':
        try:
            print('Sampling 3D coordinates...')
            return sample_coordinates_3D(mask,
                                         num_train_vols = num_train_samples,
                                         num_val_vols = num_val_samples,
                                         sample_length = sample_length)
        finally:
            print('Finished sampling coordinates')

    elif net_dim == '2D':
        try:
            print('Sampling 2D coordinates...')
            
            train_coords_set = []       
            val_coords_set = []
            
            num_slices = mask.shape[0]  # how many z-slices of tomogram (i.e. height)
            mask2D = mask[1,:,:]
            
            for i in range(num_slices):
                train_coords, val_coords = sample_coordinates_2D(np.copy(mask2D),
                                                                   num_train_slices = num_train_samples,
                                                                   num_val_slices = num_val_samples,
                                                                   sample_length = sample_length)
                train_coords_set.append(train_coords)
                val_coords_set.append(val_coords)
                    
            return train_coords_set, val_coords_set
        finally:
            print('Finished sampling coordinates')
    else:
        raise Exception('Invalid project dimensionality, please check the value of net_dim') 

            
def sample_coordinates_2D(mask, num_train_slices, num_val_slices, sample_length = 128):
    """
    Sample random coordinates for train and validation slices. The train and validation 
    slices will not overlap. The slices are only sampled from foreground regions in the mask.
    
    Parameters
    ----------
    mask : array(int)
        Binary image indicating foreground/background regions. slices will only be sampled from 
        foreground regions.
    num_train_slices : int
        Number of train-slice coordinates.
    num_val_slices : int
        Number of validation-slice coordinates.
    sample_length : tuple(int, int)
        Dimensionality of the extracted slices. Default: 128        
    Returns
    -------
    list(tuple(slice, slice))
        Training slice coordinates.
     list(tuple(slice, slice))
        Validation slice coordinates.
    """
    slice_dims = (sample_length, sample_length)
    
    cent = (np.array(slice_dims) / 2).astype(np.int32)
    mask[:cent[0]] = 0
    mask[-cent[0]:] = 0
    mask[:, :cent[1]] = 0
    mask[:, -cent[1]:] = 0
    
    tv_span = np.round(np.array(slice_dims) / 2).astype(np.int32)
    span = np.round(np.array(mask.shape) * 0.1 / 2 ).astype(np.int32)
    val_sampling_mask = mask.copy()
    val_sampling_mask[:, :span[1]] = 0
    val_sampling_mask[:, -span[1]:] = 0
    
    
    foreground_pos = np.where(val_sampling_mask)
    sample_inds = np.random.choice(len(foreground_pos[0]), 2, replace=False)
    val_sampling_mask = np.zeros(mask.shape, dtype=np.int8)
    val_sampling_inds = [fg[sample_inds] for fg in foreground_pos]
    for z, y in zip(*val_sampling_inds):
        val_sampling_mask[z - span[0]:z + span[0],
        y - span[1]:y + span[1]] = mask[z - span[0]:z + span[0],
                                        y - span[1]:y + span[1]].copy()

        mask[max(0, z - span[0] - tv_span[0]):min(mask.shape[0], z + span[0] + tv_span[0]),
        max(0, y - span[1] - tv_span[1]):min(mask.shape[1], y + span[1] + tv_span[1])] = 0

    foreground_pos = np.where(val_sampling_mask)
    sample_inds = np.random.choice(len(foreground_pos[0]), num_val_slices, replace=num_val_slices<len(foreground_pos[0]))
    val_sampling_inds = [fg[sample_inds] for fg in foreground_pos]
    val_coords = []
    for z, y in zip(*val_sampling_inds):
        val_coords.append(tuple([slice(z-tv_span[0], z + tv_span[0]),
                                 slice(y-tv_span[1], y + tv_span[1])]))

    foreground_pos = np.where(mask)
    sample_inds = np.random.choice(len(foreground_pos[0]), num_train_slices, replace=num_train_slices < len(foreground_pos[0]))
    train_sampling_inds = [fg[sample_inds] for fg in foreground_pos]
    train_coords = []
    for z, y in zip(*train_sampling_inds):
        train_coords.append(tuple([slice(z - tv_span[0], z + tv_span[0]),
                                 slice(y - tv_span[1], y + tv_span[1])]))

    return train_coords, val_coords


def sample_coordinates_3D(mask, num_train_vols, num_val_vols, sample_length = 64):
    """
    Sample random coordinates for train and validation volumes. The train and validation 
    volumes will not overlap. The volumes are only sampled from foreground regions in the mask.
    
    Parameters
    ----------
    mask : array(int)
        Binary image indicating foreground/background regions. Volumes will only be sampled from 
        foreground regions.
    num_train_vols : int
        Number of train-volume coordinates.
    num_val_vols : int
        Number of validation-volume coordinates.
    sample_length : int
        Dimensionality of the extracted volumes. Default: 64
        
    Returns
    -------
    list(tuple(slice, slice, slice))
        Training volume coordinates.
     list(tuple(slice, slice, slice))
        Validation volume coordinates.
    """
    vol_dims = (sample_length, sample_length, sample_length)
    
    cent = (np.array(vol_dims) / 2).astype(np.int32)
    mask[:cent[0]] = 0
    mask[-cent[0]:] = 0
    mask[:, :cent[1]] = 0
    mask[:, -cent[1]:] = 0
    mask[:, :, :cent[2]] = 0
    mask[:, :, -cent[2]:] = 0
    
    tv_span = np.round(np.array(vol_dims) / 2).astype(np.int32)
    span = np.round(np.array(mask.shape) * 0.1 / 2 ).astype(np.int32)
    val_sampling_mask = mask.copy()
    val_sampling_mask[:, :span[1]] = 0
    val_sampling_mask[:, -span[1]:] = 0
    val_sampling_mask[:, :, :span[2]] = 0
    val_sampling_mask[:, :, -span[2]:] = 0
    foreground_pos = np.where(val_sampling_mask)
    sample_inds = np.random.choice(len(foreground_pos[0]), 2, replace=False)

    val_sampling_mask = np.zeros(mask.shape, dtype=np.int8)
    val_sampling_inds = [fg[sample_inds] for fg in foreground_pos]
    for z, y, x in zip(*val_sampling_inds):
        val_sampling_mask[z - span[0]:z + span[0],
        y - span[1]:y + span[1],
        x - span[2]:x + span[2]] = mask[z - span[0]:z + span[0],
                                        y - span[1]:y + span[1],
                                        x - span[2]:x + span[2]].copy()

        mask[max(0, z - span[0] - tv_span[0]):min(mask.shape[0], z + span[0] + tv_span[0]),
        max(0, y - span[1] - tv_span[1]):min(mask.shape[1], y + span[1] + tv_span[1]),
        max(0, x - span[2] - tv_span[2]):min(mask.shape[2], x + span[2] + tv_span[2])] = 0

    foreground_pos = np.where(val_sampling_mask)
    sample_inds = np.random.choice(len(foreground_pos[0]), num_val_vols, replace=num_val_vols<len(foreground_pos[0]))
    val_sampling_inds = [fg[sample_inds] for fg in foreground_pos]
    val_coords = []
    for z, y, x in zip(*val_sampling_inds):
        val_coords.append(tuple([slice(z-tv_span[0], z+tv_span[0]),
                                 slice(y-tv_span[1], y+tv_span[1]),
                                 slice(x-tv_span[2], x+tv_span[2])]))

    foreground_pos = np.where(mask)
    sample_inds = np.random.choice(len(foreground_pos[0]), num_train_vols, replace=num_train_vols < len(foreground_pos[0]))
    train_sampling_inds = [fg[sample_inds] for fg in foreground_pos]
    train_coords = []
    for z, y, x in zip(*train_sampling_inds):
        train_coords.append(tuple([slice(z - tv_span[0], z + tv_span[0]),
                                 slice(y - tv_span[1], y + tv_span[1]),
                                 slice(x - tv_span[2], x + tv_span[2])]))

    return train_coords, val_coords



def normalize(img, mean, std):
    """
    Normalize image with mean and standard deviation.
    
    Parameters
    ----------
    img : array(float)
        The image to normalize
    mean : float
        The mean used for normalization.
    std : float
        The standard deviation used for normalization.
        
    Returns
    -------
    array(float)
        The normalized image.
    """
    return (img - mean) / std


def denormalize(img, mean, std):
    """
    Denormalize the image with mean and standard deviation. This inverts
    the normalization.
    
    Parameters
    ----------
    img : array(float)
        The image to denormalize.
    mean : float
        The mean which was used for normalization.
    std : float
        The standard deviation which was used for normalization.
    """
    return (img * std) + mean


# just a manager function for 2D or 3D
def extract_samples(even, odd, train_coords, val_coords, mean, std, net_dim):
    if net_dim == '3D':
        return extract_volumes(even, odd, train_coords, val_coords, mean, std)
    
    elif net_dim == '2D':
        for slice_index in range(len(train_coords)): 
            Xtemp, Ytemp, X_valtemp, Y_valtemp = extract_slices(even[slice_index,:,:],
                                                              odd[slice_index,:,:],
                                                              train_coords[slice_index],
                                                              val_coords[slice_index],
                                                              mean, std)
            # would probably be better to pre-allocate, but this works
            if slice_index==0:
                X = Xtemp
                Y = Ytemp
                X_val = X_valtemp
                Y_val = Y_valtemp
            else:
                X = np.concatenate((X,Xtemp), axis = 0)
                Y = np.concatenate((Y,Ytemp), axis = 0)
                X_val = np.concatenate((X_val,X_valtemp), axis = 0)
                Y_val = np.concatenate((Y_val,Y_valtemp), axis = 0)
        return X, Y, X_val, Y_val

    else:
        raise Exception('Invalid project dimensionality, please check the value of net_dim') 

        
def extract_volumes(even, odd, train_coords, val_coords, mean, std):
    """
    Extract train and validation volumes and normalize them.
    
    Parameters
    ----------
    even : array(float)
        Even tomogram.
    odd : array(float)
        Odd tomogram.
    train_coords : list(tuple(slice, slice, slice))
        The slices of the train-volumes.
    val_coords : list(tuple(slice, slice, slice))
        The slices of the validation-volumes.
    mean : float
        Mean used for normalization.
    std : float
        Standard deviation used for normalization.
        
    Returns
    -------
    list(array(float))
        Train data X (normalized)
    list(array(float))
        Train data Y (normalized)
    list(array(float))
        Validation data X_val (normalized)
    list(array(float))
        Validation data Y_val (normalized)
    """
    z, y, x = train_coords[0][0], train_coords[0][1], train_coords[0][2]
    train_vol_dims = (z.stop - z.start, y.stop - y.start, x.stop - x.start)
    z, y, x = val_coords[0][0], val_coords[0][1], val_coords[0][2]
    val_vol_dims = (z.stop - z.start, y.stop - y.start, x.stop - x.start)
    
    X = np.zeros((len(train_coords), *train_vol_dims), dtype=np.float32)
    Y = np.zeros((len(train_coords), *train_vol_dims), dtype=np.float32)
    X_val = np.zeros((len(val_coords), *val_vol_dims), dtype=np.float32)
    Y_val = np.zeros((len(val_coords), *val_vol_dims), dtype=np.float32)

    img_x = even
    img_x = normalize(img_x, mean, std)
    img_y = odd
    img_y = normalize(img_y, mean, std)

    for i, pos in enumerate(train_coords):
        X[i] = img_x[pos]
        Y[i] = img_y[pos]

    for i, pos in enumerate(val_coords):
        X_val[i] = img_x[pos]
        Y_val[i] = img_y[pos]


    X = X[..., np.newaxis]
    Y = Y[..., np.newaxis]
    X_val = X_val[..., np.newaxis]
    Y_val = Y_val[..., np.newaxis]

    return X, Y, X_val, Y_val


def extract_slices(even, odd, train_coords, val_coords, mean, std):
    """
    Extract train and validation volumes and normalize them.
    
    Parameters
    ----------
    even : array(float)
        Even tomogram.
    odd : array(float)
        Odd tomogram.
    train_coords : list(tuple(slice, slice, slice))
        The slices of the train-volumes.
    val_coords : list(tuple(slice, slice, slice))
        The slices of the validation-volumes.
    mean : float
        Mean used for normalization.
    std : float
        Standard deviation used for normalization.
        
    Returns
    -------
    list(array(float))
        Train data X (normalized)
    list(array(float))
        Train data Y (normalized)
    list(array(float))
        Validation data X_val (normalized)
    list(array(float))
        Validation data Y_val (normalized)
    """
    z, y = train_coords[0][0], train_coords[0][1]
    train_vol_dims = (z.stop - z.start, y.stop - y.start)
    z, y = val_coords[0][0], val_coords[0][1]
    val_vol_dims = (z.stop - z.start, y.stop - y.start)
    
    X = np.zeros((len(train_coords), *train_vol_dims), dtype=np.float32)
    Y = np.zeros((len(train_coords), *train_vol_dims), dtype=np.float32)
    X_val = np.zeros((len(val_coords), *val_vol_dims), dtype=np.float32)
    Y_val = np.zeros((len(val_coords), *val_vol_dims), dtype=np.float32)

    img_x = even
    img_x = normalize(img_x, mean, std)
    img_y = odd
    img_y = normalize(img_y, mean, std)

    for i, pos in enumerate(train_coords):
        X[i] = img_x[pos]
        Y[i] = img_y[pos]

    for i, pos in enumerate(val_coords):
        X_val[i] = img_x[pos]
        Y_val[i] = img_y[pos]


    X = X[..., np.newaxis]
    Y = Y[..., np.newaxis]
    X_val = X_val[..., np.newaxis]
    Y_val = Y_val[..., np.newaxis]

    return X, Y, X_val, Y_val


def plotTrainData(X, Y, X_val, Y_val, net_dim):
    train_index = np.random.randint(X.shape[0])
    val_index = np.random.randint(X_val.shape[0])
    if net_dim == '3D':
        plt.figure(figsize=(10,10))
        plt.subplot(2,2,1)
        plt.imshow(X[train_index,int(X.shape[1]/2),:], cmap='gray')
        plt.title('X');
        plt.subplot(2,2,2)
        plt.imshow(Y[train_index,int(X.shape[1]/2),:], cmap='gray')
        plt.title('Y');
        plt.subplot(2,2,3)
        plt.imshow(X_val[val_index,int(X.shape[1]/2),:], cmap='gray')
        plt.title('X_val');
        plt.subplot(2,2,4)
        plt.imshow(Y_val[val_index,int(X.shape[1]/2),:], cmap='gray')
        plt.title('Y_val');
    elif net_dim == '2D':
        plt.figure(figsize=(10,10))
        plt.subplot(2,2,1)
        plt.imshow(X[train_index,:], cmap='gray')
        plt.title('X');
        plt.subplot(2,2,2)
        plt.imshow(Y[train_index,:], cmap='gray')
        plt.title('Y');
        plt.subplot(2,2,3)
        plt.imshow(X_val[val_index,:], cmap='gray')
        plt.title('X_val');
        plt.subplot(2,2,4)
        plt.imshow(Y_val[val_index,:], cmap='gray')
        plt.title('Y_val');        
    else:
        raise Exception('Invalid project dimensionality, please check the value of net_dim') 
