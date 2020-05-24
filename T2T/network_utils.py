import numpy as np
from generate_train_data import normalize, compute_mean_std, denormalize


def predict_tomogram(model, tomogram, mean, std, tiles = (8,8)):
    print('> Applying denoiser to tomogram, using ' + str(model.config.n_dim) +'D network')  
    print(model.config.axes)
    if (model.config.n_dim == 3):
         # axes for a volume network should be 'ZYX' (in [04])
        tomo_denoised = denormalize(model.predict(tomogram, axes=model.config.axes[0:3], n_tiles=(8,8,8), normalizer=None), mean, std)
        
        return tomo_denoised
    elif (model.config.n_dim == 2):      
        tomo_denoised = np.zeros(tomogram.shape) # intialise
        num_tomo_slices = tomogram.shape[0] # how many z slices to process        
        for i in range(num_tomo_slices):
            print('Processing slice ' + str(i) + ' of ' + str(num_tomo_slices))
            # axes for a slice network should be 'XY' (in [04])
            tomo_denoised[i,:,:] = denormalize(model.predict(tomogram[i,:,:], axes=model.config.axes[0:2], n_tiles=tiles, normalizer=None), mean, std)
            
        return tomo_denoised        
    else:
        raise Exception('Could not detect dimensionality of trained network')