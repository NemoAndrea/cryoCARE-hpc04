# cryoCARE-hpc04

This repository holds an adapter version of the cryoCARE scripts originally developed by [the Jug Group](https://github.com/juglab) used to denoise cryo-electron tomograms.  Scripts require that the provided images originate from a `direct electron detector` (e.g. Gatan K2 Summit), in subframe form. This is because we need to be able to generate 2 images containing half the total signal at each tomogram `tilt angle`.

The scripts were built to be run on a `hpc`, but there is no reason why they cannot be run on a personal compute device. It will just take a while. Having a **fast graphics card** will really help in managing training time.

## Installation

It is recommended that you install the required dependencies via a separate **[Anaconda/Miniconda](https://docs.anaconda.com/anaconda/install/)** environment. The required dependencies are listed in `requirements.yml` and can easily be installed in a new conda environment using:

``` 
conda env create -f requirements.yml -n <name_for_your_new_environment>
```

Then run 

`conda activate <name_for_your_new_environment>` every time you want to use the scripts to make sure you are using that environment.

In addition, you need to have `IMOD` and `MotionCor2` installed on your system. 

MotionCor2 (tested with `v1.0.5`) can be downloaded from the [UCSF EM core download page](https://emcore.ucsf.edu/ucsf-software). You will have to complete a form and after that you will be provided with a download.

IMOD (tested with `v4.9.2` ) can be downloaded from the [IMOD website](https://bio3d.colorado.edu/imod/download.html). _If you are running from Windows, you will also need to install [Cygwin](https://www.cygwin.com/) to use it_

> It is possible to make things work for IMOD `4.11.x` using the `etomo --namingstyle 0` compatibility argument. 

## Structure and use

The main workflow is based on `Jupyter notebooks`. These can be found in the `T2T` directory. 

> The `.sh` files (root directory) are specific to the cluster an its installed packages and are therefore not very useful for general use

I recommend you copy the `T2T` directory (and its contents) for each dataset you want to process. That way you can always come back to the notebooks and check what options you picked, etc.

> So, clone this repository, do not touch the code in it, and just **copy the `T2T` folder** every time you want to process a new dataset. Treat this repository as a template, avoid editing files directly here

So, for each tilt series, create a folder with a name for that tomogram (e.g. `tomogram_<condition>_<date>_number_4`), and copy the `T2T` folder into it. You should then have the following structure.
```
tomogram_name_or_similar
|   └─── T2T
|           └─── frames/
|           |---- [xx]_notebook.ipynb 
|           |---- [xy]_notebook.ipynb 
|           ... 
```

Then copy your tilt images in stack form (`.tif` or `.mrc`) into the `frames/` folder. After having copied the data, you run the numbered notebook files (i.e. `.ipynb`) in numerical order. This means starting with `01_Split_Frames.ipynb`. 

If you have never used a `Jupyter notebook`, there are many online introductions such as [this one](https://realpython.com/jupyter-notebook-introduction/#starting-the-jupyter-notebook-server). To start a notebook, simply navigate into the `T2T` directory in your`Anaconda` terminal and type

```
jupyter notebook
```

This should open a notebook in browser, and you can follow the instructions from there until completion. Go through the files and adjust paths as needed, and move on to the next one after completion.

A short summary of what these notebooks do can be found below. I suggest you read that first before running anything.

## Procedure in brief

The list of steps we have to do (with corresponding notebook number) is as follows:
* [01] Split and align the raw DED frames into `even` and `odd` set (alignment using MotionCor2) 
* [02] Reconstruct two tomograms 
* [03] Sample subvolumes of the tomograms into training and validation set 
* [04] Train the network **(this takes the longest)**
* [05] Run the denoiser network on tomograms 

### Extras

* [05b] this will allow you to batch process a series of reconstructed tomograms. (denoise many using a single network)
* [06] free up storage space

## 2D training mode

This modified version of cryoCARE has the option to denoise using only 2D slices. This can be beneficial in combination with `SIRT` tomogram reconstruction  (an option in IMOD). This is done by selecting `type2D` in `[03]_training_data_generation` .

Give it a shot in combination with `SIRT` and compare it to the baseline 3D without SIRT. 

## Credit

This repository is based on the original code available [at the juglab github](https://github.com/juglab/cryoCARE_T2T), and their [associated publication](https://ieeexplore.ieee.org/document/8759519): 

```
Buchholz, Tim-Oliver, et al. "Cryo-care: Content-aware image restoration for cryo-transmission electron microscopy data." 2019 IEEE 16th International Symposium on Biomedical Imaging (ISBI 2019). IEEE, 2019.
```


