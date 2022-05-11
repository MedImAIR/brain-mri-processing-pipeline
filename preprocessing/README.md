## Data preprocessing

The main rocessing script are isolated in saparate files, applied to the whole directiory, created while dataset preparation in `Nifty` format.

The most of data processing is executed with [ANTS](http://stnava.github.io/ANTs/). Intensity normalization N4 and SUSAN are executed inside [CaPTk](https://www.med.upenn.edu/cbica/captk/). container.
You need to install [HD-BEt](https://github.com/MIC-DKFZ/HD-BET) for histagram matching and [HD-BEt](https://github.com/MIC-DKFZ/HD-BET) for skull stripping.

#### Preprocessing pipleine:
- `1_reg.py`: Inter-subject image `Rigid` registration;
- `2a_interp.py`: Inter-subject image `Rigid` registration and Interpolation to equal image size (atlas sizing);
- `2b_n4.sh`: N4 bias-field correction applied after interpolation;
- `2c_n4_susan.sh`: SUSAN denoising applied after interpolation;
- `3a_atlas.py`: Inter-subject image `Rigid` registration and `SRI24` atlas registration;
- `3b_n4.sh`: N4 bias-field correction applied after registration;
- `3c_n4_susan.sh`: N4 bias-field correction and SUSAN denoising applied after registration;
- `3d_susan.sh`: SUSAN denoising applied after registration;
- `4a_resamp.py`: Inter-subject image `Rigid` registration and Resampling to equal voxel size [1,1,1] mm;
- `4b_n4.sh`: N4 bias-field correction applied after resampling;
- `4d_susan.sh`: SUSAN denoising applied after interpolation;
- `5_ss.py`: Skull-stripping with [HD-Bet](https://github.com/MIC-DKFZ/HD-BET)
- `6_hist.py`: histogram normalization in [torchio](https://torchio.readthedocs.io/)

#### Params:
In  `params` folder you will find nesessary files: atlases, and training seed to be accaounted in statistical comparison. 

#### Logs:
All stetps are logged in python scripts and stored in folders with dataset names.
