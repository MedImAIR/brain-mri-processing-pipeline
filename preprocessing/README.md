## Data preprocessing

The main rocessing script are isolated in saparate files, applied to the whole directiory, created while dataset preparation in `*.nii` format.

The most of data processing is executed with [ANTS](http://stnava.github.io/ANTs/) with `antspyx`. 

Intensity normalization N4 and SUSAN are executed inside [CaPTk](https://www.med.upenn.edu/cbica/captk/). container.

MRi images `*.nii` conversion is done from the same image with the following command:

`dcm2niix -b n -f CT1 -z y -o ../../../data/orig/$sub/ CT1/`

See example in (datasets folder)[https://github.com/MedImAIR/brain-mri-processing-pipeline/blob/main_clean/datasets/convert_dcm2niix_captk.sh]. 


You need to install [HD-BEt](https://github.com/MIC-DKFZ/HD-BET) for histagram matching and [HD-Bet](https://github.com/MIC-DKFZ/HD-BET) for skull stripping.

### `HD-Bet` usage:

The function is descibed in `4d_skull_stripping.py`:

`command = ["hd-bet", "-i", src_path, "-o", dst_path, "-device", args.device, "-s", "1"]`
`subprocess.call(command)`

#### Preprocessing pipleine:
- `1_inter_modality_registration.py`: Inter-subject image `Rigid` registration;
- `2_resampling_to_image_size.py`: Inter-subject image `Rigid` registration and Interpolation to equal image size (`SRI24` atlas sizing);
- `3_atlas_registration.py`: Inter-subject image `Rigid` registration and `SRI24` atlas registration;
- `4_resampling_to_spacing.py`: Inter-subject image `Rigid` registration and Resampling to equal voxel size [1,1,1] mm;
- `4a_bias_field_correction.sh`: N4 bias-field correction applied after resampling;
- `4b_denoising.sh`: SUSAN denoising applied after interpolation;
- `4c_histogram_matching.py`: histogram normalization in [torchio](https://torchio.readthedocs.io/)
- `4d_skull_stripping.py`: Skull-stripping with [HD-Bet](https://github.com/MIC-DKFZ/HD-BET)

Additional test for folder similarity, to ensure expected preprocessing step was completed:
- `quality_test.py`: checks for equal sizing, absence of Nan or `inf` values, check the difference from other datasets

#### Utils:
In  `utils` folder you will find nesessary files: atlases, and training seed to be accaounted in statistical comparison. 

#### Logs:
All stetps are logged in `python` scripts and stored in folders with dataset names.
