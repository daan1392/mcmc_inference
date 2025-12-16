## Microscopic measurement training
The SAMMY input files are stored in each folder corresponding to an experiment. To run SAMMY, first configure the path to the SAMMY bin where the executable is located and make sure SAMMY works. The files were tested using SAMMY v8.1.0.
```bash
export SAMMY_INST="/path/to/SAMMY-8.1.0/bin_Lin"
export LD_LIBRARY_PATH="/path/to/SAMMY-8.1.0/lib:$LD_LIBRARY_PATH"
```

The training data is prepared in the folder /data/raw/Pérez-Maroto(2025)/cr53_thin_training/. To be consistent with the ACE data, jeff-4.0 parameters were taken.

## Acknowledgments
We would like to thank Pablo Pérez-Maroto and the nTOF collaboration for their support in providing the SAMMY inputs and measurement data.