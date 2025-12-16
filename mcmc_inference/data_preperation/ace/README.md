## Integral experiment training
The functional relationship between an input variable and effective multiplication factor is obtained by uniform sampling of the input space. ENDF-6 files are perturbed and SANDY is used to generate ACE files ready for use in SERPENT. The following steps have to be undertaken:
1. Create a template file where you store a placeholder for the value you want to replace for example: {GG1}
2. Generate perturbed ENDF-6 files using 00_generate_perturbed_endfs.py
3. Convert the ENDF-6 files into ACE files using 01_generate_aces.py
4. Update header in ACE files so it is consistent with the file name. This can be done with update_header_ace.py for each file in terminal, so for file in folder do; python 02_update_header_ace.py; done (to be included directly in the creation of ACE files)