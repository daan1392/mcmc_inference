from tqdm import tqdm
import os, glob

# Could not fully test this script

def mv_file(name, newname, job_name):
        if os.path.isfile(name):
           dst = job_name + "." + newname
           dst=os.path.join("results", dst)
           os.replace(name,dst)

def remove_sammy_files():
    inp='input'
    if os.path.isfile(inp):os.remove(inp)
    out='output'
    if os.path.isfile(out):os.remove(out)
    files = glob.glob("SAM*")
    for  f in files : os.remove(f)

def get_program_name(name):
    sammy_inst= os.environ.get('SAMMY_INST')
    program=os.path.join(sammy_inst,name)
    os.makedirs('results', exist_ok=True)
    return program

def move_standard_files(job_name):
    mv_file('SAMMY.LST','lst',job_name)
    mv_file('SAMMY.LPT','lpt',job_name)

def run_sammy_explicit_filenames(inp_file, par_file, dat_file, cov_file, job_name, emin, emax):
        sammy =  get_program_name('sammy')

        # driver input/output files to run SAMMY optionally defined
        inp='input'
        out='output'
        remove_sammy_files()

        with open(inp,'w+') as f:
                f.write(f'{inp_file}\n')
                f.write(f'{par_file}\n')
                f.write(f'{dat_file} {emin},{emax}\n')
                if os.path.isfile(f'{cov_file}'):
                        f.write(f'{cov_file}\n')

        os.system(f'{sammy}<{inp}>{out}')

        move_standard_files(job_name)
        remove_sammy_files()

if __name__ == "__main__":
    input_folder = "data/raw/Pérez-Maroto(2025)/cr53_thin_training/"
    title = "cr53_thin"
    pert_folder = "data/raw/Pérez-Maroto(2025)/cr53_thin_training/perturbed_parameters/"
    samples = 10
    emin = 1000
    emax = 10000

    for i in tqdm(range(samples)):
        perturbed_par_file = f"{pert_folder}{title}_{i}.par"
        job_name = f"cr53_thin{i}"
        run_sammy_explicit_filenames(f"{input_folder}{title}.inp", perturbed_par_file ,f"{input_folder}cr53thin_preyield_fixL_18394_17ns_250bpd.dat", '', job_name, emin, emax)