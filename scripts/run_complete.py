import subprocess
import argparse
import os
import sys

MODULE_SEQUENCE = [
    # "mcmc_inference.01_gp_train",
    "mcmc_inference.02_mcmc",
    "mcmc_inference.03_mcmc_plots"
]

def run_step(module_name, config_path):
    """Executes a single Python module using subprocess."""
    
    command = [
        sys.executable, 
        "-m", 
        module_name, 
        "--config", 
        config_path
    ]
    
    print(f"\n====> STARTING STEP: {module_name}")
    try:
        subprocess.run(command, check=True)
        print(f"----> SUCCESS: {module_name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n!!!! FAILURE: {module_name} failed with exit code {e.returncode}")
        print("!!!! Check logs above for specific error details.")
        return False
    except FileNotFoundError:
        print(f"\n!!!! FAILURE: Python environment error. Could not find module {module_name}")
        return False

def run_daan_workflow(config_path):
    """Runs the entire DAAN workflow sequentially."""
    
    header_text = rf"""
==================================================

__/\\\\\\\\\\\\________________________________________________        
 _\/\\\////////\\\______________________________________________       
  _\/\\\______\//\\\_____________________________________________      
   _\/\\\_______\/\\\__/\\\\\\\\\_____/\\\\\\\\\_____/\\/\\\\\\___     
    _\/\\\_______\/\\\_\////////\\\___\////////\\\___\/\\\////\\\__    
     _\/\\\_______\/\\\___/\\\\\\\\\\____/\\\\\\\\\\__\/\\\__\//\\\_   
      _\/\\\_______/\\\___/\\\/////\\\___/\\\/////\\\__\/\\\___\/\\\_  
       _\/\\\\\\\\\\\\/___\//\\\\\\\\/\\_\//\\\\\\\\/\\_\/\\\___\/\\\_ 
        _\////////////______\////////\//___\////////\//__\///____\///__
        
==================================================        
  Project Config: {config_path}
  Running Steps:  {len(MODULE_SEQUENCE)}
  Current Time:   {os.popen('date /t').read().strip()} {os.popen('time /t').read().strip()}
--------------------------------------------------
"""
    print(header_text)
    
    for module in MODULE_SEQUENCE:
        if not run_step(module, config_path):
            print("\nWORKFLOW ABORTED due to failure in the previous step.")
            return

    print("\n==================================================")
    print("WORKFLOW COMPLETED SUCCESSFULLY.")
    print("Posterior samples and plots are ready for analysis.")
    print("==================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DAAN: Automated GP and MCMC Inference Workflow.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to the configuration file.")
    args = parser.parse_args()
    
    run_daan_workflow(args.config)