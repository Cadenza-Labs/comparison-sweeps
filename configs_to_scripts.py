import os
import subprocess
import glob
import shutil

if __name__ == "__main__":
    # Ensure the scripts directory exists or create it
    if not os.path.exists('scripts'):
        os.makedirs('scripts')
    
    # List all JSON files in the 'configs' directory
    config_files = glob.glob("configs/*.json")

    # Iterate over each JSON file and call the original script
    for config_file in config_files:
        # Formulate the command to call the original script with the current config file
        cmd = f"python sweep-script-gen.py {config_file}"
        
        # Execute the command
        subprocess.run(cmd, shell=True)

        # Get the name from the config file to identify the output script
        base_name = os.path.basename(config_file).replace('.json', '')
        generated_file = f"sweep-not-291-{base_name}.sh"

        # Move the generated file to the 'scripts' directory
        shutil.move(generated_file, f"scripts/{generated_file}")
    
    print(f"Processed {len(config_files)} JSON files.")
