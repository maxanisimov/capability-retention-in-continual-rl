"""
Run demo.py with all configurations from demo_configs.yaml

This script iterates through all configurations and runs the demo
experiment for each one.
"""
import yaml
import subprocess
import sys

# Load all configurations
with open('demo_configs.yaml', 'r') as f:
    DEMO_CONFIGS = yaml.safe_load(f)

config_names = list(DEMO_CONFIGS.keys())
print(f"Found {len(config_names)} configurations: {config_names}")

# You can also specify which safe_state_action_data_name to use
# Options: 'Safe Training Data' or 'Safe Optimal Policy Data'
safe_data_types = ['Safe Training Data']  # Add 'Safe Optimal Policy Data' to run both

for safe_data_type in safe_data_types:
    for cfg_name in config_names:
        print(f"\n{'='*80}")
        print(f"Running experiment: {cfg_name} with {safe_data_type}")
        print(f"{'='*80}\n")
        
        # Modify demo.py to use this config
        # Read current demo.py
        with open('demo.py', 'r') as f:
            demo_code = f.read()
        
        # Replace the config name
        lines = demo_code.split('\n')
        for i, line in enumerate(lines):
            if line.startswith("cfg_name = "):
                lines[i] = f"cfg_name = '{cfg_name}'"
            elif line.startswith("safe_state_action_data_name = "):
                lines[i] = f"safe_state_action_data_name = '{safe_data_type}' #  'Safe Training Data'  or 'Safe Optimal Policy Data'"
            elif line.startswith("save_results = "):
                lines[i] = "save_results = True"  # Enable saving for batch runs
        
        # Write modified demo.py temporarily
        modified_code = '\n'.join(lines)
        with open('demo_temp.py', 'w') as f:
            f.write(modified_code)
        
        # Run the modified script
        try:
            result = subprocess.run(
                ['python', 'demo_temp.py'],
                capture_output=False,
                text=True,
                check=True
            )
            print(f"✓ Successfully completed: {cfg_name} with {safe_data_type}")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed: {cfg_name} with {safe_data_type}")
            print(f"Error: {e}")
            continue

print(f"\n{'='*80}")
print("All experiments completed!")
print(f"{'='*80}")
