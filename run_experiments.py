import subprocess

# Define the arguments for the first experiment
first_experiment_args = {
   'model_description': 'internlm-quantised',
   'exp description': 'MSE with batch size 8 and internlm-quantised model',
   'default_device': 2,
   'num_gpus' : 1
}

# Define the arguments for the second experiment
# second_experiment_args = {
#    'model_description': 'internlm-vl2',
#    'exp description': 'MSE with batch size 8 and internlm-vl2 model',
#    'default_device': 1
#    'num_gpus' : 2
# }

# Function to run the script with given arguments
def run_experiment(args):
   command = ['python', 'exp2_1_train_qbench.py']
   for key, value in args.items():
       command.append(f'--{key}={value}')
   subprocess.run(command)

# Run the first experiment
print("Running the first experiment...")
run_experiment(first_experiment_args)

# # Run the second experiment
# print("Running the second experiment...")
# run_experiment(second_experiment_args)