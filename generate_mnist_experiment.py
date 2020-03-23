import os
import sys
generated_name = str(sys.argv[1])
log_path = str(sys.argv[2])
save_path = str(sys.argv[3])
seeds = 5
base_call = "python run_mnist_experiments.py"
output_file = open(generated_name, "w")
for s in range(seeds):
    lpath = log_path + "_" + str(s)
    spath = save_path + "_" + str(s)
    final_call = base_call + " " + str(lpath) + " " + str(spath)
    print(final_call, file=output_file)

print("Experiment File Generated")
