# necessary experiments for initial tests -- see whether it will work on the edinburgh computers

import os
import sys
generated_name = str(sys.argv[1])
log_path = str(sys.argv[2])
save_path = str(sys.argv[3])
seeds = 5
base_call = "python edinburgh_weight_symmetry.py"
output_file = open(generated_name, "w")

exp_name = "backward_weight_update"
for s in range(seeds):
    lpath = log_path + "/"+str(exp_name) + "/" + str(s)
    spath = save_path + "/" + str(exp_name) + "/" + str(s)
    final_call = base_call + " " + str(lpath) + " " + str(spath) + " " + "True" + " " + "True" + " " + "True"
    print(final_call, file=output_file)

exp_name = "no_backward_weight_update"
for s in range(seeds):
    lpath = log_path + "/"+str(exp_name) + "/" + str(s)
    spath = save_path + "/" + str(exp_name) + "/" + str(s)
    final_call = base_call + " " + str(lpath) + " " + str(spath) + " " + "True" + " " + "False" + " " + "True"
    print(final_call, file=output_file)

exp_name = "no_backward_nonlinearities"
for s in range(seeds):
    lpath = log_path + "/"+str(exp_name) + "/" + str(s)
    spath = save_path + "/" + str(exp_name) + "/" + str(s)
    final_call = base_call + " " + str(lpath) + " " + str(spath) + " " + "True" + " " +"True" +" " + "False"
    print(final_call, file=output_file)

base_call = "python edinburgh_error_dispersion.py"
exp_name = "error_dispersion_with_weight_updates"
for s in range(seeds):
    lpath = log_path + "/"+str(exp_name) + "/" + str(s)
    spath = save_path + "/" + str(exp_name) + "/" + str(s)
    final_call = base_call + " " + str(lpath) + " " + str(spath) + " " + "True"
    print(final_call, file=output_file)

exp_name = "error_dispersion_without_weight_updates"
for s in range(seeds):
    lpath = log_path + "/"+str(exp_name) + "/" + str(s)
    spath = save_path + "/" + str(exp_name) + "/" + str(s)
    final_call = base_call + " " + str(lpath) + " " + str(spath) + " " + "False"
    print(final_call, file=output_file)
