# experiment generation file for running all biological plausibility experiments that I want
#these are as follows:
# 100 batches train/test on small batches but reasonable 100 * 20 = 2000 digits.
#1.) standard baselines
#2.) without weight symmetry - i.e. but without sign concordance
#3.) without weight symmetry AND no updates i.e. FA
#4.) with updates + sign concordance is 1
#5.) without updates + sign_concordance is 1
#6.) Without updates sign_concordance 0.8,0.6
#7.) without nonlinearities but with backward weights
#8.) without backward weights and without nonlinearities

import os
import sys
generated_name = str(sys.argv[1])
log_path = str(sys.argv[2])
save_path = str(sys.argv[3])
seeds = 5
base_call = (f"python biological_plausibility_experiments.py ")
output_file = open(generated_name, "w")
exp_name = "standard"
for s in range(seeds):
    lpath = log_path + "/"+str(exp_name) + "/" + str(s)
    spath = save_path + "/" + str(exp_name) + "/" + str(s)
    final_call = (f"{base_call} "
    f" --log_path {lpath}"
    f" --save_path {spath}"
    )
    print(final_call, file=output_file)

exp_name = "without_weight_symmetry"
for s in range(seeds):
    lpath = log_path + "/"+str(exp_name) + "/" + str(s)
    spath = save_path + "/" + str(exp_name) + "/" + str(s)
    final_call = (f"{base_call} "
    f" --log_path {lpath}"
    f" --save_path {spath}"
    f" --weight_symmetry False"
    )
    print(final_call, file=output_file)

exp_name = "feedback_alignment"
for s in range(seeds):
    lpath = log_path + "/"+str(exp_name) + "/" + str(s)
    spath = save_path + "/" + str(exp_name) + "/" + str(s)
    final_call = (f"{base_call} "
    f" --log_path {lpath}"
    f" --save_path {spath}"
    f" --weight_symmetry False"
    f" --backward_weight_update False"
    )
    print(final_call, file=output_file)

exp_name = "sign_concordance_update"
for s in range(seeds):
    lpath = log_path + "/"+str(exp_name) + "/" + str(s)
    spath = save_path + "/" + str(exp_name) + "/" + str(s)
    final_call = (f"{base_call} "
    f" --log_path {lpath}"
    f" --save_path {spath}"
    f" --weight_symmetry False"
    f" --backward_weight_update True"
    f" --sign_concordance_prob 1"
    )
    print(final_call, file=output_file)

exp_name = "sign_concordance_no_update"
for s in range(seeds):
    lpath = log_path + "/"+str(exp_name) + "/" + str(s)
    spath = save_path + "/" + str(exp_name) + "/" + str(s)
    final_call = (f"{base_call} "
    f" --log_path {lpath}"
    f" --save_path {spath}"
    f" --weight_symmetry False"
    f" --backward_weight_update False"
    f" --sign_concordance_prob 1"
    )
    print(final_call, file=output_file)

exp_name = "sign_concordance_08_no_update"
for s in range(seeds):
    lpath = log_path + "/"+str(exp_name) + "/" + str(s)
    spath = save_path + "/" + str(exp_name) + "/" + str(s)
    final_call = (f"{base_call} "
    f" --log_path {lpath}"
    f" --save_path {spath}"
    f" --weight_symmetry False"
    f" --backward_weight_update False"
    f" --sign_concordance_prob 0.8"
    )
    print(final_call, file=output_file)

exp_name = "sign_concordance_06_no_update"
for s in range(seeds):
    lpath = log_path + "/"+str(exp_name) + "/" + str(s)
    spath = save_path + "/" + str(exp_name) + "/" + str(s)
    final_call = (f"{base_call} "
    f" --log_path {lpath}"
    f" --save_path {spath}"
    f" --weight_symmetry False"
    f" --backward_weight_update False"
    f" --sign_concordance_prob 0.8"
    )
    print(final_call, file=output_file)

exp_name = "no_nonlinearities"
for s in range(seeds):
    lpath = log_path + "/"+str(exp_name) + "/" + str(s)
    spath = save_path + "/" + str(exp_name) + "/" + str(s)
    final_call = (f"{base_call} "
    f" --log_path {lpath}"
    f" --save_path {spath}"
    f" --weight_symmetry True"
    f" --backward_weight_update True"
    f" --use_backward_nonlinearity False"
    )
    print(final_call, file=output_file)

exp_name = "no_nonlinearities_no_weight_symmetry"
for s in range(seeds):
    lpath = log_path + "/"+str(exp_name) + "/" + str(s)
    spath = save_path + "/" + str(exp_name) + "/" + str(s)
    final_call = (f"{base_call} "
    f" --log_path {lpath}"
    f" --save_path {spath}"
    f" --weight_symmetry False"
    f" --backward_weight_update True"
    f" --use_backward_nonlinearity False"
    )
    print(final_call, file=output_file)

output_file.close()
print("Experiment File Generated")
