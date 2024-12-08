from federated_gan import federated_gan
from datetime import datetime
import time
from multiprocessing import Process

Round = 300
TEST_ACCURACY_THRESHOLD = 0.0
CLASS_TO_GENERATE = 4
CLASS_TO_ATTACK = 9
MALICIOUS_ATTACKER = 0
WARMUP_TRAINING = False
DATASET = "MNIST"
DP = False
distribution = "non-iid"
GENERATOR_ATTACK = True
experiment_name = None
run_name = None
l2_norm_clip = 1.5
noise_multiplier = 1.3
delta = 1e-6
distributionFolder = "dirichlet_distributions_01"
POISON_ATTACK = True
GPU = (0,)
NUM_TO_POISON = 500
adam_for_non_private_training = True
sgd_for_dp_training = True
global_lr = 1e-3

distributions = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 10.0]
mnist_pairs = [[0, 1], [4, 9], [8, 4]]
fmnist_pairs = [[6, 7], [7, 5], [0, 8]]


for DATASET in ["MNIST", "FMNIST"]:
    for stat_rep in range(3):
        experiment_name = (
            datetime.strftime(datetime.now(), "%Y_%m_%d_%H:%M:%S")
            + "_"
            + DATASET
            + "_distribution_influence"
        )

        pairs = mnist_pairs if DATASET == "MNIST" else fmnist_pairs

        for pair in pairs:
            for distribution in distributions:

                if pair[1] == 0:
                    MALICIOUS_ATTACKER = 3
                else:
                    MALICIOUS_ATTACKER = 0

                run_name = "gen_{}_att_{}_a_{}".format(pair[0], pair[1], distribution)
                CLASS_TO_GENERATE = pair[0]
                CLASS_TO_ATTACK = pair[1]
                proc = Process(
                    target=federated_gan,
                    args=(
                        Round,
                        TEST_ACCURACY_THRESHOLD,
                        CLASS_TO_GENERATE,
                        CLASS_TO_ATTACK,
                        MALICIOUS_ATTACKER,
                        WARMUP_TRAINING,
                        DATASET,
                        DP,
                        distribution,
                        GENERATOR_ATTACK,
                        experiment_name,
                        run_name,
                        l2_norm_clip,
                        noise_multiplier,
                        delta,
                        distributionFolder,
                        POISON_ATTACK,
                        GPU,
                        NUM_TO_POISON,
                        adam_for_non_private_training,
                        sgd_for_dp_training,
                        global_lr,
                    ),
                )
                proc.start()
                proc.join()
