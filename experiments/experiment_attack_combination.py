from federated_gan import federated_gan
from datetime import datetime
import time
from multiprocessing import Process

Round=300
TEST_ACCURACY_THRESHOLD=0.0
CLASS_TO_GENERATE=9
CLASS_TO_ATTACK=7
MALICIOUS_ATTACKER=0
WARMUP_TRAINING=False
DATASET="MNIST"
DP=False
distribution=0.0
GENERATOR_ATTACK=True
run_name=None

for DATASET in ["MNIST", "FMNIST"]:
        experiment_name=datetime.strftime(datetime.now(), "%Y_%m_%d_%H:%M:%S")+"_"+DATASET+"_attack_combination"

        for CLASS_TO_ATTACK in range(10):

                for CLASS_TO_GENERATE in range(10):
                        
                        if CLASS_TO_GENERATE == CLASS_TO_ATTACK:
                                continue

                        if CLASS_TO_ATTACK == 0:
                                MALICIOUS_ATTACKER = 3
                        else:
                                MALICIOUS_ATTACKER = 0

                        run_name = "gen_{}_att_{}".format(CLASS_TO_GENERATE, CLASS_TO_ATTACK)
                        proc = Process(target=federated_gan,
                        args=(Round, TEST_ACCURACY_THRESHOLD, CLASS_TO_GENERATE, CLASS_TO_ATTACK, MALICIOUS_ATTACKER, WARMUP_TRAINING, DATASET, DP, distribution, 
                        GENERATOR_ATTACK, experiment_name, run_name))
                        proc.start()
                        proc.join()