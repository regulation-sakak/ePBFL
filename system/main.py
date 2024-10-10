## original code produced by https://github.com/TsingZ0/HtFLlib
# !/usr/bin/env python
import torch
import platform
import argparse
import os
import json
import time
import warnings
import numpy as np
import logging

from flcore.servers.serverproto import FedProto
from flcore.servers.servertgp import FedTGP

from utils.result_utils import average_data, plotting_trial_result
from utils.mem_utils import MemReporter

logger = logging.getLogger()
logger.setLevel(logging.ERROR)

warnings.simplefilter("ignore")

torch.manual_seed(0)

def run(args):
    time_list = []
    reporter = MemReporter()

    for i in range(args.prev, args.times):
        print(f"\n============= Running time: {i}th =============")
        print("Creating server and clients ...")
        start = time.time()

        # Generate args.models
        if args.model_family == "Ht0":
            args.models = [
                'resnet8(num_classes=args.num_classes)',
                'mobilenet_v2(pretrained=False, num_classes=args.num_classes)',
                'torchvision.models.shufflenet_v2_x1_0(pretrained=False)',
                'torchvision.models.efficientnet_b0(pretrained=False)',
            ]

        else:
            raise NotImplementedError

        for model in args.models:
            print(model)

        # select algorithm
        if args.algorithm == "FedProto":
            server = FedProto(args, i)

        elif args.algorithm == "FedTGP":
            server = FedTGP(args, i)

        else:
            raise NotImplementedError

        server.train()

        time_list.append(time.time() - start)

        plotting_trial_result(args)

    print(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")

    # Global average
    average_data(dataset=args.dataset, algorithm=args.algorithm, goal=args.goal, times=args.times)

    print("All done!")

    reporter.report()


if __name__ == "__main__":
    total_start = time.time()

    parser = argparse.ArgumentParser()
    # general
    parser.add_argument('-go', "--goal", type=str, default="test",
                        help="The goal for this experiment")
    parser.add_argument('-dev', "--device", type=str, default="cuda",
                        choices=["cpu", "cuda"])
    parser.add_argument('-did', "--device_id", type=str, default="0")
    parser.add_argument('-data', "--dataset", type=str, default="Cifar10")
    parser.add_argument('-nb', "--num_classes", type=int, default=10)
    parser.add_argument('-m', "--model_family", type=str, default="Ht0")

    parser.add_argument('-lbs', "--batch_size", type=int, default=32)
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.01,
                        help="Local learning rate")
    parser.add_argument('-ld', "--learning_rate_decay", type=bool, default=False)
    parser.add_argument('-ldg', "--learning_rate_decay_gamma", type=float, default=0.99)
    parser.add_argument('-gr', "--global_rounds", type=int, default=300)
    parser.add_argument('-ls', "--local_epochs", type=int, default=1,
                        help="Multiple update steps in one local epoch.")
    parser.add_argument('-algo', "--algorithm", type=str, default="FedProto")
    parser.add_argument('-jr', "--join_ratio", type=float, default=1.0,
                        help="Ratio of clients per round")
    parser.add_argument('-rjr', "--random_join_ratio", type=bool, default=False,
                        help="Random ratio of clients per round")
    parser.add_argument('-nc', "--num_clients", type=int, default=20,
                        help="Total number of clients")
    parser.add_argument('-pv', "--prev", type=int, default=0,
                        help="Previous Running times")
    parser.add_argument('-t', "--times", type=int, default=1,
                        help="Running times")
    parser.add_argument('-eg', "--eval_gap", type=int, default=1,
                        help="Rounds gap for evaluation")
    parser.add_argument('-sfn', "--save_folder_name", type=str, default='temp')
    parser.add_argument('-ab', "--auto_break", type=bool, default=False)
    parser.add_argument('-fd', "--feature_dim", type=int, default=500)
    parser.add_argument('-vs', "--vocab_size", type=int, default=98635)
    parser.add_argument('-ml', "--max_len", type=int, default=200)

    # practical
    parser.add_argument('-cdr', "--client_drop_rate", type=float, default=0.0,
                        help="Rate for clients that train but drop out")
    parser.add_argument('-tsr', "--train_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when training locally")
    parser.add_argument('-ssr', "--send_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when sending global model")
    parser.add_argument('-ts', "--time_select", type=bool, default=False,
                        help="Whether to group and select clients at each round according to time cost")
    parser.add_argument('-tth', "--time_threthold", type=float, default=10000,
                        help="The threthold for droping slow clients")
    # FedProto
    parser.add_argument('-lam', "--lamda", type=float, default=10)

    # FedTGP
    parser.add_argument('-se', "--server_epochs", type=int, default=100)
    parser.add_argument('-mart', "--margin_threthold", type=float, default=100.0)

    # Proposed Components
    parser.add_argument('-cps', "--add_cps", action='store_true', help='prototype masking')
    parser.add_argument('-rcps', "--cps_ratio", type=int, default=1)
    parser.add_argument('-ppa', "--add_ppa", action='store_true', help='class-wise weight')
    parser.add_argument('-cpkd', "--add_cpkd", action='store_true', help='Proportional KD')
    parser.add_argument('-pms', "--prototype_shaping", type=str, default='no')
    parser.add_argument('-pmd', "--prototype_masking_decay", type=float, default=1e-4)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable.\n")
        if platform.system() == 'Linux':
            if not torch.cuda.is_available():
                args.device = "cpu"
        else:
            if torch.backends.mps.is_available():
                args.device = "mps"

    DATA_PATH = os.path.join(os.path.dirname(os.getcwd()), 'dataset')
    dataset_path = os.path.join(DATA_PATH, args.dataset)
    file_path = os.path.join(dataset_path, 'config.json')
    with open(file_path, 'r') as file:
        data = json.load(file)
        args.num_classes = data['num_classes']
        args.dataset_type = data['partition']
        data_statistic = data['Size of samples for labels in clients']

    data_dist = np.zeros((args.num_clients, args.num_classes))
    for i in range(args.num_clients):
        client_data_dist_np = np.array(data_statistic[i])
        data_dist[i, client_data_dist_np[:, 0]] = client_data_dist_np[:, 1]

    algo = args.dataset + "_" + args.algorithm
    result_path = "../results/"
    algo = algo + "_" + args.goal
    file_path = result_path + "{}_data_distribution.npy".format(algo)

    np.save(file_path, data_dist)
    args.data_dist = data_dist.tolist()

    print("=" * 50)

    print("Algorithm: {}".format(args.algorithm))
    print("Local batch size: {}".format(args.batch_size))
    print("Local steps: {}".format(args.local_epochs))
    print("Local learing rate: {}".format(args.local_learning_rate))
    print("Local learing rate decay: {}".format(args.learning_rate_decay))
    if args.learning_rate_decay:
        print("Local learing rate decay gamma: {}".format(args.learning_rate_decay_gamma))
    print("Total number of clients: {}".format(args.num_clients))
    print("Clients join in each round: {}".format(args.join_ratio))
    print("Clients randomly join: {}".format(args.random_join_ratio))
    print("Client drop rate: {}".format(args.client_drop_rate))
    print("Client select regarding time: {}".format(args.time_select))
    if args.time_select:
        print("Time threthold: {}".format(args.time_threthold))
    print("Running times: {}".format(args.times))
    print("Dataset: {}".format(args.dataset))
    print("Data Heterogeneity: {}".format(args.dataset_type))
    print("Number of classes: {}".format(args.num_classes))
    print("Backbone: {}".format(args.model_family))
    print("Using device: {}".format(args.device))
    print("Auto break: {}".format(args.auto_break))
    if not args.auto_break:
        print("Global rounds: {}".format(args.global_rounds))

    if args.algorithm == 'FedProto' or args.algorithm == 'FedTGP':
        print("prototype MSE weight: {}".format(args.lamda))
        print("TGP: margin threshold: {}".format(args.margin_threthold))
        print("TGP: server epoch: {}".format(args.server_epochs))
        print("Class-Wise Prototype Sparsification: {}".format(args.add_cps))
        print("CPS ratio: {}".format(args.cps_ratio))
        print("Privacy-Preserving Prototype Distillation: {}".format(args.add_ppa))
        print("Class-Proportional Knowledge Distillation: {}".format(args.add_cpkd))

    print("=" * 50)

    run(args)
