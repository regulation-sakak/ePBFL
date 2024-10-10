import time
import os
import numpy as np
from itertools import combinations
import torch
from flcore.clients.clientproto import clientProto
from flcore.servers.serverbase import Server
from flcore.clients.clientbase import load_item, save_item
from utils.data_utils import read_client_data
from collections import defaultdict

import matplotlib.pyplot as plt
import seaborn as sns

class FedProto(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientProto)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        self.num_classes = args.num_classes
        self.i_i = 0

        # set masking vector
        if args.add_cps:
            self.mask_dict = {}
            self.mask_idx_dict = {}
            length = args.feature_dim // args.num_classes

            vector_name = f'mask_idx_dict_fd_{args.feature_dim}_nc_{args.num_classes}_csr_{args.cps_ratio}.pth'

            if os.path.exists(vector_name):
                self.mask_idx_dict = torch.load(vector_name)
            else:
                per_mask = int(args.num_classes * (args.cps_ratio/100))
                vectors, _ = find_optimal_vectors_greedy(args.num_classes, per_mask)
                for key in range(args.num_classes):
                    mask_500d = np.zeros(args.feature_dim)
                    for i, val in enumerate(vectors[key]):
                        start_idx = i * length
                        end_idx = start_idx + length
                        mask_500d[start_idx:end_idx] = val
                    self.mask_dict[key] = torch.tensor(mask_500d.astype((np.float32)))
                    self.mask_idx_dict[key] = list(np.nonzero(mask_500d))
                torch.save(self.mask_idx_dict, vector_name)
            print("Finished creating Mask Index.")

    def train(self):
        print(f"\n-------------Local Models-------------")
        for c in self.clients:
            print(f'Client {c.id}: {c.model_name}')
            if self.args.add_cps:
                c.set_mask(self.mask_idx_dict)

        for i in range(self.global_rounds + 1):
            self.i_i = i
            s_t = time.time()
            self.selected_clients = self.select_clients()

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate heterogeneous models")
                self.evaluate()

            print("\nLocal Training.")
            for client in self.selected_clients:
                client.train()

            print("\nPrototype Aggregation.")
            self.receive_protos()

            self.Budget.append(time.time() - s_t)
            print('-' * 50, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        print(max(self.rs_test_acc))
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))

        self.save_results()

    def receive_protos(self):
        assert (len(self.selected_clients) > 0)

        self.uploaded_ids = []
        self.uploaded_data_num = []
        uploaded_protos = []
        for client in self.selected_clients:
            self.uploaded_ids.append(client.id)
            self.uploaded_data_num.append(client.num_data)
            protos = load_item(client.role, 'protos', client.save_folder_name)
            uploaded_protos.append(protos)

        global_protos\
            = proto_aggregation(self.args, uploaded_protos, self.uploaded_data_num)

        save_item(global_protos, self.role, 'global_protos', self.save_folder_name)


# https://github.com/yuetan031/fedproto/blob/main/lib/utils.py#L221
def proto_aggregation(conf, local_protos_list, data_num_list):
    agg_protos_label = defaultdict(list)
    for local_protos in local_protos_list:
        for label in local_protos.keys():
            agg_protos_label[label].append(local_protos[label])

    if conf.add_ppa:
        proto_ratio = conf.num_classes / sum(data_num_list)

    for label in agg_protos_label.keys():
        protos = torch.stack(agg_protos_label[label])
        if conf.add_ppa:
            agg_protos_label[label] = torch.sum(protos * proto_ratio, dim=0).detach()
        else:
            agg_protos_label[label] = torch.mean(protos, dim=0).detach()

    return agg_protos_label


def hamming_distance(v1, v2):
    return np.count_nonzero(v1 != v2)


def generate_random_vector(n, ones_count):
    vector = np.zeros(n, dtype=int)
    vector[:ones_count] = 1
    np.random.shuffle(vector)
    return vector


def create_start_vector(n, ones_count):
    vector = np.zeros(n, dtype=int)
    vector[-ones_count:] = 1  # Place 1s at the end
    return vector


def find_optimal_vectors_greedy(n, ones_count):
    # Start with the generalized vector
    start_vector = create_start_vector(n, ones_count)
    vectors = [start_vector]

    for _ in range(n - 1):  # We already have one vector, so generate n-1 more
        best_vector = None
        max_total_distance = -1
        attempts = 0
        max_attempts = 10000

        while attempts < max_attempts:
            candidate = generate_random_vector(n, ones_count)

            # Check if the candidate is unique
            if any(np.array_equal(candidate, v) for v in vectors):
                attempts += 1
                continue

            total_distance = sum(hamming_distance(candidate, v) for v in vectors)
            if total_distance > max_total_distance:
                max_total_distance = total_distance
                best_vector = candidate

            attempts += 1

        if best_vector is None:
            raise ValueError(f"Failed to find a unique vector after {max_attempts} attempts")

        vectors.append(best_vector)

    vectors = np.array(vectors)
    total_distance = sum(hamming_distance(v1, v2) for v1, v2 in combinations(vectors, 2))
    average_distance = total_distance / (n * (n - 1) / 2)

    return vectors, average_distance
