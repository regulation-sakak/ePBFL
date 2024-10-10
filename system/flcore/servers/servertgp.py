import time
import os
import numpy as np
from itertools import combinations

import torch
import torch.nn as nn
import torch.nn.functional as F
from flcore.clients.clienttgp import clientTGP
from flcore.servers.serverbase import Server
from flcore.clients.clientbase import load_item, save_item
from collections import defaultdict
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import seaborn as sns


class FedTGP(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientTGP)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        self.num_classes = args.num_classes

        self.server_learning_rate = args.local_learning_rate
        self.batch_size = args.batch_size
        self.server_epochs = args.server_epochs
        self.margin_threthold = args.margin_threthold

        self.feature_dim = args.feature_dim
        self.server_hidden_dim = self.feature_dim

        if args.save_folder_name == 'temp' or 'temp' not in args.save_folder_name:
            if args.add_cps:
                length = self.feature_dim // self.num_classes
                self.cps_feat_dim = length * args.cps_ratio
                self.server_hidden_dim = self.cps_feat_dim

                PROTO = Trainable_prototypes(
                    self.num_classes,
                    self.server_hidden_dim,
                    self.cps_feat_dim,
                    self.device
                ).to(self.device)
            else:
                PROTO = Trainable_prototypes(
                    self.num_classes,
                    self.server_hidden_dim,
                    self.feature_dim,
                    self.device
                ).to(self.device)

            save_item(PROTO, self.role, 'PROTO', self.save_folder_name)
            print(PROTO)
        self.CEloss = nn.CrossEntropyLoss()
        self.MSEloss = nn.MSELoss()

        self.gap = torch.ones(self.num_classes, device=self.device) * 1e9
        self.min_gap = None
        self.max_gap = None

        if args.add_cps:
            self.mask_dict = {}
            self.mask_idx_dict = {}
            length = args.feature_dim // args.num_classes

            vector_name = f'mask_idx_dict_fd_{args.feature_dim}_nc_{args.num_classes}_csr_{args.cps_ratio}.pth'

            if os.path.exists(vector_name):
                self.mask_idx_dict = torch.load(vector_name)
            else:
                per_mask = int(args.num_classes * (args.cps_ratio / 100))
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
            print("\nUpdate Prototype on Server.")
            self.update_Gen()

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
        self.uploaded_protos = []
        self.uploaded_protos_per_client = []
        self.uploaded_client_per_class = np.zeros(self.args.num_classes)
        for client in self.selected_clients:
            self.uploaded_ids.append(client.id)
            self.uploaded_data_num.append(client.num_data)
            protos = load_item(client.role, 'protos', client.save_folder_name)
            self.uploaded_client_per_class[list(protos.keys())] += 1

            for k in protos.keys():
                self.uploaded_protos.append((protos[k], k))
            self.uploaded_protos_per_client.append(protos)

        self.gap = torch.ones(self.num_classes, device=self.device) * 1e9
        avg_protos = proto_cluster(self.args, self.uploaded_protos_per_client, self.uploaded_data_num, self.uploaded_ids)

        for k1 in avg_protos.keys():
            for k2 in avg_protos.keys():
                if k1 > k2:
                    dis = torch.norm(avg_protos[k1] - avg_protos[k2], p=2)
                    self.gap[k1] = torch.min(self.gap[k1], dis)
                    self.gap[k2] = torch.min(self.gap[k2], dis)
        self.min_gap = torch.min(self.gap)
        for i in range(len(self.gap)):
            if self.gap[i] > torch.tensor(1e8, device=self.device):
                self.gap[i] = self.min_gap
        self.max_gap = torch.max(self.gap)
        print('class-wise minimum distance', self.gap)
        print('min_gap', self.min_gap)
        print('max_gap', self.max_gap)

    def update_Gen(self):
        proto_ratio = self.num_classes/sum(self.uploaded_data_num)
        PROTO = load_item(self.role, 'PROTO', self.save_folder_name)
        Gen_opt = torch.optim.SGD(PROTO.parameters(), lr=self.server_learning_rate)
        PROTO.train()
        for e in range(self.server_epochs):
            proto_loader = DataLoader(self.uploaded_protos, self.batch_size,
                                      drop_last=False, shuffle=True)
            for proto, y in proto_loader:
                if self.args.add_ppa:
                    proto = (proto *
                             proto_ratio *
                             (torch.Tensor(self.uploaded_client_per_class[y])).unsqueeze(1).
                             expand(-1, proto.size()[1]).to(self.device))

                y = torch.Tensor(y).type(torch.int64).to(self.device)

                proto_gen = PROTO(list(range(self.num_classes)))

                features_square = torch.sum(torch.pow(proto, 2), 1, keepdim=True)
                centers_square = torch.sum(torch.pow(proto_gen, 2), 1, keepdim=True)
                features_into_centers = torch.matmul(proto, proto_gen.T)
                dist = features_square - 2 * features_into_centers + centers_square.T
                dist = torch.sqrt(dist)

                one_hot = F.one_hot(y, self.num_classes).to(self.device)
                gap2 = min(self.max_gap.item(), self.margin_threthold)
                dist = dist + one_hot * gap2
                loss = self.CEloss(-dist, y)

                Gen_opt.zero_grad()
                loss.backward()
                Gen_opt.step()

        print(f'Server loss: {loss.item()}')
        save_item(PROTO, self.role, 'PROTO', self.save_folder_name)

        PROTO.eval()
        global_protos = defaultdict(list)
        for class_id in range(self.num_classes):
            global_protos[class_id] = PROTO(torch.tensor(class_id, device=self.device)).detach()
        save_item(global_protos, self.role, 'global_protos', self.save_folder_name)

        self.uploaded_protos = []


def proto_cluster(conf, local_protos_list, data_num_list=None, uploaded_id=None):
    proto_clusters = defaultdict(list)
    for protos in local_protos_list:
        for k in protos.keys():
            proto_clusters[k].append(protos[k])

    if conf.add_ppa:
        proto_ratio = conf.num_classes / sum(data_num_list)

    for k in proto_clusters.keys():
        protos = torch.stack(proto_clusters[k])
        if conf.add_ppa:
            proto_clusters[k] = torch.sum(protos * proto_ratio, dim=0).detach()
        else:
            proto_clusters[k] = torch.mean(protos, dim=0).detach()

    return proto_clusters


class Trainable_prototypes(nn.Module):
    def __init__(self, num_classes, server_hidden_dim, feature_dim, device):
        super().__init__()

        self.device = device

        self.embedings = nn.Embedding(num_classes, feature_dim)
        layers = [nn.Sequential(
            nn.Linear(feature_dim, server_hidden_dim),
            nn.ReLU()
        )]
        self.middle = nn.Sequential(*layers)
        self.fc = nn.Linear(server_hidden_dim, feature_dim)

    def forward(self, class_id):
        class_id = torch.tensor(class_id, device=self.device)

        emb = self.embedings(class_id)
        mid = self.middle(emb)
        out = self.fc(mid)

        return out


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
