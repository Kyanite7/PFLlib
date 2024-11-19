import copy
import numpy as np
import torch
import time
from flcore.clients.clientcpn import *
from flcore.servers.serverbase import Server
from utils.data_utils import read_client_data
from threading import Thread


class pFedCPN(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        self.feature_dim = list(args.model.head.parameters())[0].shape[1]
        args.GCE = GCE(in_features=self.feature_dim,
                       num_classes=args.num_classes,
                       dev=args.device).to(args.device)

        in_dim = list(args.model.head.parameters())[0].shape[1]
        # 条件策略网络
        cs = ConditionalSelection(in_dim, in_dim).to(args.device)

        # select slow clients
        self.set_slow_clients()
        for i, train_slow, send_slow in zip(range(self.num_clients), self.train_slow_clients, self.send_slow_clients):
            train_data = read_client_data(self.dataset, i, is_train=True)
            test_data = read_client_data(self.dataset, i, is_train=False)
            client = clientCPN(self.args,
                               id=i,
                               train_samples=len(train_data),
                               test_samples=len(test_data),
                               train_slow=train_slow,
                               send_slow=send_slow,
                               ConditionalSelection=cs)
            self.clients.append(client)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        self.head = None
        self.cs = None

    def train(self):
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate performance")
                self.evaluate()

            for client in self.selected_clients:
                client.train()
                client.generate_upload_head()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_models()
            self.aggregate_parameters()
            self.send_models()

            self.global_head()
            self.global_cs()

            self.global_GCE()

            self.Budget.append(time.time() - s_t)
            print('-' * 25, 'time cost', '-' * 25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))

    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        active_train_samples = 0
        for client in self.selected_clients:
            active_train_samples += client.train_samples

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []
        for client in self.selected_clients:
            self.uploaded_weights.append(client.train_samples / active_train_samples)
            self.uploaded_ids.append(client.id)
            self.uploaded_models.append(client.model.model.base)

    def global_GCE(self):
        active_train_samples = 0
        for client in self.selected_clients:
            active_train_samples += client.train_samples

        self.uploaded_weights = []
        self.uploaded_model_gs = []
        for client in self.selected_clients:
            self.uploaded_weights.append(client.train_samples / active_train_samples)
            self.uploaded_model_gs.append(client.GCE)

        self.GCE = copy.deepcopy(self.uploaded_model_gs[0])
        for param in self.GCE.parameters():
            param.data = torch.zeros_like(param.data)

        for w, client_model in zip(self.uploaded_weights, self.uploaded_model_gs):
            self.add_GCE(w, client_model)

        for client in self.clients:
            client.set_GCE(self.GCE)

    def add_GCE(self, w, GCE):
        for server_param, client_param in zip(self.GCE.parameters(), GCE.parameters()):
            server_param.data += client_param.data.clone() * w

    def global_head(self):
        self.uploaded_model_gs = []
        for client in self.selected_clients:
            self.uploaded_model_gs.append(client.model.head_g)

        self.head = copy.deepcopy(self.uploaded_model_gs[0])
        for param in self.head.parameters():
            param.data = torch.zeros_like(param.data)

        for w, client_model in zip(self.uploaded_weights, self.uploaded_model_gs):
            self.add_head(w, client_model)

        for client in self.selected_clients:
            client.set_head_g(self.head)

    def add_head(self, w, head):
        for server_param, client_param in zip(self.head.parameters(), head.parameters()):
            server_param.data += client_param.data.clone() * w

    def global_cs(self):
        self.uploaded_model_gs = []
        for client in self.selected_clients:
            self.uploaded_model_gs.append(client.model.gate.cs)

        self.cs = copy.deepcopy(self.uploaded_model_gs[0])
        for param in self.cs.parameters():
            param.data = torch.zeros_like(param.data)

        for w, client_model in zip(self.uploaded_weights, self.uploaded_model_gs):
            self.add_cs(w, client_model)

        for client in self.selected_clients:
            client.set_cs(self.cs)

    def add_cs(self, w, cs):
        for server_param, client_param in zip(self.cs.parameters(), cs.parameters()):
            server_param.data += client_param.data.clone() * w


class GCE(nn.Module):
    def __init__(self, in_features, num_classes, dev='cpu'):
        super(GCE, self).__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.embedding = nn.Embedding(num_classes, in_features)
        self.dev = dev

    def forward(self, x, label):
        embeddings = self.embedding(torch.tensor(range(self.num_classes), device=self.dev))
        cosine = F.linear(F.normalize(x), F.normalize(embeddings))
        one_hot = torch.zeros(cosine.size(), device=self.dev)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        softmax_value = F.log_softmax(cosine, dim=1)
        softmax_loss = one_hot * softmax_value
        softmax_loss = - torch.mean(torch.sum(softmax_loss, dim=1))

        return softmax_loss


class ConditionalSelection(nn.Module):
    def __init__(self, in_dim, h_dim):
        super(ConditionalSelection, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(in_dim, h_dim * 2),
            nn.LayerNorm([h_dim * 2]),
            nn.ReLU(),
        )

    def forward(self, x, tau=1, hard=False):
        shape = x.shape
        x = self.fc(x)
        x = x.view(shape[0], 2, -1)
        x = F.gumbel_softmax(x, dim=1, tau=tau, hard=hard)
        return x[:, 0, :], x[:, 1, :]
