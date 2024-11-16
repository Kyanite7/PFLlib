import copy
import torch
import numpy as np
import time
from flcore.clients.clientbase import Client
from flcore.optimizers.fedoptimizer import SAMOptimizer  # 引入之前定义的SAM优化器


class clientSAM(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        # 初始化SAM优化器
        self.base_optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.optimizer = SAMOptimizer(self.base_optimizer.param_groups, lr=self.learning_rate, rho=args.rho)

    def train(self):
        trainloader = self.load_train_data()
        self.model.to(self.device)  # 确保模型在正确的设备上
        self.model.train()  # 确保模型处于训练模式

        # 确保模型参数启用梯度
        for param in self.model.parameters():
            param.requires_grad = True  # 强制要求所有参数的梯度

        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for epoch in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                # 确保输入数据在正确设备上并为浮点类型
                x = x.to(self.device, dtype=torch.float32)
                y = y.to(self.device)

                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))

                # 定义闭包函数
                def closure():
                    self.optimizer.zero_grad()
                    output = self.model(x)

                    # 检查模型输出是否要求梯度
                    if not output.requires_grad:
                        raise RuntimeError(f"Model output does not require gradients. Output shape: {output.shape}")

                    loss = self.loss(output, y)

                    # 检查损失是否需要梯度
                    if not loss.requires_grad:
                        raise RuntimeError("Loss does not require gradients. Check model and data.")

                    loss.backward()
                    return loss

                # 使用 SAM 优化器的两步训练
                self.optimizer.step(closure)

        self.model.cpu()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time
