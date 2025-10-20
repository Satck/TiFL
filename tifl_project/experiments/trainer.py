import numpy as np
import time
from tqdm import tqdm

class FederatedTrainer:
    def __init__(self, clients, server, config, strategy_name, tiers=None, scheduler=None):
        self.clients = clients
        self.server = server
        self.config = config
        self.strategy_name = strategy_name
        self.tiers = tiers
        self.scheduler = scheduler
        
        # 结果记录
        self.metrics = {
            'round': [],
            'accuracy': [],
            'loss': [],
            'training_time': [],
            'wall_clock_time': []
        }
    
    def train(self):
        """执行联邦学习训练"""
        from strategies.selector import ClientSelector
        
        num_rounds = self.config['num_rounds']
        clients_per_round = self.config['clients_per_round']
        
        print(f"\n{'='*60}")
        print(f"Training: {self.strategy_name} strategy")
        print(f"Rounds: {num_rounds}, Clients/round: {clients_per_round}")
        print(f"{'='*60}\n")
        
        total_start = time.time()
        
        for round_num in tqdm(range(num_rounds), desc="Training"):
            round_start = time.time()
            
            # 1. 选择客户端
            if self.strategy_name == 'vanilla':
                selected_ids = ClientSelector.vanilla(self.clients, clients_per_round)
            elif self.strategy_name == 'uniform':
                selected_ids = ClientSelector.uniform(self.tiers, clients_per_round)
            elif self.strategy_name == 'fast':
                selected_ids = ClientSelector.fast(self.tiers, clients_per_round)
            elif self.strategy_name == 'slow':
                selected_ids = ClientSelector.slow(self.tiers, clients_per_round)
            elif self.strategy_name == 'adaptive':
                selected_ids, tier_id = ClientSelector.adaptive(
                    self.scheduler, self.tiers, clients_per_round, round_num
                )
            
            # 2. 客户端训练
            global_weights = self.server.get_weights()
            client_weights = []
            client_sizes = []
            client_times = []
            
            for cid in selected_ids:
                client = self.clients[cid]
                weights, train_time, _ = client.train(global_weights)
                client_weights.append(weights)
                client_sizes.append(client.data_size)
                client_times.append(train_time)
            
            # 3. 聚合
            self.server.aggregate(client_weights, client_sizes)
            
            # 4. 评估
            accuracy, loss = self.server.evaluate()
            
            # 5. 记录指标
            round_time = time.time() - round_start
            wall_time = time.time() - total_start
            
            self.metrics['round'].append(round_num)
            self.metrics['accuracy'].append(float(accuracy))
            self.metrics['loss'].append(float(loss))
            self.metrics['training_time'].append(max(client_times))
            self.metrics['wall_clock_time'].append(wall_time)
            
            # 6. 更新调度器（adaptive策略）
            if self.strategy_name == 'adaptive' and self.scheduler:
                self.scheduler.update_tier_accuracy(tier_id, accuracy)
            
            # 打印进度
            if (round_num + 1) % 50 == 0:
                print(f"Round {round_num+1}: Acc={accuracy:.4f}, Loss={loss:.4f}, Time={round_time:.2f}s")
        
        print(f"\nTraining completed! Total time: {time.time()-total_start:.2f}s")
        return self.metrics
