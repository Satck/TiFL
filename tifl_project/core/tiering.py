import numpy as np

class TieringSystem:
    def __init__(self, num_tiers=5):
        self.num_tiers = num_tiers
        self.tiers = {}
    
    def profile_clients(self, clients, global_weights, sync_rounds=3):
        """性能分析：测量客户端延迟"""
        latencies = {} 
        for client in clients:
            times = []
            for _ in range(sync_rounds):  
                _, t, _ = client.train(global_weights, epochs=1)    
                ##这里只要部分返回值t，即可就是时间，其他的都不要  client.train 返回了部分参数
                times.append(t) 
            latencies[client.client_id] = np.mean(times)  ##计算平均值 通过计算平均值，可以得出该客户端在多轮训练中的平均延迟。
        return latencies
    
    def create_tiers(self, latencies):
        """根据延迟创建层级"""
        sorted_clients = sorted(latencies.items(), key=lambda x: x[1])   ## atencies中的数据传 lambda 函数提取每组数据中的第一个元素然后进行排序
        clients_per_tier = len(sorted_clients) // self.num_tiers
        
        self.tiers = {}
        for tier_id in range(self.num_tiers):
            start = tier_id * clients_per_tier
            end = start + clients_per_tier if tier_id < self.num_tiers - 1 else len(sorted_clients)
            
            tier_clients = [cid for cid, _ in sorted_clients[start:end]]
            tier_latencies = [lat for _, lat in sorted_clients[start:end]]
            
            self.tiers[tier_id] = {
                'clients': tier_clients,
                'avg_latency': np.mean(tier_latencies)
            }
        return self.tiers

class AdaptiveScheduler:
    def __init__(self, tiers, interval=50, initial_credits=10):
        self.tiers = tiers
        self.num_tiers = len(tiers)
        self.interval = interval
        self.credits = {i: initial_credits for i in range(self.num_tiers)}
        self.probs = {i: 1.0/self.num_tiers for i in range(self.num_tiers)}
        self.tier_accuracies = {i: [] for i in range(self.num_tiers)}
    
    def select_tier(self, current_round):
        """选择层级（带Credits约束）"""
        # 更新概率（每interval轮）
        if current_round > 0 and current_round % self.interval == 0:
            self._update_probabilities()
        
        # 选择层级
        while True:
            tier_id = np.random.choice(
                list(self.probs.keys()),
                p=list(self.probs.values())
            )
            if self.credits[tier_id] > 0:
                self.credits[tier_id] -= 1
                return tier_id
            # 如果所有Credits耗尽，重置
            if all(c == 0 for c in self.credits.values()):
                self._reset_credits()
    
    def _update_probabilities(self):
        """基于准确率调整概率（降低表现差的层级）"""
        if len(self.tier_accuracies[0]) > 1:
            for tier_id in range(self.num_tiers):
                accs = self.tier_accuracies[tier_id]
                if len(accs) >= 2 and accs[-1] < accs[-2]:
                    self.probs[tier_id] *= 0.9  # 降低概率
            # 归一化
            total = sum(self.probs.values())
            self.probs = {k: v/total for k, v in self.probs.items()}
    
    def _reset_credits(self):
        for tier_id in self.credits:
            self.credits[tier_id] = 10
    
    def update_tier_accuracy(self, tier_id, accuracy):
        self.tier_accuracies[tier_id].append(accuracy)
