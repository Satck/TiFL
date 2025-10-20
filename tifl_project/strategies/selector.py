import numpy as np

class ClientSelector:
    @staticmethod
    def vanilla(clients, num_select):
        """随机选择"""
        client_ids = [c.client_id for c in clients]
        return np.random.choice(client_ids, num_select, replace=False).tolist()
    
    @staticmethod
    def uniform(tiers, num_select):
        """均匀层级选择"""
        tier_id = np.random.randint(0, len(tiers))
        tier_clients = tiers[tier_id]['clients']
        return np.random.choice(tier_clients, min(num_select, len(tier_clients)), replace=False).tolist()
    
    @staticmethod
    def fast(tiers, num_select):
        """只选最快层"""
        fastest_clients = tiers[0]['clients']
        return np.random.choice(fastest_clients, min(num_select, len(fastest_clients)), replace=False).tolist()
    
    @staticmethod
    def slow(tiers, num_select):
        """只选最慢层"""
        slowest_tier = len(tiers) - 1
        slowest_clients = tiers[slowest_tier]['clients']
        return np.random.choice(slowest_clients, min(num_select, len(slowest_clients)), replace=False).tolist()
    
    @staticmethod
    def adaptive(scheduler, tiers, num_select, current_round):
        """TiFL自适应策略"""
        tier_id = scheduler.select_tier(current_round)
        tier_clients = tiers[tier_id]['clients']
        selected = np.random.choice(tier_clients, min(num_select, len(tier_clients)), replace=False).tolist()
        return selected, tier_id
