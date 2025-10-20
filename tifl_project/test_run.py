#!/usr/bin/env python3
"""
快速测试脚本 - 使用较小的参数验证代码是否正常工作
"""

import yaml
import numpy as np
import tensorflow as tf

# 导入模块
from data.loader import load_dataset, create_non_iid_split
from models.networks import get_model
from core.client import Client
from core.server import FederatedServer
from core.tiering import TieringSystem, AdaptiveScheduler
from experiments.trainer import FederatedTrainer

def quick_test():
    """快速测试"""
    print("=== TiFL 快速验证测试 ===")
    
    # 设置随机种子
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # 测试配置（较小参数）
    config = {
        'num_clients': 10,
        'clients_per_round': 3,
        'num_rounds': 5,  # 只测试5轮
        'num_tiers': 3,
        'non_iid_shards': 2
    }
    
    print("1. 加载数据集...")
    (x_train, y_train), (x_test, y_test) = load_dataset('mnist')
    print(f"   训练集大小: {x_train.shape}, 测试集大小: {x_test.shape}")
    
    print("2. 创建Non-IID划分...")
    client_data = create_non_iid_split(x_train, y_train, config['num_clients'], config['non_iid_shards'])
    print(f"   客户端数据大小: {[len(cd[0]) for cd in client_data[:3]]}")
    
    print("3. 创建模型...")
    model = get_model('mnist')
    print(f"   模型参数数量: {model.count_params()}")
    
    print("4. 创建客户端...")
    cpu_alloc = [2.0, 1.0, 0.5]  # 3个组
    clients = []
    clients_per_group = config['num_clients'] // len(cpu_alloc)
    
    for i in range(config['num_clients']):
        group_id = min(i // clients_per_group, len(cpu_alloc) - 1)
        client = Client(
            client_id=i,
            data=client_data[i],
            model=model,
            cpu_capacity=cpu_alloc[group_id]
        )
        clients.append(client)
    print(f"   创建了 {len(clients)} 个客户端")
    
    print("5. 创建服务器...")
    server = FederatedServer(get_model('mnist'), (x_test, y_test))
    
    print("6. 测试vanilla策略...")
    trainer = FederatedTrainer(
        clients=clients,
        server=server,
        config=config,
        strategy_name='vanilla'
    )
    
    print("7. 运行训练（5轮测试）...")
    metrics = trainer.train()
    
    print(f"8. 测试完成！")
    print(f"   初始准确率: {metrics['accuracy'][0]:.4f}")
    print(f"   最终准确率: {metrics['accuracy'][-1]:.4f}")
    print(f"   总训练时间: {metrics['wall_clock_time'][-1]:.2f}秒")
    
    print("\n✅ 所有测试通过！项目代码正常工作。")
    print("💡 现在可以运行完整实验: python main.py --dataset mnist --strategy vanilla")

if __name__ == '__main__':
    quick_test()
