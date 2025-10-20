#!/usr/bin/env python3
"""
TiFL完整复现主脚本
使用方法: python main.py --dataset mnist --strategy adaptive
"""

import argparse
import yaml
import numpy as np
import tensorflow as tf
import json
import os
from datetime import datetime

# 导入模块
from data.loader import load_dataset, create_non_iid_split, create_non_iid_cifar
from models.networks import get_model
from core.client import Client
from core.server import FederatedServer
from core.tiering import TieringSystem, AdaptiveScheduler
from experiments.trainer import FederatedTrainer

def set_seed(seed):
    """设置随机种子"""
    np.random.seed(seed)
    tf.random.set_seed(seed)

def setup_clients(dataset_name, num_clients, cpu_alloc, config):
    """创建客户端"""
    print(f"Loading {dataset_name} dataset...")
    (x_train, y_train), (x_test, y_test) = load_dataset(dataset_name)
    
    # Non-IID划分
    print("Creating Non-IID split...")
    if dataset_name in ['mnist', 'fashion_mnist']:
        client_data = create_non_iid_split(x_train, y_train, num_clients, 
                                          config['non_iid_shards'])
    else:  # cifar10
        client_data = create_non_iid_cifar(x_train, y_train, num_clients,
                                          config['non_iid_classes'])
    
    # 创建模型
    model = get_model(dataset_name)
    
    # 创建客户端
    print(f"Creating {num_clients} clients...")
    clients = []
    clients_per_group = num_clients // len(cpu_alloc)
    
    for i in range(num_clients):
        group_id = min(i // clients_per_group, len(cpu_alloc) - 1)
        cpu_capacity = cpu_alloc[group_id]
        
        client = Client(
            client_id=i,
            data=client_data[i],
            model=model,
            cpu_capacity=cpu_capacity
        )
        clients.append(client)
    
    return clients, (x_test, y_test), model

def main():
    parser = argparse.ArgumentParser(description='TiFL Reproduction')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['mnist', 'fashion_mnist', 'cifar10'])
    parser.add_argument('--strategy', type=str, required=True,
                       choices=['vanilla', 'uniform', 'fast', 'slow', 'adaptive'])
    parser.add_argument('--config', type=str, default='config.yaml')
    
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # 设置随机种子
    set_seed(config['common']['seed'])
    
    # 合并配置
    dataset_config = {**config['common'], **config['datasets'][args.dataset]}
    dataset_config['dataset'] = args.dataset  # 添加数据集名称
    
    # 创建客户端
    clients, test_data, model = setup_clients(
        args.dataset,
        dataset_config['num_clients'],
        dataset_config['cpu_alloc'],
        dataset_config
    )
    
    # 创建服务器
    print("Creating federated server...")
    server = FederatedServer(get_model(args.dataset), test_data)
    
    # 分层和调度器（除vanilla外）
    tiers = None
    scheduler = None
    
    if args.strategy != 'vanilla':
        print("Profiling clients and creating tiers...")
        tiering = TieringSystem(num_tiers=dataset_config['num_tiers'])
        global_weights = server.get_weights()
        latencies = tiering.profile_clients(clients, global_weights, sync_rounds=3)
        tiers = tiering.create_tiers(latencies)
        
        for tid, info in tiers.items():
            print(f"  Tier {tid}: {len(info['clients'])} clients, "
                  f"avg latency: {info['avg_latency']:.2f}s")
        
        if args.strategy == 'adaptive':
            scheduler = AdaptiveScheduler(tiers, interval=50, initial_credits=10)
    
    # 训练
    trainer = FederatedTrainer(
        clients=clients,
        server=server,
        config=dataset_config,
        strategy_name=args.strategy,
        tiers=tiers,
        scheduler=scheduler
    )
    
    metrics = trainer.train()
    
    # 保存结果
    save_dir = f"results/metrics/{args.dataset}_{args.strategy}"
    os.makedirs(save_dir, exist_ok=True)
    
    with open(f"{save_dir}/metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nResults saved to {save_dir}/metrics.json")
    print(f"Final accuracy: {metrics['accuracy'][-1]:.4f}")
    print(f"Total time: {metrics['wall_clock_time'][-1]:.2f}s")

if __name__ == '__main__':
    main()
