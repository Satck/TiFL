#!/usr/bin/env python3
"""
可视化实验结果
使用方法: python visualize.py --dataset mnist
"""

import argparse
import json
import os
import matplotlib.pyplot as plt
import numpy as np

def plot_comparison(dataset_name, strategies):
    """对比不同策略的结果"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    for strategy in strategies:
        metrics_file = f"results/metrics/{dataset_name}_{strategy}/metrics.json"
        if not os.path.exists(metrics_file):
            print(f"Warning: {metrics_file} not found")
            continue
        
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        # 1. Accuracy over rounds
        axes[0, 0].plot(metrics['round'], metrics['accuracy'], label=strategy, linewidth=2)
        
        # 2. Loss over rounds
        axes[0, 1].plot(metrics['round'], metrics['loss'], label=strategy, linewidth=2)
        
        # 3. Accuracy over time
        axes[1, 0].plot(metrics['wall_clock_time'], metrics['accuracy'], label=strategy, linewidth=2)
        
        # 4. Training time per round
        axes[1, 1].plot(metrics['round'], metrics['training_time'], label=strategy, linewidth=2)
    
    # 设置标题和标签
    axes[0, 0].set_xlabel('Round')
    axes[0, 0].set_ylabel('Test Accuracy')
    axes[0, 0].set_title('Accuracy vs Round')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_xlabel('Round')
    axes[0, 1].set_ylabel('Test Loss')
    axes[0, 1].set_title('Loss vs Round')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_xlabel('Wall-clock Time (s)')
    axes[1, 0].set_ylabel('Test Accuracy')
    axes[1, 0].set_title('Accuracy vs Time')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_xlabel('Round')
    axes[1, 1].set_ylabel('Training Time (s)')
    axes[1, 1].set_title('Training Time per Round')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(f'{dataset_name.upper()} - Strategy Comparison', fontsize=16)
    plt.tight_layout()
    
    # 保存
    save_path = f"results/plots/{dataset_name}_comparison.png"
    os.makedirs('results/plots', exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    args = parser.parse_args()
    
    strategies = ['vanilla', 'uniform', 'fast', 'slow', 'adaptive']
    plot_comparison(args.dataset, strategies)

if __name__ == '__main__':
    main()
