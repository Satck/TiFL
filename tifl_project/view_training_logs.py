#!/usr/bin/env python3
"""
查看训练日志脚本
使用方法: python view_training_logs.py --strategy adaptive --rounds 50
"""

import argparse
import json
import os

def display_training_log(strategy, max_rounds=None, interval=10):
    """显示训练日志"""
    metrics_file = f"results/metrics/mnist_{strategy}/metrics.json"
    
    if not os.path.exists(metrics_file):
        print(f"❌ 未找到 {strategy} 策略的结果文件")
        return
    
    print(f"📊 {strategy.upper()} 策略训练日志")
    print("="*80)
    
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    rounds = metrics['round']
    accuracies = metrics['accuracy']
    losses = metrics['loss']
    training_times = metrics['training_time']
    wall_times = metrics['wall_clock_time']
    
    total_rounds = len(rounds)
    if max_rounds:
        display_rounds = min(max_rounds, total_rounds)
    else:
        display_rounds = total_rounds
    
    print(f"轮次范围: 1-{display_rounds} (总共 {total_rounds} 轮)")
    print(f"显示间隔: 每 {interval} 轮显示一次")
    print("-"*80)
    print(f"{'轮次':>6} {'准确率':>10} {'损失值':>10} {'训练时间':>12} {'累计时间':>12} {'进度':>8}")
    print("-"*80)
    
    for i in range(0, display_rounds, interval):
        round_num = rounds[i] + 1  # 从1开始计数
        accuracy = accuracies[i]
        loss = losses[i]
        train_time = training_times[i]
        wall_time = wall_times[i]
        progress = (i + 1) / total_rounds * 100
        
        print(f"{round_num:6d} {accuracy:10.4f} {loss:10.4f} {train_time:10.2f}s {wall_time:10.1f}s {progress:6.1f}%")
    
    # 显示最后一轮（如果不在间隔显示中）
    if (display_rounds - 1) % interval != 0:
        i = display_rounds - 1
        round_num = rounds[i] + 1
        accuracy = accuracies[i]
        loss = losses[i]
        train_time = training_times[i]
        wall_time = wall_times[i]
        progress = 100.0
        
        print(f"{round_num:6d} {accuracy:10.4f} {loss:10.4f} {train_time:10.2f}s {wall_time:10.1f}s {progress:6.1f}%")
    
    print("-"*80)
    print(f"📈 最终结果:")
    print(f"   准确率: {accuracies[-1]:.4f}")
    print(f"   损失值: {losses[-1]:.4f}")
    print(f"   总时间: {wall_times[-1]:.1f}秒")
    print(f"   平均每轮: {wall_times[-1]/len(rounds):.2f}秒")

def compare_strategies(strategies, round_num):
    """对比特定轮次各策略的表现"""
    print(f"\n🔍 第 {round_num} 轮各策略对比")
    print("="*80)
    print(f"{'策略':>12} {'准确率':>10} {'损失值':>10} {'训练时间':>12} {'累计时间':>12}")
    print("-"*80)
    
    for strategy in strategies:
        metrics_file = f"results/metrics/mnist_{strategy}/metrics.json"
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            
            if round_num - 1 < len(metrics['round']):
                idx = round_num - 1
                accuracy = metrics['accuracy'][idx]
                loss = metrics['loss'][idx]
                train_time = metrics['training_time'][idx]
                wall_time = metrics['wall_clock_time'][idx]
                
                print(f"{strategy:>12} {accuracy:10.4f} {loss:10.4f} {train_time:10.2f}s {wall_time:10.1f}s")

def main():
    parser = argparse.ArgumentParser(description='查看TiFL训练日志')
    parser.add_argument('--strategy', type=str, 
                       choices=['vanilla', 'uniform', 'fast', 'slow', 'adaptive', 'all'],
                       default='all', help='选择策略')
    parser.add_argument('--rounds', type=int, help='显示的轮数上限')
    parser.add_argument('--interval', type=int, default=10, help='显示间隔')
    parser.add_argument('--compare', type=int, help='对比特定轮次的各策略表现')
    
    args = parser.parse_args()
    
    strategies = ['vanilla', 'uniform', 'fast', 'slow', 'adaptive']
    
    if args.compare:
        compare_strategies(strategies, args.compare)
        return
    
    if args.strategy == 'all':
        for strategy in strategies:
            display_training_log(strategy, args.rounds, args.interval)
            print("\n" + "="*80 + "\n")
    else:
        display_training_log(args.strategy, args.rounds, args.interval)

if __name__ == '__main__':
    main()
