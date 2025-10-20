import tensorflow as tf
import numpy as np

class FederatedServer:
    def __init__(self, model, test_data):
        self.global_model = model
        self.x_test, self.y_test = test_data
        self.global_model.compile(
            optimizer='rmsprop',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def aggregate(self, client_weights_list, client_data_sizes):
        """
        FedAvg聚合: w_global = sum(w_i * n_i) / sum(n_i)
        """
        total_size = sum(client_data_sizes)
        
        # 加权平均
        new_weights = []
        for layer_idx in range(len(client_weights_list[0])):
            layer_weights = np.array([
                weights[layer_idx] * client_data_sizes[i] 
                for i, weights in enumerate(client_weights_list)
            ])
            new_layer_weights = np.sum(layer_weights, axis=0) / total_size
            new_weights.append(new_layer_weights)
        
        self.global_model.set_weights(new_weights)
        return new_weights
    
    def evaluate(self):
        """评估全局模型"""
        loss, accuracy = self.global_model.evaluate(
            self.x_test, self.y_test, verbose=0
        )
        return accuracy, loss
    
    def get_weights(self):
        return self.global_model.get_weights()
