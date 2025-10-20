import tensorflow as tf
import numpy as np
import time

class Client:
    def __init__(self, client_id, data, model, cpu_capacity):
        self.client_id = client_id
        self.x_train, self.y_train = data
        self.model = tf.keras.models.clone_model(model)
        self.cpu_capacity = cpu_capacity
        self.data_size = len(self.x_train)
    
    def train(self, global_weights, lr=0.01, decay=0.995, epochs=1, batch_size=10):
        """
        本地训练
        Returns: (updated_weights, simulated_time, loss)
        """
        # 设置全局权重
        self.model.set_weights(global_weights)
        
        # 编译模型
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr, decay=decay)
        self.model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # 训练
        start_time = time.time()
        history = self.model.fit(
            self.x_train, self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0
        )
        actual_time = time.time() - start_time
        
        # 模拟延迟（根据CPU容量）
        simulated_time = actual_time / self.cpu_capacity
        
        # 获取更新后的权重
        updated_weights = self.model.get_weights()
        loss = history.history['loss'][-1]
        
        return updated_weights, simulated_time, loss
    
    def get_loss(self, global_weights):
        """计算本地损失（用于选择策略）"""
        self.model.set_weights(global_weights)
        self.model.compile(
            optimizer='rmsprop',
            loss='sparse_categorical_crossentropy'
        )
        loss = self.model.evaluate(self.x_train, self.y_train, verbose=0)
        return loss
