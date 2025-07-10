# import SoccerNet
# from SoccerNet.Downloader import SoccerNetDownloader
# mySoccerNetDownloader=SoccerNetDownloader(LocalDirectory="/data1/zxd/SoccerNet")

# mySoccerNetDownloader.downloadDataTask(task="tracking", split=["train", "test", "challenge"])
# mySoccerNetDownloader.downloadDataTask(task="tracking-2023", split=["train", "test", "challenge"])

from sklearn.linear_model import Ridge
import numpy as np

def recursive_ridge_prediction(history, n_future_steps=10, window_size=10):
    """
    Args:
        history: np.array of shape (T, D), T 是历史帧数，D 是维度（如 229）
        n_future_steps: 要预测多少帧
        window_size: 每次预测使用的最近几帧（滑动窗口）
    
    Returns:
        predictions: np.array of shape (n_future_steps, D)
    """
    predictions = []
    current_history = history.copy()

    for _ in range(n_future_steps):
        # 取最近 window_size 帧作为输入
        X = np.arange(len(current_history))[-window_size:].reshape(-1, 1)  # 帧号作为输入
        Y = current_history[-window_size:]

        # 训练 Ridge 模型（也可以放在外面只训练一次）
        model = Ridge(alpha=1.0)
        model.fit(X, Y)

        # 预测下一帧
        next_x = np.array([[len(current_history)]])
        next_pred = model.predict(next_x)  # shape: (1, D)

        # 加入预测结果
        predictions.append(next_pred[0])
        current_history = np.vstack([current_history, next_pred])

    return np.array(predictions)


# 假设我们有 100 帧的历史姿态特征
history = np.random.randn(100, 229)

# 预测未来 10 帧
future_preds = recursive_ridge_prediction(history, n_future_steps=10)

print("Predicted future poses shape:", future_preds.shape)
# 输出: Predicted future poses shape: (10, 229)