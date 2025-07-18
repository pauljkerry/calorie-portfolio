import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import pickle

class FertilizerDataset(Dataset):
    """PyTorch用のカスタムデータセット"""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def preprocess_data(train_df, target_col, test_df=None, save_preprocessors=True):
    """
    データの前処理を行う関数
    
    Args:
        train_df: 訓練データ
        target_col: ターゲット列名
        test_df: テストデータ（オプション）
        save_preprocessors: 前処理オブジェクトを保存するか
    
    Returns:
        dict: 前処理済みデータと各種オブジェクト
    """
    print("Starting data preprocessing...")
    
    # ターゲット変数を分離
    X_train = train_df.drop(columns=[target_col]).copy()
    y_train = train_df[target_col].copy()
    
    # ターゲットのラベルエンコーディング
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    
    print(f"Target classes: {label_encoder.classes_}")
    print(f"Number of classes: {len(label_encoder.classes_)}")
    print(f"Class distribution: {np.bincount(y_train_encoded)}")
    
    # 特徴量のスケーリング
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    print(f"Feature shape: {X_train_scaled.shape}")
    print(f"Features scaled - mean: {X_train_scaled.mean():.6f}, std: {X_train_scaled.std():.6f}")
    
    # 訓練・検証データに分割
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train_scaled, y_train_encoded, 
        test_size=0.2, 
        random_state=42, 
        stratify=y_train_encoded
    )
    
    print(f"Train set size: {X_train_split.shape[0]}")
    print(f"Validation set size: {X_val_split.shape[0]}")
    
    # PyTorchデータセットを作成
    train_dataset = FertilizerDataset(X_train_split, y_train_split)
    val_dataset = FertilizerDataset(X_val_split, y_val_split)
    
    # テストデータの処理（存在する場合）
    test_dataset = None
    X_test_scaled = None
    if test_df is not None:
        X_test = test_df.copy()
        X_test_scaled = scaler.transform(X_test)
        # テストデータはダミーのターゲットを作成（実際の予測時は使用しない）
        dummy_y_test = np.zeros(len(X_test))
        test_dataset = FertilizerDataset(X_test_scaled, dummy_y_test)
        print(f"Test set size: {X_test_scaled.shape[0]}")
    
    # 前処理オブジェクトを保存
    if save_preprocessors:
        with open('../artifacts/label_encoder.pkl', 'wb') as f:
            pickle.dump(label_encoder, f)
        with open('../artifacts/scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        print("Preprocessors saved!")
    
    result = {
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'test_dataset': test_dataset,
        'label_encoder': label_encoder,
        'scaler': scaler,
        'num_features': X_train_scaled.shape[1],
        'num_classes': len(label_encoder.classes_),
        'X_train_scaled': X_train_scaled,
        'y_train_encoded': y_train_encoded,
        'X_test_scaled': X_test_scaled
    }
    
    print("Preprocessing completed!")
    return result

def create_data_loaders(train_dataset, val_dataset, test_dataset=None, batch_size=64):
    """データローダーを作成"""
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2,
        pin_memory=True
    )
    
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=2,
            pin_memory=True
        )
    
    return train_loader, val_loader, test_loader

# 使用例
if __name__ == "__main__":
    # データの読み込み例
    # train_df = pd.read_csv('../data/train.csv')
    # test_df = pd.read_csv('../data/test.csv')  # オプション
    
    # 前処理の実行
    # result = preprocess_data(train_df, 'target_column_name', test_df)
    
    # データローダーの作成
    # train_loader, val_loader, test_loader = create_data_loaders(
    #     result['train_dataset'], 
    #     result['val_dataset'], 
    #     result['test_dataset']
    # )
    
    print("Data preprocessing module ready!")