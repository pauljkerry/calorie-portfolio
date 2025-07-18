import pandas as pd
import numpy as np
import itertools
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import KFold, train_test_split
import torch
from torch.utils.data import Dataset, DataLoader

def feature_engineering(tr_df, test_df):
    all_data = pd.concat([tr_df, test_df])

    # === 1) 数値変数のみ
    num_df = all_data.select_dtypes(include=[np.number])
    num_cols = num_df.columns

    # ==== 2) カテゴリ変数のみ
    cat_df = all_data[["Soil Type", "Crop Type"]]
    cat_cols = cat_df.columns

    # ==== 3) target
    target_df = all_data["target"]
    
    # === 4) 交互作用
    inter_df = pd.DataFrame(index=all_data.index)
    
    for c1, c2 in itertools.combinations_with_replacement(num_cols, 2):
        inter_df[f"{c1}_{c2}_inter"] = all_data[c1] * all_data[c2]
    
    # === 5) OHE
    ohe_df = pd.get_dummies(cat_df).astype(np.float32)
    ohe_df.index = all_data.index

    # === 6) Target Encoding
    # targetをoheで2値変換
    target_cols = tr_df["target"].astype(str).unique()
    tmp_df = all_data.copy()
    for t in target_cols:
        tmp_df[t] = (all_data["target"] == t).astype(int)
    
    tr_te_df = tmp_df.iloc[:len(tr_df)].copy()
    test_te_df = tmp_df.iloc[len(tr_df):].copy()
    
    # test用
    for c in cat_cols:
        for t in target_cols:
            means = tr_te_df.groupby(c)[t].mean()
            test_te_df[f"{c}_{t}_te"] = test_te_df[c].map(means)
    
    # train用
    kf = KFold(n_splits=4, shuffle=True, random_state=41)
    
    for c in cat_cols:
        for t in target_cols:
            tr_te_df[f"{c}_{t}_te"] = 0.0  # 初期化
    
        for tr_idx, val_idx in kf.split(tr_df):
            tr_fold = tr_te_df.iloc[tr_idx]
            val_fold = tr_te_df.iloc[val_idx]
    
            for t in target_cols:
                means = tr_fold.groupby(c)[t].mean()
                colname = f"{c}_{t}_te"
                tr_te_df.loc[val_idx, colname] = val_fold[c].map(means).to_numpy()

    te_df = pd.concat([tr_te_df, test_te_df], axis=0, ignore_index=True)
    te_df.index = all_data.index
    te_df = te_df.drop(columns=np.concatenate([tr_df.columns, target_cols]))

    # === 7) 追加の農業関連特徴量
    various_ratio_df = pd.DataFrame(index=all_data.index)
    
    npk_total = all_data["N"] + all_data["P"] + all_data["K"] + 1e-6
    various_ratio_df["N_ratio"] = all_data["N"] / npk_total
    various_ratio_df["P_ratio"] = all_data["P"] / npk_total
    various_ratio_df["K_ratio"] = all_data["K"] / npk_total

    # 環境ストレス指標
    various_ratio_df["(Hum-Moi)/Tem"] = (all_data["Humidity"] - all_data["Moisture"]) / (all_data["Temperature"] + 1e-6)
    various_ratio_df["(Tem-Hum)/Moi"] = (all_data["Temperature"] - all_data["Humidity"]) / (all_data["Moisture"] + 1e-6)
    various_ratio_df["(Tem-Moi)/Hum"] = (all_data["Temperature"] - all_data["Moisture"]) / (all_data["Humidity"] + 1e-6)

    # 効率性指標
    various_ratio_df["Hum/Tem"] = all_data["Humidity"] / (all_data["Temperature"] + 1e-6)
    various_ratio_df["Moi/Hum"] = all_data["Moisture"] / (all_data["Humidity"] + 1e-6)
    various_ratio_df["Moi/Tem"] = all_data["Moisture"] / (all_data["Temperature"] + 1e-6)

    # NPK比率
    various_ratio_df["NPK_total"] = npk_total
    various_ratio_df["P/K"] = all_data["P"] / (all_data["K"] + 1e-6)
    various_ratio_df["N/K"] = all_data["N"] / (all_data["K"] + 1e-6)
    various_ratio_df["P/N"] = all_data["P"] / (all_data["N"] + 1e-6)

    # 特徴量のスケーリング
    scaler = StandardScaler()
    scaled_df = pd.concat([num_df, inter_df, te_df, various_ratio_df], axis=1)
    scaled_df = pd.DataFrame(
        scaler.fit_transform(scaled_df),
        columns=scaled_df.columns,
        index=all_data.index
    ).astype(np.float32)
                          
    # dfを結合
    all_data = pd.concat([scaled_df, ohe_df, target_df], axis=1)

    tr_df = all_data.iloc[:len(tr_df)]
    test_df = all_data.iloc[len(tr_df):].drop("target", axis=1)

    return tr_df, test_df
    
    """ここからNN専用データ整形"""
    
    X_train = tr_df.drop("target", axis=1).copy()
    y_train = tr_df["target"].copy()

    # ターゲットのラベルエンコーディング
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)

    # 訓練／検証データに分割
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        Xtrain, y_train_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_train_encoded
    )

    # PyTorchデータセットを作成
    train_dataset = FertilizerDataset(X_train_split, y_train_split)
    val_dataset = FertilizerDataset(X_val_split, y_val_split)

    dummy_y_test = np.zeros(len(test_df))
    test_dataset = FertilizerDataset(test_df, dummy_y_test)

    