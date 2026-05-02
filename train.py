"""
train.py — SybilGAT training pipeline for cresci-2017

Usage (Google Colab):
    Mount Drive at /content/drive/MyDrive/
    Place cresci-2017 data at /content/drive/MyDrive/cresci/
    Run: python train.py

Expected file structure:
    cresci/
    ├── genuine_accounts.csv/genuine_accounts.csv/users.csv
    ├── genuine_accounts.csv/genuine_accounts.csv/tweets.csv
    ├── fake_followers.csv/fake_followers.csv/users.csv
    ... (same pattern for all 9 groups)
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import kneighbors_graph
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from scipy import stats

from sybilgat import SybilGAT, BotGCN, BotSAGE

# ── CONFIG ────────────────────────────────────────────────────────────────────
BASE       = '/content/drive/MyDrive/cresci'
MAX_TWEETS = 200
KNN_K      = 10
TFIDF_DIM  = 200
EPOCHS     = 300
LR         = 5e-3
WEIGHT_DECAY = 1e-4
HIDDEN     = 64
HEADS      = 4
DROPOUT    = 0.5
PATIENCE   = 20

GROUPS = {
    'genuine_accounts':       0,
    'fake_followers':         1,
    'social_spambots_1':      1,
    'social_spambots_2':      1,
    'social_spambots_3':      1,
    'traditional_spambots_1': 1,
    'traditional_spambots_2': 1,
    'traditional_spambots_3': 1,
    'traditional_spambots_4': 1,
}

# ── HELPERS ───────────────────────────────────────────────────────────────────
def data_path(group, fname):
    return os.path.join(BASE, f'{group}.csv', f'{group}.csv', fname)

def parse_age_days(series):
    ref = pd.Timestamp('2015-05-01', tz='UTC')
    dt  = pd.to_datetime(series, utc=True, errors='coerce')
    return (ref - dt).dt.total_seconds() / 86400

def build_profile_features(df):
    df = df.copy()
    df['age_days']   = parse_age_days(df['created_at']).clip(lower=1)
    df['has_url']    = df['url'].notna().astype(float)
    df['has_desc']   = df['description'].notna().astype(float)
    df['name_len']   = df['name'].fillna('').str.len().astype(float)
    df['screen_len'] = df['screen_name'].fillna('').str.len().astype(float)
    df['ff_ratio']   = df['followers_count'] / (df['friends_count'] + 1)
    df['tweet_rate'] = df['statuses_count'] / df['age_days']
    df['fav_rate']   = df['favourites_count'] / df['age_days']

    cols = [
        'followers_count', 'friends_count', 'statuses_count',
        'favourites_count', 'listed_count',
        'verified', 'geo_enabled', 'default_profile',
        'has_url', 'has_desc',
        'name_len', 'screen_len', 'age_days',
        'ff_ratio', 'tweet_rate', 'fav_rate',
        'statuses_count',
    ]
    feat = df[cols].apply(pd.to_numeric, errors='coerce').fillna(0)
    return feat.values.astype(np.float32)

def stream_tweet_text(group):
    fpath = data_path(group, 'tweets.csv')
    if not os.path.exists(fpath):
        return {}
    user_tweets = {}
    for chunk in pd.read_csv(fpath, chunksize=50_000,
                              usecols=['user_id', 'text'],
                              dtype={'user_id': str, 'text': str},
                              encoding='latin-1',
                              on_bad_lines='skip'):
        for uid, grp in chunk.groupby('user_id'):
            existing = user_tweets.get(uid, [])
            if len(existing) < MAX_TWEETS:
                texts = grp['text'].dropna().tolist()
                user_tweets[uid] = (existing + texts)[:MAX_TWEETS]
    return user_tweets

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
def load_data():
    print("Loading groups...")
    all_users = []
    for group, label in GROUPS.items():
        upath = data_path(group, 'users.csv')
        if not os.path.exists(upath):
            print(f"  SKIP {group}")
            continue
        df = pd.read_csv(upath, dtype={'id': str}, on_bad_lines='skip')
        df['_group'] = group
        df['_label'] = label
        all_users.append(df)
        print(f"  {group}: {len(df)} users")

    users_df = pd.concat(all_users, ignore_index=True)
    print(f"\nTotal: {len(users_df)} users | "
          f"Bots: {(users_df['_label']==1).sum()} | "
          f"Humans: {(users_df['_label']==0).sum()}")

    # Profile features
    profile_feat = build_profile_features(users_df)

    # Tweet features
    print("\nStreaming tweets...")
    all_tweet_text = {}
    for group in GROUPS:
        if not os.path.exists(data_path(group, 'tweets.csv')):
            continue
        print(f"  {group}...")
        all_tweet_text.update(stream_tweet_text(group))

    uid_col = users_df['id'].astype(str).tolist()
    corpus  = [' '.join(all_tweet_text.get(uid, [''])) for uid in uid_col]

    print("Fitting TF-IDF...")
    tfidf     = TfidfVectorizer(max_features=TFIDF_DIM, sublinear_tf=True,
                                strip_accents='unicode', min_df=3)
    tfidf_arr = tfidf.fit_transform(corpus).toarray().astype(np.float32)

    X = np.concatenate([profile_feat, tfidf_arr], axis=1)
    y = users_df['_label'].values.astype(np.int64)
    print(f"Feature matrix: {X.shape}")

    # k-NN graph
    print(f"Building {KNN_K}-NN graph...")
    A = kneighbors_graph(X, n_neighbors=KNN_K, mode='connectivity',
                         include_self=False, n_jobs=-1)
    A = A + A.T
    A.data[:] = 1
    cx = A.tocoo()
    edge_index = torch.tensor(np.vstack([cx.row, cx.col]), dtype=torch.long)
    print(f"Edges: {edge_index.shape[1]:,}")

    # PyG Data object
    data = Data(
        x          = torch.tensor(X, dtype=torch.float),
        edge_index = edge_index,
        y          = torch.tensor(y, dtype=torch.long),
    )

    n   = data.num_nodes
    idx = torch.randperm(n)
    t1, t2 = int(0.70 * n), int(0.85 * n)
    data.train_mask = torch.zeros(n, dtype=torch.bool)
    data.val_mask   = torch.zeros(n, dtype=torch.bool)
    data.test_mask  = torch.zeros(n, dtype=torch.bool)
    data.train_mask[idx[:t1]]    = True
    data.val_mask  [idx[t1:t2]]  = True
    data.test_mask [idx[t2:]]    = True

    return data, users_df

# ── TRAINING ──────────────────────────────────────────────────────────────────
def train_model(model, data, cw, device):
    model     = model.to(device)
    data_d    = data.to(device)
    opt       = torch.optim.Adam(model.parameters(), lr=LR,
                                  weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    opt, factor=0.5, patience=10)
    criterion = nn.CrossEntropyLoss(weight=cw)

    best_val_f1  = 0
    best_state   = None
    patience_cnt = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        opt.zero_grad()
        out  = model(data_d.x, data_d.edge_index)
        loss = criterion(out[data_d.train_mask], data_d.y[data_d.train_mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        model.eval()
        with torch.no_grad():
            preds    = model(data_d.x, data_d.edge_index).argmax(dim=1)
            val_f1   = f1_score(data_d.y[data_d.val_mask].cpu(),
                                preds[data_d.val_mask].cpu(), average='macro')
        scheduler.step(1 - val_f1)

        if val_f1 > best_val_f1:
            best_val_f1  = val_f1
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1

        if patience_cnt >= PATIENCE:
            print(f"  Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        preds = model(data.x.to(device),
                      data.edge_index.to(device)).argmax(dim=1).cpu().numpy()

    y_true = data.y[data.test_mask].numpy()
    y_pred = preds[data.test_mask]
    return model, accuracy_score(y_true, y_pred), f1_score(y_true, y_pred, average='macro')

def run_baselines(data):
    X, y = data.x.numpy(), data.y.numpy()
    X_tr, y_tr = X[data.train_mask], y[data.train_mask]
    X_te, y_te = X[data.test_mask],  y[data.test_mask]

    results = {}
    for name, clf in [
        ('Logistic Regression', LogisticRegression(max_iter=1000, class_weight='balanced', n_jobs=-1)),
        ('Random Forest',       RandomForestClassifier(n_estimators=300, class_weight='balanced', n_jobs=-1, random_state=42)),
    ]:
        print(f"  Training {name}...")
        clf.fit(X_tr, y_tr)
        p = clf.predict(X_te)
        results[name] = (accuracy_score(y_te, p), f1_score(y_te, p, average='macro'))
    return results

# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    data, users_df = load_data()

    n_humans = (data.y == 0).sum().item()
    n_bots   = (data.y == 1).sum().item()
    cw = torch.tensor([n_bots / n_humans, 1.0], dtype=torch.float).to(device)
    print(f"Class weights: human={cw[0]:.2f}  bot={cw[1]:.2f}\n")

    in_ch = data.num_node_features

    print("=== Baselines ===")
    baseline_results = run_baselines(data)

    print("\n=== GNN Models ===")
    gnn_results = {}
    for name, model in [
        ('GCN',       BotGCN(in_ch)),
        ('GraphSAGE', BotSAGE(in_ch)),
        ('SybilGAT',  SybilGAT(in_ch)),
    ]:
        print(f"  Training {name}...")
        _, acc, f1 = train_model(model, data, cw, device)
        gnn_results[name] = (acc, f1)
        print(f"  {name}: Acc={acc:.4f}  F1={f1:.4f}")

    print("\n" + "=" * 52)
    print(f"{'Model':<22} {'Accuracy':>10} {'Macro F1':>10}")
    print("=" * 52)
    for name, (acc, f1) in {**baseline_results, **gnn_results}.items():
        print(f"{name:<22} {acc:>10.4f} {f1:>10.4f}")
    print("=" * 52)
