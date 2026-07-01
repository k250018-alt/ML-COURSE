
import os
import sys
import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "tree"))

from C5 import C5
from CART import Cart

DATA_PATH = os.path.join(
    os.path.dirname(__file__), "test data", "tree", "loan_approval.xlsx"
)
TARGET_COL = "Approved"
SEED = 42
np.random.seed(SEED)  # C5.Boost() uses the global np.random state, not a local Generator


def load_data():
    df = pd.read_excel(DATA_PATH, sheet_name="LoanApplications_Encoded", header=2)
    X = df.drop(columns=[TARGET_COL]).to_numpy(dtype=float)
    Y = df[TARGET_COL].to_numpy(dtype=int)
    return X, Y, df.drop(columns=[TARGET_COL]).columns.tolist()


def train_test_split(X, Y, test_frac=0.25, seed=SEED):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(Y))
    n_test = int(len(Y) * test_frac)
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    return X[train_idx], Y[train_idx], X[test_idx], Y[test_idx]


def test_c5():
    print("=" * 60)
    print("C5 (boosted C4.5 ensemble)")
    print("=" * 60)
    X, Y, feature_names = load_data()
    X_train, Y_train, X_test, Y_test = train_test_split(X, Y)

    model = C5(n=10)
    model.fit(X_train, Y_train)
    acc = model.check_accuracy(X_test, Y_test)
    print(f"Train size: {len(Y_train)}, Test size: {len(Y_test)}")
    print(f"Test accuracy: {acc:.2f}%")

    assert 0 <= acc <= 100, "accuracy out of range"
    assert acc > 52, f"C5 accuracy too low ({acc:.2f}%) - check Boost()/winnowing()"
    print("PASS\n")
    return acc


def test_cart():
    print("=" * 60)
    print("CART (Gini + cost-complexity pruning)")
    print("=" * 60)
    X, Y, feature_names = load_data()
    X_train, Y_train, X_test, Y_test = train_test_split(X, Y)

    model = Cart(alpha=2.0, min_split=5, depth=10)
    model.fit(X_train, Y_train)
    acc = model.check_accuracy(X_test, Y_test)
    n_leaves = model.count_leaves(model._tree)
    print(f"Train size: {len(Y_train)}, Test size: {len(Y_test)}")
    print(f"Test accuracy: {acc:.2f}%")
    print(f"Leaves after pruning: {n_leaves}")

    assert 0 <= acc <= 100, "accuracy out of range"
    assert acc > 68, f"CART accuracy too low ({acc:.2f}%) - check Gini split direction / alpha"
    assert n_leaves >= 1
    print("PASS\n")
    return acc


def test_cart_pruning_reduces_or_maintains_leaves():
    # A very high alpha should prune aggressively (fewer or equal leaves
    # than a near-zero alpha), since alpha penalizes leaf count directly.
    X, Y, _ = load_data()
    X_train, Y_train, _, _ = train_test_split(X, Y)

    light = Cart(alpha=0.5, min_split=5, depth=10)
    light.fit(X_train, Y_train)
    heavy = Cart(alpha=5.0, min_split=5, depth=10)
    heavy.fit(X_train, Y_train)

    leaves_light = light.count_leaves(light._tree)
    leaves_heavy = heavy.count_leaves(heavy._tree)
    print(f"Leaves (alpha=0.5): {leaves_light}   Leaves (alpha=5.0): {leaves_heavy}")
    assert leaves_heavy <= leaves_light, "higher alpha should not produce MORE leaves"
    print("PASS\n")


if __name__ == "__main__":
    c5_acc = test_c5()
    cart_acc = test_cart()
    test_cart_pruning_reduces_or_maintains_leaves()
    print("All tests completed.")
    print(f"Summary -> C5: {c5_acc:.2f}%   CART: {cart_acc:.2f}%")