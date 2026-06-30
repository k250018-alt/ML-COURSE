import numpy as np
import pandas as pd
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'tree'))
from IB3 import *
from C45 import *

# ── helpers ──────────────────────────────────────────────────────────────────

EDUCATION_MAP = {"High School": 0, "Bachelor": 1, "Master": 2, "PhD": 3}

def load_data(path=None):
    if path is None:
        base = os.path.dirname(__file__)
        path = os.path.join(base, "test data", "tree", "loan_data.xlsx")
    df = pd.read_excel(path, sheet_name="Loan Approval Data")
    df["Education"] = df["Education"].map(EDUCATION_MAP)
    X = df[["Age","Salary","Years_Employed","Credit_Score",
            "Existing_Loans","Education","Married"]].values.tolist()
    Y = df["Approved"].values.tolist()
    return X, Y

def train_test_split(X, Y, test_size=0.2, seed=42):
    np.random.seed(seed)
    n       = len(Y)
    indices = np.random.permutation(n)
    split   = int(n * (1 - test_size))
    train_i, test_i = indices[:split], indices[split:]
    X_arr = np.array(X)
    Y_arr = np.array(Y)
    return (X_arr[train_i].tolist(), Y_arr[train_i].tolist(),
            X_arr[test_i].tolist(),  Y_arr[test_i].tolist())

def print_separator(char="─", width=55):
    print(char * width)

def print_confusion_matrix(y_true, y_pred):
    tp = sum(1 for a, b in zip(y_true, y_pred) if a == 1 and b == 1)
    tn = sum(1 for a, b in zip(y_true, y_pred) if a == 0 and b == 0)
    fp = sum(1 for a, b in zip(y_true, y_pred) if a == 0 and b == 1)
    fn = sum(1 for a, b in zip(y_true, y_pred) if a == 1 and b == 0)
    print(f"  {'':12} Predicted 0   Predicted 1")
    print(f"  {'Actual 0':12} {tn:^13} {fp:^13}")
    print(f"  {'Actual 1':12} {fn:^13} {tp:^13}")

# ── user input ────────────────────────────────────────────────────────────────

def get_user_input():
    print_separator()
    print("  ENTER APPLICANT DETAILS")
    print_separator()

    def ask_int(prompt, lo, hi):
        while True:
            try:
                v = int(input(f"  {prompt} ({lo}-{hi}): "))
                if lo <= v <= hi:
                    return v
                print(f"  Please enter a value between {lo} and {hi}.")
            except ValueError:
                print("  Please enter a whole number.")

    def ask_choice(prompt, choices):
        print(f"  {prompt}")
        for idx, c in enumerate(choices, 1):
            print(f"    {idx}. {c}")
        while True:
            try:
                v = int(input("  Choice: "))
                if 1 <= v <= len(choices):
                    return choices[v - 1]
                print(f"  Enter a number between 1 and {len(choices)}.")
            except ValueError:
                print("  Please enter a number.")

    age      = ask_int("Age", 18, 80)
    salary   = ask_int("Annual Salary ($)", 10000, 500000)
    years    = ask_int("Years Employed", 0, 50)
    credit   = ask_int("Credit Score", 300, 850)
    loans    = ask_int("Number of Existing Loans", 0, 10)
    edu_str  = ask_choice("Education Level", list(EDUCATION_MAP.keys()))
    edu      = EDUCATION_MAP[edu_str]
    married  = ask_choice("Marital Status", ["Single", "Married"])
    married  = 1 if married == "Married" else 0

    return [age, salary, years, credit, loans, edu, married]

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print_separator("═")
    print("  DECISION TREE MODEL TESTER")
    print("  ID3  vs  C4.5")
    print_separator("═")

    # ── load & split ──
    print("\n  Loading data...")
    try:
        X, Y = load_data()
    except FileNotFoundError:
        print("  ERROR: loan_data.xlsx not found.")
        print("  Make sure loan_data.xlsx is in the same folder.")
        return

    X_train, Y_train, X_test, Y_test = train_test_split(X, Y, test_size=0.2)
    print(f"  Train size : {len(Y_train)} samples")
    print(f"  Test  size : {len(Y_test)}  samples")

    # ── train & evaluate ID3 ──
    print("\n  Training ID3...")
    id3 = ID3()
    id3.fit(X_train, Y_train)
    id3_train_acc = id3.check_accuracy(X_train, Y_train)
    id3_test_acc  = id3.check_accuracy(X_test,  Y_test)
    id3_preds     = id3.predict(X_test)

    # ── train & evaluate C4.5 ──
    print("  Training C4.5...")
    c45 = C45()
    c45.fit(X_train, Y_train)
    c45_train_acc = c45.check_accuracy(X_train, Y_train)
    c45_test_acc  = c45.check_accuracy(X_test,  Y_test)
    c45_preds     = c45.predict(X_test)

    # ── results ──
    print_separator("═")
    print("  ACCURACY RESULTS")
    print_separator("═")
    print(f"  {'Model':<10} {'Train Acc':>12} {'Test Acc':>12}")
    print_separator()
    print(f"  {'ID3':<10} {id3_train_acc:>11.1f}% {id3_test_acc:>11.1f}%")
    print(f"  {'C4.5':<10} {c45_train_acc:>11.1f}% {c45_test_acc:>11.1f}%")
    print_separator()

    winner = "ID3" if id3_test_acc > c45_test_acc else "C4.5" if c45_test_acc > id3_test_acc else "Tie"
    print(f"\n  Best model on test set: {winner}")

    # ── confusion matrices ──
    print_separator("═")
    print("  CONFUSION MATRICES  (test set)")
    print_separator("═")
    print("\n  ID3:")
    print_separator()
    print_confusion_matrix(Y_test, id3_preds)
    print("\n  C4.5:")
    print_separator()
    print_confusion_matrix(Y_test, c45_preds)

    # ── per-sample predictions ──
    print_separator("═")
    print("  PER-SAMPLE PREDICTIONS  (test set)")
    print_separator("═")
    print(f"  {'#':<4} {'Actual':>8} {'ID3':>8} {'C4.5':>8} {'Match?':>8}")
    print_separator()
    for i, (actual, p_id3, p_c45) in enumerate(zip(Y_test, id3_preds, c45_preds), 1):
        match = "✓" if p_id3 == actual and p_c45 == actual else "✗"
        print(f"  {i:<4} {actual:>8} {p_id3:>8} {p_c45:>8} {match:>8}")

    # ── user prediction ──
    while True:
        print_separator("═")
        print("  PREDICT FOR A NEW APPLICANT")
        print_separator("═")
        sample = get_user_input()

        id3_pred = id3.predictone(sample, id3._tree)
        c45_pred = c45.predictone(sample, c45._tree)

        print_separator()
        print("  PREDICTION RESULTS")
        print_separator()
        label = {1: "APPROVED ✓", 0: "REJECTED ✗"}
        print(f"  ID3  says: {label[id3_pred]}")
        print(f"  C4.5 says: {label[c45_pred]}")
        print_separator()

        again = input("\n  Test another applicant? (y/n): ").strip().lower()
        if again != "y":
            break

    print_separator("═")
    print("  Done.")
    print_separator("═")


if __name__ == "__main__":
    main()