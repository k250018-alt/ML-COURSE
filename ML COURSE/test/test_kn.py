import sys
import numpy as np
import pandas as pd

sys.path.append('../KN')

from KMeans import KMeans
from KNNClassifcation import KNearestNeighbour
from KNNRegression import KNNREgression

# ─────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────

student_df  = pd.read_excel('test data/KN/student_data.xlsx')
house_df    = pd.read_excel('test data/KN/house_data.xlsx')
customer_df = pd.read_excel('test data/KN/customer_data.xlsx')

student_X  = student_df[['hours_studied', 'attendance_%', 'prev_grade']].to_numpy(dtype=float)
student_Z  = student_df['pass'].to_numpy()

house_X    = house_df[['size_sqft', 'bedrooms', 'distance_to_city_km']].to_numpy(dtype=float)
house_Z    = house_df['price_$1000s'].to_numpy(dtype=float)

customer_X = customer_df[['age', 'annual_income_$1000s', 'spending_score']].to_numpy(dtype=float)


# ─────────────────────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────────────────────

def print_header(title):
    print("\n" + "=" * 55)
    print(f"  {title}")
    print("=" * 55)

def print_result(label, value):
    print(f"  {label:<35} {value}")


# ─────────────────────────────────────────────────────────────
# TEST 1 — KNN CLASSIFICATION (Student Pass/Fail)
# ─────────────────────────────────────────────────────────────

print_header("KNN CLASSIFICATION — Student Pass/Fail")

try:
    knn = KNearestNeighbour()
    knn.fit(student_X, student_Z)

    test_cases = np.array([
        [7.0, 88, 80],   # should Pass
        [1.5, 35, 38],   # should Fail
        [5.0, 72, 68],   # should Pass
    ])
    predictions = knn.predict(test_cases)
    labels = ["Pass" if p == 1 else "Fail" for p in predictions]

    print_result("Test [7h, 88% attend, grade80]:", labels[0])
    print_result("Test [1.5h, 35% attend, grade38]:", labels[1])
    print_result("Test [5h, 72% attend, grade68]:", labels[2])

    accuracy = knn.check_accuracy(
        knn._KNearestNeighbour__Xtest,
        knn._KNearestNeighbour__Ztest
    )
    print_result("Accuracy:", f"{accuracy * 100:.2f}%")

except Exception as e:
    print(f"  ERROR: {e}")


# ─────────────────────────────────────────────────────────────
# TEST 2 — KNN REGRESSION (House Prices)
# ─────────────────────────────────────────────────────────────

print_header("KNN REGRESSION — House Price Prediction")

try:
    knnr = KNNREgression()
    knnr.fit(house_X, house_Z)

    test_houses = np.array([
        [1600, 3, 6],
        [3000, 5, 2],
        [750,  1, 17],
    ])
    preds = knnr.predict(test_houses)

    print_result("Test [1600sqft, 3bed, 6km]:", f"${preds[0]:.1f}k  (expected ~$265k)")
    print_result("Test [3000sqft, 5bed, 2km]:", f"${preds[1]:.1f}k  (expected ~$520k)")
    print_result("Test [750sqft,  1bed, 17km]:", f"${preds[2]:.1f}k  (expected ~$110k)")

    mse  = knnr.check_accuracy(
        knnr._KNNREgression__Xtest,
        knnr._KNNREgression__Ztest
    )
    rmse = np.sqrt(mse)
    accuracy_pct = max(0, (1 - rmse / np.mean(house_Z)) * 100)

    print_result("MSE:", f"{mse:.2f}")
    print_result("RMSE:", f"${rmse:.2f}k  (avg error per prediction)")
    print_result("Accuracy (1 - RMSE/mean price):", f"{accuracy_pct:.2f}%")

except Exception as e:
    print(f"  ERROR: {e}")


# ─────────────────────────────────────────────────────────────
# TEST 3 — KMEANS CLUSTERING (Mall Customers)
# ─────────────────────────────────────────────────────────────

print_header("KMEANS CLUSTERING — Mall Customer Segments")

for model_num, model_name in [(1,"Euclidean"),(2,"Manhattan"),(3,"Hamming"),(4,"Cosine")]:
    try:
        km = KMeans(model_num)
        km.fit(customer_X, 3)

        new_customers = np.array([
            [24, 86, 84],   # young high spender
            [41, 55, 50],   # middle aged medium
            [63, 29, 20],   # older low spender
        ])

        c1 = km.predict(new_customers[0])
        c2 = km.predict(new_customers[1])
        c3 = km.predict(new_customers[2])

        sil, inertia = km.check_accuracy(customer_X, 3)
        sil_pct = (sil + 1) / 2 * 100

        print(f"\n  [{model_name}]")
        print_result("  Young high spender → cluster:", str(c1))
        print_result("  Middle aged medium → cluster:", str(c2))
        print_result("  Older low spender  → cluster:", str(c3))
        print_result("  Silhouette Score:", f"{sil:.4f}")
        print_result("  Inertia (WCSS):",   f"{inertia:.4f}")
        print_result("  Cluster Quality %:", f"{sil_pct:.2f}%")

    except Exception as e:
        print(f"  [{model_name}] ERROR: {e}")

print("\n" + "=" * 55)
print("  DONE")
print("=" * 55 + "\n")