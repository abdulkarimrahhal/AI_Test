import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

iris_df = pd.read_csv("IrisDataset.csv")
print(iris_df.head())
print()

# معالجة القيم الفارغة
print(iris_df.info())
for attr in iris_df.columns:
    if not iris_df[attr].isna().sum() == 0:
        iris_df[attr].fillna(iris_df[attr].mean(), inplace=True)
        print(f"the null values in {attr} column were replaced with mean value")
print()

x = iris_df.drop(columns="species")

#  تطبيع البيانات (normalization)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(x)

# تقسيم البيانات الى 70% تدريب و 30% اختبار
y = iris_df["species"]
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

# تدريب النموذج
model = RandomForestClassifier(n_estimators=100, max_depth=1, random_state=42) # ملاحظة : max_depth =1 لكي لا يصبح لدينا overfitting
model.fit(X_train, y_train)

# قياس دقة النموذج
y_predict = model.predict(X_test)
accuracy = accuracy_score(y_test, y_predict)
print(f"\n Accuracy : {accuracy} ")

# رسم المصفوفة
class_names = sorted(list(set(y_test) | set(y_predict)))
conf_matrix = confusion_matrix(y_test, y_predict)
plt.figure(figsize=(8, 6))
sns.heatmap(
    conf_matrix,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names,
)
plt.title(" Confusion Matrix ")
plt.xlabel(" Predicted ")
plt.ylabel(" Actual ")
plt.show()
