import pandas as pd
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import joblib

# Load data
df = pd.read_csv("fake_job_postings.csv")
df.fillna('', inplace=True)

# Combine important text columns
df['text'] = df['title'] + ' ' + df['company_profile'] + ' ' + df['description'] + ' ' + df['requirements']
X_text = df['text']
y = df['fraudulent']

# Vectorize
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X = vectorizer.fit_transform(X_text)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# Model: SGDClassifier
model = SGDClassifier(loss='log_loss', max_iter=1, warm_start=True)

# Epoch training
epochs = 10
train_acc = []
test_acc = []

# Class labels for partial_fit
classes = list(set(y))

# Track training time
start_time = time.time()

# Training loop with progress bar
for epoch in tqdm(range(epochs), desc="Training Epochs"):
    model.partial_fit(X_train, y_train, classes=classes)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_accuracy = accuracy_score(y_train, train_pred)
    test_accuracy = accuracy_score(y_test, test_pred)

    train_acc.append(train_accuracy)
    test_acc.append(test_accuracy)

    print(f"Epoch {epoch+1}/{epochs} | Train Acc: {train_accuracy:.4f} | Test Acc: {test_accuracy:.4f}")

end_time = time.time()
print(f"\n🕒 Training completed in {end_time - start_time:.2f} seconds")

# Save accuracy log
pd.DataFrame({
    "epoch": list(range(1, epochs+1)),
    "train_accuracy": train_acc,
    "test_accuracy": test_acc
}).to_csv("epoch_accuracy_log.csv", index=False)

# Plot accuracy
plt.figure(figsize=(10, 5))
plt.plot(range(1, epochs+1), train_acc, label='Train Accuracy', marker='o')
plt.plot(range(1, epochs+1), test_acc, label='Test Accuracy', marker='s')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Model Learning Over Epochs")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("training_progress.png")
plt.show()

# Confusion matrix and classification report
y_pred = model.predict(X_test)
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Real', 'Fake'])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.show()

# Save model and vectorizer
joblib.dump(model, 'fake_job_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

print("✅ Model and vectorizer saved.")
