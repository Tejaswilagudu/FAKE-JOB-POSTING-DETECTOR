import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
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

# SGDClassifier (logistic regression-like with partial_fit)
model = SGDClassifier(loss='log_loss', max_iter=1, warm_start=True)

# Epoch training
epochs = 10
train_acc = []
test_acc = []

# You must call partial_fit with `classes=` in first call
classes = list(set(y))

for epoch in range(epochs):
    model.partial_fit(X_train, y_train, classes=classes)
    
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_acc.append(accuracy_score(y_train, train_pred))
    test_acc.append(accuracy_score(y_test, test_pred))

    print(f"Epoch {epoch+1}/{epochs} | Train Acc: {train_acc[-1]:.4f} | Test Acc: {test_acc[-1]:.4f}")

# Plot accuracy graph
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

# Save model and vectorizer
joblib.dump(model, 'fake_job_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
