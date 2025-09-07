import pickle
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier

def train_and_evaluate(X_train, y_train, X_test, y_test):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    with open("reports/metrics.txt", "w") as f:
        f.write("Accuracy: {:.4f}\n".format(accuracy_score(y_test, y_pred)))
        f.write("\nClassification Report:\n")
        f.write(classification_report(y_test, y_pred))

    with open("data/processed/model.pkl", "wb") as f:
        pickle.dump(model, f)

    return model
