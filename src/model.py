import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def train_model(X_train, y_train, n_estimators=100):
    model = RandomForestClassifier(n_estimators=n_estimators)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    return accuracy_score(y_test, predictions)

def save_model(model, filename):
    joblib.dump(model, filename)