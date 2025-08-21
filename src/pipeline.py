 
from .data_processing import load_data, preprocess_data
from .model import evaluate_model, save_model, train_model


def run_pipeline(data_path, model_save_path):
    # Load and preprocess data
    df = load_data(data_path)
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate model
    accuracy = evaluate_model(model, X_test, y_test)
    print(f"Model Accuracy: {accuracy:.2f}")
    
    # Save model
    save_model(model, model_save_path)
    
    return model, accuracy