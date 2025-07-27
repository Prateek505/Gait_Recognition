import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib

def train_model():
    """Trains an SVM model and saves it."""
    # Load the dataset
    try:
        df = pd.read_csv('gait_features.csv')
    except FileNotFoundError:
        print("Error: gait_features.csv not found. Please run process_data.py first.")
        return

    # Prepare the data
    X = df.drop('person', axis=1)
    y = df['person']

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

    # Train the SVM model
    print("Training SVM model...")
    model = SVC(kernel='rbf', probability=True, C=1.0, gamma='scale', random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel training complete!")
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    
    # Show detailed report
    print("\nClassification Report:")
    # Get original labels for the report
    target_names = label_encoder.inverse_transform(sorted(list(set(y_encoded))))
    print(classification_report(y_test, y_pred, target_names=target_names))

    # Save the trained model and the label encoder
    joblib.dump(model, 'gait_model.pkl')
    joblib.dump(label_encoder, 'label_encoder.pkl')
    print("Model and label encoder saved as gait_model.pkl and label_encoder.pkl")

if __name__ == "__main__":
    train_model()