import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.initializers import GlorotUniform, HeNormal, RandomNormal
import matplotlib.pyplot as plt
from io import BytesIO

# --- Ackley Function ---
def ackley_function(x):
    a = 20
    b = 0.2
    c = 2 * np.pi
    sum_sq_term = -b * np.sqrt(np.mean(x**2, axis=1))
    cos_term = -np.mean(np.cos(c * x), axis=1)
    return a + np.exp(1) + sum_sq_term + cos_term

# --- Generate Data ---
def generate_data():
    X_train = np.random.uniform(-5.0, 5.0, (10000, 2))  # Training data
    y_train = ackley_function(X_train)  # Apply Ackley function
    X_unseen = np.random.uniform(-5.0, 5.0, (2000, 2))  # Test data
    y_unseen = ackley_function(X_unseen)
    return X_train, y_train, X_unseen, y_unseen

# --- Build Model ---
def build_model(hidden_units, dropout_rate, weight_initializer):
    model = Sequential([
        Dense(hidden_units, input_dim=2, activation='relu', kernel_initializer=weight_initializer),
        BatchNormalization(),
        Dropout(dropout_rate),
        Dense(hidden_units, activation='relu', kernel_initializer=weight_initializer),
        BatchNormalization(),
        Dropout(dropout_rate),
        Dense(1, kernel_initializer=weight_initializer)  # Output layer
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# --- Ensemble Prediction ---
def ensemble_predict(models, X):
    predictions = [model.predict(X, verbose=0) for model in models]
    return np.mean(predictions, axis=0)

# --- Plot Functions ---
def plot_loss(history):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history.history['loss'], label='Training Loss')
    ax.plot(history.history['val_loss'], label='Validation Loss')
    ax.set_title('Training and Validation Loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend()
    st.pyplot(fig)

def plot_predictions(models, X_unseen, y_unseen):
    y_pred = ensemble_predict(models, X_unseen)
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Prediction vs Actual
    ax[0].scatter(y_unseen, y_pred, alpha=0.3)
    ax[0].plot([y_unseen.min(), y_unseen.max()], [y_unseen.min(), y_unseen.max()], 'r--')
    ax[0].set_title('Actual vs Predicted')
    ax[0].set_xlabel('Actual Values')
    ax[0].set_ylabel('Predicted Values')

    # Residuals
    residuals = y_unseen - y_pred.flatten()
    ax[1].scatter(y_pred, residuals, alpha=0.3)
    ax[1].hlines(0, y_pred.min(), y_pred.max(), colors='r', linestyles='dashed')
    ax[1].set_title('Residuals')
    ax[1].set_xlabel('Predicted Values')
    ax[1].set_ylabel('Residuals')

    st.pyplot(fig)

# --- Streamlit App ---
def main():
    st.title("Ackley Function Approximation with Neural Networks (Ensemble)")

    st.sidebar.header("Hyperparameters")

    # Hyperparameter inputs
    hidden_units = st.sidebar.slider("Hidden Units", min_value=32, max_value=512, step=32, value=256)
    dropout_rate = st.sidebar.slider("Dropout Rate", min_value=0.0, max_value=0.5, step=0.1, value=0.2)
    batch_size = st.sidebar.slider("Batch Size", min_value=16, max_value=128, step=16, value=32)
    epochs = st.sidebar.slider("Epochs", min_value=10, max_value=200, step=10, value=50)
    weight_initializer = st.sidebar.selectbox("Weight Initializer", ["glorot_uniform", "he_normal", "random_normal"])
    num_models = st.sidebar.slider("Number of Models in Ensemble", min_value=1, max_value=10, step=1, value=3)

    # Generate data
    X_train, y_train, X_unseen, y_unseen = generate_data()

    if st.sidebar.button("Train Ensemble"):
        st.write("### Training the Ensemble Models...")
        initializers = {
            "glorot_uniform": GlorotUniform(),
            "he_normal": HeNormal(),
            "random_normal": RandomNormal()
        }
        models = []
        for i in range(num_models):
            st.write(f"Training Model {i + 1}/{num_models}")
            model = build_model(hidden_units, dropout_rate, initializers[weight_initializer])
            history = model.fit(X_train, y_train, 
                                epochs=epochs, 
                                batch_size=batch_size, 
                                validation_split=0.2, 
                                verbose=0)
            models.append(model)
            plot_loss(history)

        st.success("Ensemble training complete!")

        st.write("### Predictions on Unseen Data")
        plot_predictions(models, X_unseen, y_unseen)

if __name__ == "__main__":
    main()
