# Flexible ANN Template
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

# =========================
# 1. Load your dataset
# =========================
# Example CSV: replace 'your_data.csv' with your file
data = pd.read_csv('your_data.csv')

# Assume last column is target
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# =========================
# 2. Preprocess features
# =========================
# Example: automatic scaling for numerical columns
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =========================
# 3. Preprocess target
# =========================
if len(np.unique(y)) > 2:
    # Multi-class classification
    encoder = OneHotEncoder(sparse=False)
    y_processed = encoder.fit_transform(y.reshape(-1, 1))
    output_neurons = y_processed.shape[1]
    loss_function = 'categorical_crossentropy'
    output_activation = 'softmax'
else:
    # Binary classification or regression
    if y.dtype == np.object or np.array_equal(np.unique(y), [0, 1]):
        # Binary classification
        y_processed = y
        output_neurons = 1
        loss_function = 'binary_crossentropy'
        output_activation = 'sigmoid'
    else:
        # Regression
        y_processed = y
        output_neurons = 1
        loss_function = 'mean_squared_error'
        output_activation = 'linear'

# =========================
# 4. Train/test split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_processed, test_size=0.2, random_state=42
)

# =========================
# 5. Build the ANN model
# =========================
model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(output_neurons, activation=output_activation)
])

# =========================
# 6. Compile the model
# =========================
model.compile(optimizer='adam', loss=loss_function, metrics=['accuracy'])

# =========================
# 7. Train the model
# =========================
model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1)

# =========================
# 8. Evaluate the model
# =========================
y_pred = model.predict(X_test)

if loss_function == 'mean_squared_error':
    print("MSE:", mean_squared_error(y_test, y_pred))
else:
    if output_neurons > 1:
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_test_classes = np.argmax(y_test, axis=1)
    else:
        y_pred_classes = (y_pred > 0.5).astype(int)
        y_test_classes = y_test
    print("Accuracy:", accuracy_score(y_test_classes, y_pred_classes))

# =========================
# 9. Predictions for new data
# =========================
# new_X_scaled = scaler.transform(new_X)
# predictions = model.predict(new_X_scaled)


# Add XOR dataset example and modern ANN model implementation
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

# XOR dataset
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
y = np.array([0, 1, 1, 0])

# Build ANN model (modern style)
model = Sequential([
    Input(shape=(2,)),
    Dense(4, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train
model.fit(X, y, epochs=5000, verbose=0)  # Increase epochs for XOR

# Predict
predictions = model.predict(X)
print("Predictions:")
print(predictions)
