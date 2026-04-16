import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

model = Sequential()

# Input layer
model.add(Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)))

# Hidden layers
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))

# Output layer (3 classes)
model.add(Dense(3, activation='softmax'))

# Compile
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()


history = model.fit(
    X_train_scaled, y_train_enc,
    validation_split=0.2,
    epochs=20,
    batch_size=32,
    verbose=1
)

from sklearn.metrics import classification_report

# Predictions
y_pred_probs = model.predict(X_test_scaled)
y_pred = y_pred_probs.argmax(axis=1)

# Convert back to labels
y_pred_labels = le_target.inverse_transform(y_pred)

print(classification_report(y_test, y_pred_labels))

from sklearn.metrics import accuracy_score, f1_score

dl_accuracy = accuracy_score(y_test, y_pred_labels)
dl_f1 = f1_score(y_test, y_pred_labels, average='weighted')

print("Deep Learning Model:")
print("Accuracy:", dl_accuracy)
print("F1 Score:", dl_f1)