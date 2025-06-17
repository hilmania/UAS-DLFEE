import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, SimpleRNN, Dense, Dropout
from tensorflow.keras.optimizers import Adam

def build_rnn_model(input_shape, num_classes, rnn_units_1=64, rnn_units_2=32, dropout_rate=0.2, learning_rate=0.001):
    """
    Membangun arsitektur model Simple RNN dengan hyperparameter yang bisa diatur.
    """
    model = Sequential()
    model.add(Input(shape=input_shape))

    # Menggunakan parameter untuk fleksibilitas
    model.add(SimpleRNN(rnn_units_1, activation='tanh', return_sequences=True))
    model.add(Dropout(dropout_rate))
    model.add(SimpleRNN(rnn_units_2, activation='tanh'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(num_classes, activation='softmax'))

    # Menggunakan learning rate yang bisa diatur
    optimizer = Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def evaluate_model(model, X_test, y_test):
    """
    Mengevaluasi model pada data pengujian.
    (Fungsi ini tetap sama dan bisa dipertahankan)
    """
    print("\n--- Mengevaluasi Model pada Data Uji ---")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    return loss, accuracy
