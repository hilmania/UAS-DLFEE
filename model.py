import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

def build_rnn_model(input_shape, num_classes):
    """
    Membangun arsitektur model Simple RNN.

    Args:
        input_shape (tuple): Bentuk data input (timestep, fitur).
        num_classes (int): Jumlah kelas target.

    Returns:
        tf.keras.Model: Model RNN yang sudah dikompilasi.
    """
    model = Sequential([
        SimpleRNN(64, activation='relu', input_shape=input_shape, return_sequences=True),
        Dropout(0.2),
        SimpleRNN(32, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax') # Softmax untuk klasifikasi multi-kelas
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy', # Cocok untuk target integer
        metrics=['accuracy']
    )
    return model

def train_model(model, X_train, y_train, epochs=50, batch_size=32, validation_split=0.2):
    """
    Melatih model dengan data pelatihan.

    Args:
        model (tf.keras.Model): Model yang akan dilatih.
        X_train, y_train: Data pelatihan.
        epochs (int): Jumlah epoch.
        batch_size (int): Ukuran batch.
        validation_split (float): Proporsi data pelatihan untuk validasi.

    Returns:
        tf.keras.callbacks.History: Objek riwayat pelatihan.
    """
    # Early stopping untuk mencegah overfitting
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    print("\n--- Memulai Pelatihan Model ---")
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=[early_stopping],
        verbose=1
    )
    print("--- Pelatihan Selesai ---")
    return history

def evaluate_model(model, X_test, y_test):
    """
    Mengevaluasi model pada data pengujian.

    Args:
        model (tf.keras.Model): Model yang telah dilatih.
        X_test, y_test: Data pengujian.

    Returns:
        tuple: loss dan akurasi pada data tes.
    """
    print("\n--- Mengevaluasi Model pada Data Uji ---")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    return loss, accuracy
