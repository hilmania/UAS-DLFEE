# model_builder.py

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, GRU, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

def build_model(model_type, input_shape, num_classes, units=64, dropout_rate=0.2, learning_rate=0.001):
    """
    Membangun berbagai jenis model RNN berdasarkan tipe yang ditentukan.

    Args:
        model_type (str): Tipe model ('simple_rnn', 'bidirectional_rnn', 'lstm', 'gru').
        input_shape (tuple): Bentuk data input.
        num_classes (int): Jumlah kelas target.
        units (int): Jumlah unit di lapisan RNN.
        dropout_rate (float): Rate untuk Dropout.
        learning_rate (float): Learning rate untuk optimizer.

    Returns:
        tf.keras.Model: Model yang sudah dikompilasi.
    """
    model = Sequential()

    # Menentukan lapisan RNN berdasarkan model_type
    if model_type == 'simple_rnn':
        model.add(SimpleRNN(units, activation='relu', input_shape=input_shape))
    elif model_type == 'bidirectional_rnn':
        model.add(Bidirectional(SimpleRNN(units, activation='relu'), input_shape=input_shape))
    elif model_type == 'lstm':
        model.add(LSTM(units, input_shape=input_shape))
    elif model_type == 'gru':
        model.add(GRU(units, input_shape=input_shape))
    else:
        raise ValueError("Tipe model tidak dikenal. Pilih dari: 'simple_rnn', 'bidirectional_rnn', 'lstm', 'gru'")

    model.add(Dropout(dropout_rate))
    model.add(Dense(num_classes, activation='softmax'))

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

def train_model(model, X_train, y_train, epochs=30, batch_size=32):
    """
    Melatih model dengan data pelatihan.
    """
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=0  # Set ke 0 agar output tidak terlalu ramai selama perbandingan
    )
    return history
