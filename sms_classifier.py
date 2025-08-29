import os
import sys
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers

MODEL_PATH = "sms_model.keras"  # modelo guardado (incluye TextVectorization en el grafo)

def load_dataset(path="sms.tsv"):
    """
    Carga dataset tabulado con columnas: label \t message
    Convierte labels a 0=ham, 1=spam
    Split 80/20 (estratificado) de forma reproducible
    """
    if not os.path.exists(path):
        raise FileNotFoundError("No se encontró sms.tsv. Verifica descarga o ruta.")
    df = pd.read_csv(path, sep="\t")
    # normalizar nombres
    cols = [c.strip().lower().replace(" ", "_") for c in df.columns]
    df.columns = cols
    # algunos repos usan 'message' o 'text' como columna de contenido
    text_col = "message" if "message" in df.columns else ("text" if "text" in df.columns else None)
    label_col = "label" if "label" in df.columns else None
    if text_col is None or label_col is None:
        raise ValueError("El TSV debe contener columnas 'label' y 'message' (o 'text').")

    df = df[[label_col, text_col]].dropna()
    df[label_col] = df[label_col].str.strip().str.lower()
    df["y"] = (df[label_col] == "spam").astype("int32")
    df["x"] = df[text_col].astype(str)

    # split 80/20 estratificado
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        df["x"].values, df["y"].values, test_size=0.2, random_state=42, stratify=df["y"].values
    )
    return X_train, X_test, y_train, y_test

def make_model(max_tokens=20000, seq_len=100, embed_dim=64):
    """
    Modelo string->prob(spam):
      Input (string) -> TextVectorization -> Embedding -> GlobalAvgPool -> Dense -> Sigmoid
    """
    text_input = tf.keras.Input(shape=(1,), dtype=tf.string, name="text")
    vectorize = layers.TextVectorization(
        max_tokens=max_tokens, output_mode="int", output_sequence_length=seq_len
    )
    # La capa se adaptará con el texto de entrenamiento antes de compilar.
    x = vectorize(text_input)
    x = layers.Embedding(max_tokens + 1, embed_dim)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    output = layers.Dense(1, activation="sigmoid", name="spam_prob")(x)

    model = tf.keras.Model(text_input, output)
    model.vectorize = vectorize  # para acceder y adaptar fácilmente
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="binary_crossentropy",
                  metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])
    return model

def train(save_path=MODEL_PATH, epochs=8, batch_size=32):
    X_train, X_test, y_train, y_test = load_dataset()

    model = make_model()
    # Adaptar el TextVectorization con el texto de entrenamiento
    model.vectorize.adapt(tf.data.Dataset.from_tensor_slices(X_train).batch(256))

    # tf.data para rendir mejor
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size).prefetch(1)
    val_ds   = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size).prefetch(1)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=2, restore_best_weights=True, verbose=1)
    ]

    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, verbose=2, callbacks=callbacks)

    loss, acc, auc = model.evaluate(val_ds, verbose=0)
    print(f"Validation -> acc: {acc:.4f} | auc: {auc:.4f}")

    model.save(save_path)
    print(f"Modelo guardado en {save_path}")
    return model

# --- API requerida por el reto ---
def predict_message(message: str):
    """
    Retorna: [probabilidad_de_spam (0..1), 'spam'|'ham']
    Nota: prob = 1.0 => más probable 'spam'; 0.0 => 'ham'.
    """
    if not os.path.exists(MODEL_PATH):
        # entrena si no existe modelo (con defaults)
        print("⚠️  No se encontró modelo entrenado; entrenando uno rápido...")
        _ = train()

    model = tf.keras.models.load_model(MODEL_PATH)
    # El TextVectorization está dentro del grafo del modelo (input string)
    prob_spam = float(model.predict([message], verbose=0)[0][0])
    label = "spam" if prob_spam >= 0.5 else "ham"
    return [prob_spam, label]

def main():
    parser = argparse.ArgumentParser(description="SMS Text Classifier (ham/spam) con Keras")
    parser.add_argument("--train", action="store_true", help="Entrenar y guardar modelo")
    parser.add_argument("--epochs", type=int, default=8, help="Épocas de entrenamiento")
    parser.add_argument("--message", type=str, default=None, help="Mensaje a clasificar")
    args = parser.parse_args()

    if args.train:
        train(epochs=args.epochs)

    if args.message is not None:
        res = predict_message(args.message)
        print(res)

    if not args.train and args.message is None:
        # demo rápida
        if not os.path.exists(MODEL_PATH):
            train(epochs=args.epochs)
        samples = [
            "FREE entry in 2 a wkly comp to win FA Cup final tkts! Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's",
            "Hey, are we still on for dinner tonight?",
            "URGENT! You have won a 1 week FREE membership in our £100,000 Prize Jackpot! Txt the word: WIN to 80086 NOW."
        ]
        for s in samples:
            print(s, "->", predict_message(s))

if __name__ == "__main__":
    main()
