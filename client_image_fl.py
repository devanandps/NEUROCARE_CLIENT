import argparse
from pathlib import Path
import flwr as fl
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
import numpy as np

# ================================
# Paths
# ================================
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)
IMAGE_MODEL_FILE = MODEL_DIR / "image_model.h5"
CLIENT_MODELS_DIR = MODEL_DIR / "client_saved"
CLIENT_MODELS_DIR.mkdir(exist_ok=True)

# ================================
# Build Model
# ================================
def build_model(num_classes: int) -> Model:
    base = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    base.trainable = False
    inputs = Input(shape=(224, 224, 3))
    x = base(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Dense(256, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)
    outputs = Dense(num_classes, activation="softmax")(x)
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(1e-4), loss="categorical_crossentropy", metrics=["accuracy"])
    return model

# ================================
# Data Generator
# ================================
def get_train_generator(client_data_dir: Path, batch_size: int = 4):
    datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    return datagen.flow_from_directory(
        client_data_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
    )

# ================================
# Flower Client
# ================================
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, train_gen, client_id, local_epochs=1):
        self.model = model
        self.train_gen = train_gen
        self.client_id = client_id
        self.local_epochs = local_epochs

    def get_parameters(self, config=None):
        # Convert all model weights to NumPy arrays
        return [np.array(w, copy=True) for w in self.model.get_weights()]

    def fit(self, parameters, config):
        print(f"[Client {self.client_id}] Starting local training...")
        self.model.set_weights(parameters)
        self.train_gen.reset()  # Reset generator to avoid placeholder errors
        steps = max(1, len(self.train_gen))
        history = self.model.fit(
            self.train_gen,
            epochs=self.local_epochs,
            steps_per_epoch=steps,
            verbose=1,
        )

        # Save updated local model
        try:
            self.model.save(CLIENT_MODELS_DIR / f"image_model_{self.client_id}.h5", save_format="h5")
            print(f"[Client {self.client_id}] Model saved successfully")
        except Exception as e:
            print(f"[Client {self.client_id}] Error saving model: {e}")

        # Convert metrics to Python floats
        train_loss = float(history.history["loss"][-1])
        train_acc = float(history.history.get("accuracy", [0.0])[-1])
        num_examples = int(len(self.train_gen) * self.train_gen.batch_size)

        print(f"[Client {self.client_id}] Training done: loss={train_loss:.4f}, acc={train_acc:.4f}")
        return self.get_parameters(), num_examples, {"loss": train_loss, "accuracy": train_acc}

    def evaluate(self, parameters, config):
        print(f"[Client {self.client_id}] Evaluating...")
        self.model.set_weights(parameters)
        self.train_gen.reset()  # Reset generator to avoid placeholder errors
        steps = max(1, len(self.train_gen))
        loss, accuracy = self.model.evaluate(self.train_gen, steps=steps, verbose=0)

        # Convert metrics to Python floats
        loss = float(loss)
        accuracy = float(accuracy)
        num_examples = int(len(self.train_gen) * self.train_gen.batch_size)

        print(f"[Client {self.client_id}] Eval done: loss={loss:.4f}, acc={accuracy:.4f}")
        return loss, num_examples, {"accuracy": accuracy}

# ================================
# Main
# ================================
def main(server: str, client_id: str, local_epochs: int = 1):
    client_data_dir = Path("CLIENT_DATA") / "MRI"
    if not client_data_dir.exists():
        raise RuntimeError(f"No MRI data for client at {client_data_dir}")

    train_gen = get_train_generator(client_data_dir, batch_size=4)
    num_classes = len(train_gen.class_indices)

    # Load model if exists, else build new
    if IMAGE_MODEL_FILE.exists():
        try:
            print("Loading existing model...")
            model = load_model(str(IMAGE_MODEL_FILE))
        except Exception as e:
            print(f"⚠️ Could not load saved model ({e}), building new model")
            model = build_model(num_classes)
    else:
        print("Building new model...")
        model = build_model(num_classes)

    client = FlowerClient(model, train_gen, client_id, local_epochs=local_epochs)

    fl.client.start_numpy_client(
        server_address=server,
        client=client,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", type=str, required=True)
    parser.add_argument("--client_id", type=str, required=True)
    parser.add_argument("--local_epochs", type=int, default=1)
    args = parser.parse_args()
    main(args.server, args.client_id, args.local_epochs)
