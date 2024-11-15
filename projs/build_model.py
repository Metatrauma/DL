import tensorflow as tf
from tensorflow.keras import layers, models
import keras_tuner as kt
import numpy as np



def build_model(hp):
    model = models.Sequential()
    model.add(layers.Dense(
        units=hp.Int('units', min_value=32, max_value=256, step=32),
        activation='relu',
        input_shape=(10,)
    ))
    model.add(layers.Dense(
        units=hp.Int('hidden_units', min_value=16, max_value=128, step=16),
        activation='relu'
    ))
    model.add(layers.Dense(1))
    return model

def train_model(hp):
    model = build_model(hp)
    hp_learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')
    optimizer = tf.keras.optimizers.Adam(learning_rate=hp_learning_rate)

    model.compile(optimizer=optimizer, loss='mse')
    return model

def create_data():
    X = np.random.randn(1000, 10)
    y = np.random.randn(1000, 1)
    return X, y


tuner = kt.BayesianOptimization(
    train_model,
    objective='val_loss',
    max_trials=5,
    directory='tuning_dir',
    project_name='bayesian_optimization_demo'
)

# Create data and split it
X, y = create_data()
X_train, X_val, y_train, y_val = X[:800], X[800:], y[:800], y[800:]

# Run the search
tuner.search(X_train, y_train, epochs=10, validation_data=(X_val, y_val), batch_size=32)

#to retrieve the best model
best_model = tuner.get_best_models(num_models=1)[0]
best_hp = tuner.get_best_hyperparameters(1)[0]

print(f"Best number of units: {best_hp.get('units')}")
print(f"Best hidden units: {best_hp.get('hidden_units')}")
print(f"Best learning rate: {best_hp.get('learning_rate')}")