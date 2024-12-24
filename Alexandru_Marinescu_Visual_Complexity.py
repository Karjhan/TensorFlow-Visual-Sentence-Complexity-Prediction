import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import StackingRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import TextVectorization, Embedding, LSTM, Dense, Dropout, Bidirectional, GlobalAveragePooling1D, BatchNormalization, LayerNormalization, Conv1D, MaxPooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import load_model
from tensorflow.keras import Sequential

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

###

training_data = pd.read_csv("data/train.csv")
validation_data = pd.read_csv("data/val.csv")
testing_data = pd.read_csv("data/test.csv")

def plot_scores_histogram(scores):
    plt.figure(figsize=(8, 6))
    plt.hist(scores, bins=int((scores.max() - scores.min()) / 0.25), edgecolor='black', color='skyblue')
    plt.title("Histogram of Train Data Scores", fontsize=16)
    plt.xlabel("Score", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

print(f"Training data length is {training_data.size} with scores having: mean {training_data["score"].mean()} and standard deviation {training_data["score"].std()}")
print(training_data.head())
plot_scores_histogram(training_data["score"])

print(f"Validation data length is {validation_data.size} with scores having: mean {validation_data["score"].mean()} and standard deviation {validation_data["score"].std()}")
print(validation_data.head())
plot_scores_histogram(validation_data["score"])

print(testing_data.head())

###

neural_network_score, random_forest_score, ridge_score, lasso_score, support_vector_score, combined_model_score = [0, 0, 0, 0, 0, 0]

training_texts, training_scores = training_data["text"], training_data["score"]
validation_texts, validation_scores = validation_data["text"], validation_data["score"]

testing_texts = testing_data["text"]
testing_ids = testing_data["id"]

lemmatizer = WordNetLemmatizer()
english_stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalnum() and word not in english_stop_words]
    return ' '.join(tokens)

training_texts_processed = [preprocess_text(text) for text in training_texts]
validation_texts_processed = [preprocess_text(text) for text in validation_texts]
testing_texts_processed = [preprocess_text(text) for text in testing_texts]

print(training_texts[0])
print(training_texts_processed[0])

###

nonNN_vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
nonNN_training_features = nonNN_vectorizer.fit_transform(training_texts_processed).toarray()
nonNN_validation_features = nonNN_vectorizer.transform(validation_texts_processed).toarray()
nonNN_testing_features = nonNN_vectorizer.transform(testing_texts_processed).toarray()

print(training_texts[0])
print(training_texts_processed[0])
print(nonNN_testing_features[0])

max_tokens = 5000
sequence_length = 100

NN_vectorizer = TextVectorization(
    max_tokens=max_tokens,
    output_sequence_length=sequence_length,
    standardize="lower_and_strip_punctuation",
    split="whitespace"
)
NN_vectorizer.adapt(training_texts)

print(training_texts[0])
print(training_texts_processed[0])
print(NN_vectorizer(training_texts[0]))

###

embedding_dimmension = 128
tf.random.set_seed(42)

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=100000,
    decay_rate=0.96,
    staircase=True
)

model = Sequential([
    NN_vectorizer,
    Embedding(input_dim=max_tokens, output_dim=embedding_dimmension, mask_zero=True),
    Bidirectional(LSTM(128, return_sequences=True, dropout=0.5, recurrent_regularizer=l2(0.01), kernel_regularizer=l2(0.01))),
    GlobalAveragePooling1D(),
    Dense(64, activation="relu", kernel_regularizer=l2(0.03)),
    Dropout(0.5),
    Dense(1)
])

model.compile(
    loss="mse",
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    metrics=["mae"]
)

history = model.fit(
    training_texts,
    training_scores,
    validation_data=(validation_texts, validation_scores),
    epochs=18,
    batch_size=32
)

epochs = range(1, len(history.history["loss"]) + 1)
loss_evolution = history.history["loss"]
val_loss_evolution = history.history["val_loss"]
mae_evolution = history.history["mae"]

plt.figure(figsize=(12, 8))

plt.plot(epochs, loss_evolution, label="Loss", marker="o")
plt.plot(epochs, val_loss_evolution, label="Validation Loss", marker="o")

plt.plot(epochs, mae_evolution, label="Mean Absolute Error (MAE)", marker="o")

plt.title("Neural Network Metrics Evaluation", fontsize=16)
plt.xlabel("Epochs", fontsize=14)
plt.ylabel("Value", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)

plt.show()

nn_predictions = model.predict(validation_texts).flatten()
neural_network_score = spearmanr(validation_scores, nn_predictions).correlation
print(f"Neural Network Spearman Score: {neural_network_score}")

print(model.summary())
###

######
random_forest_parameters = {
    'n_estimators': [10, 50, 100, 150, 200, 250]
}
random_forest_grid_search = GridSearchCV(RandomForestRegressor(), random_forest_parameters, cv=5, scoring='neg_mean_squared_error')
random_forest_grid_search.fit(nonNN_training_features, training_scores)
print(random_forest_grid_search.best_params_)

random_forest_model = RandomForestRegressor(n_estimators=100, random_state=42)
print(random_forest_model.fit(nonNN_training_features, training_scores))

######
ridge_parameters = {
    'alpha': [0.001, 0.1, 1.0, 10, 20, 30, 40, 50, 60, 70, 100]
}
ridge_grid_search = GridSearchCV(Ridge(), ridge_parameters, cv=5, scoring='neg_mean_squared_error')
ridge_grid_search.fit(nonNN_training_features, training_scores)
print(ridge_grid_search.best_params_)

ridge_model = Ridge(alpha=6.0, solver='lsqr')
print(ridge_model.fit(nonNN_training_features, training_scores))

######
lasso_parameters = {
    'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10, 20, 30, 40, 50, 60, 70, 100]
}
lasso_grid_search = GridSearchCV(Lasso(max_iter=10000), lasso_parameters, cv=5, scoring='neg_mean_squared_error')
lasso_grid_search.fit(nonNN_training_features, training_scores)
print(lasso_grid_search.best_params_)

lasso_model = Lasso(alpha=0.0005, max_iter=10000, random_state=42)
print(lasso_model.fit(nonNN_training_features, training_scores))

######
svr_param_grid = {
    "C": [0.01, 0.1, 1],
    "epsilon": [0.01, 0.1, 1]
}
svr_grid_search = GridSearchCV(SVR(), svr_param_grid, scoring="neg_mean_squared_error", cv=3)
svr_grid_search.fit(nonNN_training_features, training_scores)
print(svr_grid_search.best_params_)

svr_model = SVR(kernel="linear", C=0.1, epsilon=0.1)
svr_model.fit(nonNN_training_features, training_scores)

###

######
random_forest_score_predictions = random_forest_model.predict(nonNN_validation_features)
print(f"RANDOM FOREST - Validation data length is {random_forest_score_predictions.size} with scores having: mean {random_forest_score_predictions.mean()} and standard deviation {random_forest_score_predictions.std()}")
plot_scores_histogram(random_forest_score_predictions)

######
ridge_score_predictions = ridge_model.predict(nonNN_validation_features)
print(f"RIDGE - Validation data length is {ridge_score_predictions.size} with scores having: mean {ridge_score_predictions.mean()} and standard deviation {ridge_score_predictions.std()}")
plot_scores_histogram(ridge_score_predictions)

######
lasso_score_predictions = lasso_model.predict(nonNN_validation_features)
print(f"LASSO - Validation data length is {lasso_score_predictions.size} with scores having: mean {lasso_score_predictions.mean()} and standard deviation {lasso_score_predictions.std()}")
plot_scores_histogram(lasso_score_predictions)

######
svr_score_predictions = svr_model.predict(nonNN_validation_features)
print(f"SVR - Validation data length is {svr_score_predictions.size} with scores having: mean {svr_score_predictions.mean()} and standard deviation {svr_score_predictions.std()}")
plot_scores_histogram(svr_score_predictions)

######
random_forest_score = spearmanr(validation_scores, random_forest_score_predictions).correlation
ridge_score = spearmanr(validation_scores, ridge_score_predictions).correlation
lasso_score = spearmanr(validation_scores, lasso_score_predictions).correlation
support_vector_score = spearmanr(validation_scores, svr_score_predictions).correlation

print(f"Random Forest: {random_forest_score}, Ridge: {ridge_score}, Lasso: {lasso_score}, Support Vector: {support_vector_score}")

######
total_score = random_forest_score + ridge_score + lasso_score + support_vector_score
random_forest_weight = random_forest_score / total_score
ridge_weight = ridge_score / total_score
lasso_weight = lasso_score / total_score
support_vector_weight = support_vector_score / total_score

print(f"Random Forest Weight: {random_forest_weight}, Ridge Weight: {ridge_weight}, Lasso Weight: {lasso_weight}, Support Vector Weight: {support_vector_weight}")

# Weighted average of predictions
final_predictions = (random_forest_weight * random_forest_score_predictions +
                     ridge_weight * ridge_score_predictions +
                     lasso_weight * lasso_score_predictions +
                     support_vector_weight * svr_score_predictions)

final_score = spearmanr(validation_scores, final_predictions).correlation
print(f"Combined Model Spearman Score: {final_score}")

###

######
testing_score_predictions = ridge_model.predict(nonNN_testing_features)
print(f"BEST - Validation data length is {testing_score_predictions.size} with scores having: mean {testing_score_predictions.mean()} and standard deviation {testing_score_predictions.std()}")
plot_scores_histogram(testing_score_predictions)

output_df = pd.DataFrame({
    "id": np.array(testing_ids).flatten(),
    "score": testing_score_predictions.flatten()
})

output_df.to_csv("saves/submission_ridge_clear.csv", index=False)

######
random_forest_score_test_predictions = random_forest_model.predict(nonNN_testing_features)
ridge_score_test_predictions = ridge_model.predict(nonNN_testing_features)
lasso_score_test_predictions = lasso_model.predict(nonNN_testing_features)
svr_score_test_predictions = svr_model.predict(nonNN_testing_features)

final_test_predictions = (random_forest_weight * random_forest_score_test_predictions +
                     ridge_weight * ridge_score_test_predictions +
                     lasso_weight * lasso_score_test_predictions +
                     support_vector_weight * svr_score_test_predictions)

print(f"BEST - Validation data length is {final_test_predictions.size} with scores having: mean {final_test_predictions.mean()} and standard deviation {final_test_predictions.std()}")
plot_scores_histogram(testing_score_predictions)

output_df = pd.DataFrame({
    "id": np.array(testing_ids).flatten(),
    "score": final_test_predictions.flatten()
})

output_df.to_csv("saves/submission_combined_clear.csv", index=False)