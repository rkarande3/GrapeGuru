#preprocessing

import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertModel
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D, GRU, Dropout, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences



def normalize_text(text):
    # Normalize to NFKD Unicode normal form which separates base characters from diacritics
    text = text.str.normalize('NFKD')
    # Encode to ASCII bytes, ignore non-ASCII characters, then decode back to Unicode string
    text = text.str.encode('ascii', errors='ignore').str.decode('ascii')
    # Replace any non-word character (excluding space) with an empty string
    return text.str.replace('[^\w\s]', ' ', regex=True)


data = pd.read_csv("datasets/winemag-data_280k.csv")
df = pd.DataFrame(data)
#drop duplicates based on all columns
df = df.drop_duplicates()
#drop all rows where description is not populated
df = df.dropna(subset=['description'])
#drop all rows where points is not populated
df = df.dropna(subset=['points'])
#populate empty region 2 values with the value in region 1
# df['region_2'] = df['region_1'] # Error

#populate empty region1 region2 like: province -> region1 -> region2
df['region_1'].fillna(df['province'], inplace=True)
df['region_2'].fillna(df['region_1'], inplace=True)


#preprocess points
df['points'] = df['points'].astype(float)
df = df[df['points'] <= 100]
df = df[df['points'] >= 0]

#z score normalization for points
mean = df['points'].mean()
std = df['points'].std()
df['points'] = (df['points'] - mean) / std

average_price = df['price'].mean()

# Fill NaN values in the 'price' column with the average price
df['price'].fillna(average_price, inplace=True)

#preprocess all text fields
text_fields = ['country', 'description', 'designation', 'province', 'region_1', 'region_2', 'variety', 'winery']
df[text_fields] = df[text_fields].apply(lambda x: x.str.lower().astype(str))
df[text_fields] = df[text_fields].apply(lambda x: x.str.lower().astype(str)).apply(normalize_text)

#move country to right side
country_ci = df.columns.get_loc("country")
province_ci = df.columns.get_loc("province")

country_col = df.pop("country")
df.insert(province_ci - 1, "country", country_col)

df.to_csv('datasets/wine_updated_ds.csv', index=False)

df['description'][0]


cleaned_df = pd.read_csv('datasets/wine_updated_ds.csv')
df = cleaned_df.dropna(subset=['description', 'variety'])

#model

tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['description'])
total_words = len(tokenizer.word_index) + 1

# Convert text descriptions to sequences
sequences = tokenizer.texts_to_sequences(df['description'])
padded_sequences = pad_sequences(sequences)



# Encode the target variable
label_encoder = LabelEncoder()
df['variety'] = label_encoder.fit_transform(df['variety'])

'''

# Combine the labels from both training and pred_df datasets
all_labels = pd.concat([df['variety'], pred_df['variety']], axis=0)

# Encode the target variable in both training and pred_df datasets
label_encoder = LabelEncoder()
all_labels_encoded = label_encoder.fit_transform(all_labels)

# Update the encoding for both datasets
df['variety'] = all_labels_encoded[:len(df)]
pred_df['variety'] = all_labels_encoded[len(df):]

'''
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, df['variety'], test_size=0.30, random_state=42)

# Build the model
model = Sequential()
model.add(Embedding(input_dim=total_words, output_dim=100, input_length=padded_sequences.shape[1]))
model.add(SpatialDropout1D(0.2))
model.add(GRU(128, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(len(label_encoder.classes_), activation='softmax'))  # Change the number of units

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

X_test, X_pred, y_test, y_pred = train_test_split(X_test, y_test, test_size=0.33, random_state=42)

# Train the model
history = model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))

# Evaluate the model
# loss, accuracy = model.evaluate(X_test, y_test)
# print(f'Loss: {loss}, Accuracy: {accuracy}')

model.save('wine_recommender.h5')