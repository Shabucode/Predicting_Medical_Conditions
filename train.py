import pandas as pd
import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import string

#load the dataset
df = pd.read_csv("data/trials.csv")

#Data Preprocessing
def data_preprocessing(df):
    
    #convert to lowercase
    df['description'] = df['description'].str.lower()
    #remove punctuation
    df['description'] = df['description'].str.translate(str.maketrans('','',string.punctuation)) 

    #REmove stopwords
    df['description'] = df['description'].apply(lambda x: ' '.join([word for word in x.split() if word not in ENGLISH_STOP_WORDS])) #remove punctuation

    #Tokenizationpyth
    import nltk
    from nltk.tokenize import word_tokenize
    df['description'] =df['description'].apply(word_tokenize) #List of words

    #Lemmatization
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    df['description'] = df['description'].apply(lambda x: ' '. join(lemmatizer.lemmatize(word) for word in x))

    #Label Encoding
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['label'])
    
    return df

df = data_preprocessing(df)  #call data preprocessing function

# Split data into train and test and validation sets
train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
#split temp_df into test_df and val_df
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42)


#Vectorization of train, test and validation sets
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(train_df['description'])
X_val = vectorizer.transform(val_df['description'])
X_test = vectorizer.transform(test_df['description'])

y_train = train_df['label']
y_val = val_df['label']
y_test = test_df['label']

#Training the model
model = LogisticRegression()
model.fit(X_train, y_train)

#Predicting the labels for the validation set
y_val_pred = model.predict(X_val)

#Predicting the labels for the test set
y_test_pred = model.predict(X_test)


# Save model
joblib.dump(model, "lr_model.pkl")
print("Model trained and saved successfully!")
