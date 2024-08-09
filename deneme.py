import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import joblib


veri = pd.read_csv("sonuc.csv")

veri.drop("i",axis=1,inplace=True)
veri.drop("index",axis=1,inplace=True)



# Etiketleri ve mesajları  ayırıyoruz
X = veri['text']
y = veri['sonuc']

# Veriyi eğitim ve test seti olarak ayırıyoruz
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Adım 3: Verilerin özellik çıkartılması
vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)

# Adım 4: Modelin Eğitilmesi
model = MultinomialNB()
model.fit(X_train_counts, y_train)


# Adım 5: Modelin Test Edilmesi
y_pred = model.predict(X_test_counts)
accuracy = accuracy_score(y_test, y_pred)
print(f"Doğruluk Oranı: {accuracy * 100:.2f}%")
print(classification_report(y_test, y_pred))



# Modeli ve vektörizer'ı kaydediyoruz
joblib.dump(model, 'text_classifier_model.pkl')
joblib.dump(vectorizer, 'text_vectorizer.pkl')
