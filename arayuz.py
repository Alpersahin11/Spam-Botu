import joblib
import tkinter as tk
from tkinter import scrolledtext, messagebox

# Modeli ve vektörizerı yüklüyoruz
model = joblib.load('text_classifier_model.pkl')
vectorizer = joblib.load('text_vectorizer.pkl')

# Metni sınıflandıran fonksiyon
def classify_text():
    message = text_input.get("1.0", tk.END).strip()
    if message:
        message_counts = vectorizer.transform([message])
        prediction = model.predict(message_counts)
        if prediction[0] == "spam":
            result = "Spam"
        else:
            result= "Spam Değil"
        messagebox.showinfo("Tahmin Sonucu", f"Mesaj: {message}\nSınıflandırma: {result}")
    else:
        messagebox.showwarning("Giriş Hatası", "Sınıflandırılacak bir mesaj girin.")

# GUI'yi oluşturuyoruz
root = tk.Tk()
root.title("Metin Sınıflandırıcı")

# Pencere boyutunu sabitlemek için sabit genişlik ve yükseklik ayarlıyoruz
root.geometry("500x400")
root.resizable(False, False)

# Etiket ekliyoruz
tk.Label(root, text="Mesajınızı girin:").pack(pady=10)

# Metin giriş kutusu ekliyoruz
text_input = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=60, height=15)
text_input.pack(padx=10, pady=10)


# Sınıflandırma düğmesi ekliyoruz
classify_button = tk.Button(root, text="Sınıflandır", command=classify_text)
classify_button.pack(pady=10)

# GUI'yi çalıştırıyoruz
root.mainloop()
