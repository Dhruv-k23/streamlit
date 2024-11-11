import streamlit as st
from transformers import MarianMTModel, MarianTokenizer

# Load the model and tokenizer
model_name = 'Helsinki-NLP/opus-mt-hi-en'
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# Define the translation function
def translate_text(input_text):
    if input_text:
        # Tokenize the input and generate translation
        inputs = tokenizer(input_text, return_tensors="pt", padding=True)
        translated = model.generate(**inputs)
        translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
        return translated_text
    return "No text provided for translation."

# Streamlit app UI
st.title("Hindi to English Translation")
st.write("Enter text in Hindi and click the 'Translate' button to see the English translation.")

# Text input from the user
input_text = st.text_area("Enter text in Hindi:")

# Translate button
if st.button("Translate"):
    translated_text = translate_text(input_text)
    st.write("Translated Text:", translated_text)
