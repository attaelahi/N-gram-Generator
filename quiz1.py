import streamlit as st
import nltk
from string import punctuation

# Download NLTK data
nltk.download('punkt')

# Custom CSS styles
custom_css = """
<style>
    body {
        font-family: 'Arial', sans-serif;
        background-color: #f4f4f4;
        color: #333333;
    }

    .st-bd {
        max-width: 800px;
        margin: 0 auto;
    }

    .stTextInput {
        width: 100%;
    }

    .stRadio {
        padding: 10px;
        border: 1px solid #ddd;
        border-radius: 5px;
        margin-top: 10px;
    }

    .stButton button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        cursor: pointer;
        border-radius: 5px;
    }

    .stButton button:hover {
        background-color: #45a049;
    }

    .stSubheader {
        margin-top: 20px;
    }

    .stMarkdown {
        margin-top: 20px;
    }
</style>
"""

# Inject custom CSS
st.markdown(custom_css, unsafe_allow_html=True)

def preprocess_text(text):
    # Remove punctuation and make lowercase
    text = text.translate(str.maketrans('', '', punctuation))
    text = text.lower()

    # Tokenize the text
    tokens = nltk.word_tokenize(text)

    return tokens

@st.cache_data
def generate_ngrams(tokens, n):
    ngrams = []
    for i in range(len(tokens) - (n - 1)):
        ngrams.append(tokens[i:i+n])

    return ngrams

def main():
    st.title("N-gram Generator")

    # Get the user input text
    text_input = st.text_area("Enter your text here:")

    # Check if user has entered text
    if text_input:
        # Allow user to select n-grams
        n_gram_choice = st.radio("Select n-grams:", options=["Bigrams", "Trigrams", "Four-grams"])

        # Map user choice to n value
        n_values = {"Bigrams": 2, "Trigrams": 3, "Four-grams": 4}
        n = n_values[n_gram_choice]

        # Button to trigger n-gram generation
        if st.button("Generate N-grams"):
            # Preprocess the text
            tokens = preprocess_text(text_input)

            # Generate n-grams
            selected_ngrams = generate_ngrams(tokens, n)

            # Display the selected n-grams
            st.subheader(f"{n}-grams:")
            st.write(selected_ngrams)

if __name__ == "__main__":
    main()
