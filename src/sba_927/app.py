import nltk
from nltk import pos_tag, ne_chunk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import string


# Download NLTK resources if not already present
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

# Function for Tokenization and Preprocessing


def tokenize_and_preprocess(text):
    # Tokenize the text
    tokens = word_tokenize(text)

    # Remove stop words and punctuation
    stop_words = set(stopwords.words('english'))
    punctuation = set(string.punctuation)

    filtered_tokens = [
        word.lower() for word in tokens
        if (word.isalpha() and word.lower() not in stop_words and word not in punctuation)
    ]
    return filtered_tokens


# Read the external file
# Read the external file from the project root
file_path = '../../SBA927.txt'
with open(file_path, 'r', encoding='utf-8') as file:
    # Read the entire content of the file
    dataset = file.read().splitlines()

# Tokenization and Preprocessing for the entire dataset
processed_dataset = [tokenize_and_preprocess(text) for text in dataset]

# Display the results
# for i, text_tokens in enumerate(processed_dataset):
#     print(f"Original Text {i + 1}: {dataset[i]}")
#     print(f"Processed Tokens {i + 1}: {text_tokens}")
#     print("\n")


for i, text_tokens in enumerate(processed_dataset):
    print("-" * 60)
    print(f"[{i + 1}] Original Text: {dataset[i]}")
    print(f"[{i + 1}] Processed Tokens: {text_tokens}")
print("-" * 60)


# Define a function for stemming and lemmatization


def stem_and_lemmatize(tokens):
    # Initialize stemming and lemmatization tools
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()

    # Apply stemming
    stemmed_tokens = [stemmer.stem(token) for token in tokens]

    # Apply lemmatization
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]

    return stemmed_tokens, lemmatized_tokens


# Assume 'processed_dataset' contains the preprocessed text data
# Apply stemming and lemmatization to the entire dataset
stemmed_and_lemmatized_dataset = [
    stem_and_lemmatize(tokens) for tokens in processed_dataset]

# Display the results
for i, (stemmed_tokens, lemmatized_tokens) in enumerate(stemmed_and_lemmatized_dataset):
    print("-" * 60)
    print(f"Original Tokens {i + 1}: {processed_dataset[i]}")
    print(f"Stemmed Tokens {i + 1}: {stemmed_tokens}")
    print(f"Lemmatized Tokens {i + 1}: {lemmatized_tokens}")
print("-" * 60)


# Function to tokenize text into sentences


def tokenize_into_sentences(text):
    sentences = sent_tokenize(text)
    return sentences

# Function to perform part-of-speech tagging on sentences


def part_of_speech_tagging(sentences):
    pos_tagged_sentences = [pos_tag(word_tokenize(sentence))
                            for sentence in sentences]
    return pos_tagged_sentences

# Function to extract named entities from POS-tagged sentences


def extract_named_entities(pos_tagged_sentences):
    named_entities = []
    for sentence in pos_tagged_sentences:
        tree = ne_chunk(sentence)
        for subtree in tree:
            if hasattr(subtree, 'label'):  # Check if subtree is a named entity
                entity = " ".join([word for word, tag in subtree.leaves()])
                named_entities.append((entity, subtree.label()))
    return named_entities

# Define a function for Named Entity Recognition


def named_entity_recognition(text):
    # Tokenize and process the text
    sentences = tokenize_into_sentences(text)
    pos_tags = part_of_speech_tagging(sentences)
    named_entities = extract_named_entities(pos_tags)
    return named_entities


# Apply Named Entity Recognition to the entire dataset
ner_results_dataset = [named_entity_recognition(text) for text in dataset]

# Display the results with clean formatting
for i, ner_results in enumerate(ner_results_dataset):
    print("-" * 60)
    print(f"Original Text {i + 1}: {dataset[i]}")
    print(f"Named Entities {i + 1}: {ner_results}")
print("-" * 60)


# Define a function for POS tagging
def pos_tagging(text):
    # Tokenize and process the text
    sentences = sent_tokenize(text)
    pos_tags = []

    # Process each sentence
    for sentence in sentences:
        words = word_tokenize(sentence)
        pos_tags.extend(pos_tag(words))

    return pos_tags


# Assume 'dataset' contains the original text data
# Apply POS tagging to the entire dataset
pos_tags_dataset = [pos_tagging(text) for text in dataset]

# Display the results with clean formatting
for i, pos_tags in enumerate(pos_tags_dataset):
    print("-" * 60)
    print(f"Original Text {i + 1}: {dataset[i]}")
    print(f"POS Tags {i + 1}: {pos_tags}")
print("-" * 60)
