import json
from collections import defaultdict
from nltk import bigrams, trigrams
from nltk.tokenize import word_tokenize

# Load the JSON data from your emails.json file
with open('emails.json', 'r', encoding='utf-8') as file:
    email_data = json.load(file)

# Extract the "Snippet" part from the email data
snippets = [email['Snippet'] for email in email_data]

# Tokenize the snippets into words
tokenized_snippets = [word_tokenize(snippet) for snippet in snippets]

# Create an inverted index
inverted_index = defaultdict(list)
for snippet in tokenized_snippets:
  for word in snippet:
    inverted_index[word].append(snippet)

# Define a function to run queries on the inverted index
def run_query(inverted_index, query):
  query_terms = query.split()
  results = []
  for query_term in query_terms:
    if query_term not in inverted_index:
      continue
    num_hits = len(inverted_index[query_term])
    for snippet in inverted_index[query_term]:
      results.append(snippet)
  return results

# Run a sample query
query = "Hi"
results = run_query(inverted_index, query)

# Print the results
print("Results for query:", query)
for snippet, num_hits in results:
  print(f"{snippet} ({num_hits} hits)")

# Create and train a bigram model
bigram_model = list(bigrams(tokenized_snippets))

# Create and train a trigram model
trigram_model = list(trigrams(tokenized_snippets))

import random
import itertools

def generate_text(model, n=2):

  # Start with a random n-gram from the model.
  start_gram = random.choice(model)

  # Generate text by adding one word at a time, using the previous n-1 words as the context.
  generated_text = []
  for i in range(n - 1, len(start_gram)):
    generated_text.append(start_gram[i])

  while True:
    # Get the next word in the n-gram model, given the previous n-1 words.
    next_words = [next_gram[n-1] for next_gram in model if next_gram[:n - 1] == generated_text[-n + 1:]]

    # If the list of next n-grams is empty, stop generating text.
    if not next_words:
      break

    # Randomly select a next word from the list of next n-grams.

    next_word = random.choice(next_words)

    # Add the next word to the generated text and update the context.
    generated_text.append(next_word)

    # If we hit a punctuation mark, stop generating text.
    if next_word in [".", "!", "?"]:
      break

  # Flatten the generated text list before joining it into a string.
  generated_text = list(itertools.chain.from_iterable(generated_text))

  # Convert the generated text list to a string.
  generated_text = " ".join(generated_text)

  return generated_text

# Generate a sentence using the bigram model.
print(generate_text(bigram_model, n=2))