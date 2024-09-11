from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Carico il tokenizer e il modello addestrato
tokenizer = GPT2Tokenizer.from_pretrained('./Model_1/saved_tokenizer_1')
model = GPT2LMHeadModel.from_pretrained('./Model_1/saved_model_1')

# Codice di esempio per la generazione
prompt = 'pragma solidity ^0.4.6; \n contract myBank {'  # Inizio con un prompt di esempio per il modello
inputs = tokenizer.encode(prompt, return_tensors='pt')

# Genera la sequenza di codice
generated_sequences = model.generate(
    inputs, 
    max_length=200, 
    num_return_sequences=5,
    no_repeat_ngram_size=2, # Evita la ripetizione di n-grammi, in questo caso di 2 parole
    #early_stopping=True, # Ferma la generazione quando il modello ha finito di generare il testo
    do_sample=True,  # Abilito il campionamento, ovvero la generazione casuale
    top_k=40,  # Numero di token da considerare per il campionamento
    temperature=0.6  # Controllo la casualità del campionamento
)

# Decode the generated sequences into Solidity code
decoded_sequences = []
for i, sequence in enumerate(generated_sequences):
    decoded_sequence = tokenizer.decode(sequence, skip_special_tokens=False)
    decoded_sequences.append(decoded_sequence)

# Print the generated code sequences
for i, sequence in enumerate(decoded_sequences):
    print(f"\n\nGenerated code {i+1}:\n\n")
    print(sequence)