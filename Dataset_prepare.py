import shutil, os, re
from transformers import GPT2Tokenizer
from tqdm import tqdm

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

def file_clear(percorso_file):
    with open(percorso_file, 'r') as file:
        contenuto = file.read()
        contenuto = re.sub(r'//.*', '\n', contenuto)  # Sostituisci i commenti su una singola linea con una riga vuota
        contenuto = re.sub(r'/\*.*?\*/', '\n', contenuto, flags=re.DOTALL)  # Sostituisci i commenti su più linee con una riga vuota
        contenuto = re.sub(r'\n\s*\n', '\n', contenuto)  # Rimuovi le righe vuote in eccesso
        contenuto = re.sub(r'0x\w+', 'a_address', contenuto)  # Sostituisci le stringhe che iniziano per 0x con "a_address"

    with open(percorso_file, 'w') as file:
        file.write(contenuto)

# Lista dei file .sol in una specifica cartella
percorso_cartella = '../Dataset/Train'
lista_file = [os.path.join(percorso_cartella, nome_file) for nome_file in os.listdir(percorso_cartella) if nome_file.endswith('.sol')]

# Specifico la cartella di output
cartella_output = '../Dataset/Train_Clear'

# Creo la cartella di output se non esiste
if not os.path.exists(cartella_output):
    os.makedirs(cartella_output)

# Rimuovo i commenti da ogni file e salva nella cartella di output
for percorso_file in tqdm(lista_file, desc="Processing files"):
    # Ottengo il nome del file dal percorso originale
    nome_file = os.path.basename(percorso_file)
    # Creo il nuovo percorso del file nella cartella di output
    nuovo_percorso_file = os.path.join(cartella_output, nome_file)
    # Copia il file nella cartella di output
    shutil.copyfile(percorso_file, nuovo_percorso_file)
    # Rimuovi i commenti dal file nella cartella di output
    file_clear(nuovo_percorso_file)
    # Ottengo la dimensione del file e calcolo la lunghezza in token dopo aver eliminato i commenti
    file_length = open(nuovo_percorso_file, 'r').read()
    token_length = len(tokenizer.tokenize(file_length))
    if token_length == 0 or token_length > 1024:
        os.remove(nuovo_percorso_file)

# Lista dei file .sol in una specifica cartella
def concatenate_files(path, extension):
    files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(extension)]
    with open(os.path.join(path, 'concatenated.txt'), 'w') as outfile:
        for fname in files:
            with open(fname) as infile:
                outfile.write(infile.read())

train_path =  '../dataset/Train_Clear'
test_path = '../dataset/Test_Clear'

# Concateno i file .sol nel dataset di train e di test
concatenate_files(train_path, "sol")
concatenate_files(test_path, "sol")
