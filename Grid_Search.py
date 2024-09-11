from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
import os, json, re
import tensorflow as tf

# Carico il tokenizer e il modello pre-addestrato
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

train_path =  '../dataset/Train_Clear'
test_path = '../dataset/Test_Clear'

def concatenate_files(path, extension):
    files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(extension)]
    with open(os.path.join(path, 'concatenated.txt'), 'w') as outfile:
        for fname in files:
            with open(fname) as infile:
                outfile.write(infile.read())

# Concateno i file .sol nel dataset di train e di test
concatenate_files(train_path, "sol")
concatenate_files(test_path, "sol")

# Carico il dataset di train
train_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path=os.path.join(train_path, 'concatenated.txt'),
    block_size=128
)

# Carico il dataset di test
test_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path=os.path.join(test_path, 'concatenated.txt'),
    block_size=128
)

# Definisco il data collator, ovvero l'oggetto che si occupa di preparare i dati per l'addestramento
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

# Definisci i range dei parametri da testare
learning_rates = [0.5, 0.05, 0.005, 0.0005, 0.00005]
batch_sizes = [6, 5, 4, 3, 2, 1]
warmup_steps = [70, 50, 30, 10]
weights_decay = [0.5, 0.05, 0.005, 0.0005]

# Percorso dove salvare i risultati
results_dir = "./json_data"
os.makedirs(results_dir, exist_ok=True)

# Esegui il fine-tuning con diversi set di parametri
for lr in learning_rates:
    for bs in batch_sizes:
        for ws in warmup_steps:
            for wd in weights_decay:
                # Definisco gli argomenti di addestramento
                training_args = TrainingArguments(
                    output_dir="./model_test/results/",
                    overwrite_output_dir=True,       
                    num_train_epochs=0.001,              
                    per_device_train_batch_size=bs,    
                    eval_steps=500,                 # Ogni quanti step di addestramento valutare il modello          
                    save_steps=500,                 # Ogni quanti step di addestramento salvare il modello
                    save_total_limit=3,             # Numero massimo di modelli salvati
                    warmup_steps=ws,               # Numero di step di riscaldamento in cui la learning rate aumenta da 0 al valore adatto        
                    #prediction_loss_only=True,     # Calcolo solo la loss di predizione, meglio toglierla per metriche complete
                    logging_dir='./logs/',
                    logging_steps=150,              # Ogni quanti step di addestramento loggare i risultati
                    learning_rate=lr,            # Valore di learning rate
                    weight_decay=wd,              # Valore di decay dei pesi, utile per evitare l'overfitting
                )

                # Inizializzo il Trainer
                trainer = Trainer(
                    model=model,
                    args=training_args,
                    data_collator=data_collator,
                    train_dataset=train_dataset,
                    eval_dataset=test_dataset,
                )

                # Avvio l'addestramento
                trainer.train()

                def extract_train_loss(log_dir = './logs/'):
                    # Ottieni tutti i file nella cartella
                    files = os.listdir(log_dir)
                    # Verifica se c'è solo un file nella cartella
                    if len(files) == 1:
                        # Crea il percorso completo del file
                        file_dir = os.path.join(log_dir, files[0])
                        # Carica il file di log
                        data = tf.compat.v1.train.summary_iterator(file_dir)
                        
                        # Estrai i valori di train_loss
                        for event in data:
                            for value in event.summary.value:
                                if value.tag == 'train/train_loss':
                                    train_loss = value.simple_value

                        # Cancella il file dalla cartella logs
                        os.remove(file_dir)
                        return train_loss

                # Salva i risultati
                with open(f"{results_dir}/results.json", "r") as file:
                    data = json.load(file)
                
                data.append({
                    "learning_rate": lr,
                    "batch_size": bs,
                    "warmup_steps": ws,
                    "weight_decay": wd,
                    "train_loss": extract_train_loss()
                })

                with open(f"{results_dir}/results.json", "w") as file:
                    json.dump(data, file, indent=4)