from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
import os

# Carico il tokenizer e il modello pre-addestrato
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

train_path =  '../dataset/Train_Clear'
test_path = '../dataset/Test_Clear2'

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

# Definisco gli argomenti di addestramento
training_args = TrainingArguments(
    output_dir='./results',          
    overwrite_output_dir=True,       
    num_train_epochs=0.001,              
    per_device_train_batch_size=2,   
    per_device_eval_batch_size=2,    
    #eval_steps=500,                 # Ogni quanti step di addestramento valutare il modello          
    save_steps=500,                 # Ogni quanti step di addestramento salvare il modello
    warmup_steps=200,               # Numero di step di riscaldamento in cui la learning rate aumenta da 0 al valore adatto        
    #prediction_loss_only=True,     # Calcolo solo la loss di predizione, meglio toglierla per metriche complete
    logging_dir='./logs',
    logging_steps=50,              # Ogni quanti step di addestramento loggare i risultati
    learning_rate=5e-2,            # Valore di learning rate
    weight_decay=0.01,              # Valore di decay dei pesi, utile per evitare l'overfitting
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

# Salvo il modello addestrato
#model.save_pretrained('./saved_model')

# Salvo il tokenizer addestrato
#tokenizer.save_pretrained('./saved_tokenizer')

