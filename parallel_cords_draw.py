import json
import os
import plotly.express as px

# Percorso della cartella contenente i file JSON
file_path = './results.json'

# Crea una lista vuota per i dati
data_list = []

with open(file_path, 'r') as json_file:
    data = json.load(json_file)
    
    for element in data:
        data_list.append(element)

# Crea il grafico a coordinate parallele
dimensions = [
    dict(range=[0.00001, 0.001], label='Learning Rate', values=[data['learning_rate'] for data in data_list], tickvals=[0.001, 0.0001, 0.00001]),
    dict(range=[1, 3], label='Batch Size', values=[data['batch_size'] for data in data_list], tickvals=[1, 2, 3]),
    dict(range=[200, 700], label='Warmup Steps', values=[data['warmup_steps'] for data in data_list], tickvals=[200, 500, 700]),
    dict(range=[0.0001, 0.01], label='Weight Decay', values=[data['weight_decay'] for data in data_list], tickvals=[0.0001, 0.001, 0.01]),
    dict(range=[0, 1], label='Train Loss', values=[data['train_loss'] for data in data_list]),
]

fig = px.parallel_coordinates(
    data_list,
    color='train_loss',
    dimensions={'train_loss', 'learning_rate', 'batch_size', 'warmup_steps', 'weight_decay'},
    color_continuous_scale=px.colors.diverging.Tealrose,
    title='Hyperparameter Tuning Results',
    color_continuous_midpoint=2,
)

# Mostra il grafico
fig.write_image('./plot.png', scale=4)