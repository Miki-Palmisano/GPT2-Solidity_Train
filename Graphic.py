import json
import matplotlib.pyplot as plt

# Carica il file JSON
with open('./model_1/trainer_state_1.json', 'r') as f:
    data = json.load(f)

# Estrai i dati di interesse
epochs = [log['epoch'] for log in data['log_history']]
losses = [log['loss'] for log in data['log_history']]
learning_rates = [log['learning_rate'] for log in data['log_history']]
grad_norms = [log['grad_norm'] for log in data['log_history']]

# Crea un grafico per la perdita
plt.figure(figsize=(10, 5))
plt.plot(epochs, losses, label='Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Time')
plt.legend()
plt.show()

# Crea un grafico per il tasso di apprendimento
plt.figure(figsize=(10, 5))
plt.plot(epochs, learning_rates, label='Learning Rate')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('Learning Rate over Time')
plt.legend()
plt.show()

# Crea un grafico per la norma del gradiente
plt.figure(figsize=(10, 5))
plt.plot(epochs, grad_norms, label='Gradient Norm')
plt.xlabel('Epoch')
plt.ylabel('Gradient Norm')
plt.title('Gradient Norm over Time')
plt.legend()
plt.show()