import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
from customtkinter import *
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r', encoding="utf-8") as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Sam"


# Função para responder à entrada do usuário
def respond():
    user_input = user_input_box.get("1.0", "end-1c")
    user_input_box.delete("1.0", "end")  # Limpar a caixa de entrada
    user_input = tokenize(user_input)
    X = bag_of_words(user_input, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                response = random.choice(intent['responses'])
    else:
        response = "Não entendi..."

    user_input_str = ' '.join(user_input)  # Converte a lista de palavras em uma única string
    chat_log.configure(state="normal")
    chat_log.insert("end", f"Você: {user_input_str}\n")
    chat_log.insert("end", f"{bot_name}: {response}\n\n")
    chat_log.configure(state="disabled")
    chat_log.see("end")  # Rolagem automática para o final


# Criar a janela principal
root = CTk()
root.title("Chatbot pyTorch")
root.geometry("500x400")

label = CTkLabel(root, text="Pergunte algo...", fg_color="transparent", text_color="#FFCC70", font=("Arial", 22))
label.pack()
# Caixa de texto para entrada do usuário
user_input_box = CTkTextbox(root, height=5, width=400, text_color="#FFCC70")
user_input_box.pack(pady=10)

# Botão para enviar a mensagem
send_button = CTkButton(root, text="Enviar", command=respond, fg_color="transparent", border_color="#FFCC70",
                        border_width=2, text_color="#FFCC70")
send_button.pack()

# Caixa de texto para exibir a conversa
chat_log = CTkTextbox(root, height=300, width=400, text_color="#FFCC70")
chat_log.pack(padx=10, pady=10)
chat_log.configure(state="disabled")  # Apenas para leitura

root.mainloop()
