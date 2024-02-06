import gym
import numpy as np
import tensorflow as tf

# Definimos as configurações do algoritmo
GAMMA = 0.99
MAX_EPISODES = 1000
MAX_STEPS = 1000

# Definimos a rede neural
class QNetwork(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(2)
        ])

    def call(self, state):
        return self.model(state)

# Inicializamos a rede neural
q_network = QNetwork()

# Inicializamos o ambiente
env = gym.make('CartPole-v1')

# Iniciamos o treinamento
for episode in range(MAX_EPISODES):
    state = env.reset()
    for step in range(MAX_STEPS):
        # Realizamos a ação
        action = q_network(np.expand_dims(state, axis=0)).argmax()

        # Prosseguimos com o ambiente
        next_state, reward, done, _ = env.step(action)

        # Atualizamos a rede neural
        target = reward + GAMMA * q_network(np.array([next_state])).max()
        q_network.train_on_batch([state], [target])

        # Atualizamos o estado
        state = next_state

        # Verificamos se o jogo terminou
        if done:
            break

# Imprimimos o resultado
print("O agente conseguiu equilibrar o carrinho por {} episódios!".format(episode))
