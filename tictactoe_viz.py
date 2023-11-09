import matplotlib as plt
from tictactoe import TicTacToe
import torch
from alpha_zero import AlphaZero
from model import ResNet
import matplotlib.pyplot as plt

torch.manual_seed(0)

tictactoe = TicTacToe()

state = tictactoe.get_initial_state()
state = tictactoe.get_next_state(state, 2, -1)
state = tictactoe.get_next_state(state, 4, -1)
state = tictactoe.get_next_state(state, 6  , 1)
state = tictactoe.get_next_state(state, 8 , 1)
# state = tictactoe.get_next_state(state, 2 , 1)
# state = tictactoe.get_next_state(state, 7 , -1)

encoded_state = tictactoe.get_encoded_state(state)

tensor_state = torch.tensor(encoded_state).unsqueeze(0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = ResNet(tictactoe, 4, 64, device)
model.load_state_dict(torch.load('models/model_2.pt', map_location=device))
model.eval()

policy, value = model(tensor_state)
value = value.item()
policy = torch.softmax(policy, axis=1).squeeze(0).detach().cpu().numpy()

print(value)
print(state)
print(tensor_state)
print(policy)

plt.bar(range(tictactoe.action_size), policy )
plt.show()



