import QLearningModule as qlm
from IPython.display import clear_output


qlearner = qlm.QLearn()

# Agent training
for i in range(10001):
    q_table = qlearner.q_learning()
    if i % 100 == 0:
        clear_output(wait=True)
        print(f"Training Episode: {i}")
print("Training finished. \n")

# Agent evaluation
total_epochs, total_penalties = 0, 0
episodes = 100

for _ in range(episodes):
    if _ % 10 == 0:
        clear_output(wait=True)
        print(f"Evaluation Episode: {_}")

    epochs, penalties = qlearner.q_evaluation()

    total_epochs += epochs
    total_penalties += penalties

print("Evaluation finished.\n")
print(f"Total epochs: {total_epochs}\nTotal penalties: {total_penalties}")




