from tensorboard.backend.event_processing import event_accumulator
import numpy as np
import matplotlib.pyplot as plt

# List of log directories
log_path = '/Olympus-ws/Takeoff/runs/LongJump/summaries/events.out.tfevents.1702396208.ntnu08987'
event_acc = event_accumulator.EventAccumulator(log_path)
event_acc.Reload()

# Specify the tags for the desired rewards/metrics

data_to_plot = {
    'rewards/iter' : ['Total game reward','Reward'],
    'len_dev_all/iter': ['Len deviation','deviation'],
    'est_jump_length_y/iter': ['Exit angle','angle'],
    'spin_at_takeoff/iter': ['Take off angular velocity','velocity'],
    'losses/a_loss' : ['Actor loss','loss'],
    'losses/bounds_loss' : ['Bounds loss','loss'],
    'losses/c_loss' : ['Critic Loss','loss'],
    'losses/entropy' : ['Entropy','entropy'],

}
import tikzplotlib
import os
steps = np.arange(1,2000,1)
for tag in data_to_plot.keys():
    # Create a new plot for each tag
    plt.figure(figsize=(6, 2.5))

    reward_events = event_acc.Scalars(tag)
    rewards = [event.value for event in reward_events]
    if len(rewards) > len(steps):
        rewards = rewards[:len(steps)]
    # Plot the metrics
    plt.plot(steps, rewards)

    # Set title and labels
    plt.title(data_to_plot[tag][0], fontsize=10)
    # plt.xlabel('Epochs', fontsize=10)
    plt.ylabel(data_to_plot[tag][1], fontsize=10)


    #make dir
    dir = f'{tag.replace("/", "_")}'
    path = os.path.join(dir,"tikz.tex")
    os.makedirs(dir,exist_ok=True)
    tikzplotlib.save(path,externalize_tables=True)
    # Save the plot as a PDF and PGF. NB: DO NOT USE EPS
    #plt.savefig(f'{tag.replace("/", "_")}_plot.pdf', format='pdf')