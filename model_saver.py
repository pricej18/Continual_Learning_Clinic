# model_saver.py

import os
import torch


def save_model(model, save_path, task_id):
    """
    Save the model state dictionary and other relevant information.

    Args:
        model (torch.nn.Module): The model to be saved.
        save_path (str): The directory where the model will be saved.
        task_id (int): The current task ID.
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path) # Create the "Save_path" directory if it does not exist

    # saves the model and assigns attributes to it (task_id, model_state_dict)
    save_file = os.path.join(save_path, f'model_task_{task_id}.pth.tar') # The path where the model will be saved
    torch.save({
        'task_id': task_id,   # The task ID
        'model_state_dict': model.state_dict(), # The model state dictionary
    }, save_file)
    print(f'Model for task {task_id} saved at {save_file}') # Print the path where the model is saved
