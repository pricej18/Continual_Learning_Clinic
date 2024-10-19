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
    # Create the "saved_models" directory inside the save_path if it does not exist
    save_path = os.path.join(save_path, 'saved_models')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Save the model and assign attributes to it (task_id, model_state_dict)
    save_file = os.path.join(save_path, f'model_task_{task_id}.pth.tar')
    torch.save({
        'task_id': task_id,
        'model_state_dict': model.state_dict(),
    }, save_file)
    print(f'Model for task {task_id} saved at {save_file}')
