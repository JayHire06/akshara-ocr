import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add root directory to sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.insert(0, root_dir)

from data.dataset import OCRDataset
from model.crnn import CRNN
from model.train import train

def collate_fn(batch):
    """
    Custom collate function to handle variable-width images and CTC target flattening.
    Batch is a list of dicts: {'image': tensor, 'label': str, 'label_encoded': tensor, 'language': str}
    """
    images = [item['image'] for item in batch]
    labels_encoded = [item['label_encoded'] for item in batch]
    
    # 1. Pad images to match the maximum width in the batch
    max_width = max(img.shape[2] for img in images)
    
    padded_images = []
    for img in images:
        h = img.shape[1]
        w = img.shape[2]
        
        # Resize height to 32, proportional width
        if h != 32:
            new_w = int(w * (32 / h))
            img = nn.functional.interpolate(img.unsqueeze(0), size=(32, new_w), mode='bilinear', align_corners=False).squeeze(0)
            
        w = img.shape[2]
        
        # Re-calc max width dynamically
        pad_w = max_width - w if max_width > w else 0
        
        # Pad width to max_width of the original batch or newly calculated
        padded_img = nn.functional.pad(img, (0, pad_w, 0, 0))
        padded_images.append(padded_img)
        
    # Re-verify max padding since resizing might have changed the global max width
    actual_max_width = max(img.shape[2] for img in padded_images)
    final_padded_images = [nn.functional.pad(img, (0, actual_max_width - img.shape[2], 0, 0)) for img in padded_images]
    
    images_tensor = torch.stack(final_padded_images) # (batch, 1, 32, max_width)
    
    # 2. Flatten targets for CTCLoss and compute target lengths
    target_lengths = torch.tensor([len(lbl) for lbl in labels_encoded], dtype=torch.long)
    targets = torch.cat(labels_encoded) # 1D tensor
    
    return images_tensor, targets, target_lengths

def test_training_pipeline():
    print("Starting Training Pipeline Verification Check...")
    
    data_dir = os.path.join(root_dir, 'data', 'train')
    labels_csv = os.path.join(data_dir, 'labels.csv')
    vocab_path = os.path.join(root_dir, 'vocab.json')
    
    # Ensure data directory and labels exist
    if not os.path.exists(data_dir) or not os.path.exists(labels_csv):
        print(f"Error: Could not find dataset paths! {data_dir} or {labels_csv}")
        return
        
    print(f"Loading dataset from {data_dir}...")
    # Using same dataset for both train and val for a quick pipeline test
    dataset = OCRDataset(data_dir, labels_csv, vocab_path=vocab_path)
    print(f"Dataset Size: {len(dataset)} items")
    
    # Build a simple index-to-char vocabulary list mapping
    # Assuming vocab is a dict of char -> index
    vocab_dict = dataset.vocab
    
    # Length of vocab_dict includes <PAD> and <UNK>. Max index will be len(vocab_dict) - 1. 
    vocab_size = len(vocab_dict)
    
    # Needs to be a string list where index = char for the CTC Decoder during evaluate()
    vocab_list = [""] * vocab_size
    for char, idx in vocab_dict.items():
        vocab_list[idx] = char
        
    # Get <PAD> or <BLANK> id
    blank_id = vocab_dict.get('<PAD>', 0)
    
    print(f"Vocab size compiled: {vocab_size} classes")
    
    train_loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    val_loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    print("Initializing Model...")
    config = {
        'in_channels': 1,
        'lstm_input_size': 512,
        'lstm_hidden_size': 256,
        'lstm_num_layers': 2,
        'lstm_dropout': 0.3,
        'blank_id': blank_id,
        'lr': 1e-3,
        'lr_factor': 0.5,
        'lr_patience': 3,
        'epochs': 1, # FORCING JUST 1 EPOCH FOR QUICK TEST
        'save_interval': 1, # Force save on epoch 1
        'clip_grad_norm': 5.0,
        'save_dir': os.path.join(root_dir, 'model', 'checkpoints')
    }
    
    # Pass 'epochs': config['epochs'] exactly.
    model = CRNN(vocab_size=vocab_size, config=config)
    
    print("Starting Train Loop (1 Epoch)...")
    # Setting wandb mode to offline or disabled so it doesn't freeze or prompt auth
    os.environ['WANDB_MODE'] = 'disabled'
    
    try:
        train(model, train_loader, val_loader, config, vocab_list)
        print("Training pipeline executed successfully without crashes.")
        
        # Check if checkpoint exists
        checkpoint_path = os.path.join(config['save_dir'], 'checkpoint_epoch_1.pth')
        if os.path.exists(checkpoint_path):
            print(f"Success: Checkpoint verified at {checkpoint_path}")
        else:
            print("Error: Checkpoint file was not saved!")
            
    except Exception as e:
        print(f"Training Failed explicitly: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_training_pipeline()
