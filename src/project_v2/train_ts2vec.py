
import torch
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())



import os
import numpy as np
import argparse
from temporal_ts2vec import train_ts2vec_model
from path_utils import normalize_runtime_paths, train_data_path, ts2vec_model_path


def load_times_from_trajectories(trajs):

    times_data = []
    print("Extracting time vectors from trajectories...")
    
    for idx, traj in enumerate(trajs):
        times = []
        for point in traj:
            if len(point) == 2:
                times.append(point[1])
        
        if len(times) > 0:
            times_data.append(np.array(times, dtype=np.float32))
        
        if (idx + 1) % 10000 == 0:
            print(f"Processed {idx + 1}/{len(trajs)} trajectories...")
    
    print(f"Extracted {len(times_data)} time sequences")
    return times_data


def main():

    parser = argparse.ArgumentParser(description='Train TS2Vec temporal embedding model')
    parser.add_argument('--dataset', type=str, default='porto',
                       choices=['porto', 'cd'],
                       help='Dataset name (porto or cd)')
    parser.add_argument('--output_dims', type=int, default=128,
                       help='Output embedding dimensions (default: 128)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs (default: 50)')
    parser.add_argument('--batch_size', type=int, default=512,
                       help='Batch size (default: 512)')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--use_official', action='store_true',
                       help='Use official TS2Vec library (requires installation)')
    parser.add_argument('--device', type=str, default=None,

                       help='Device (cuda:0 or cpu). If None, use cuda:0')
    parser.add_argument('--use_contrastive', action='store_true', default=True,
                       help='Use contrastive learning (InfoNCE loss)')
    parser.add_argument('--no_contrastive', action='store_true',
                       help='Disable contrastive learning (simplified training)')
    parser.add_argument('--temperature', type=float, default=0.5,
                       help='Temperature for contrastive loss (default: 0.5)')
    parser.add_argument('--aug_type', type=str, default='mixed',
                       choices=['jitter', 'scale', 'crop', 'mixed'],
                       help='Data augmentation type (default: mixed)')
    parser.add_argument('--project_root', type=str, default='')
    parser.add_argument('--data_root', type=str, default='')
    parser.add_argument('--dataset_data_dir', type=str, default='')
    parser.add_argument('--temporal_model_dir', type=str, default='')
    parser.add_argument('--ts2vec_model_path', type=str, default='')

    train_args = parser.parse_args()
    train_args = normalize_runtime_paths(train_args)


    if train_args.no_contrastive:
        train_args.use_contrastive = False
    

    dataset = train_args.dataset
    device = train_args.device if train_args.device else 'cuda:0'
    

    if device.startswith('cuda'):
        try:
            import torch
            if torch.cuda.is_available():

                try:
                    test_tensor = torch.zeros(1).cuda()
                    del test_tensor
                    print(f"CUDA device detected: {device}")
                except Exception as e:
                    print(f"Warning: CUDA error detected ({str(e)}), switching to CPU")
                    print("This is likely due to CUDA capability mismatch (e.g., RTX 5060 Ti)")
                    device = 'cpu'
            else:
                print("Warning: CUDA not available, switching to CPU")
                device = 'cpu'
        except Exception as e:
            print(f"Warning: CUDA check failed ({str(e)}), switching to CPU")
            device = 'cpu'
    
    print("=" * 60)
    print("TS2Vec Temporal Embedding Training")
    print("=" * 60)
    print(f"Dataset: {dataset}")
    print(f"Device: {device}")
    print(f"Output dimensions: {train_args.output_dims}")
    print(f"Training epochs: {train_args.epochs}")
    print(f"Batch size: {train_args.batch_size}")
    print(f"Learning rate: {train_args.lr}")
    print(f"Use official TS2Vec: {train_args.use_official}")
    print()
    print(f"Contrastive learning: {train_args.use_contrastive}")
    if train_args.use_contrastive:
        print(f"  - Temperature: {train_args.temperature}")
        print(f"  - Augmentation: {train_args.aug_type}")
    print("=" * 60)
    

    current_train_data_path = train_data_path(train_args, 'init')
    if not os.path.exists(current_train_data_path):
        print(f"Error: Training data not found at {current_train_data_path}")
        print("Please run preprocessing first!")
        return
    
    print(f"\nLoading training data from {current_train_data_path}...")
    train_trajs = np.load(current_train_data_path, allow_pickle=True)
    print(f"Loaded {len(train_trajs)} trajectories")
    

    times_data = load_times_from_trajectories(train_trajs)
    
    if len(times_data) == 0:
        print("Error: No time data extracted!")
        return
    

    seq_lens = [len(t) for t in times_data]
    print(f"\nTime sequence statistics:")
    print(f"  Total sequences: {len(times_data)}")
    print(f"  Min length: {min(seq_lens)}")
    print(f"  Max length: {max(seq_lens)}")
    print(f"  Average length: {np.mean(seq_lens):.2f}")

    model_save_path = str(ts2vec_model_path(train_args))
    
    print(f"\nStarting TS2Vec training...")
    print(f"Model will be saved to: {model_save_path}")
    
    model = train_ts2vec_model(
        times_data=times_data,
        device=device,
        output_dims=train_args.output_dims,
        save_path=model_save_path,
        epochs=train_args.epochs,
        batch_size=train_args.batch_size,
        lr=train_args.lr,
        use_official=train_args.use_official,
        use_contrastive=train_args.use_contrastive,
        temperature=train_args.temperature,
        aug_type=train_args.aug_type
    )
    
    print("\n" + "=" * 60)
    print("TS2Vec training completed!")
    print(f"Model saved to: {model_save_path}")
    print("=" * 60)
    print("\nTraining mode:", "Contrastive Learning" if train_args.use_contrastive else "Simplified")
    print("\nNext steps:")
    print("1. Set temporal_embedding_type='ts2vec' in config.py")
    print("2. Run training: python train.py --task train --dataset", dataset)
    print("\nTo compare both modes:")
    print("  - Contrastive: python train_ts2vec.py --dataset", dataset, "--use_contrastive")
    print("  - Simplified:  python train_ts2vec.py --dataset", dataset, "--no_contrastive")
    print("=" * 60)


if __name__ == "__main__":

    import sys
    if len(sys.argv) == 1:

        print("=" * 60)
        print("One-click run mode: Use default parameters")
        print("Dataset: porto, Epochs: 50")
        
        print("Mode: Contrastive Learning (InfoNCE)")
        print()
        print("To customize parameters, please use the command line:")
        print("  ct: python train_ts2vec.py --dataset porto --use_contrastive")
        print("  ez: python train_ts2vec.py --dataset porto --no_contrastive")
        print("=" * 60)
        sys.argv = ['train_ts2vec.py', '--dataset', 'porto', '--epochs', '45', '--use_contrastive']   
    main()

