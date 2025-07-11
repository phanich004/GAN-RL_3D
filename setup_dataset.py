"""
Script to setup the ShapeNet dataset for RL-GAN-Net training
Can use either synthetic data for testing or real ShapeNet data
"""

import argparse
from pathlib import Path
from utils.dataset import setup_dataset, create_dataloader


def main():
    parser = argparse.ArgumentParser(description='Setup RL-GAN-Net dataset')
    parser.add_argument('--data-dir', type=str, default='./data/shapenet',
                       help='Directory to store dataset')
    parser.add_argument('--synthetic', action='store_true',
                       help='Create synthetic dataset for testing')
    parser.add_argument('--samples-per-category', type=int, default=100,
                       help='Number of samples per category for synthetic data')
    parser.add_argument('--test', action='store_true',
                       help='Test data loading after setup')
    
    args = parser.parse_args()
    
    print("Setting up RL-GAN-Net dataset...")
    print(f"Data directory: {args.data_dir}")
    print(f"Using synthetic data: {args.synthetic}")
    
    # Setup dataset
    if args.synthetic:
        print(f"Creating synthetic dataset with {args.samples_per_category} samples per category...")
    
    setup_dataset(args.data_dir, synthetic=args.synthetic)
    
    # Test data loading if requested
    if args.test:
        print("\nTesting data loading...")
        
        try:
            # Test train loader
            train_loader = create_dataloader(
                args.data_dir,
                split='train',
                batch_size=4,
                num_workers=2,
                num_points=2048
            )
            
            print(f"Train dataset size: {len(train_loader.dataset)}")
            
            # Load a sample batch
            batch = next(iter(train_loader))
            print(f"Batch keys: {list(batch.keys())}")
            print(f"Complete point cloud shape: {batch['complete_pc'].shape}")
            print(f"Incomplete point cloud shape: {batch['incomplete_pc'].shape}")
            print(f"Categories: {batch['category']}")
            
            # Test validation loader
            val_loader = create_dataloader(
                args.data_dir,
                split='test',
                batch_size=4,
                num_workers=2,
                num_points=2048
            )
            
            print(f"Validation dataset size: {len(val_loader.dataset)}")
            
            print("✅ Data loading test passed!")
            
        except Exception as e:
            print(f"❌ Data loading test failed: {e}")
            return 1
    
    print("\n📁 Dataset setup complete!")
    if args.synthetic:
        print("📌 Note: Using synthetic data for testing.")
        print("   For real ShapeNet data, follow instructions in utils/dataset.py")
    
    print(f"📁 Dataset location: {Path(args.data_dir).absolute()}")
    print("🚀 Ready to train RL-GAN-Net!")
    
    return 0


if __name__ == "__main__":
    exit(main()) 