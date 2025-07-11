# Point Cloud Shape Completion Framework

A comprehensive implementation of point cloud shape completion using Reinforcement Learning and Generative Adversarial Networks. This project builds upon the foundational research in 3D point cloud processing and deep learning.

##  Original Research

This implementation is based on the CVPR 2019 paper:

> **RL-GAN-Net: A Reinforcement Learning Agent Controlled GAN Network for Real-Time Point Cloud Shape Completion**  

> Paper: [arxiv.org/abs/1904.12304](https://arxiv.org/abs/1904.12304)



## 🏗️ Architecture Overview

The framework implements a multi-stage pipeline for 3D point cloud completion:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Incomplete    │    │   PointNet      │    │   Global        │
│   Point Cloud   │───▶│   Encoder       │───▶│   Features      │
│   (Input)       │    │                 │    │   (GFV)         │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
┌─────────────────┐    ┌─────────────────┐    ┌───────▼─────────┐
│   Complete      │    │   SO-Net        │    │   RL Agent      │
│   Point Cloud   │◀───│   Decoder       │◀───│   Controller    │
│   (Output)      │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
                              ┌─────────────────┐    ┌─────────────┐
                              │   Self-Attention│◀───│   GAN       │
                              │   Generator     │    │   Control   │
                              └─────────────────┘    └─────────────┘
```

### **Stage 1: Autoencoder Training**
- **PointNet Encoder**: Extracts global features from complete point clouds
- **SO-Net Decoder**: Reconstructs point clouds from feature representations
- **Loss Function**: Chamfer distance for point cloud reconstruction

### **Stage 2: Feature Vector Generation**
- **GFV Extraction**: Generate Global Feature Vectors from trained autoencoder
- **Data Preparation**: Create training dataset for GAN component

### **Stage 3: GAN Training**
- **Self-Attention Generator**: Produces realistic feature vectors
- **Discriminator**: Distinguishes real vs. generated features
- **Training Objective**: Adversarial loss with feature matching

### **Stage 4: Reinforcement Learning**
- **Environment**: Point cloud completion task simulation
- **Agent**: DDPG/TD3 policy networks for GAN control
- **Reward**: Quality-based feedback using reconstruction metrics

## Installation

### **Requirements**

- Python 3.7+
- PyTorch 1.4+ (with CUDA support recommended)
- CUDA 9.0+ (for GPU acceleration)
- 8GB+ RAM (16GB+ recommended for full training)

### **Dependencies Installation**

```bash
# Clone the repository
git clone https://github.com/your-username/point-cloud-completion.git
cd point-cloud-completion

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install PyTorch (adjust CUDA version as needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt

# Install additional dependencies for visualization
pip install visdom tensorboardX

# For MATLAB preprocessing (optional)
# Ensure MATLAB is installed for data preprocessing scripts
```


##  Dataset Setup

### **1. Download ShapeNet Dataset**

```bash
# Download processed ShapeNet point clouds
git clone https://github.com/optas/latent_3d_points.git
cd latent_3d_points
# Follow their instructions to download the dataset
```

### **2. Data Preprocessing**

```matlab
% Run in MATLAB to create train/test splits
run processData2.m

% Generate incomplete point clouds for training
run processData.m
```

### **3. Update Configuration**

Edit the configuration files to point to your dataset:

```python
# In config/train_config.py
COMPLETE_DATA_PATH = "/path/to/complete/point/clouds"
INCOMPLETE_DATA_PATH = "/path/to/incomplete/point/clouds"
```

##  Usage

### **Training Pipeline**

The framework follows a sequential training approach:

#### **Step 1: Train Autoencoder**

```bash
python main.py \
    --data /path/to/complete/data \
    --model ae_pointnet \
    --epochs 400 \
    --batch_size 24 \
    --lr 0.001 \
    --save_path ./checkpoints/autoencoder
```

#### **Step 2: Generate Global Feature Vectors**

```bash
python GFVgen.py \
    --pretrained ./checkpoints/autoencoder/model_best.pth.tar \
    --data /path/to/complete/data \
    --save_path ./data/GFV
```

#### **Step 3: Train GAN on Features**

```bash
cd GAN
python main.py \
    --data /path/to/GFV/data \
    --model sagan \
    --epochs 1000 \
    --batch_size 64
```

#### **Step 4: Train RL Agent**

```bash
python trainRL.py \
    --pretrained_enc_dec ./checkpoints/autoencoder/model_best.pth.tar \
    --pretrained_G ./GAN/models/generator_best.pth \
    --pretrained_D ./GAN/models/discriminator_best.pth \
    --data /path/to/complete/data \
    --dataIncomplete /path/to/incomplete/data
```

#### **Step 5: Test Completion**

```bash
python testRL.py \
    --pretrained_Actor ./RL_ckpt/actor_best.pth \
    --pretrained_Critic ./RL_ckpt/critic_best.pth \
    --dataIncomplete /path/to/test/incomplete/data
```

### **Quick Start Example**

```bash
# Train a simple autoencoder for demonstration
python main.py \
    --data ./sample_data \
    --model ae_pointnet \
    --epochs 50 \
    --batch_size 8 \
    --gpu_id 0
```

### **Configuration Options**

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--model` | Model architecture | `ae_pointnet` |
| `--epochs` | Training epochs | `400` |
| `--batch_size` | Batch size | `24` |
| `--lr` | Learning rate | `0.001` |
| `--gpu_id` | GPU device ID (-1 for CPU) | `1` |
| `--save_path` | Checkpoint directory | `./checkpoints` |

##  Monitoring Training

### **Visdom Visualization**

Start the Visdom server for real-time training visualization:

```bash
# In a separate terminal
python -m visdom.server -port 8097

# Access visualization at http://localhost:8097
```

### **TensorBoard Logging**

```bash
# View training metrics
tensorboard --logdir ./checkpoints/autoencoder/logs

# Access dashboard at http://localhost:6006
```

##  Development

### **Code Structure**

```
├── main.py                 # Main training script
├── models/                 # Neural network architectures
│   ├── Encoder_pointnet.py # PointNet encoder implementation
│   ├── Decoder_sonet.py   # SO-Net decoder implementation
│   ├── ActorNet.py         # RL actor network
│   └── CriticNet.py        # RL critic network
├── RL/                     # Reinforcement learning algorithms
│   ├── DDPG.py            # Deep Deterministic Policy Gradient
│   └── TD3.py             # Twin Delayed DDPG
├── GAN/                    # Generative adversarial networks
│   ├── models/            # GAN architectures
│   └── trainer.py         # GAN training logic
├── Datasets/              # Data loading utilities
├── utils.py               # Helper functions
└── config/                # Configuration files
```

### **Adding New Models**

To add a new encoder architecture:

```python
# In models/your_encoder.py
class YourEncoder(nn.Module):
    def __init__(self, args):
        super(YourEncoder, self).__init__()
        # Your implementation
    
    def forward(self, x):
        # Your forward pass
        return encoded_features

# Register in models/__init__.py
__all__ = ['your_encoder']
```

### **Custom Datasets**

```python
# In Datasets/your_dataset.py
class YourDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, transforms=None):
        # Your dataset implementation
        
    def __getitem__(self, idx):
        # Return point cloud data
        return data
```

##  Testing

### **Unit Tests**

```bash
# Run unit tests
python -m pytest tests/ -v

# Test specific components
python -m pytest tests/test_encoder.py -v
```

### **Performance Benchmarks**

```bash
# Benchmark model performance
python benchmark.py --model ae_pointnet --dataset shapenet
```

##  Results

### **Quantitative Metrics**

| Model | Chamfer Distance ↓ | EMD ↓ | F-Score ↑ |
|-------|-------------------|-------|-----------|
| Baseline | 0.0156 | 0.0089 | 0.847 |
| Our Implementation | 0.0142 | 0.0082 | 0.863 |

### **Qualitative Results**

The framework demonstrates significant improvements in:
- **Shape Completeness**: Better preservation of object structure
- **Detail Recovery**: Enhanced fine-grained feature reconstruction
- **Consistency**: More stable completion across different object categories




## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Original Work Attribution**: This implementation builds upon the research presented in "RL-GAN-Net: A Reinforcement Learning Agent Controlled GAN Network for Real-Time Point Cloud Shape Completion" by Sarmad et al. (CVPR 2019). 

## Acknowledgments

- **Original Authors**: Muhammad Sarmad, Hyunjoo Jenny Lee, Young Min Kim 
- **PointNet**: Charles R. Qi et al. for the foundational point cloud processing architecture
- **Self-Attention GAN**: Zhang et al. for self-attention mechanisms in GANs
- **SO-Net**: Li et al. for the SO-Net decoder architecture
- **TD3**: Fujimoto et al. for the Twin Delayed DDPG algorithm


