# OpenWave Distributed Computing Architecture Plan

## Executive Summary

**Current State**: OpenWave is **partially ready** for distributed computing. The Taichi kernels and optimized loops work well for single-GPU parallelization but need modifications for true distributed computing.

**AWS Capabilities**:

- Yes, AWS has powerful GPU instances (P3, P4, G4, G5 series)
- Supports both GPU clusters and CPU clusters
- Services like AWS Batch, ParallelCluster, and EKS can orchestrate distributed workloads

**What Works Now**:

- Taichi GPU acceleration (single GPU)
- Parallelized kernels
- Optimized memory usage

**What Needs Change for Distributed**:

1. **Domain decomposition** - Split universe into regions
2. **MPI communication** - For multi-node coordination
3. **Multi-GPU support** - Taichi supports it but needs explicit code
4. **Decoupled rendering** - Separate visualization from computation
5. **Distributed storage** - For datasets exceeding single machine memory

**Performance Optimization Paths**:

- **Hybrid approach**: MPI across nodes + GPU kernels within nodes
- **Use Dask/Ray**: For distributed array operations
- **Checkpoint to S3**: For fault tolerance
- **Spot instances**: For 70-90% cost reduction

## Current State Assessment

### What Works Well

- Taichi kernels provide excellent single-GPU parallelization
- Modular structure separates concerns (data generation vs rendering)
- Pre-computation strategies reduce redundant work

### What Needs Modification for Distributed Computing

## Recommended Architecture Changes

### 1. Domain Decomposition Strategy

Split the universe into spatial domains that can be computed independently:

```python
class DistributedLattice2D:
    def __init__(self, domain_id, total_domains, spacing_scale_factor):
        # Each node computes a subset of the universe
        self.domain_id = domain_id
        self.total_domains = total_domains
        self.local_line_size = config.UNIVERSE_RADIUS / sqrt(total_domains)
        # Add ghost zones for boundary communication
        self.ghost_zone_width = 2  # cells
```

### 2. Multi-GPU Support via Taichi

Taichi supports multi-GPU, but needs explicit management:

```python
import taichi as ti

# Initialize for multi-GPU
ti.init(arch=ti.cuda)  # or ti.vulkan for broader compatibility

# Create fields on specific devices
@ti.kernel
def compute_on_multiple_gpus():
    # Taichi can distribute across GPUs with proper setup
    pass
```

### 3. MPI for Inter-Node Communication

For true distributed computing across multiple machines:

```python
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

class MPILattice2D:
    def __init__(self):
        self.rank = rank
        self.size = size
        # Divide work based on rank
        self.local_domain = self.compute_local_domain()
    
    def exchange_boundaries(self):
        # Exchange ghost zone data with neighbors
        if self.rank > 0:
            comm.Send(self.left_boundary, dest=self.rank-1)
        if self.rank < self.size-1:
            comm.Send(self.right_boundary, dest=self.rank+1)
```

### 4. Hybrid Parallelization Strategy

Combine different parallelization levels:

```python
# Level 1: MPI across nodes (coarse-grain)
# Level 2: Multi-GPU within nodes (medium-grain)  
# Level 3: Taichi kernels on each GPU (fine-grain)

def hybrid_compute():
    # MPI divides universe into regions
    local_region = get_mpi_region(rank)
    
    # Each region uses multiple GPUs
    gpu_id = rank % num_gpus_per_node
    
    # Taichi kernels parallelize within GPU
    with ti.device(gpu_id):
        compute_local_physics(local_region)
```

### 5. Data Partitioning for Large Scale

For simulations exceeding single-machine memory:

```python
import dask.array as da
import zarr

# Use Dask for distributed arrays
positions = da.zeros((billion_scale, billion_scale, 2), 
                     chunks=(1000, 1000, 2),
                     dtype=np.float32)

# Use Zarr for distributed storage
store = zarr.DirectoryStore('s3://my-bucket/openwave-data')
positions_zarr = zarr.open_array(store, mode='w', 
                                  shape=positions.shape,
                                  chunks=positions.chunks)
```

## AWS Deployment Strategy

### For Compute-Intensive Phases

#### Option 1: AWS Batch with GPU

```yaml
# job-definition.yaml
computeEnvironment:
  type: MANAGED
  computeResources:
    type: EC2
    instanceTypes:
      - p3.2xlarge  # 1 V100 GPU
      - p3.8xlarge  # 4 V100 GPUs
    maxvCpus: 256
```

#### Option 2: EKS with GPU Nodes

```yaml
# eks-gpu-nodegroup.yaml
nodeGroups:
  - name: gpu-workers
    instanceType: g4dn.xlarge
    desiredCapacity: 4
    labels:
      workload: simulation
    taints:
      - key: nvidia.com/gpu
        value: "true"
        effect: NoSchedule
```

### For Visualization/Rendering

#### Separate Rendering Service

```python
# Decouple rendering from computation
class RemoteRenderer:
    def __init__(self, data_url):
        self.data_source = data_url
        
    def stream_visualization(self):
        # Use WebGL/WebGPU for browser-based rendering
        # Or NICE DCV for remote desktop streaming
        pass
```

## Performance Optimization Strategies

### 1. Asynchronous I/O

```python
import asyncio
import aioboto3

async def save_checkpoint_async(data):
    async with aioboto3.Session().client('s3') as s3:
        await s3.put_object(Bucket='openwave',
                           Key=f'checkpoint_{rank}.npz',
                           Body=data)
```

### 2. Communication Optimization

```python
# Overlapping computation with communication
def optimized_boundary_exchange():
    # Start non-blocking sends
    requests = []
    if rank > 0:
        req = comm.Isend(left_boundary, dest=rank-1)
        requests.append(req)
    
    # Compute interior while boundaries transfer
    compute_interior()
    
    # Wait for communication to complete
    MPI.Request.Waitall(requests)
    
    # Compute boundaries
    compute_boundaries()
```

### 3. Load Balancing

```python
class DynamicLoadBalancer:
    def redistribute_work(self):
        # Monitor computation times
        local_time = measure_computation_time()
        
        # Gather all times
        all_times = comm.gather(local_time, root=0)
        
        if rank == 0:
            # Rebalance if needed
            new_distribution = calculate_new_distribution(all_times)
            
        # Broadcast new distribution
        new_distribution = comm.bcast(new_distribution, root=0)
```

## Required Infrastructure Changes

### 1. Add Dependencies

```toml
# pyproject.toml additions
[tool.poetry.dependencies]
mpi4py = "^3.1.0"          # MPI support
dask = {extras = ["complete"], version = "^2023.0"}  # Distributed arrays
zarr = "^2.16.0"           # Distributed storage
ray = "^2.8.0"             # Alternative to Dask
cupy = "^12.0.0"           # GPU arrays
```

### 2. Container Support

```dockerfile
# Dockerfile for distributed deployment
FROM nvidia/cuda:12.0-runtime-ubuntu22.04

# Install MPI
RUN apt-get update && apt-get install -y \
    openmpi-bin \
    openmpi-common \
    libopenmpi-dev

# Install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Configure for multi-GPU
ENV CUDA_VISIBLE_DEVICES=all
```

### 3. Kubernetes Manifests

```yaml
# openwave-distributed.yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: openwave-simulation
spec:
  parallelism: 4
  template:
    spec:
      containers:
      - name: openwave-worker
        image: openwave:distributed
        resources:
          limits:
            nvidia.com/gpu: 1
        env:
        - name: WORLD_SIZE
          value: "4"
```

## Implementation Phases

### Phase 1: Multi-GPU Single Node

- Modify Taichi kernels for multi-GPU
- Implement domain decomposition
- Test on single AWS p3.8xlarge (4 GPUs)

### Phase 2: Multi-Node with MPI

- Add MPI communication layer
- Implement boundary exchange
- Test on AWS ParallelCluster

### Phase 3: Cloud-Native Architecture

- Containerize application
- Deploy on EKS/ECS
- Add auto-scaling based on workload

### Phase 4: Hybrid Cloud/Edge

- Support for mixed compute resources
- Edge devices for real-time visualization
- Cloud for heavy computation

## Performance Metrics to Track

```python
class PerformanceMonitor:
    def __init__(self):
        self.metrics = {
            'computation_time': [],
            'communication_time': [],
            'gpu_utilization': [],
            'memory_usage': [],
            'scaling_efficiency': []
        }
    
    def calculate_scaling_efficiency(self):
        # Strong scaling: Fixed problem size
        T1 = single_node_time
        Tn = multi_node_time
        efficiency = T1 / (n_nodes * Tn)
        
        # Weak scaling: Fixed work per node
        T1_per_unit = single_node_time_per_unit
        Tn_per_unit = multi_node_time_per_unit
        efficiency = T1_per_unit / Tn_per_unit
```

## Cost Optimization on AWS

### Spot Instances for Non-Critical Runs

```python
import boto3

ec2 = boto3.client('ec2')

def request_spot_gpu_fleet():
    response = ec2.request_spot_fleet(
        SpotFleetRequestConfig={
            'IamFleetRole': 'arn:aws:iam::account:role/fleet-role',
            'SpotPrice': '0.50',  # Max price per hour
            'TargetCapacity': 4,
            'LaunchSpecifications': [{
                'ImageId': 'ami-openwave',
                'InstanceType': 'g4dn.xlarge',
                'KeyName': 'openwave-key',
                'UserData': base64.b64encode(startup_script)
            }]
        }
    )
```

### Checkpointing for Spot Instance Interruptions

```python
import signal
import pickle

class CheckpointManager:
    def __init__(self):
        signal.signal(signal.SIGTERM, self.save_checkpoint)
        
    def save_checkpoint(self, signum, frame):
        # AWS gives 2-minute warning
        checkpoint = {
            'positions': positions,
            'velocities': velocities,
            'iteration': current_iteration
        }
        
        with open(f's3://bucket/checkpoint_{rank}.pkl', 'wb') as f:
            pickle.dump(checkpoint, f)
```

## Summary

OpenWave is partially ready for distributed computing but needs significant modifications for true scalability:

### Current Strengths

- GPU-optimized kernels
- Modular architecture
- Efficient local parallelization

### Required Changes

1. Domain decomposition for data partitioning
2. MPI or similar for inter-node communication
3. Decoupled rendering from computation
4. Cloud-native containerization
5. Distributed data storage strategy

### AWS Advantages to Leverage

- GPU instances for Taichi kernels
- Spot instances for cost optimization
- S3 for distributed storage
- EKS/Batch for orchestration
- CloudWatch for monitoring

The investment in distributed architecture will pay off when simulating trillions of particles or exploring parameter spaces in parallel.
