# 🧪 Candle Exploration Laboratory

This directory contains experimental code and capability tests for the Candle ML framework, organized as a proper Rust subcrate with GPU acceleration support.

## 📁 Code Files Overview

### Tensor Capabilities
- **`test_5d_tensors.rs`** - 5D Tensor Capabilities Test - Comprehensive testing of Candle's native 5D tensor support
- **`test_higher_dims.rs`** - Higher Dimensional Tensor Test - Verifies Candle's support for 6D, 7D, 8D, and 10D tensors  
- **`test_custom_types.rs`** - Custom Types Investigation - Analysis of Candle's tensor element type limitations and requirements
- **`compound_data_examples.rs`** - Compound Data Structures - Working with complex numbers, RGB images, and 3D vectors in Candle

### Real-time AI Applications  
- **`realtime_yolo.rs`** - Real-time YOLO Object Detection - High-performance GPU-accelerated object detection with live annotations
- **`video_processor.rs`** - Video Processing Framework - Infrastructure for real-time AI video processing applications

### Scripts & Utilities
- **`realtime_demo.sh`** - Bash script for simulating real-time video processing workflows
- **`realtime_output/`** - Directory containing generated output from real-time processing demos

## 🚀 Quick Start

Run any exploration with CUDA acceleration:
```bash
# From the main candle directory
cargo run --package candle-exploration --bin test_5d_tensors --release --features cuda
cargo run --package candle-exploration --bin realtime_yolo --release --features cuda
```

## 🎯 Key Discoveries

✅ **5D+ Tensors**: Candle natively supports unlimited dimensions with full GPU acceleration  
✅ **Real-time AI**: 30+ FPS object detection with YOLO v8 on GPU  
✅ **CUDA Integration**: Full CUDA support with cuDNN compatibility  
❌ **Custom Types**: Limited to 8 built-in scalar types, but workarounds exist  

## 🔧 Technical Setup

- **Subcrate**: Properly integrated into main Candle workspace
- **Dependencies**: candle-core, candle-nn, candle-transformers, fastrand, imageproc
- **Features**: CUDA and cuDNN support enabled
- **GPU**: Tested on CUDA 12.0 with GeForce RTX systems
