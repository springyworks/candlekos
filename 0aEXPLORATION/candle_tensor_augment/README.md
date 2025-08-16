# Candle Tensor Augment

A trait-based extension library for enhancing Candle tensors with additional mathematical and utility operations.

## Overview

This crate provides extension traits that add useful methods to Candle's `Tensor` type, focusing on mathematical expression evaluation and tensor generation utilities. It's designed to complement the core Candle functionality with specialized operations commonly needed in scientific computing and machine learning workflows.

## Features

### 🧮 **Mathematical Expression Filling**
- Generate tensors from mathematical expressions using string-based formulas
- Support for normalized coordinate variables (`x`, `y`) in the range [0,1]
- Built-in mathematical functions (sin, cos, tan, sqrt, abs)
- Flexible tensor shape specification

### 📊 **Tensor Utilities**
- Element summation operations
- Extensible trait-based design for adding new operations

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
candle_tensor_augment = { path = "path/to/candle_tensor_augment" }
candle-core = "0.9.1"
```

### Basic Usage

```rust
use candle_core::{Device, Tensor};
use candle_tensor_augment::{TensorAugment, TensorMathFill};

let device = Device::Cpu;

// Generate a tensor from a mathematical expression
let tensor = Tensor::fill_with_expr(
    100, 100,           // 100x100 tensor
    "sin(x * 6.28) * cos(y * 6.28)",  // Wave interference pattern
    &device
)?;

// Use utility methods
let sum = tensor.sum_elements()?;
println!("Sum of all elements: {}", sum);
```

## Mathematical Expression Syntax

The expression parser supports standard mathematical operations and functions:

### Variables
- `x`: Normalized x-coordinate [0, 1]
- `y`: Normalized y-coordinate [0, 1]

### Supported Operations
- Basic arithmetic: `+`, `-`, `*`, `/`, `^` (power)
- Parentheses for grouping: `(`, `)`

### Built-in Functions
- `sin(x)`: Sine function
- `cos(x)`: Cosine function  
- `tan(x)`: Tangent function
- `sqrt(x)`: Square root
- `abs(x)`: Absolute value

### Example Expressions

```rust
// Simple gradient
"x"

// Radial gradient  
"sqrt((x - 0.5)^2 + (y - 0.5)^2)"

// Wave interference
"sin(x * 10) * cos(y * 10)"

// Gaussian-like bump
"exp(-((x - 0.5)^2 + (y - 0.5)^2) * 10)"

// Checkerboard pattern
"((floor(x * 8) + floor(y * 8)) % 2)"
```

## API Reference

### `TensorMathFill` Trait

#### `fill_with_expr(h: usize, w: usize, expr: &str, device: &Device) -> Result<Tensor>`

Creates a new tensor of shape `(h, w)` filled with values computed from the mathematical expression.

**Parameters:**
- `h`: Height of the tensor (number of rows)
- `w`: Width of the tensor (number of columns)  
- `expr`: Mathematical expression string
- `device`: Candle device for tensor allocation

**Returns:**
- `Result<Tensor>`: The generated tensor or an error if expression parsing/evaluation fails

### `TensorAugment` Trait

#### `sum_elements(&self) -> Result<f32>`

Computes the sum of all elements in the tensor.

**Returns:**
- `Result<f32>`: Sum of all tensor elements (f32 tensors only)

## Error Handling

The crate uses Candle's error handling system:

- **Parse errors**: Invalid mathematical expression syntax
- **Evaluation errors**: Runtime errors during expression evaluation (e.g., division by zero)
- **Tensor errors**: Standard Candle tensor operation errors

```rust
match Tensor::fill_with_expr(100, 100, "invalid expression", &device) {
    Ok(tensor) => println!("Success!"),
    Err(e) => eprintln!("Error: {}", e),
}
```

## Performance Considerations

- Expression parsing occurs once per `fill_with_expr` call
- Evaluation happens for each tensor element
- For large tensors, consider vectorized operations where possible
- GPU tensors are supported but evaluation occurs on CPU then transferred

## Integration with Candle Ecosystem

This crate integrates seamlessly with:

- **candle-core**: Core tensor operations
- **candle-notebooks**: Visualization and display utilities
- **candle-nn**: Neural network building blocks

### Example: Generating Training Data

```rust
use candle_core::{Device, Tensor};
use candle_tensor_augment::TensorMathFill;

let device = Device::Cpu;

// Generate synthetic data with known patterns
let input = Tensor::fill_with_expr(256, 256, "sin(x * 3.14) + cos(y * 3.14)", &device)?;
let target = Tensor::fill_with_expr(256, 256, "x^2 + y^2", &device)?;

// Use with candle-nn for training...
```

## Extending the Library

The trait-based design makes it easy to add new functionality:

```rust
use candle_core::Tensor;

pub trait MyTensorExtensions {
    fn my_custom_operation(&self) -> candle_core::Result<Tensor>;
}

impl MyTensorExtensions for Tensor {
    fn my_custom_operation(&self) -> candle_core::Result<Tensor> {
        // Your implementation here
        todo!()
    }
}
```

## Dependencies

- **candle-core**: Core Candle tensor library
- **meval**: Mathematical expression parser and evaluator

## Contributing

This crate is part of the Candle exploration ecosystem. Contributions are welcome for:

- Additional mathematical functions
- New tensor generation patterns
- Performance optimizations
- Documentation improvements

## License

Same as the parent Candle project: MIT OR Apache-2.0

## See Also

- [Candle Core Documentation](../../candle-core/)
- [Candle Notebooks](../research/notebooks/candle_notebooks/)
- [FFT Demo](../demos/fft_basic_demo.rs.ipynb)
- [Mathematical Visualization Examples](../research/notebooks/)