//! Compound Data Structures - Working with complex numbers, RGB images, and 3D vectors in Candle
//! Demonstrates practical approaches for handling compound types using multiple tensors and packed representations

//! Practical examples of working with compound data (complex numbers, RGB, vectors) in Candle
//! Demonstrates multiple tensors, packed representation, and interleaved data approaches

// Example: Working with compound data in Candle using workarounds
use candle_core::{Device, Result, Tensor, DType, IndexOp};

// Example: Complex numbers using multiple tensors approach
struct ComplexTensor {
    real: Tensor,
    imag: Tensor,
}

impl ComplexTensor {
    fn new(real: Tensor, imag: Tensor) -> Result<Self> {
        if real.shape() != imag.shape() {
            panic!("Real and imaginary parts must have same shape");
        }
        Ok(Self { real, imag })
    }

    fn zeros(shape: impl Into<candle_core::Shape>, device: &Device) -> Result<Self> {
        let shape = shape.into();
        let real = Tensor::zeros(shape.clone(), DType::F32, device)?;
        let imag = Tensor::zeros(shape, DType::F32, device)?;
        Ok(Self { real, imag })
    }

    fn shape(&self) -> &candle_core::Shape {
        self.real.shape()
    }

    // Complex addition: (a + bi) + (c + di) = (a + c) + (b + d)i
    fn add(&self, other: &ComplexTensor) -> Result<ComplexTensor> {
        let real = (&self.real + &other.real)?;
        let imag = (&self.imag + &other.imag)?;
        Ok(ComplexTensor { real, imag })
    }

    // Complex multiplication: (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
    fn mul(&self, other: &ComplexTensor) -> Result<ComplexTensor> {
        let ac = (&self.real * &other.real)?;
        let bd = (&self.imag * &other.imag)?;
        let ad = (&self.real * &other.imag)?;
        let bc = (&self.imag * &other.real)?;
        
        let real = (&ac - &bd)?;
        let imag = (&ad + &bc)?;
        Ok(ComplexTensor { real, imag })
    }

    // Magnitude: |a + bi| = sqrt(a² + b²)
    fn magnitude(&self) -> Result<Tensor> {
        let real_sq = (&self.real * &self.real)?;
        let imag_sq = (&self.imag * &self.imag)?;
        let sum = (&real_sq + &imag_sq)?;
        sum.sqrt()
    }
}

// Example: RGB pixels using packed representation
struct RgbTensor {
    data: Tensor, // Shape: [..., 3] where last dim is [R, G, B]
}

impl RgbTensor {
    fn new(data: Tensor) -> Result<Self> {
        let dims = data.dims();
        if dims.is_empty() || dims[dims.len() - 1] != 3 {
            panic!("Last dimension must be 3 for RGB");
        }
        Ok(Self { data })
    }

    fn zeros(mut shape: Vec<usize>, device: &Device) -> Result<Self> {
        shape.push(3); // Add RGB dimension
        let data = Tensor::zeros(shape, DType::F32, device)?;
        Ok(Self { data })
    }

    fn shape(&self) -> &candle_core::Shape {
        self.data.shape()
    }

    fn red(&self) -> Result<Tensor> {
        self.data.i((.., 0))
    }

    fn green(&self) -> Result<Tensor> {
        self.data.i((.., 1))
    }

    fn blue(&self) -> Result<Tensor> {
        self.data.i((.., 2))
    }

    // Grayscale conversion: 0.299*R + 0.587*G + 0.114*B
    fn to_grayscale(&self) -> Result<Tensor> {
        let r = self.red()? * 0.299;
        let g = self.green()? * 0.587;
        let b = self.blue()? * 0.114;
        
        ((r? + g?)? + b?)
    }
}

fn main() -> Result<()> {
    println!("🎨 Compound Data Examples in Candle");
    
    let device = Device::cuda_if_available(0)?;
    println!("Using device: {device:?}\n");
    
    // Example 1: Complex numbers
    println!("🔢 Example 1: Complex Numbers");
    let c1 = ComplexTensor::new(
        Tensor::full(3.0f32, (2, 2), &device)?, // 3 + 0i
        Tensor::full(4.0f32, (2, 2), &device)?, // 0 + 4i
    )?; // Result: 3 + 4i
    
    let c2 = ComplexTensor::new(
        Tensor::full(1.0f32, (2, 2), &device)?, // 1 + 0i  
        Tensor::full(2.0f32, (2, 2), &device)?, // 0 + 2i
    )?; // Result: 1 + 2i
    
    let c_sum = c1.add(&c2)?;
    let c_product = c1.mul(&c2)?;
    let magnitude = c1.magnitude()?;
    
    println!("  Complex tensor shape: {:?}", c1.shape());
    println!("  (3+4i) + (1+2i) = sum with shape {:?}", c_sum.shape());
    println!("  (3+4i) * (1+2i) = product with shape {:?}", c_product.shape());
    println!("  |3+4i| = magnitude with shape {:?}", magnitude.shape());
    
    // Example 2: RGB images
    println!("\n🌈 Example 2: RGB Images");
    let rgb_image = RgbTensor::zeros(vec![4, 4], &device)?; // 4x4 RGB image
    println!("  RGB image shape: {:?}", rgb_image.shape()); // [4, 4, 3]
    
    let red_channel = rgb_image.red()?;
    let green_channel = rgb_image.green()?;
    let blue_channel = rgb_image.blue()?;
    
    println!("  Red channel shape: {:?}", red_channel.shape());   // [4, 4]
    println!("  Green channel shape: {:?}", green_channel.shape()); // [4, 4]
    println!("  Blue channel shape: {:?}", blue_channel.shape());  // [4, 4]
    
    let grayscale = rgb_image.to_grayscale()?;
    println!("  Grayscale shape: {:?}", grayscale.shape()); // [4, 4]
    
    // Example 3: Custom data structures using interleaved approach
    println!("\n📊 Example 3: Vector3D using interleaved data");
    let vectors_3d = Tensor::zeros((100, 3), DType::F32, &device)?; // 100 3D vectors [x,y,z]
    println!("  3D vectors shape: {:?}", vectors_3d.shape());
    
    let x_coords = vectors_3d.i((.., 0))?; // All X coordinates
    let y_coords = vectors_3d.i((.., 1))?; // All Y coordinates  
    let z_coords = vectors_3d.i((.., 2))?; // All Z coordinates
    
    println!("  X coordinates: {:?}", x_coords.shape());
    println!("  Y coordinates: {:?}", y_coords.shape());
    println!("  Z coordinates: {:?}", z_coords.shape());
    
    // Vector magnitude: sqrt(x² + y² + z²)
    let x_sq = (&x_coords * &x_coords)?;
    let y_sq = (&y_coords * &y_coords)?;
    let z_sq = (&z_coords * &z_coords)?;
    let magnitude_3d = ((&x_sq + &y_sq)? + &z_sq)?.sqrt()?;
    println!("  Vector magnitudes: {:?}", magnitude_3d.shape());
    
    println!("\n✅ All compound data examples work with standard Candle tensors!");
    println!("💡 Key insight: Use shape dimensions cleverly to encode structure");
    
    Ok(())
}
