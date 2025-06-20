use candle_core::{Device, DType, Tensor};
use parking_lot::Mutex;
use smallvec::SmallVec;
use std::collections::HashMap;
use std::sync::Arc;

type ShapeKey = SmallVec<[usize; 4]>;

pub struct TensorPool {
    pools: Arc<Mutex<HashMap<(ShapeKey, DType), Vec<Tensor>>>>,
    device: Device,
    max_pool_size: usize,
}

impl TensorPool {
    pub fn new(device: Device) -> Self {
        Self {
            pools: Arc::new(Mutex::new(HashMap::new())),
            device,
            max_pool_size: 100,
        }
    }
    
    /// Get a tensor from pool or create new one
    pub fn get(&self, shape: &[usize], dtype: DType) -> candle_core::Result<Tensor> {
        let key = (ShapeKey::from_slice(shape), dtype);
        
        let mut pools = self.pools.lock();
        if let Some(pool) = pools.get_mut(&key) {
            if let Some(tensor) = pool.pop() {
                return Ok(tensor);
            }
        }
        
        // Create new tensor
        Tensor::zeros(shape, dtype, &self.device)
    }
    
    /// Return tensor to pool
    pub fn return_tensor(&self, tensor: Tensor) -> candle_core::Result<()> {
        let shape = tensor.shape();
        let dtype = tensor.dtype();
        let key = (ShapeKey::from_slice(shape.dims()), dtype);
        
        let mut pools = self.pools.lock();
        let pool = pools.entry(key).or_insert_with(Vec::new);
        
        if pool.len() < self.max_pool_size {
            // Zero out tensor before returning to pool
            let zeros = Tensor::zeros(shape.dims(), dtype, &self.device)?;
            tensor.copy_from(&zeros)?;
            pool.push(tensor);
        }
        
        Ok(())
    }
    
    /// Clear all pooled tensors
    pub fn clear(&self) {
        self.pools.lock().clear();
    }
} 