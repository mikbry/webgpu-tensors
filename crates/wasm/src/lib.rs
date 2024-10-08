// #[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;
// #[cfg(target_arch = "wasm32")]
use webgpu_tensors::{Tensor, TensorOptions, Tensors, TensorsOperations};
// #[cfg(target_arch = "wasm32")]
use serde::{Deserialize, Serialize};
// #[cfg(target_arch = "wasm32")]
use serde_wasm_bindgen::{from_value, to_value};

// #[cfg(target_arch = "wasm32")]
#[derive(Serialize, Deserialize)]
pub struct JSTensorOptions {
    pub dtype: Option<String>,
    pub device: Option<String>,
    pub shape: Option<Vec<usize>>,
}

impl JSTensorOptions {
    fn to_options(&self) -> TensorOptions {
        TensorOptions::default()
    }
}

// #[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub struct WASMTensors {
    #[wasm_bindgen(skip)]
    pub instance: Tensors,

    pub device_type: usize,
}

// #[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl WASMTensors {
    // #[cfg(target_arch = "wasm32")]
    #[wasm_bindgen(constructor)]
    pub fn new() -> WASMTensors {
        let instance = Tensors::default();
        WASMTensors {
            instance,
            device_type: 0,
        }
    }

    #[inline]
    pub fn init(&mut self) {
        let _ = self.instance.init(None);
    }

    #[inline]
    pub fn reset(&mut self) {
        self.instance.reset();
    }

    #[inline]
    pub fn compute(&mut self) {
        self.instance.compute();
    }

    #[inline]
    pub fn destroy(&mut self) {
        self.instance.destroy();
    }

    #[wasm_bindgen]
    pub fn empty(&self, shape: Vec<usize>, options: JsValue) -> Result<JsValue, JsValue> {
        let tensor_options = self.parse_options(options);
        let tensor = self.instance.empty(shape, Some(tensor_options.to_options()));
        to_value(&tensor).map_err(|e| e.into())
    }

    #[wasm_bindgen]
    pub fn ones(&self, shape: Vec<usize>, options: JsValue) -> Result<JsValue, JsValue> {
        let tensor_options = self.parse_options(options);
        let tensor = self.instance.ones(shape, Some(tensor_options.to_options()));
        to_value(&tensor).map_err(|e| e.into())
    }

    #[wasm_bindgen]
    pub fn rand(&self, shape: Vec<usize>, options: JsValue) -> Result<JsValue, JsValue> {
        let tensor_options = self.parse_options(options);
        let tensor = self.instance.rand(shape, Some(tensor_options.to_options()));
        to_value(&tensor).map_err(|e| e.into())
    }

    #[wasm_bindgen]
    pub fn randn(&self, shape: Vec<usize>, options: JsValue) -> Result<JsValue, JsValue> {
        let tensor_options = self.parse_options(options);
        let tensor = self.instance.randn(shape, Some(tensor_options.to_options()));
        to_value(&tensor).map_err(|e| e.into())
    }

    #[wasm_bindgen]
    pub fn zeros(&self, shape: Vec<usize>, options: JsValue) -> Result<JsValue, JsValue> {
        let tensor_options = self.parse_options(options);
        let tensor = self.instance.zeros(shape, Some(tensor_options.to_options()));
        to_value(&tensor).map_err(|e| e.into())
    }

    #[wasm_bindgen]
    pub fn tensor(&self, array: JsValue, options: JsValue) -> Result<JsValue, JsValue> {
        let tensor_options = self.parse_options(options);
        let shape = tensor_options.shape.clone().unwrap_or([].to_vec());
        let data: Vec<f32> = from_value(array).map_err(|e| e.to_string())?;
        let tensor = self.instance.tensor((data, shape), Some(tensor_options.to_options()));
        to_value(&tensor).map_err(|e| e.into())
    }

    #[wasm_bindgen]
    pub fn matmul(&self, a: JsValue, b: JsValue) -> Result<JsValue, JsValue> {
        let tensor_a: Tensor = from_value(a).map_err(|e| e.to_string())?;
        let tensor_b: Tensor = from_value(b).map_err(|e| e.to_string())?;
        let result = self.instance.matmul(&tensor_a, &tensor_b);
        to_value(&result).map_err(|e| e.into())
    }

    #[wasm_bindgen]
    pub fn mul(&self, a: JsValue, b: JsValue) -> Result<JsValue, JsValue> {
        let tensor_a: Tensor = from_value(a).map_err(|e| e.to_string())?;
        let tensor_b: Tensor = from_value(b).map_err(|e| e.to_string())?;
        let result = self.instance.mul(&tensor_a, &tensor_b);
        to_value(&result).map_err(|e| e.into())
    }

    #[wasm_bindgen]
    pub fn sub(&self, a: JsValue, b: JsValue) -> Result<JsValue, JsValue> {
        let tensor_a: Tensor = from_value(a).map_err(|e| e.to_string())?;
        let tensor_b: Tensor = from_value(b).map_err(|e| e.to_string())?;
        let result = self.instance.sub(&tensor_a, &tensor_b);
        to_value(&result).map_err(|e| e.into())
    }

    #[wasm_bindgen]
    pub fn pow(&self, tensor: JsValue, exponent: f32) -> Result<JsValue, JsValue> {
        let tensor: Tensor = from_value(tensor).map_err(|e| e.to_string())?;
        let result = self.instance.pow(&tensor, exponent);
        to_value(&result).map_err(|e| e.into())
    }

    #[wasm_bindgen]
    pub fn gt(&self, tensor: JsValue, value: f32) -> Result<JsValue, JsValue> {
        let tensor: Tensor = from_value(tensor).map_err(|e| e.to_string())?;
        let result = self.instance.gt(&tensor, value);
        to_value(&result).map_err(|e| e.into())
    }

    #[wasm_bindgen]
    pub fn transpose(&self, tensor: JsValue) -> Result<JsValue, JsValue> {
        let tensor: Tensor = from_value(tensor).map_err(|e| e.to_string())?;
        let result = self.instance.transpose(&tensor);
        to_value(&result).map_err(|e| e.into())
    }

    #[wasm_bindgen]
    pub fn maximum(&self, tensor: JsValue, value: f32) -> Result<JsValue, JsValue> {
        let tensor: Tensor = from_value(tensor).map_err(|e| e.to_string())?;
        let result = self.instance.maximum(&tensor, value);
        to_value(&result).map_err(|e| e.into())
    }

    #[wasm_bindgen]
    pub fn relu(&self, tensor: JsValue) -> Result<JsValue, JsValue> {
        let tensor: Tensor = from_value(tensor).map_err(|e| e.to_string())?;
        let result = self.instance.relu(&tensor);
        to_value(&result).map_err(|e| e.into())
    }

    #[wasm_bindgen]
    pub fn max(&self, tensor: JsValue) -> Result<JsValue, JsValue> {
        let tensor: Tensor = from_value(tensor).map_err(|e| e.to_string())?;
        let result = self.instance.max(&tensor);
        to_value(&result).map_err(|e| e.into())
    }

    #[wasm_bindgen]
    pub fn mean(&self, tensor: JsValue) -> Result<JsValue, JsValue> {
        let tensor: Tensor = from_value(tensor).map_err(|e| e.to_string())?;
        let result = self.instance.mean(&tensor);
        to_value(&result).map_err(|e| e.into())
    }

    #[wasm_bindgen]
    pub fn sigmoid(&self, tensor: JsValue) -> Result<JsValue, JsValue> {
        let tensor: Tensor = from_value(tensor).map_err(|e| e.to_string())?;
        let result = self.instance.sigmoid(&tensor);
        to_value(&result).map_err(|e| e.into())
    }

    #[wasm_bindgen]
    pub fn clone(&self, tensor: JsValue) -> Result<JsValue, JsValue> {
        let tensor: Tensor = from_value(tensor).map_err(|e| e.to_string())?;
        let result = self.instance.clone(&tensor);
        to_value(&result).map_err(|e| e.into())
    }

    #[wasm_bindgen]
    pub fn item(&self, tensor: JsValue) -> Result<f32, JsValue> {
        let tensor: Tensor = from_value(tensor).map_err(|e| e.to_string())?;
        Ok(self.instance.item(&tensor))
    }

    #[wasm_bindgen]
    pub fn mul_scalar(&self, tensor: JsValue, scalar: f32) -> Result<JsValue, JsValue> {
        let tensor: Tensor = from_value(tensor).map_err(|e| e.to_string())?;
        let result = self.instance.mul_scalar(&tensor, scalar);
        to_value(&result).map_err(|e| e.into())
    }

    #[wasm_bindgen]
    pub fn copy(&self, src: JsValue, dst: JsValue) -> Result<(), JsValue> {
        let src_tensor: Tensor = from_value(src).map_err(|e| e.to_string())?;
        let mut dst_tensor: Tensor = from_value(dst).map_err(|e| e.to_string())?;
        self.instance
            .copy(&src_tensor, &mut dst_tensor)
            .map_err(|e| e.into())
    }

    #[wasm_bindgen]
    pub fn read_array(&self, tensor: JsValue) -> Result<JsValue, JsValue> {
        let tensor: Tensor = from_value(tensor).map_err(|e| e.to_string())?;
        let result = tensor.read_array().map_err(|e| e.to_string())?;
        to_value(&result).map_err(|e| e.into())
    }

    #[wasm_bindgen]
    pub fn read_float32(&self, tensor: JsValue) -> Result<JsValue, JsValue> {
        let tensor: Tensor = from_value(tensor).map_err(|e| e.to_string())?;
        let result = tensor.read_float32().map_err(|e| e.to_string())?;
        to_value(&result).map_err(|e| e.into())
    }

    fn parse_options(&self, options: JsValue) -> JSTensorOptions {
        from_value(options).unwrap_or_else(|_| JSTensorOptions {
            dtype: None,
            device: None,
            shape: None,
        })
    }
}
