#[cfg(target_arch = "wasm32")]
use wasm_bindgen::{prelude::wasm_bindgen, JsValue};
#[cfg(target_arch = "wasm32")]
use webgpu_tensors::{RSTensors, Tensors, RSTensor, TensorOptions};
use serde::{Serialize, Deserialize};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[derive(Serialize, Deserialize)]
pub struct JSTensorOptions {
    pub dtype: Option<String>,
    pub device: Option<String>,
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub struct WASMTensorsImpl {
    #[wasm_bindgen(skip)]
    pub instance: RSTensors,

    pub device_type: usize,
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl WASMTensorsImpl {
    #[cfg(target_arch = "wasm32")]
    #[wasm_bindgen(constructor)]
    pub fn new() -> WASMTensorsImpl {
        WASMTensorsImpl { instance: RSTensors, device_type: 0 }
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
    pub fn randn(&self, shape: Vec<usize>, options: JsValue) -> Result<JsValue, JsValue> {
        let js_options: JSTensorOptions = options.into_serde().unwrap_or_else(|_| JSTensorOptions { dtype: None, device: None });
        let tensor_options = TensorOptions {
            usage: 0, // Set appropriate usage value
            mapped_at_creation: None,
            readable: true,
        };
        let tensor = self.instance.randn(shape, Some(tensor_options));
        JsValue::from_serde(&tensor).map_err(|e| e.into())
    }

    #[wasm_bindgen]
    pub fn matmul(&self, a: JsValue, b: JsValue) -> Result<JsValue, JsValue> {
        let tensor_a: RSTensor = a.into_serde().map_err(|e| e.to_string())?;
        let tensor_b: RSTensor = b.into_serde().map_err(|e| e.to_string())?;
        let result = self.instance.matmul(&tensor_a, &tensor_b);
        JsValue::from_serde(&result).map_err(|e| e.into())
    }

    #[wasm_bindgen]
    pub fn mul(&self, a: JsValue, b: JsValue) -> Result<JsValue, JsValue> {
        let tensor_a: RSTensor = a.into_serde().map_err(|e| e.to_string())?;
        let tensor_b: RSTensor = b.into_serde().map_err(|e| e.to_string())?;
        let result = self.instance.mul(&tensor_a, &tensor_b);
        JsValue::from_serde(&result).map_err(|e| e.into())
    }
}
