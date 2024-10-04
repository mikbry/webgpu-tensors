#[cfg(target_arch = "wasm32")]
use wasm_bindgen::{prelude::wasm_bindgen, JsValue};
#[cfg(target_arch = "wasm32")]
use webgpu_tensors::{RSTensors, Tensors, RSTensor, TensorOptions};
use serde::{Serialize, Deserialize};
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
    pub fn randn(&self, shape: Vec<usize>, options: JsValue) -> Result<JsValue, 
JsValue> {
        let js_options: JSTensorOptions = 
options.into_serde().unwrap_or(JSTensorOptions { dtype: None, device: None });
        let tensor_options = TensorOptions {
            dtype: js_options.dtype.map(|d| d.parse().unwrap()),
            device: js_options.device.map(|d| d.parse().unwrap()),
        };
        let tensor = self.instance.randn(shape, Some(tensor_options));
        Ok(JsValue::from_serde(&tensor).unwrap())
    }

    #[wasm_bindgen]
    pub fn matmul(&self, a: JsValue, b: JsValue) -> Result<JsValue, JsValue> {
        let tensor_a: RSTensor = a.into_serde().unwrap();
        let tensor_b: RSTensor = b.into_serde().unwrap();
        let result = self.instance.matmul(&tensor_a, &tensor_b);
        Ok(JsValue::from_serde(&result).unwrap())
    }

    #[wasm_bindgen]
    pub fn mul(&self, a: JsValue, b: JsValue) -> Result<JsValue, JsValue> {
        let tensor_a: RSTensor = a.into_serde().unwrap();
        let tensor_b: RSTensor = b.into_serde().unwrap();
        let result = self.instance.mul(&tensor_a, &tensor_b);
        Ok(JsValue::from_serde(&result).unwrap())
    }
}
