#[cfg(target_arch = "wasm32")]
use wasm_bindgen::{prelude::wasm_bindgen, JsCast, UnwrapThrowExt};
#[cfg(target_arch = "wasm32")]
use webgpu_tensors::{RSTensors, Tensors};

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

    #[inline]
    pub fn randn(&mut self) {
        // TODO
    }
}