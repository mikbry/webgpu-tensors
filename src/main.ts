import './style.css'
import { startRendering, endRendering, webgpuInit, WebGPUInstance } from './webgpu'

let instance = await webgpuInit();

const vertices = new Float32Array([
  //   X,    Y,
  -0.8, -0.8, // Triangle 1 (Blue)
  0.8, -0.8,
  0.8, 0.8,

  -0.8, -0.8, // Triangle 2 (Red)
  0.8, 0.8,
  -0.8, 0.8,

]);

const vertexBufferLayout: GPUVertexBufferLayout = {
  arrayStride: 8,
  attributes: [{
    format: "float32x2",
    offset: 0,
    shaderLocation: 0, // Position, see vertex shader
  }],
};


const { device, format } = instance;
const vertexBuffer = device.createBuffer({
  label: "Cell vertices",
  size: vertices.byteLength,
  usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
});
device.queue.writeBuffer(vertexBuffer, /*bufferOffset=*/0, vertices);

const cellShaderModule = device.createShaderModule({
  label: "Cell shader",
  code: `
    @vertex
    fn vertexMain(@location(0) pos: vec2f) ->
      @builtin(position) vec4f {
      return vec4f(pos, 0, 1);
    }

    @fragment
    fn fragmentMain() -> @location(0) vec4f {
      return vec4f(1, 0, 0, 1); // (Red, Green, Blue, Alpha)
    }
  `
});

const cellPipeline = device.createRenderPipeline({
  label: "Cell pipeline",
  layout: "auto",
  vertex: {
    module: cellShaderModule,
    entryPoint: "vertexMain",
    buffers: [vertexBufferLayout]
  },
  fragment: {
    module: cellShaderModule,
    entryPoint: "fragmentMain",
    targets: [{
      format
    }]
  },
  multisample: {
    count: 4,
  },
});

function render(instance: WebGPUInstance) {
  if (!instance.render) {
    throw new Error("Renderer not found.");
  }
  const { pass } = instance.render;
  pass.setPipeline(cellPipeline);
  pass.setVertexBuffer(0, vertexBuffer);
  pass.draw(vertices.length / 2);
  pass.end();

}


instance = startRendering(instance, { r: 0.6, g: 0, b: 0.4, a: 1 });
render(instance);
instance = endRendering(instance);