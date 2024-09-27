async function main() {
    if (!navigator.gpu) {
        console.log("WebGPU not supported on this browser.");
        return;
    }

    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
        console.log("Failed to get GPU adapter.");
        return;
    }
    const device = await adapter.requestDevice();

    // Define matrices
    const matrixA = new Float32Array([1, 2, 3, 4, 5, 6]);
    const matrixB = new Float32Array([7, 8, 9, 10, 11, 12]);
    const resultMatrix = new Float32Array(4);

    // Create buffers
    const bufferA = device.createBuffer({
        size: matrixA.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const bufferB = device.createBuffer({
        size: matrixB.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const bufferResult = device.createBuffer({
        size: resultMatrix.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    // Write data to buffers
    device.queue.writeBuffer(bufferA, 0, matrixA);
    device.queue.writeBuffer(bufferB, 0, matrixB);

    // Create compute pipeline
    const shaderModule = device.createShaderModule({
        code: `
            @group(0) @binding(0) var<storage, read> a: array<f32>;
            @group(0) @binding(1) var<storage, read> b: array<f32>;
            @group(0) @binding(2) var<storage, read_write> result: array<f32>;

            @compute @workgroup_size(1)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                let row = global_id.x;
                let col = global_id.y;
                var sum = 0.0;
                for (var i = 0u; i < 3u; i = i + 1u) {
                    sum = sum + a[row * 3u + i] * b[i * 2u + col];
                }
                result[row * 2u + col] = sum;
            }
        `
    });

    const computePipeline = device.createComputePipeline({
        layout: 'auto',
        compute: {
            module: shaderModule,
            entryPoint: 'main'
        }
    });

    // Create bind group
    const bindGroup = device.createBindGroup({
        layout: computePipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: bufferA } },
            { binding: 1, resource: { buffer: bufferB } },
            { binding: 2, resource: { buffer: bufferResult } },
        ]
    });

    // Create command encoder and pass
    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(computePipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(2, 2);
    passEncoder.end();

    // Get result
    const gpuReadBuffer = device.createBuffer({
        size: resultMatrix.byteLength,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });
    commandEncoder.copyBufferToBuffer(bufferResult, 0, gpuReadBuffer, 0, resultMatrix.byteLength);
    device.queue.submit([commandEncoder.finish()]);

    // Read result
    await gpuReadBuffer.mapAsync(GPUMapMode.READ);
    const arrayBuffer = gpuReadBuffer.getMappedRange();
    const result = new Float32Array(arrayBuffer);
    console.log("Result matrix:", result);
    gpuReadBuffer.unmap();
}

main();
