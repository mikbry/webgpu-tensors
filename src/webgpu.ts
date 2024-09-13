export interface WebGPUInstance {
    canvas: HTMLCanvasElement;
    context: GPUCanvasContext;
    device: GPUDevice;
    format: GPUTextureFormat;

    renderTarget: GPUTexture | undefined;
    renderTargetView: GPUTextureView | undefined;

    render?: {
        pass: GPURenderPassEncoder;
        encoder: GPUCommandEncoder;
    }
}

function getDevicePixelContentBoxSize(entry: ResizeObserverEntry) {
    // Safari does not support devicePixelContentBoxSize
    if (entry.devicePixelContentBoxSize) {
        return {
            width: entry.devicePixelContentBoxSize[0].inlineSize,
            height: entry.devicePixelContentBoxSize[0].blockSize,
        };
    } else {
        // These values not correct but they're as close as you can get in Safari
        return {
            width: entry.contentBoxSize[0].inlineSize * devicePixelRatio,
            height: entry.contentBoxSize[0].blockSize * devicePixelRatio,
        };
    }
}

function resizeCanvas(entry: ResizeObserverEntry, canvas: HTMLCanvasElement, device: GPUDevice) {
    const { width, height } = getDevicePixelContentBoxSize(entry);

    // A size of 0 will cause an error when we call getCurrentTexture.
    // A size > maxTextureDimension2D will also an error when we call getCurrentTexture.
    canvas.width = Math.max(1, Math.min(width, device.limits.maxTextureDimension2D));
    canvas.height = Math.max(1, Math.min(height, device.limits.maxTextureDimension2D));
}

async function webGPUInit(canvasId?: string): Promise<WebGPUInstance> {
    if (!navigator.gpu) {
        throw new Error("WebGPU not supported on this browser.");
    }

    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
        throw new Error("No appropriate GPUAdapter found.");
    }

    const device = await adapter.requestDevice();

    const canvas = document.querySelector<HTMLCanvasElement>(canvasId ? `#${canvasId}` : 'canvas');
    if (!canvas) {
        throw new Error("No appropriate Canvas found.");
    }
    // resizeCanvas(canvas, device)
    const context = canvas.getContext("webgpu");
    if (!context) {
        throw new Error("No appropriate WebGPU Context found.");
    }
    const format = navigator.gpu.getPreferredCanvasFormat();
    context.configure({
        device,
        format,
        alphaMode: 'premultiplied',
    });


    return { canvas, context, device, format, renderTarget: undefined, renderTargetView: undefined };
}

function initRendering(instance: WebGPUInstance, renderLoop: () => {}) {
    const { canvas, device } = instance;
    const observer = new ResizeObserver(([entry]) => {
        resizeCanvas(entry, canvas, device);
        renderLoop();
    });
    observer.observe(canvas);
}

function startRendering(instance: WebGPUInstance, backgroundColor?: GPUColor): WebGPUInstance {
    const { context, device, canvas, format } = instance;
    let { renderTarget, renderTargetView } = instance;
    const encoder = device.createCommandEncoder();
    const currentWidth = canvas.clientWidth * devicePixelRatio;
    const currentHeight = canvas.clientHeight * devicePixelRatio;
    if (
        (currentWidth !== canvas.width ||
            currentHeight !== canvas.height ||
            !renderTargetView) &&
        currentWidth &&
        currentHeight
    ) {
        if (renderTarget !== undefined) {
            // Destroy the previous render target
            renderTarget.destroy();
        }

        // Setting the canvas width and height will automatically resize the textures returned
        // when calling getCurrentTexture() on the context.
        canvas.width = currentWidth;
        canvas.height = currentHeight;

        // Resize the multisampled render target to match the new canvas size.
        renderTarget = device.createTexture({
            label: 'renderTarget',
            size: [canvas.width, canvas.height],
            format: format,
            usage: GPUTextureUsage.RENDER_ATTACHMENT,
            sampleCount: 4,
        });

        renderTargetView = renderTarget.createView();
    }

    if (!renderTarget || !renderTargetView) {
        throw new Error("RenderTarget not available.");
    }
    const pass = encoder.beginRenderPass({
        colorAttachments: [{
            view: renderTargetView,
            resolveTarget: context.getCurrentTexture().createView(),
            loadOp: "clear",
            clearValue: backgroundColor || { r: 0, g: 0, b: 0.4, a: 1 },
            storeOp: "store",
        }]
    });

    return { ...instance, renderTarget, renderTargetView, render: { pass, encoder } };
}

function endRendering(instance: WebGPUInstance) {
    const { device, render } = instance;
    if (!render) {
        throw new Error("Renderer not found.");
    }
    const { encoder } = render;
    device.queue.submit([encoder.finish()]);

    return { ...instance, render: undefined };
}

export { webGPUInit as webgpuInit, initRendering, startRendering, endRendering };
