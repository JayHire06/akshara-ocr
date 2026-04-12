import * as ort from 'onnxruntime-web';

// Pre-define specific mapping options perfectly suited for WebAssembly
ort.env.wasm.numThreads = 1;
ort.env.wasm.simd = true;

/**
 * Native Canvas Preprocessor
 * Loads an arbitrary Image blob, forces High-Res 32px height downscaling,
 * Grayscales it natively, and converts it to a Float32 Tensor mapping structure.
 */
function preprocessImage(imageElement) {
    const targetHeight = 32;
    const scale = targetHeight / imageElement.height;
    const targetWidth = Math.max(1, Math.floor(imageElement.width * scale));

    const canvas = document.createElement('canvas');
    canvas.width = targetWidth;
    canvas.height = targetHeight;
    const ctx = canvas.getContext('2d');
    
    // Smooth bilinear interpolation trick inside Canvas
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(imageElement, 0, 0, targetWidth, targetHeight);

    // Extract native pixel bytes natively
    const imgData = ctx.getImageData(0, 0, targetWidth, targetHeight).data;
    const floatArray = new Float32Array(targetWidth * targetHeight);

    // Grayscale (Luma weighted logic) and normalize to 0..1
    for (let c = 0, p = 0; c < imgData.length; c += 4, p++) {
        const gray = (imgData[c] * 0.299 + imgData[c+1] * 0.587 + imgData[c+2] * 0.114) / 255.0;
        floatArray[p] = gray;
    }

    // ONNX expects (Batch=1, Channel=1, Height=32, Width=targetWidth)
    const tensor = new ort.Tensor('float32', floatArray, [1, 1, targetHeight, targetWidth]);
    return tensor;
}

/**
 * JS Greedy Decoder mapped perfectly mimicking PyTorch's native logic
 */
function greedyDecoder(logitsTensor, vocab) {
    const sequenceLen = logitsTensor.dims[0];
    const vocabSize = logitsTensor.dims[2];
    const data = logitsTensor.data;
    
    let chars = [];
    let lastCharIdx = -1;

    for (let t = 0; t < sequenceLen; t++) {
        let maxProb = -Infinity;
        let bestIdx = 0;
        
        // Find Argmax for frame
        for (let v = 0; v < vocabSize; v++) {
            const prob = data[t * vocabSize + v];
            if (prob > maxProb) {
                maxProb = prob;
                bestIdx = v;
            }
        }
        
        // Blank ID conventionally mapped to 0 natively
        if (bestIdx !== 0 && bestIdx !== lastCharIdx) {
            // Find char in vocab dict
            const charEntry = Object.entries(vocab).find(([k, v]) => v === bestIdx);
            if (charEntry) chars.push(charEntry[0]);
        }
        lastCharIdx = bestIdx;
    }
    
    return chars.join('');
}

/**
 * Core Subsystem Invoker perfectly localized 
 */
export async function runLocalInference(imageFile, vocabFile = '/vocab.json', modelFile = '/model.onnx') {
    const startTime = performance.now();
    try {
        // Load Dictionary
        const vocabRes = await fetch(vocabFile);
        if (!vocabRes.ok) throw new Error("Could not load native vocab.json. Did you export it?");
        const vocab = await vocabRes.json();

        // Instantiate ONNX Context Wrapper
        const session = await ort.InferenceSession.create(modelFile, { executionProviders: ['wasm'] });

        // Load image into native standard Image element
        const imageElement = new Image();
        const objectUrl = URL.createObjectURL(imageFile);
        
        await new Promise((resolve, reject) => {
            imageElement.onload = resolve;
            imageElement.onerror = reject;
            imageElement.src = objectUrl;
        });

        // 1. Preprocess
        const inputTensor = preprocessImage(imageElement);
        URL.revokeObjectURL(objectUrl);

        // 2. Inference Execute (WASM)
        const feeds = { input: inputTensor };
        const results = await session.run(feeds);
        const outputTensor = results.output; // Returns dynamically sized logits output!

        // 3. Decode
        const parsedText = greedyDecoder(outputTensor, vocab);
        
        const executionTime = performance.now() - startTime;
        
        return {
            status: 'done',
            text: parsedText,
            processingTimeMs: Math.round(executionTime),
        };
    } catch (e) {
        console.error("Native ONNX Local Extraction Error:", e);
        throw new Error(`Local inference failed natively: ${e.message}`);
    }
}
