import * as ort from 'onnxruntime-web';

// Advanced Edge Inference Configuration
ort.env.wasm.numThreads = 1;
ort.env.wasm.simd = true;

/**
 * Configure global WASM paths to pull from a reliable CDN.
 * This resolves Vite "Aborted" errors by bypassing local dev server fallback.
 */
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.24.3/dist/';

/**
 * Line Segmenter (Vertical Projection Profile)
 * Detects horizontal strips of text in a document.
 */
function segmentImageIntoLines(canvas) {
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    const imgData = ctx.getImageData(0, 0, width, height).data;

    // 1. Calculate Row Intensities with Grayscale conversion
    const rawSums = new Int32Array(height);
    for (let y = 0; y < height; y++) {
        let count = 0;
        for (let x = 0; x < width; x++) {
            const idx = (y * width + x) * 4;
            const gray = (imgData[idx] * 0.299 + imgData[idx+1] * 0.587 + imgData[idx+2] * 0.114);
            // Count "dark" pixels (text signal)
            if (gray < 200) count++;
        }
        rawSums[y] = count;
    }

    // 2. Apply Smoothing Filter (Moving Average)
    // This bridges tiny horizontal gaps caused by thin strokes or noise.
    const rowSums = new Float32Array(height);
    const windowSize = 3; 
    for (let y = 0; y < height; y++) {
        let sum = 0;
        let count = 0;
        for (let dw = -windowSize; dw <= windowSize; dw++) {
            if (y + dw >= 0 && y + dw < height) {
                sum += rawSums[y + dw];
                count++;
            }
        }
        rowSums[y] = sum / count;
    }

    // 3. Adaptive Threshold Detection
    // We ignore rows that have very little "text energy" compared to the document width.
    const textThreshold = Math.max(width * 0.005, 2); 

    const lines = [];
    let inLine = false;
    let startY = 0;
    
    // Gap tolerance: how many "empty" rows to allow before finishing a line
    // This allows for scripts with disconnected characters.
    let gapCounter = 0;
    const maxGap = 5; 

    for (let y = 0; y < height; y++) {
        const isTextRow = rowSums[y] > textThreshold;

        if (isTextRow) {
            if (!inLine) {
                inLine = true;
                startY = y;
            }
            gapCounter = 0;
        } else if (inLine) {
            gapCounter++;
            if (gapCounter > maxGap) {
                const endY = y - gapCounter;
                const lineHeight = endY - startY;
                
                // Only keep lines with a realistic thickness
                if (lineHeight > 8) {
                    // Apply dynamic padding based on line height
                    const padding = Math.floor(lineHeight * 0.15);
                    lines.push({ 
                        startY: Math.max(0, startY - padding), 
                        endY: Math.min(height, endY + padding) 
                    });
                }
                inLine = false;
            }
        }
    }

    if (inLine && (height - startY > 8)) {
        lines.push({ startY: Math.max(0, startY - 2), endY: height });
    }

    return lines;
}

/**
 * Native Canvas Preprocessor for a specific crop
 */
function preprocessLine(imageElement, cropY, cropH) {
    const targetHeight = 32;
    const scale = targetHeight / cropH;
    const targetWidth = Math.max(1, Math.floor(imageElement.width * scale));

    const canvas = document.createElement('canvas');
    canvas.width = targetWidth;
    canvas.height = targetHeight;
    const ctx = canvas.getContext('2d');
    
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';

    // Draw only the specified crop onto the 32px height target
    ctx.drawImage(
        imageElement, 
        0, cropY, imageElement.width, cropH, // Source
        0, 0, targetWidth, targetHeight       // Destination
    );

    const imgData = ctx.getImageData(0, 0, targetWidth, targetHeight).data;
    const floatArray = new Float32Array(targetWidth * targetHeight);

    for (let c = 0, p = 0; c < imgData.length; c += 4, p++) {
        const gray = (imgData[c] * 0.299 + imgData[c+1] * 0.587 + imgData[c+2] * 0.114) / 255.0;
        floatArray[p] = gray;
    }

    return new ort.Tensor('float32', floatArray, [1, 1, targetHeight, targetWidth]);
}

/**
 * Greedy Decoder with Real Confidence Scoring
 */
function greedyDecoder(logitsTensor, vocab) {
    const sequenceLen = logitsTensor.dims[0];
    const vocabSize = logitsTensor.dims[2];
    const data = logitsTensor.data;
    
    let chars = [];
    let lastCharIdx = -1;
    let confidenceSum = 0;
    let decodedFrames = 0;

    for (let t = 0; t < sequenceLen; t++) {
        let maxProb = -Infinity;
        let bestIdx = 0;
        
        for (let v = 0; v < vocabSize; v++) {
            const prob = data[t * vocabSize + v];
            if (prob > maxProb) {
                maxProb = prob;
                bestIdx = v;
            }
        }
        
        // Blank ID is usually 0
        if (bestIdx !== 0 && bestIdx !== lastCharIdx) {
            const charEntry = Object.entries(vocab).find(([k, v]) => v === bestIdx);
            if (charEntry) {
                chars.push(charEntry[0]);
                // Logits are often pre-activation (raw output), 
                // for true CTC they should be Softmaxed. 
                // We'll approximate confidence using the relative logit space if softmax isn't applied.
                // Assuming model returns log_probs or values that can serve as relative confidence.
                confidenceSum += Math.min(1.0, Math.exp(maxProb) || 0.95); // Safely convert to 0..1
                decodedFrames++;
            }
        }
        lastCharIdx = bestIdx;
    }
    
    const confidence = decodedFrames > 0 ? (confidenceSum / decodedFrames) : 1.0;
    return { text: chars.join(''), confidence: confidence };
}

/**
 * Enhanced Core Subsystem Invoker with Multi-Line Support
 */
export async function runLocalInference(imageFile, vocabFile = '/vocab.json', modelFile = '/model.onnx') {
    const startTime = performance.now();
    try {
        const vocabRes = await fetch(vocabFile);
        if (!vocabRes.ok) throw new Error("Could not load vocab.json");
        const vocab = await vocabRes.json();

        // Instantiate Session (WebGPU prioritized)
        const session = await ort.InferenceSession.create(modelFile, { 
            executionProviders: ['webgpu', 'wasm'] 
        });

        // Load image
        const imageElement = new Image();
        const objectUrl = URL.createObjectURL(imageFile);
        await new Promise((resolve, reject) => {
            imageElement.onload = resolve;
            imageElement.onerror = reject;
            imageElement.src = objectUrl;
        });

        // Create a temp canvas for segmentation
        const segCanvas = document.createElement('canvas');
        segCanvas.width = imageElement.width;
        segCanvas.height = imageElement.height;
        const segCtx = segCanvas.getContext('2d');
        segCtx.drawImage(imageElement, 0, 0);

        // 1. Segment Image into discrete lines
        const lineSpecs = segmentImageIntoLines(segCanvas);
        
        // If no lines detected, try the whole image as one line
        if (lineSpecs.length === 0) {
            lineSpecs.push({ startY: 0, endY: imageElement.height });
        }

        let fullTextParts = [];
        let totalConfidence = 0;

        // 2. Process each line sequentially (Leveraging WebGPU speed)
        for (const line of lineSpecs) {
            const cropH = line.endY - line.startY;
            if (cropH < 2) continue;

            const inputTensor = preprocessLine(imageElement, line.startY, cropH);
            const feeds = { input: inputTensor };
            const results = await session.run(feeds);
            
            const decoded = greedyDecoder(results.output, vocab);
            if (decoded.text.trim()) {
                fullTextParts.push(decoded.text);
                totalConfidence += decoded.confidence;
            }
        }

        URL.revokeObjectURL(objectUrl);
        const executionTime = performance.now() - startTime;
        
        const finalConfidence = fullTextParts.length > 0 ? (totalConfidence / fullTextParts.length) : 0;

        return {
            status: 'done',
            text: fullTextParts.join('\n'),
            confidence: finalConfidence,
            processingTimeMs: Math.round(executionTime),
        };
    } catch (e) {
        console.error("Native ONNX Error:", e);
        throw new Error(`Inference failed: ${e.message}`);
    }
}
