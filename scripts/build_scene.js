// Builds an accurate LeNet-5 architecture VRML + WebGL scene:
//  - 3D graphic of the LeNet-5 handwritten-digit classifier
//  - input is an image, output is 10 classes (0-9), shown as 10 distinct
//    nodes (not one box merely sized "10")
//  - every layer (including the input image) is shown as a box whose
//    spatial array size (X/Y) and channel depth (Z) are both quantitatively
//    accurate - C1/S2/C3/S4 further split into one box PER CHANNEL, since a
//    layer's C output channels are C independently-varying feature maps,
//    not one blended average
//  - each weight-bearing layer's box is textured with a grayscale pixel
//    image of its own REAL PRETRAINED weights (scripts/train_lenet5.py: an
//    actual LeNet-5 trained on real MNIST, ~98% test accuracy) - not
//    random init, and not decorative: the exact same numbers drive both
//    the texture AND the real conv/pool/fc forward pass computed live in
//    the browser (lib/vrml.js's runInference())
const path = require('path');
const fs = require('fs');

const ROOT_DIR = path.dirname(__dirname);
const vrml = require(path.join(ROOT_DIR, 'lib/vrml'));

const OUTPUT_DIR = path.join(ROOT_DIR, 'output');
const TRAINED_WEIGHTS_PATH = path.join(ROOT_DIR, 'data/lenet5_trained_weights.json');
const SAMPLE_MANIFEST_PATH = path.join(ROOT_DIR, 'data/mnist_sample_manifest.json');

if (!fs.existsSync(TRAINED_WEIGHTS_PATH)) {
  console.error('Missing ' + TRAINED_WEIGHTS_PATH + ' - run scripts/train_lenet5.py first.');
  process.exit(1);
}
if (!fs.existsSync(SAMPLE_MANIFEST_PATH)) {
  console.error('Missing ' + SAMPLE_MANIFEST_PATH + ' - run scripts/export_mnist_samples.py first.');
  process.exit(1);
}

const trained = JSON.parse(fs.readFileSync(TRAINED_WEIGHTS_PATH, 'utf8'));
const sampleImages = JSON.parse(fs.readFileSync(SAMPLE_MANIFEST_PATH, 'utf8'));
// The scene's starting INPUT image - one of the exported MNIST test
// samples (relative URL, resolved against wherever output/ is served
// from). The re-infer button then swaps between all of sampleImages.
const INPUT_IMAGE_URL = sampleImages[0].url;

const SPATIAL_SCALE = 0.12;   // pixels -> world units (X/Y footprint)
const FC_HEIGHT_SCALE = 0.03; // neuron count -> world units (vertical rod height)
const FC_THIN = 0.4;          // FC layer's thin fixed X/Z
const GAP = 1.0;
const NODE_RADIUS = 0.2;      // one output node's sphere radius
const NODE_GAP = 0.15;        // vertical gap between stacked output nodes

function squareGrid(n) {
  const w = Math.max(1, Math.round(Math.sqrt(n)));
  const h = Math.max(1, Math.ceil(n / w));
  return [w, h];
}

const shapes = [];
let cursorX = 0;
let prevHalfX = 0;

// PER_CHANNEL_THICKNESS/CHANNEL_GAP: a conv/pool layer's C output channels
// are C independently-varying feature maps, not one blended signal - a
// single box averaging them together would hide real per-channel
// differences (one filter reacts strongly to this digit's strokes,
// another barely at all) behind one shared brightness value. This instead
// stacks C thin slabs along Z (still centered on the layer's own X slot,
// so the rest of the row's layout is unaffected - only sizeX drives
// horizontal spacing), one slab per channel, each with its own real
// weight slice (conv) or its own decorative noise (pooling) and its own
// independently-computed activation glow.
const PER_CHANNEL_THICKNESS = 0.09;
const CHANNEL_GAP = 0.025;

function pushChannelStack(name, w, h, channelCount, color, channelWeightsFn, isInput) {
  const sizeX = w * SPATIAL_SCALE;
  const heightY = h * SPATIAL_SCALE;
  const halfX = sizeX / 2;
  const x = shapes.length === 0 ? 0 : cursorX + prevHalfX + GAP + halfX;
  cursorX = x;
  prevHalfX = halfX;

  const step = PER_CHANNEL_THICKNESS + CHANNEL_GAP;
  const totalDepth = channelCount * step - CHANNEL_GAP;
  const indices = [];
  for (let ch = 0; ch < channelCount; ch++) {
    const z = -totalDepth / 2 + ch * step + PER_CHANNEL_THICKNESS / 2;
    const weights = channelWeightsFn ? channelWeightsFn(ch) : null;
    const shape = {
      type: 'box',
      color,
      size: sizeX,
      heightY,
      thickness: PER_CHANNEL_THICKNESS,
      x,
      y: heightY / 2,
      z,
      interactive: false,
      // Only the first channel carries the layer's own name (HUD button)
      // and dimension label - one button/label per LAYER, not per
      // channel, keeps the HUD from exploding into near-duplicate entries.
      ...(ch === 0 ? { name: `${name} ${w}x${h}x${channelCount}`, label3d: `${w}x${h}x${channelCount}` } : { showButton: false }),
    };
    if (weights) {
      // Real per-channel kernel weights, sized to the kernel's own true
      // shape (e.g. 5x5=25 for a single input channel) - an exact 1
      // texel = 1 real weight image, not cyclic tiling of a handful of
      // real values across hundreds of texels.
      shape.texture = 'weights';
      shape.weightGrid = squareGrid(weights.length);
      shape.weights = weights;
    } else if (isInput) {
      // INPUT: channelCount=1 (grayscale), no weights (it's the raw
      // image, not a learned layer) - texture it with the current input
      // digit image instead. pickRandomInputImage() (lib/vrml.js) swaps
      // this same shape's texture on every re-infer click.
      shape.textureUrl = INPUT_IMAGE_URL;
    } else {
      // Pooling channel: no real weights - decorative noise, independently
      // random per channel, sized to the spatial footprint.
      shape.texture = 'weights';
      shape.weightGrid = squareGrid(w * h);
    }
    shapes.push(shape);
    indices.push(shapes.length - 1);
  }
  return indices;
}

function pushFcBox(name, n, color, weights) {
  const sizeX = FC_THIN;
  const heightY = Math.max(0.4, n * FC_HEIGHT_SCALE);
  const thickness = FC_THIN;
  const halfX = sizeX / 2;
  const x = cursorX + prevHalfX + GAP + halfX;
  cursorX = x;
  prevHalfX = halfX;
  shapes.push({
    type: 'box',
    name: `${name} FC${n}`,
    label3d: `${n}`,
    color,
    size: sizeX,
    heightY,
    thickness,
    x,
    y: heightY / 2,
    z: 0,
    texture: 'weights',
    weightGrid: squareGrid(n),
    weights,
    interactive: false,
  });
  return shapes.length - 1;
}

// ---- real pretrained LeNet-5 weights (data/lenet5_trained_weights.json) ----
const c1Weights = trained.c1.weights;
const c1Bias = trained.c1.bias;
const c3Weights = trained.c3.weights;
const c3Bias = trained.c3.bias;
const c5Weights = trained.c5.weights;
const c5Bias = trained.c5.bias;
const f6Weights = trained.f6.weights;
const f6Bias = trained.f6.bias;
const outputBias = trained.output.bias;
const outputNodeWeights = trained.output.weights;

// ---- build shapes in forward-pass order ----
// INPUT (channelCount=1) and C1/S2/C3/S4 all go through the same
// pushChannelStack - INPUT is just the degenerate N=1 case. Each conv
// channel's own kernel slice is a contiguous chunk of the real trained
// weight array (PyTorch's (outC,inC,kH,kW) layout is already contiguous
// per output channel: channel ch occupies [ch*kernelSize, (ch+1)*kernelSize)).
const idxInputList = pushChannelStack('INPUT', 32, 32, 1, [0.82, 0.82, 0.82], null, true);
const idxInput = idxInputList[0];
const c1KernelSize = 1 * 5 * 5;
const idxC1List = pushChannelStack('C1', 28, 28, 6, [0.35, 0.55, 0.95], (ch) => c1Weights.slice(ch * c1KernelSize, (ch + 1) * c1KernelSize));
const idxS2List = pushChannelStack('S2', 14, 14, 6, [0.95, 0.65, 0.25], null);
const c3KernelSize = 6 * 5 * 5;
const idxC3List = pushChannelStack('C3', 10, 10, 16, [0.30, 0.50, 0.90], (ch) => c3Weights.slice(ch * c3KernelSize, (ch + 1) * c3KernelSize));
const idxS4List = pushChannelStack('S4', 5, 5, 16, [0.95, 0.65, 0.25], null);
const idxC5 = pushFcBox('C5', 120, [0.35, 0.85, 0.45], c5Weights);
const idxF6 = pushFcBox('F6', 84, [0.35, 0.85, 0.45], f6Weights);

const outputColor = [0.95, 0.30, 0.30];
const outputIndices = [];
for (let d = 0; d < 10; d++) {
  shapes.push({
    type: 'sphere',
    name: `OUTPUT ${d}`,
    label3d: `${d}`,
    labelSide: 'right',
    showButton: false,
    color: outputColor,
    size: NODE_RADIUS * 2,
    x: cursorX + prevHalfX + GAP + NODE_RADIUS,
    y: NODE_RADIUS + d * (NODE_RADIUS * 2 + NODE_GAP),
    z: 0,
    texture: 'weights',
    weightGrid: squareGrid(84),
    weights: outputNodeWeights[d],
    interactive: false,
  });
  outputIndices.push(shapes.length - 1);
}

const scene = {
  floor: true,
  background: { top: [0.05, 0.07, 0.10], bottom: [0.16, 0.19, 0.25], vertical: true },
  camera: { theta: 0.42, phi: 0.24, radiusScale: 0.55 },
  shapes,
  inference: {
    inputImageUrl: INPUT_IMAGE_URL,
    inputSize: 32,
    trained: true,
    inputShapeIndex: idxInput,
    sampleImages,
    layers: [
      { type: 'conv', shapeIndices: idxC1List, inC: 1, outC: 6, k: 5, bias: c1Bias },
      { type: 'avgpool', k: 2, shapeIndices: idxS2List },
      { type: 'conv', shapeIndices: idxC3List, inC: 6, outC: 16, k: 5, bias: c3Bias },
      { type: 'avgpool', k: 2, shapeIndices: idxS4List },
      { type: 'fc', shapeIndex: idxC5, inN: 400, outN: 120, bias: c5Bias },
      { type: 'fc', shapeIndex: idxF6, inN: 120, outN: 84, bias: f6Bias },
    ],
    output: { nodeShapeIndices: outputIndices, bias: outputBias },
  },
};

fs.mkdirSync(OUTPUT_DIR, { recursive: true });
// No baseUrl - the .wrl's embedded absolute-URL fallback is only needed by
// legacy standalone VRML browsers reading the .wrl outside this page; the
// primary experience (the WebGL viewer HTML) resolves textureUrl as a
// plain relative path against wherever output/ is served from.
fs.writeFileSync(path.join(OUTPUT_DIR, 'lenet5.wrl'), vrml.generateWrl(scene, ''));
fs.writeFileSync(path.join(OUTPUT_DIR, 'lenet5-viewer.html'), vrml.generateViewerDocument(scene, 'LeNet-5 3D 실시간 추론'));

console.log('OK', { shapes: scene.shapes.length, outputIndices, inputImage: INPUT_IMAGE_URL });
console.log('Wrote output/lenet5.wrl and output/lenet5-viewer.html');
console.log('Serve ./output over HTTP and open lenet5-viewer.html (file:// will hit canvas security restrictions on the texture images).');
