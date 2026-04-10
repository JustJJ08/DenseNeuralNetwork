#include "network.hpp"

const char* networkKernelSource = R"CLC(
#define ACTIVATION_LINEAR 0
#define ACTIVATION_RELU 1
#define ACTIVATION_LEAKY_RELU 2
#define ACTIVATION_SIGMOID 3
#define ACTIVATION_TANH 4
#define ACTIVATION_SOFTMAX 5

#ifndef ACTIVATION
#define ACTIVATION ACTIVATION_LEAKY_RELU
#endif

inline float sigmoid(float x) {
if (x >= 0.0f)
return 1.0f / (1.0f + exp(-x));
else {
float z = exp(x);
return z / (1.0f + z);
}
}

inline float activate(float x) {
#if ACTIVATION == ACTIVATION_LINEAR
return x;
#elif ACTIVATION == ACTIVATION_RELU
return fmax(0.0f, x);
#elif ACTIVATION == ACTIVATION_LEAKY_RELU
return fmax(0.1f * x, x);
#elif ACTIVATION == ACTIVATION_SIGMOID
return sigmoid(x);
#elif ACTIVATION == ACTIVATION_TANH
return tanh(x);
#elif ACTIVATION == ACTIVATION_SOFTMAX
return x;
#endif
}

inline float activationDerivative(float x) {
#if ACTIVATION == ACTIVATION_LINEAR
return 1;
#elif ACTIVATION == ACTIVATION_RELU
return (x < 0) ? 0 : 1;
#elif ACTIVATION == ACTIVATION_LEAKY_RELU
return (x < 0) ? 0.1 : 1;
#elif ACTIVATION == ACTIVATION_SIGMOID
float s = sigmoid(x);
return s * (1 - s);
#elif ACTIVATION == ACTIVATION_TANH
float t = tanh(x);
return 1.0f - t * t;
#elif ACTIVATION == ACTIVATION_SOFTMAX
return 1;
#endif
}

__kernel void softmax(__global const float* input, __global float* output, int neurons) {
int i = get_global_id(0);

float maxVal = input[i * neurons];
for (int x = 1; x < neurons; x++) { if (input[i * neurons + x] > maxVal) maxVal = input[i * neurons + x]; }

float sumExp = 0.0f;
for (int x = 0; x < neurons; x++) {
float e = exp(input[i * neurons + x] - maxVal);
output[i * neurons + x] = e;
sumExp += e;
}

for (int x = 0; x < neurons; x++) { output[i * neurons + x] /= sumExp; }
}

__kernel void copy_softmax_to_activations(__global const float* probs, __global float* activations) {
int idx = get_global_id(0);
activations[idx] = probs[idx];
}

__kernel void add_bias_and_activate(__global const float* BIASES, __global float* OUT, int neurons) {
int batch = get_global_id(0);
int neuron = get_global_id(1);

OUT[batch * neurons + neuron] = activate(OUT[batch * neurons + neuron] + BIASES[neuron]);
}

__kernel void get_activations_and_zs(__global const float* BIASES, __global float* OUTPUTBUFFER, __global float* ACTIVATIONS, __global float* ZS, __global uchar* mask, int NEURONS, float dropoutRate, uint seed) {
const int batch = get_global_id(0);
const int neuron = get_global_id(1);
const int out = batch * NEURONS + neuron;

float z = OUTPUTBUFFER[out] + BIASES[neuron];
float a = activate(z);
float a_d = a;

uint state = seed + out;
state = 1664525u * state + 1013904223u;
float rand = (float) (state & 0x00FFFFFF) / (float) 0x01000000;

if (rand < dropoutRate) a_d = 0.0f;
else a_d = a / (1.0f - dropoutRate);

ACTIVATIONS[out] = a_d;
OUTPUTBUFFER[out] = a_d;
ZS[out] = z;
mask[out] = rand >= dropoutRate;
}

__kernel void compute_output_deltas(__global const float* activations, __global const float* zs, __global const float* expected_output, __global float* deltas) {
int i = get_global_id(0);
deltas[i] = (activations[i] - expected_output[i]) * activationDerivative(zs[i]);
}

__kernel void compute_output_deltas_single_action_reinforcement(__global const float* activations, __global const float* chosen_action, __global float* deltas, __global const float* rewards, int neurons) {
int i = get_global_id(0);
int batch = i / neurons;

if (chosen_action[i] == 0) { deltas[i] = activations[i] * rewards[batch]; }
else { deltas[i] = (activations[i] - 1) * rewards[batch]; }
}

__kernel void compute_output_deltas_output_vector_reinforcement(
__global const float* activations, __global const float* zs, __global const float* chosenActivation, __global float* deltas, __global const float* rewards, float sigmaSquared, int neurons) {
int i = get_global_id(0);
int batch = i / neurons;

deltas[i] = rewards[batch] * ((chosenActivation[i] - activations[i]) / sigmaSquared) * activationDerivative(zs[i]);
}

__kernel void compute_hidden_deltas(__global const float* zs, __global float* deltas, __global uchar* mask, float dropoutRate, int neurons) {
int id = get_global_id(0);
deltas[id] *= activationDerivative(zs[id]) * (float)mask[id];
}

__kernel void average_deltas_and_update_network_biases(__global const float* biasGradients, __global float* biases, __global float* expAvgB, __global float* expAvgSqB, 
float learningRate, float BETA1, float BETA2, float pcBETA1, float pcBETA2, float pcBeta1PowT, float pcBeta2PowT, float EPSILON) {
int neuron = get_global_id(0);
float g = biasGradients[neuron];

expAvgB[neuron] = BETA1 * expAvgB[neuron] + pcBETA1 * g;
expAvgSqB[neuron] = BETA2 * expAvgSqB[neuron] + pcBETA2 * g * g;

float mHat = expAvgB[neuron] / pcBeta1PowT;
float vHat = expAvgSqB[neuron] / pcBeta2PowT;

biases[neuron] -= learningRate * mHat / (sqrt(vHat) + EPSILON);
}

__kernel void average_deltas_and_update_network_weights(__global const float* weightGradients, __global float* weights, __global float* expAvgW, __global float* expAvgSqW, 
float learningRate, float BETA1, float BETA2, float pcBETA1, float pcBETA2, float pcBeta1PowT, float pcBeta2PowT, float EPSILON, float WEIGHT_DECAY) {
int neuron = get_global_id(0);
int previous = get_global_id(1);
int idx = neuron * get_global_size(1) + previous;

float g = weightGradients[idx];

expAvgW[idx] = BETA1 * expAvgW[idx] + pcBETA1 * g;
expAvgSqW[idx] = BETA2 * expAvgSqW[idx] + pcBETA2 * g * g;

float mHat = expAvgW[idx] / pcBeta1PowT;
float vHat = expAvgSqW[idx] / pcBeta2PowT;

weights[idx] -= learningRate * (mHat / (sqrt(vHat) + EPSILON));
weights[idx] -= learningRate * WEIGHT_DECAY * weights[idx];
}

__kernel void average_deltas_and_update_network_biases_no_optimizer(__global const float* biasGradients, __global float* biases, float learningRate) {
int neuron = get_global_id(0);
biases[neuron] -= learningRate * biasGradients[neuron];
}

__kernel void average_deltas_and_update_network_weights_no_optimizer(__global const float* weightGradients, __global float* weights, float learningRate) {
int neuron = get_global_id(0);
int previous = get_global_id(1);
int idx = neuron * get_global_size(1) + previous;
weights[idx] -= learningRate * weightGradients[idx];
}
)CLC";

Network::Network(const std::vector<int>& neuronsInLayers, Activation hiddenActivationFunction, Activation outputActivationFunction, float learningRate, bool fastMath, bool optimizer, bool printDeviceInfo, int usedDeviceIndex) {
  this -> learningRate = learningRate;
  useOptimizer = optimizer;

  if (neuronsInLayers.size() < 2) throw std::invalid_argument("You can't create a network with only one layer");

  std::vector<Activation> activationFunctions(neuronsInLayers.size() - 1, hiddenActivationFunction);
  activationFunctions.back() = outputActivationFunction;

  // Get the GPU stuff
  device = getDevice(printDeviceInfo, usedDeviceIndex);
  context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
  clCheckErr(err, "clCreateContext");
  
  queue = clCreateCommandQueueWithProperties(context, device, nullptr, &err);
  clCheckErr(err, "clCreateCommandQueueWithProperties");

  const char* options = "-cl-fast-relaxed-math";
  std::vector<std::string> kernelSources(neuronsInLayers.size() - 1);
  for (int a = 0; a < kernelSources.size(); a++) kernelSources[a] = getNetworkKernelSource(activationFunctions[a]);
  
  programs.resize(kernelSources.size());

  for (int l = 0; l < kernelSources.size(); l++) {
    const char* str = kernelSources[l].c_str();
    programs[l] = clCreateProgramWithSource(context, 1, &str, nullptr, &err);

    err = clBuildProgram(programs[l], 1, &device, (fastMath ? options : nullptr), nullptr, nullptr);
    if (err != CL_SUCCESS) {
      size_t logSize;
      clGetProgramBuildInfo(programs[l], device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
      std::string log(logSize, '\0');
      clGetProgramBuildInfo(programs[l], device, CL_PROGRAM_BUILD_LOG, logSize, log.data(), nullptr);
      throw std::runtime_error("Building program failed:\n" + log);
    }
  }
  
  createKernels();

  // Create the storage for the layers
  layers.reserve(neuronsInLayers.size() - 1);
  for (int x = 1; x < neuronsInLayers.size(); x++) { // Create the layers
    layers.emplace_back(context, err, queue, neuronsInLayers[x - 1], neuronsInLayers[x], activationFunctions[x - 1]);

    if (neuronsInLayers[x] <= 0) throw std::invalid_argument("You can't create a network with a layer with 0 neurons");
    if (neuronsInLayers[x - 1] <= 0) throw std::invalid_argument("You can't create a network with a layer with 0 neurons");
  }

  networkInputs = neuronsInLayers[0];
}

Network::Network(const std::vector<int>& neuronsInLayers, const std::vector<Activation>& activationFunctions, float learningRate, bool fastMath, bool optimizer, bool printDeviceInfo, int usedDeviceIndex) {
  this -> learningRate = learningRate;
  useOptimizer = optimizer;

  if (neuronsInLayers.size() < 2) throw std::invalid_argument("You can't create a network with only one layer");
  if (activationFunctions.size() != neuronsInLayers.size() - 1) throw std::invalid_argument("You need to specify an activation function for each layer in your network except the input layer.\n  Provided functions: " + std::to_string(activationFunctions.size()) + "\n  Expected functions: " + std::to_string(neuronsInLayers.size() - 1));

  // Get the GPU stuff
  device = getDevice(printDeviceInfo, usedDeviceIndex);
  context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
  clCheckErr(err, "clCreateContext");
  
  queue = clCreateCommandQueueWithProperties(context, device, nullptr, &err);
  clCheckErr(err, "clCreateCommandQueueWithProperties");

  const char* options = "-cl-fast-relaxed-math";
  std::vector<std::string> kernelSources(neuronsInLayers.size() - 1);
  for (int a = 0; a < kernelSources.size(); a++) kernelSources[a] = getNetworkKernelSource(activationFunctions[a]);
  
  programs.resize(kernelSources.size());

  for (int l = 0; l < kernelSources.size(); l++) {
    const char* str = kernelSources[l].c_str();
    programs[l] = clCreateProgramWithSource(context, 1, &str, nullptr, &err);

    err = clBuildProgram(programs[l], 1, &device, (fastMath ? options : nullptr), nullptr, nullptr);
    if (err != CL_SUCCESS) {
      size_t logSize;
      clGetProgramBuildInfo(programs[l], device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
      std::string log(logSize, '\0');
      clGetProgramBuildInfo(programs[l], device, CL_PROGRAM_BUILD_LOG, logSize, log.data(), nullptr);
      throw std::runtime_error("Building program failed:\n" + log);
    }
  }
  
  createKernels();

  // Create the storage for the layers
  layers.reserve(neuronsInLayers.size() - 1);
  for (int x = 1; x < neuronsInLayers.size(); x++) { // Create the layers
    layers.emplace_back(context, err, queue, neuronsInLayers[x - 1], neuronsInLayers[x], activationFunctions[x - 1]);

    if (neuronsInLayers[x] <= 0) throw std::invalid_argument("You can't create a network with a layer with 0 neurons");
    if (neuronsInLayers[x - 1] <= 0) throw std::invalid_argument("You can't create a network with a layer with 0 neurons");
  }

  networkInputs = neuronsInLayers[0];
}

Network::Network(const std::string& path, bool fastMath, bool optimizer, bool printDeviceInfo, int usedDeviceIndex) {
  useOptimizer = optimizer;
  
  device = getDevice(printDeviceInfo, usedDeviceIndex);
  context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
  clCheckErr(err, "clCreateContext");

  queue = clCreateCommandQueueWithProperties(context, device, nullptr, &err);
  clCheckErr(err, "clCreateCommandQueueWithProperties");
  
  loadFromFile(path);
  programs.resize(layers.size());

  const char* options = "-cl-fast-relaxed-math";

  std::vector<std::string> kernelSources(layers.size());
  for (int a = 0; a < layers.size(); a++) kernelSources[a] = getNetworkKernelSource(layers[a].activationFunction);

  for (int l = 0; l < layers.size(); l++) {
    const char* str = kernelSources[l].c_str();
    programs[l] = clCreateProgramWithSource(context, 1, &str, nullptr, &err);

    err = clBuildProgram(programs[l], 1, &device, (fastMath ? options : nullptr), nullptr, nullptr);
    if (err != CL_SUCCESS) {
      size_t logSize;
      clGetProgramBuildInfo(programs[l], device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
      std::string log(logSize, '\0');
      clGetProgramBuildInfo(programs[l], device, CL_PROGRAM_BUILD_LOG, logSize, log.data(), nullptr);
      throw std::runtime_error("Building program failed:\n" + log);
    }
  }

  createKernels();
}

Network::~Network() {
  layers.clear();

  for (cl_kernel &k : biasAndActivateKernels) safeReleaseKernel(k);
  for (cl_kernel &k : zGetKernels) safeReleaseKernel(k);
  for (cl_kernel &k : deltaKernels) safeReleaseKernel(k);

  safeReleaseKernel(averageNetworkUpdateBiasKernel);
  safeReleaseKernel(averageNetworkUpdateWeightKernel);
  safeReleaseKernel(softMaxKernel);
  safeReleaseKernel(copyProbsToActivationKernel);

  safeReleaseKernel(outputDeltaSingleActionReinforcementKernel);
  safeReleaseKernel(outputDeltaOutputVectorReinforcementKernel);
  
  safeReleaseMemory(lastRunInputBuffer);
  safeReleaseMemory(lastRunOutputBuffer);

  safeReleaseMemory(lastTrainInputBuffer);
  safeReleaseMemory(lastTrainInputBuffer2);
  safeReleaseMemory(lastTrainOutputBuffer);
  safeReleaseMemory(lastTrainOutputDeltas);
  safeReleaseMemory(lastTrainRewardBuffer);
  safeReleaseMemory(lastTrainExpectedOutputs);

  safeReleaseMemory(onesBuffer);

  for (cl_mem &m : lastTrainActivations) safeReleaseMemory(m);
  for (cl_mem &m : lastTrainZs) safeReleaseMemory(m);
  for (cl_mem &m : lastTrainHiddenDeltas) safeReleaseMemory(m);
  for (cl_mem &m : lastTrainDropoutMask) safeReleaseMemory(m);
  for (cl_mem &m : weightGradientBuffer) safeReleaseMemory(m);
  for (cl_mem &m : biasGradientBuffer) safeReleaseMemory(m);

  for (cl_program &p : programs) if (p) clReleaseProgram(p);
  if (queue) clReleaseCommandQueue(queue);
  if (context) clReleaseContext(context);
}

void Network::setLearningRate(float l) { learningRate = l; }
void Network::setBETA1(float b) { BETA1 = b; }
void Network::setBETA2(float b) { BETA2 = b; }
void Network::setEpsilon(float e) { EPSILON = e; }
void Network::setWeightDecay(float w) { WEIGHT_DECAY = w; }
void Network::setOptimizerEnabled(bool o) { useOptimizer = o; }

float Network::getLearningRate() { return learningRate; }
float Network::getBETA1() { return BETA1; }
float Network::getBETA2() { return BETA2; }
float Network::getEpsilon() { return EPSILON; }
float Network::getWeightDecay() { return WEIGHT_DECAY; }
bool Network::getOptimizerEnabled() { return useOptimizer; }

void Network::setDropoutRate(std::vector<float> rates) {
  if (rates.size() != layers.size() - 1)
    throw std::invalid_argument("You need to set a dropout rate for all layers except the input and output layers\nProvided dropout rates: " + std::to_string(rates.size()) + "\nExpected: " + std::to_string(layers.size() - 1));

  for (int x = 0; x < rates.size(); x++) layers[x].setDropoutRate(rates[x]);
}

std::vector<float> Network::getDropoutRates() {
  std::vector<float> rates(layers.size() - 1);
  for (int x = 0; x < rates.size(); x++) rates[x] = layers[x].getDropoutRate();
  return rates;
}

Network::Layer::Layer(cl_context context, cl_int err, cl_command_queue queue, int previousNeurons, int neurons, Network::Activation activationFunction) {
  this -> neurons = neurons;
  this -> previousNeurons = previousNeurons;
  this -> context = context;
  this -> err = err;
  this -> queue = queue;
  this -> activationFunction = activationFunction;
  
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> initDist(-std::sqrt(6.0f / (previousNeurons + neurons)), std::sqrt(6.0f / (previousNeurons + neurons)));

  int weightCount = neurons * previousNeurons;
  std::vector<float> m1B(neurons, 0.0f);
  std::vector<float> m1W(weightCount, 0.0f);
  
  std::vector<float> biases(neurons);
  for (int x = 0; x < neurons; x++) biases[x] = initDist(gen) / 10.0f; // Fill biases with near zeros

  std::vector<float> weights(weightCount);
  for (int x = 0; x < weightCount; x++) weights[x] = initDist(gen); // Fill weights with Xavier initializers

  biasesBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * neurons, nullptr, &err);
  clCheckErr(err, "Creating cl_mem buffer for biases");
  weightsBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * weightCount, nullptr, &err);
  clCheckErr(err, "Creating cl_mem buffer for weights");

  expAvgB = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * neurons, nullptr, &err);
  clCheckErr(err, "Creating cl_mem buffer for adam first bias momentums");
  expAvgSqB = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * neurons, nullptr, &err);
  clCheckErr(err, "Creating cl_mem buffer for adam second bias momentums");
  expAvgW = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * weightCount, nullptr, &err);
  clCheckErr(err, "Creating cl_mem buffer for adam first weight momentums");
  expAvgSqW = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * weightCount, nullptr, &err);
  clCheckErr(err, "Creating cl_mem buffer for adam second weight momentums");
  
  err = clEnqueueWriteBuffer(queue, biasesBuffer, CL_FALSE, 0, sizeof(float) * neurons, biases.data(), 0, nullptr, nullptr);
  err |= clEnqueueWriteBuffer(queue, weightsBuffer, CL_FALSE, 0, sizeof(float) * weightCount, weights.data(), 0, nullptr, nullptr);

  err |= clEnqueueWriteBuffer(queue, expAvgB, CL_FALSE, 0, sizeof(float) * neurons, m1B.data(), 0, nullptr, nullptr);
  err |= clEnqueueWriteBuffer(queue, expAvgSqB, CL_FALSE, 0, sizeof(float) * neurons, m1B.data(), 0, nullptr, nullptr);
  err |= clEnqueueWriteBuffer(queue, expAvgW, CL_FALSE, 0, sizeof(float) * weightCount, m1W.data(), 0, nullptr, nullptr);
  err |= clEnqueueWriteBuffer(queue, expAvgSqW, CL_FALSE, 0, sizeof(float) * weightCount, m1W.data(), 0, nullptr, nullptr);
  clCheckErr(err, "Writing cl_mem buffers: \"biasesBuffer\", \"weightsBuffer\", \"expAvgB\", \"expAvgSqB\", \"expAvgW\", \"expAvgSqW\"");

  gen.seed(std::random_device{}());
  dist = std::uniform_int_distribution<uint32_t>(0, std::numeric_limits<uint32_t>::max());
    
  clFinish(queue);
}

Network::Layer::~Layer() {
  Network::safeReleaseMemory(biasesBuffer);
  Network::safeReleaseMemory(weightsBuffer);
  Network::safeReleaseMemory(expAvgB);
  Network::safeReleaseMemory(expAvgSqB);
  Network::safeReleaseMemory(expAvgW);
  Network::safeReleaseMemory(expAvgSqW);
}

void Network::Layer::setBiases(cl_mem biases) {
  safeReleaseMemory(biasesBuffer);
  biasesBuffer = biases;
}

void Network::Layer::setWeights(cl_mem weights) {
  safeReleaseMemory(weightsBuffer);
  weightsBuffer = weights;
}

void Network::Layer::setExpAvgB(cl_mem buffer) {
  safeReleaseMemory(expAvgB);
  expAvgB = buffer;
}

void Network::Layer::setExpAvgSqB(cl_mem buffer) {
  safeReleaseMemory(expAvgSqB);
  expAvgSqB = buffer;
}

void Network::Layer::setExpAvgW(cl_mem buffer) {
  safeReleaseMemory(expAvgW);
  expAvgW = buffer;
}

void Network::Layer::setExpAvgSqW(cl_mem buffer) {
  safeReleaseMemory(expAvgSqW);
  expAvgSqW = buffer;
}

void Network::Layer::setDropoutRate(float rate) { dropoutRate = rate; }
float Network::Layer::getDropoutRate() { return dropoutRate; }
int Network::Layer::countNeurons() { return neurons; }

cl_device_id Network::getDevice(bool printDeviceInfo, int usedDeviceIndex) {
  cl_uint numPlatforms;
  cl_int err = clGetPlatformIDs(0, nullptr, &numPlatforms);
  clCheckErr(err, "Getting amount of platforms");

  std::vector<cl_platform_id> platforms(numPlatforms);
  err = clGetPlatformIDs(numPlatforms, platforms.data(), nullptr);
  clCheckErr(err, "Getting platforms");

  if (platforms.size() == 0) throw std::runtime_error("No platform found");

  int idx = -1;
  cl_device_id device;
  
  for (cl_platform_id platform : platforms) {
    cl_uint numDevices;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &numDevices);
    clCheckErr(err, "Gathering amount of devices");

    if (numDevices == 0) continue;

    std::vector<cl_device_id> devices(numDevices);
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, numDevices, devices.data(), nullptr);
    clCheckErr(err, "Getting device IDs");

    if (printDeviceInfo) {
      std::cout << "\n" << "Information about available devices" << "\n" << std::endl;
      std::cout << "Index" << "\t  |" << " Name" << "\t\t\t  |" << " VRAM" << "\t  |" << " Compute units" << std::endl;
      std::cout << "––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––" << std::endl;

      for (int x = 0; x < devices.size(); x++) {
	idx++;

	if (idx == usedDeviceIndex) device = devices[x];
	
	char name[256];
	clGetDeviceInfo(devices[x], CL_DEVICE_NAME, sizeof(name), name, nullptr);

	cl_ulong vram;
	clGetDeviceInfo(devices[x], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(vram), &vram, nullptr);

	cl_uint computeUnits;
	clGetDeviceInfo(devices[x], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(computeUnits), &computeUnits, nullptr);

	std::cout << x << "\t  | " << name << "\t  | "  << vram << "\t  | " << computeUnits << std::endl;
      }

      std::cout << std::endl;
    } else {
      for (int x = 0; x < devices.size(); x++) {
	idx++;
	if (idx == usedDeviceIndex) return devices[x];
      }
    }
  }

  if (idx < 0) throw std::runtime_error("No GPU found");
  
  return device;
}

std::string Network::getNetworkKernelSource(Activation activationFunction) {
  std::string source = "#define ACTIVATION ACTIVATION_";
  
  switch (activationFunction) {
  case Activation::Linear:
    source += "LINEAR";
    break;
  case Activation::ReLU:
    source += "RELU";
    break;
  case Activation::LeakyReLU:
    source += "LEAKY_RELU";
    break;
  case Activation::Sigmoid:
    source += "SIGMOID";
    break;
  case Activation::Tanh:
    source += "TANH";
    break;
  case Activation::Softmax:
    source += "SOFTMAX";
    break;
  default:
    throw std::length_error("This is not a valid argument for an activation function");
  }

  return source + "\n" + networkKernelSource;
}

void Network::createKernels() {
  biasAndActivateKernels.resize(programs.size());
  zGetKernels.resize(programs.size());
  deltaKernels.resize(programs.size());
  
  averageNetworkUpdateBiasKernel = clCreateKernel(programs.back(), "average_deltas_and_update_network_biases", &err);
  clCheckErr(err, "Creation of Kernel \"averageNetworkUpdateBiasKernel\"");
  averageNetworkUpdateWeightKernel = clCreateKernel(programs.back(), "average_deltas_and_update_network_weights", &err);
  clCheckErr(err, "Creation of Kernel \"averageNetworkUpdateWeightKernel\"");
  averageNetworkUpdateBiasKernelNoOptimizer = clCreateKernel(programs.back(), "average_deltas_and_update_network_biases_no_optimizer", &err);
  clCheckErr(err, "Creation of Kernel \"averageNetworkUpdateBiasKernelNoOptimizer\"");
  averageNetworkUpdateWeightKernelNoOptimizer = clCreateKernel(programs.back(), "average_deltas_and_update_network_weights_no_optimizer", &err);
  clCheckErr(err, "Creation of Kernel \"averageNetworkUpdateWeightKernelNoOptimizer\"");
  softMaxKernel = clCreateKernel(programs.back(), "softmax", &err);
  clCheckErr(err, "Creation of Kernel \"softMaxKernel\"");
  copyProbsToActivationKernel = clCreateKernel(programs.back(), "copy_softmax_to_activations", &err);
  clCheckErr(err, "Creation of Kernel \"copyProbsToActivationKernel\"");

  for (int l = 0; l < programs.size(); l++) {
    biasAndActivateKernels[l] = clCreateKernel(programs[l], "add_bias_and_activate", &err);
    clCheckErr(err, "Creation of Kernel \"biasAndActivateKernel\"");

    zGetKernels[l] = clCreateKernel(programs[l], "get_activations_and_zs", &err);
    clCheckErr(err, "Creation of Kernel \"zGetKernel\"");

    deltaKernels[l] = clCreateKernel(programs[l], (l != programs.size() - 1 ? "compute_hidden_deltas" : "compute_output_deltas"), &err);
    clCheckErr(err, "Creation of Kernel \"deltaKernel\"");
  }

  outputDeltaSingleActionReinforcementKernel = clCreateKernel(programs.back(), "compute_output_deltas_single_action_reinforcement", &err);
  clCheckErr(err, "Creation of Kernel \"outputDeltaSingleActionReinforcementKernel\"");
  outputDeltaOutputVectorReinforcementKernel = clCreateKernel(programs.back(), "compute_output_deltas_output_vector_reinforcement", &err);
  clCheckErr(err, "Creation of Kernel \"outputDeltaOutputVectorReinforcementKernel\"");
}

std::vector<float> Network::flat2DArray(const std::vector<std::vector<float>>& array) {
  std::vector<float> flatArray(array.size() * array[0].size());
  for (int x = 0; x < array.size(); x++) memcpy(flatArray.data() + x * array[0].size(), array[x].data(), sizeof(float) * array[0].size());

  return flatArray;
}

std::vector<std::vector<float>> Network::construct2DArray(int x, int y, const std::vector<float>& array) {
  std::vector<std::vector<float>> array2D(x, std::vector<float>(y));
  for (int d = 0; d < x; d++) memcpy(array2D[d].data(), &array[d * y], sizeof(float) * y);
							       
  return array2D;
}

std::vector<std::vector<float>> Network::run(const std::vector<std::vector<float>>& inputs) {
  int batchSize = inputs.size();
  cl_mem out = runCL_MEM(inputs);
  
  std::vector<float> flatOutputs(layers.back().countNeurons() * batchSize);
  err = clEnqueueReadBuffer(queue, out, CL_TRUE, 0, sizeof(float) * flatOutputs.size(), flatOutputs.data(), 0, nullptr, nullptr);
  clCheckErr(err, "Reading results to CPU");

  std::vector<std::vector<float>> outputs = construct2DArray(batchSize, layers.back().countNeurons(), flatOutputs);
  
  err = clFinish(queue);
  clCheckErr(err, "Finishing run call");
  return outputs;
}

std::vector<std::vector<float>> Network::run(cl_mem inputs, int batchSize) {
  cl_mem outputBuffer = runCL_MEM(inputs, batchSize);
  
  std::vector<float> flatOutputs(layers.back().countNeurons() * batchSize);
  err = clEnqueueReadBuffer(queue, outputBuffer, CL_TRUE, 0, sizeof(float) * flatOutputs.size(), flatOutputs.data(), 0, nullptr, nullptr);
  clCheckErr(err, "Reading results to CPU");

  std::vector<std::vector<float>> outputs = construct2DArray(batchSize, layers.back().countNeurons(), flatOutputs);
  
  err = clFinish(queue);
  clCheckErr(err, "Finishing run call");
  return outputs;
}

cl_mem Network::runCL_MEM(cl_mem inputs, int batchSize) {
  int maxNeurons = networkInputs;
  for (Layer& l : layers) maxNeurons = (l.countNeurons() > maxNeurons) ? l.countNeurons() : maxNeurons;
  maxNeurons *= batchSize;

  cl_mem inputBuffer = nullptr;
  cl_mem outputBuffer = nullptr;
  
  if (batchSize != lastRunBatchSize) {
    safeReleaseMemory(lastRunInputBuffer);
    safeReleaseMemory(lastRunOutputBuffer);

    inputBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * maxNeurons, nullptr, &err);
    clCheckErr(err, "Creation of cl_mem: \"inputBuffer\"");
    outputBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * maxNeurons, nullptr, &err);
    clCheckErr(err, "Creation of cl_mem: \"outputBuffer\"");
    
    lastRunInputBuffer = inputBuffer;
    lastRunOutputBuffer = outputBuffer;
    lastRunBatchSize = batchSize;
  } else {
    inputBuffer = lastRunInputBuffer;
    outputBuffer = lastRunOutputBuffer;
  }

  int x = 0;
  for (Layer& l : layers) {
    l.run(biasAndActivateKernels[x], (x == 0 ? inputs : inputBuffer), outputBuffer, batchSize);
    std::swap(inputBuffer, outputBuffer);
    x++;
  }

  int outputSize = layers.back().countNeurons();

  if (layers.back().activationFunction == Activation::Softmax) {    
    err = clSetKernelArg(softMaxKernel, 0, sizeof(cl_mem), &inputBuffer);
    err |= clSetKernelArg(softMaxKernel, 1, sizeof(cl_mem), &outputBuffer);
    err |= clSetKernelArg(softMaxKernel, 2, sizeof(int), &outputSize);
    clCheckErr(err, "Setting kernel arguments: \"softMaxKernel\"");

    size_t globalSize = batchSize;
    err = clEnqueueNDRangeKernel(queue, softMaxKernel, 1, nullptr, &globalSize, nullptr, 0, nullptr, nullptr);
    clCheckErr(err, "Launching kernel: \"softMaxKernel\"");
  }

  return layers.back().activationFunction == Activation::Softmax ? outputBuffer : inputBuffer;
}

cl_mem Network::runCL_MEM(const std::vector<std::vector<float>>& inputs) {
  int batchSize = inputs.size();
  std::vector<float> flatInputs = flat2DArray(inputs);

  int maxNeurons = networkInputs;
  for (Layer& l : layers) maxNeurons = (l.countNeurons() > maxNeurons) ? l.countNeurons() : maxNeurons;
  maxNeurons *= batchSize;
  
  if (flatInputs.size() / batchSize != networkInputs)
    throw std::invalid_argument("The number of network inputs and provided inputs must be the same. \nNetwork inputs: " + std::to_string(networkInputs) + "\nInputs: " + std::to_string(flatInputs.size() / batchSize));

  cl_mem inputBuffer = nullptr;
  cl_mem outputBuffer = nullptr;
  
  if (batchSize != lastRunBatchSize) {
    safeReleaseMemory(lastRunInputBuffer);
    safeReleaseMemory(lastRunOutputBuffer);
    
    inputBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * maxNeurons, flatInputs.data(), &err);
    clCheckErr(err, "Creation of cl_mem: \"inputBuffer\"");
    
    outputBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * maxNeurons, nullptr, &err);
    clCheckErr(err, "Creation of cl_mem: \"inputBuffer\"");

    lastRunInputBuffer = inputBuffer;
    lastRunOutputBuffer = outputBuffer;
    lastRunBatchSize = batchSize;
  } else
    inputBuffer = lastRunInputBuffer;
  
  err = clEnqueueWriteBuffer(queue, inputBuffer, CL_FALSE, 0, sizeof(float) * flatInputs.size(), flatInputs.data(), 0, nullptr, nullptr);
  clCheckErr(err, "Writing to cl_mem: \"inputBuffer\"");

  return runCL_MEM(inputBuffer, batchSize);
}

void Network::Layer::run(cl_kernel kernel, cl_mem inputBuffer, cl_mem outputBuffer, int batchSize) {
  auto status = clblast::Gemm(clblast::Layout::kRowMajor, clblast::Transpose::kNo, clblast::Transpose::kYes,
			      batchSize, neurons, previousNeurons, 1.0f, inputBuffer, 0, previousNeurons, weightsBuffer, 0,
			      previousNeurons, 0.0f, outputBuffer, 0, neurons, &queue, nullptr);

  if (status != clblast::StatusCode::kSuccess) std::cout << "GEMM error: " << static_cast<int>(status) << std::endl;
  
  err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &biasesBuffer);
  err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &outputBuffer);
  err |= clSetKernelArg(kernel, 2, sizeof(int), &neurons);
  clCheckErr(err, "Setting kernel arguments: \"kernel\"");

  size_t globalSize[2] = { (size_t) batchSize, (size_t) neurons };  
  err = clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, globalSize, nullptr, 0, nullptr, nullptr);
  clCheckErr(err, "Launching kernel: \"kernel\"");
}

void Network::Layer::run(cl_kernel kernel, cl_mem inputBuffer, cl_mem outputBuffer, cl_mem activations, cl_mem zs, cl_mem mask, int batchSize) {
  auto status = clblast::Gemm(clblast::Layout::kRowMajor, clblast::Transpose::kNo, clblast::Transpose::kYes,
			      batchSize, neurons, previousNeurons, 1.0f, inputBuffer, 0, previousNeurons, weightsBuffer, 0,
			      previousNeurons, 0.0f, outputBuffer, 0, neurons, &queue, nullptr);

  if (status != clblast::StatusCode::kSuccess) std::cout << "GEMM error: " << static_cast<int>(status) << std::endl;

  uint32_t seed = dist(gen);
  
  err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &biasesBuffer);
  err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &outputBuffer);
  err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &activations);
  err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &zs);
  err |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &mask);
  err |= clSetKernelArg(kernel, 5, sizeof(int), &neurons);
  err |= clSetKernelArg(kernel, 6, sizeof(float), &dropoutRate);
  err |= clSetKernelArg(kernel, 7, sizeof(uint32_t), &seed);
  clCheckErr(err, "Setting kernel arguments: \"kernel\"");

  size_t globalSize[2] = { (size_t) batchSize, (size_t) neurons };  
  err = clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, globalSize, nullptr, 0, nullptr, nullptr);
  clCheckErr(err, "Launching kernel: \"kernel\"");
}

void Network::trainSupervised(const std::vector<std::vector<float>>& inputs, const std::vector<std::vector<float>>& expectedOutputs, cl_mem inputErrorBuffer) {
  timestepAdam++;

  if (inputs.size() != expectedOutputs.size())
    throw std::invalid_argument("The size of the batch size should be equal for inputs and outputs. \n    Input batch size: " + std::to_string(inputs.size()) + "\n    Output batch size: " + std::to_string(expectedOutputs.size()));
  
  int batchSize = inputs.size();
  int maxNeurons = inputs[0].size();
  for (Layer& l : layers) maxNeurons = (l.countNeurons() > maxNeurons) ? l.countNeurons() : maxNeurons;

  std::vector<float> flatInputs = flat2DArray(inputs);
  std::vector<float> flatOutputs = flat2DArray(expectedOutputs);

  if (flatInputs.size() / batchSize != networkInputs)
    throw std::invalid_argument("The number of network inputs and provided inputs must be the same. \n    Network inputs: " + std::to_string(networkInputs) + "\nProvided inputs: " + std::to_string(flatInputs.size() / batchSize));
  if (flatOutputs.size() / batchSize != layers.back().countNeurons())
    throw std::invalid_argument("The number of network output and provided outputs must be the same. \n    Network outputs: " + std::to_string(layers.back().countNeurons()) + "\nProvided inputs: " + std::to_string(flatOutputs.size() / batchSize));

  if (batchSize != lastTrainBatchSize) {
    lastTrainBatchSize = batchSize;
    reinitializeClMemObjects(maxNeurons);
  }

  err = clEnqueueWriteBuffer(queue, lastTrainInputBuffer2, CL_FALSE, 0, sizeof(float) * flatInputs.size(), flatInputs.data(), 0, nullptr, nullptr);
  err |= clEnqueueWriteBuffer(queue, lastTrainInputBuffer, CL_FALSE, 0, sizeof(float) * flatInputs.size(), flatInputs.data(), 0, nullptr, nullptr);
  err |= clEnqueueWriteBuffer(queue, lastTrainExpectedOutputs, CL_FALSE, 0, sizeof(float) * flatOutputs.size(), flatOutputs.data(), 0, nullptr, nullptr);
  clCheckErr(err, "Writing cl_mem: \"lastTrainInputBuffer\", \"lastTrainInputBuffer2\", \"lastTrainExpectedOutputs\"");
  
  getActivationsAndZs();
  computeOutputDeltasSupervised();
  computeHiddenDeltas();
  updateNetwork();

  calculateInputError(inputErrorBuffer);
}

void Network::trainSupervised(cl_mem inputs, cl_mem expectedOutputs, int batchSize, cl_mem inputErrorBuffer) {
  timestepAdam++;

  int maxNeurons = networkInputs;
  for (Layer& l : layers) maxNeurons = (l.countNeurons() > maxNeurons) ? l.countNeurons() : maxNeurons;

  if (lastTrainBatchSize != batchSize) {
    lastTrainBatchSize = batchSize;
    reinitializeClMemObjects(maxNeurons);
  }

  err = clEnqueueCopyBuffer(queue, inputs, lastTrainInputBuffer2, 0, 0, (size_t) (sizeof(float) * networkInputs * batchSize), 0, nullptr, nullptr);
  err |= clEnqueueCopyBuffer(queue, inputs, lastTrainInputBuffer, 0, 0, (size_t) (sizeof(float) * networkInputs * batchSize), 0, nullptr, nullptr);
  err |= clEnqueueCopyBuffer(queue, expectedOutputs, lastTrainExpectedOutputs, 0, 0, (size_t) (sizeof(float) * layers.back().countNeurons() * batchSize), 0, nullptr, nullptr);
  clCheckErr(err, "Copying cl_mem");

  getActivationsAndZs();
  computeOutputDeltasSupervised();
  computeHiddenDeltas();
  updateNetwork();

  calculateInputError(inputErrorBuffer);
}

void Network::trainReinforcement(const std::vector<std::vector<float>>& inputs, const std::vector<int>& chosenActions, std::vector<float> rewards, cl_mem inputErrorBuffer) { // Single action
  timestepAdam++;
  
  int batchSize = inputs.size();
  int maxNeurons = inputs[0].size();
  for (Layer& l : layers) maxNeurons = (l.countNeurons() > maxNeurons) ? l.countNeurons() : maxNeurons;

  std::vector<float> flatInputs = flat2DArray(inputs);
  std::vector<float> actionChosen(layers.back().countNeurons() * batchSize, 0.0f);
  for (int x = 0; x < batchSize; x++) actionChosen[x * layers.back().countNeurons() + chosenActions[x]] = 1.0f;

  if (flatInputs.size() / batchSize != networkInputs)
    throw std::invalid_argument("The number of network inputs and provided inputs must be the same. \nNetwork inputs: " + std::to_string(networkInputs) + "\nProvided inputs: " + std::to_string(flatInputs.size() / batchSize));

  if (batchSize != lastTrainBatchSize) {
    lastTrainBatchSize = batchSize;
    reinitializeClMemObjects(maxNeurons);
    lastTrainRewardBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * batchSize, rewards.data(), &err);
    clCheckErr(err, "Creation of cl_mem buffer: \"lastTrainRewardBuffer\"");
  } else {
    err = clEnqueueWriteBuffer(queue, lastTrainRewardBuffer, CL_FALSE, 0, sizeof(float) * batchSize, rewards.data(), 0, nullptr, nullptr);
    clCheckErr(err, "Writing cl_mem buffer: \"lastTrainRewardBuffer\"");
  }
  
  err = clEnqueueWriteBuffer(queue, lastTrainInputBuffer2, CL_FALSE, 0, sizeof(float) * flatInputs.size(), flatInputs.data(), 0, nullptr, nullptr);
  err |= clEnqueueWriteBuffer(queue, lastTrainInputBuffer, CL_FALSE, 0, sizeof(float) * flatInputs.size(), flatInputs.data(), 0, nullptr, nullptr);
  err |= clEnqueueWriteBuffer(queue, lastTrainExpectedOutputs, CL_FALSE, 0, sizeof(float) * actionChosen.size(), actionChosen.data(), 0, nullptr, nullptr);
  clCheckErr(err, "Writing cl_mem: \"lastTrainInputBuffer\", \"lastTrainInputBuffer2\", \"lastTrainExpectedOutputs\"");

  getActivationsAndZs();
  computeOutputDeltasSingleActionReinforcement();
  computeHiddenDeltas();
  updateNetwork();

  calculateInputError(inputErrorBuffer);
}

void Network::trainReinforcement(cl_mem inputs, cl_mem chosenActions, cl_mem rewards, int batchSize, cl_mem inputErrorBuffer) {
  timestepAdam++;
  
  int maxNeurons = networkInputs;
  for (Layer& l : layers) maxNeurons = (l.countNeurons() > maxNeurons) ? l.countNeurons() : maxNeurons;

  if (lastTrainBatchSize != batchSize) {
    lastTrainBatchSize = batchSize;
    reinitializeClMemObjects(maxNeurons);
  }

  err = clEnqueueCopyBuffer(queue, inputs, lastTrainInputBuffer2, 0, 0, (size_t) (sizeof(float) * networkInputs * batchSize), 0, nullptr, nullptr);
  err |= clEnqueueCopyBuffer(queue, inputs, lastTrainInputBuffer, 0, 0, (size_t) (sizeof(float) * networkInputs * batchSize), 0, nullptr, nullptr);
  err |= clEnqueueCopyBuffer(queue, chosenActions, lastTrainExpectedOutputs, 0, 0, (size_t) (sizeof(float) * batchSize), 0, nullptr, nullptr);
  err |= clEnqueueCopyBuffer(queue, rewards, lastTrainRewardBuffer, 0, 0, (size_t) (sizeof(float) * batchSize), 0, nullptr, nullptr);
  clCheckErr(err, "Copying cl_mem");

  getActivationsAndZs();
  computeOutputDeltasSingleActionReinforcement();
  computeHiddenDeltas();
  updateNetwork();

  calculateInputError(inputErrorBuffer);
}

void Network::trainReinforcement(const std::vector<std::vector<float>>& inputs, const std::vector<std::vector<float>>& actions, std::vector<float> rewards, float sigma, cl_mem inputErrorBuffer) { // Whole output combination
  timestepAdam++;

  int batchSize = inputs.size();
  int maxNeurons = inputs[0].size();
  for (Layer& l : layers) maxNeurons = (l.countNeurons() > maxNeurons) ? l.countNeurons() : maxNeurons;

  std::vector<float> flatInputs = flat2DArray(inputs);
  std::vector<float> flatActions = flat2DArray(actions);

  if (flatInputs.size() / batchSize != networkInputs)
    throw std::invalid_argument("The number of network inputs and provided inputs must be the same. \nNetwork inputs: " + std::to_string(networkInputs) + "\nProvided inputs: " + std::to_string(flatInputs.size() / batchSize));

  if (batchSize != lastTrainBatchSize) {
    lastTrainBatchSize = batchSize;
    reinitializeClMemObjects(maxNeurons);
    lastTrainRewardBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * batchSize, rewards.data(), &err);
    clCheckErr(err, "Creation of cl_mem: \"lastTrainRewardBuffer\"");
  } else {
    err = clEnqueueWriteBuffer(queue, lastTrainRewardBuffer, CL_FALSE, 0, sizeof(float) * batchSize, rewards.data(), 0, nullptr, nullptr);
    clCheckErr(err, "Writing cl_mem buffer: \"lastTrainRewardBuffer\"");
  }

  err = clEnqueueWriteBuffer(queue, lastTrainInputBuffer2, CL_FALSE, 0, sizeof(float) * flatInputs.size(), flatInputs.data(), 0, nullptr, nullptr);
  err |= clEnqueueWriteBuffer(queue, lastTrainInputBuffer, CL_FALSE, 0, sizeof(float) * flatInputs.size(), flatInputs.data(), 0, nullptr, nullptr);
  err |= clEnqueueWriteBuffer(queue, lastTrainExpectedOutputs, CL_FALSE, 0, sizeof(float) * flatActions.size(), flatActions.data(), 0, nullptr, nullptr);
  clCheckErr(err, "Writing cl_mem: \"lastTrainInputBuffer\", \"lastTrainInputBuffer2\", \"lastTrainExpectedOutputs\"");

  getActivationsAndZs();
  computeOutputDeltasOutputVectorReinforcement(sigma);
  computeHiddenDeltas();
  updateNetwork();

  calculateInputError(inputErrorBuffer);
}

void Network::trainReinforcement(cl_mem inputs, cl_mem actions, cl_mem rewards, int batchSize, float sigma, cl_mem inputErrorBuffer) {
  timestepAdam++;
  
  int maxNeurons = networkInputs;
  for (Layer& l : layers) maxNeurons = (l.countNeurons() > maxNeurons) ? l.countNeurons() : maxNeurons;

  if (lastTrainBatchSize != batchSize) {
    lastTrainBatchSize = batchSize;
    reinitializeClMemObjects(maxNeurons);
  }

  lastTrainInputBuffer = inputs;
  lastTrainExpectedOutputs = actions;
  lastTrainRewardBuffer = rewards;

  err = clEnqueueCopyBuffer(queue, inputs, lastTrainInputBuffer2, 0, 0, (size_t) (sizeof(float) * networkInputs * batchSize), 0, nullptr, nullptr);
  err |= clEnqueueCopyBuffer(queue, inputs, lastTrainInputBuffer, 0, 0, (size_t) (sizeof(float) * networkInputs * batchSize), 0, nullptr, nullptr);
  err |= clEnqueueCopyBuffer(queue, actions, lastTrainExpectedOutputs, 0, 0, (size_t) (sizeof(float) * batchSize), 0, nullptr, nullptr);
  err |= clEnqueueCopyBuffer(queue, rewards, lastTrainRewardBuffer, 0, 0, (size_t) (sizeof(float) * batchSize), 0, nullptr, nullptr);
  clCheckErr(err, "Copying cl_mem");

  getActivationsAndZs();
  computeOutputDeltasOutputVectorReinforcement(sigma);
  computeHiddenDeltas();
  updateNetwork();

  calculateInputError(inputErrorBuffer);
}

void Network::reinitializeClMemObjects(int maxNeurons) {
  int batchSize = lastTrainBatchSize;

  safeReleaseMemory(lastTrainInputBuffer);
  safeReleaseMemory(lastTrainInputBuffer2);
  safeReleaseMemory(lastTrainOutputBuffer);
  safeReleaseMemory(lastTrainOutputDeltas);
  safeReleaseMemory(lastTrainExpectedOutputs);
  safeReleaseMemory(lastTrainRewardBuffer);
  safeReleaseMemory(onesBuffer);
  
  lastTrainInputBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * maxNeurons * batchSize, nullptr, &err);
  clCheckErr(err, "Creation of cl_mem: \"lastTrainInputBuffer\"");
  lastTrainInputBuffer2 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * networkInputs * batchSize, nullptr, &err);
  clCheckErr(err, "Creation of cl_mem: \"lastTrainInputBuffer2\"");
  lastTrainOutputBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * maxNeurons * batchSize, nullptr, &err);
  clCheckErr(err, "Creation of cl_mem: \"lastTrainOutputBuffer\"");
  lastTrainOutputDeltas = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * layers.back().countNeurons() * batchSize, nullptr, &err);
  clCheckErr(err, "Creation of cl_mem: \"lastTrainOutputDeltas\"");
  lastTrainExpectedOutputs = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * layers.back().countNeurons() * batchSize, nullptr, &err);
  clCheckErr(err, "Creation of cl_mem: \"lastTrainExpectedOutputs\"");
  onesBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * batchSize, std::vector<float>(batchSize, 1.0f).data(), &err);
  clCheckErr(err, "Creation of cl_mem: \"onesBuffer\"");

  lastTrainActivations.resize(layers.size());
  lastTrainZs.resize(layers.size());
  weightGradientBuffer.resize(layers.size());
  biasGradientBuffer.resize(layers.size());
  lastTrainDropoutMask.resize(layers.size());
  for (int x = 0; x < layers.size(); x++) {
    safeReleaseMemory(lastTrainActivations[x]);
    safeReleaseMemory(lastTrainZs[x]);
    safeReleaseMemory(weightGradientBuffer[x]);
    safeReleaseMemory(biasGradientBuffer[x]);
    safeReleaseMemory(lastTrainDropoutMask[x]);
    
    lastTrainActivations[x] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * layers[x].countNeurons() * batchSize, nullptr, &err);
    clCheckErr(err, "Creation of cl_mem: \"lastTrainActivations\"");
    lastTrainZs[x] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * layers[x].countNeurons() * batchSize, nullptr, &err);
    clCheckErr(err, "Creation of cl_mem: \"lastTrainZs\"");
    weightGradientBuffer[x] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * layers[x].countNeurons() * ((x == 0) ? networkInputs : layers[x - 1].countNeurons()), nullptr, &err);
    clCheckErr(err, "Creation of cl_mem: \"weightGradientBuffer\"");
    biasGradientBuffer[x] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * layers[x].countNeurons(), nullptr, &err);
    clCheckErr(err, "Creation of cl_mem: \"biasGradientBuffer\"");
    lastTrainDropoutMask[x] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(unsigned char) * layers[x].countNeurons() * batchSize, nullptr, &err);
    clCheckErr(err, "Creation of cl_mem: \"lastTrainDropoutMask\"");
  }
  
  lastTrainHiddenDeltas.resize(layers.size() - 1);
  for (int x = 0; x < layers.size() - 1; x++) {
    safeReleaseMemory(lastTrainHiddenDeltas[x]);
    lastTrainHiddenDeltas[x] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * layers[x].countNeurons() * batchSize, nullptr, &err);
    clCheckErr(err, "Creation of cl_mem: \"lastTrainHiddenDeltas\"");
  }
}

void Network::getActivationsAndZs() {
  for (int x = 0; x < layers.size(); x++) {
    Layer& l = layers[x];
    l.run(zGetKernels[x], lastTrainInputBuffer, lastTrainOutputBuffer, lastTrainActivations[x], lastTrainZs[x], lastTrainDropoutMask[x], lastTrainBatchSize);
    std::swap(lastTrainInputBuffer, lastTrainOutputBuffer);
  }

  if (layers.back().activationFunction == Activation::Softmax) {
    int outputSize = layers.back().countNeurons();
    err = clSetKernelArg(softMaxKernel, 0, sizeof(cl_mem), &lastTrainInputBuffer);
    err |= clSetKernelArg(softMaxKernel, 1, sizeof(cl_mem), &lastTrainOutputBuffer);    
    err |= clSetKernelArg(softMaxKernel, 2, sizeof(int), &outputSize);
    clCheckErr(err, "Setting kernel arguments: \"softMaxKernel\"");

    size_t globalSize = lastTrainBatchSize;
    err = clEnqueueNDRangeKernel(queue, softMaxKernel, 1, nullptr, &globalSize, nullptr, 0, nullptr, nullptr);
    clCheckErr(err, "Launching kernel: \"softMaxKernel\"");

    err = clSetKernelArg(copyProbsToActivationKernel, 0, sizeof(cl_mem), &lastTrainOutputBuffer);
    err |= clSetKernelArg(copyProbsToActivationKernel, 1, sizeof(cl_mem), &lastTrainActivations.back());
    clCheckErr(err, "Setting kernel arguments: \"copyProbsToActivationKernel\"");

    size_t copyGlobal = (size_t) (lastTrainBatchSize * outputSize);
    err = clFinish(queue);
    clCheckErr(err, "Finishing kernel run");
    
    err = clEnqueueNDRangeKernel(queue, copyProbsToActivationKernel, 1, nullptr, &copyGlobal, nullptr, 0, nullptr, nullptr);
    clCheckErr(err, "Launching kernel: \"copyProbsToActivationKernel\"");
  }
}

void Network::computeOutputDeltasSupervised() {
  err = clSetKernelArg(deltaKernels.back(), 0, sizeof(cl_mem), &lastTrainActivations.back());
  err |= clSetKernelArg(deltaKernels.back(), 1, sizeof(cl_mem), &lastTrainZs.back());
  err |= clSetKernelArg(deltaKernels.back(), 2, sizeof(cl_mem), &lastTrainExpectedOutputs);
  err |= clSetKernelArg(deltaKernels.back(), 3, sizeof(cl_mem), &lastTrainOutputDeltas);
  clCheckErr(err, "Setting kernel arguments: \"outputDeltaKernel\"");

  size_t globalSize = layers.back().countNeurons() * lastTrainBatchSize;
  err = clEnqueueNDRangeKernel(queue, deltaKernels.back(), 1, nullptr, &globalSize, nullptr, 0, nullptr, nullptr);
  clCheckErr(err, "Launching kernel: \"outputDeltaKernel\"");
}

void Network::computeOutputDeltasSingleActionReinforcement() {
  int outs = layers.back().countNeurons();
    
  err = clSetKernelArg(outputDeltaSingleActionReinforcementKernel, 0, sizeof(cl_mem), &lastTrainActivations.back());
  err |= clSetKernelArg(outputDeltaSingleActionReinforcementKernel, 1, sizeof(cl_mem), &lastTrainExpectedOutputs);
  err |= clSetKernelArg(outputDeltaSingleActionReinforcementKernel, 2, sizeof(cl_mem), &lastTrainOutputDeltas);
  err |= clSetKernelArg(outputDeltaSingleActionReinforcementKernel, 3, sizeof(cl_mem), &lastTrainRewardBuffer);
  err |= clSetKernelArg(outputDeltaSingleActionReinforcementKernel, 4, sizeof(int), &outs);
  clCheckErr(err, "Setting kernel arguments: \"outputDeltaSingleActionReinforcementKernel\"");

  size_t globalSize = outs * lastTrainBatchSize;
  err = clEnqueueNDRangeKernel(queue, outputDeltaSingleActionReinforcementKernel, 1, nullptr, &globalSize, nullptr, 0, nullptr, nullptr);
  clCheckErr(err, "Launching kernel: \"outputDeltaSingleActionReinforcementKernel\"");
}

void Network::computeOutputDeltasOutputVectorReinforcement(float sigma) {
  int outs = layers.back().countNeurons();
  float sigmaSquared = sigma * sigma;

  err = clSetKernelArg(outputDeltaOutputVectorReinforcementKernel, 0, sizeof(cl_mem), &lastTrainActivations.back());
  err |= clSetKernelArg(outputDeltaOutputVectorReinforcementKernel, 1, sizeof(cl_mem), &lastTrainZs.back());
  err |= clSetKernelArg(outputDeltaOutputVectorReinforcementKernel, 2, sizeof(cl_mem), &lastTrainExpectedOutputs);
  err |= clSetKernelArg(outputDeltaOutputVectorReinforcementKernel, 3, sizeof(cl_mem), &lastTrainOutputDeltas);
  err |= clSetKernelArg(outputDeltaOutputVectorReinforcementKernel, 4, sizeof(cl_mem), &lastTrainRewardBuffer);
  err |= clSetKernelArg(outputDeltaOutputVectorReinforcementKernel, 5, sizeof(float), &sigmaSquared);
  err |= clSetKernelArg(outputDeltaOutputVectorReinforcementKernel, 6, sizeof(int), &outs);
  clCheckErr(err, "Setting kernel arguments: \"outputDeltaOutputVectorReinforcementKernel\"");

  size_t globalSize = outs * lastTrainBatchSize;
  err = clEnqueueNDRangeKernel(queue, outputDeltaOutputVectorReinforcementKernel, 1, nullptr, &globalSize, nullptr, 0, nullptr, nullptr);
  clCheckErr(err, "Launching kernel: \"outputDeltaOutputVectorReinforcementKernel\"");
}

void Network::computeHiddenDeltas() {
  int batchSize = lastTrainBatchSize;
    
  for (int x = layers.size() - 2; x >= 0; x--) {
    int neurons = layers[x].countNeurons();
    int nextNeurons = layers[x + 1].countNeurons();
    
    auto status = clblast::Gemm(clblast::Layout::kRowMajor, clblast::Transpose::kNo, clblast::Transpose::kNo,
				batchSize, neurons, nextNeurons, 1.0f, (x == layers.size() - 2) ? lastTrainOutputDeltas : lastTrainHiddenDeltas[x + 1], 0, nextNeurons, layers[x + 1].weightsBuffer, 0,
				neurons, 0.0f, lastTrainHiddenDeltas[x], 0, neurons, &queue, nullptr);

    if (status != clblast::StatusCode::kSuccess) throw std::runtime_error("GEMM error: " + std::to_string(static_cast<int>(status)));
    float dr = layers[x].getDropoutRate();
    
    err = clSetKernelArg(deltaKernels[x], 0, sizeof(cl_mem), &lastTrainZs[x]);
    err |= clSetKernelArg(deltaKernels[x], 1, sizeof(cl_mem), &lastTrainHiddenDeltas[x]);
    err |= clSetKernelArg(deltaKernels[x], 2, sizeof(cl_mem), &lastTrainDropoutMask[x]);
    err |= clSetKernelArg(deltaKernels[x], 3, sizeof(float), &dr);
    err |= clSetKernelArg(deltaKernels[x], 4, sizeof(int), &neurons);
    clCheckErr(err, "Setting kernel arguments: \"hiddenDeltaKernel\"");

    size_t globalSize = batchSize * layers[x].countNeurons();
    err = clEnqueueNDRangeKernel(queue, deltaKernels[x], 1, nullptr, &globalSize, nullptr, 0, nullptr, nullptr);
    clCheckErr(err, "Launching kernel: \"deltaKernel\"");
  }
}

void Network::updateNetwork() {
  float pcBETA1 = 1.0f - BETA1;
  float pcBETA2 = 1.0f - BETA2;
  
  float beta1PowT = std::pow(BETA1, timestepAdam);
  float beta2PowT = std::pow(BETA2, timestepAdam);
  
  float pcBeta1PowT = 1.0f - beta1PowT;
  float pcBeta2PowT = 1.0f - beta2PowT;

  float invBatch = 1.0f / (float) lastTrainBatchSize;

  cl_kernel kb = useOptimizer ? averageNetworkUpdateBiasKernel : averageNetworkUpdateBiasKernelNoOptimizer;
  cl_kernel kw = useOptimizer ? averageNetworkUpdateWeightKernel : averageNetworkUpdateWeightKernelNoOptimizer;
 
  err = clSetKernelArg(kb, useOptimizer ? 4 : 2, sizeof(float), &learningRate);
  if (useOptimizer) {
    err |= clSetKernelArg(kb, 5, sizeof(float), &BETA1);
    err |= clSetKernelArg(kb, 6, sizeof(float), &BETA2);
    err |= clSetKernelArg(kb, 7, sizeof(float), &pcBETA1);
    err |= clSetKernelArg(kb, 8, sizeof(float), &pcBETA2);
    err |= clSetKernelArg(kb, 9, sizeof(float), &pcBeta1PowT);
    err |= clSetKernelArg(kb, 10, sizeof(float), &pcBeta2PowT);
    err |= clSetKernelArg(kb, 11, sizeof(float), &EPSILON);
  }
  clCheckErr(err, "Setting kernel arguments: \"averageNetworkUpdateBiasKernel\"");
  
  err = clSetKernelArg(kw, useOptimizer ? 4 : 2, sizeof(float), &learningRate);
  if (useOptimizer) {
    err |= clSetKernelArg(kw, 5, sizeof(float), &BETA1);
    err |= clSetKernelArg(kw, 6, sizeof(float), &BETA2);
    err |= clSetKernelArg(kw, 7, sizeof(float), &pcBETA1);
    err |= clSetKernelArg(kw, 8, sizeof(float), &pcBETA2);
    err |= clSetKernelArg(kw, 9, sizeof(float), &pcBeta1PowT);
    err |= clSetKernelArg(kw, 10, sizeof(float), &pcBeta2PowT);
    err |= clSetKernelArg(kw, 11, sizeof(float), &EPSILON);
    err |= clSetKernelArg(kw, 12, sizeof(float), &WEIGHT_DECAY);
  }
  clCheckErr(err, "Setting kernel arguments: \"averageNetworkUpdateWeightKernel\"");

  for (int x = 0; x < layers.size(); x++) {
    int neurons = layers[x].countNeurons();
    int previousNeurons = (x == 0) ? networkInputs : layers[x - 1].countNeurons();

    cl_mem deltas = (x == layers.size() - 1) ? lastTrainOutputDeltas : lastTrainHiddenDeltas[x];
    cl_mem prevActs = (x == 0) ? lastTrainInputBuffer2 : lastTrainActivations[x - 1];

    auto status = clblast::Gemv(clblast::Layout::kRowMajor, clblast::Transpose::kYes,
				lastTrainBatchSize, neurons,
				invBatch, deltas, 0, neurons,
				onesBuffer, 0, 1, 0.0f,
				biasGradientBuffer[x], 0, 1,
				&queue, nullptr);

    if (status != clblast::StatusCode::kSuccess) throw std::runtime_error("GEMM error: " + std::to_string(static_cast<int>(status)));

    status = clblast::Gemm(clblast::Layout::kRowMajor,
				clblast::Transpose::kYes, clblast::Transpose::kNo,
				neurons, previousNeurons, lastTrainBatchSize,
				invBatch, deltas, 0, neurons,
				prevActs, 0, previousNeurons,
				0.0f,
				weightGradientBuffer[x], 0, previousNeurons,
				&queue, nullptr);

    if (status != clblast::StatusCode::kSuccess) throw std::runtime_error("GEMM error: " + std::to_string(static_cast<int>(status)));

    err = clSetKernelArg(kb, 0, sizeof(cl_mem), &biasGradientBuffer[x]);
    err |= clSetKernelArg(kb, 1, sizeof(cl_mem), &layers[x].biasesBuffer);
    if (useOptimizer) {
      err |= clSetKernelArg(kb, 2, sizeof(cl_mem), &layers[x].expAvgB);
      err |= clSetKernelArg(kb, 3, sizeof(cl_mem), &layers[x].expAvgSqB);
    }
    clCheckErr(err, "Setting kernel arguments: \"averageNetworkUpdateBiasKernel\"");

    {
      size_t globalSize = (size_t) neurons;
      err = clEnqueueNDRangeKernel(queue, kb, 1, nullptr, &globalSize, nullptr, 0, nullptr, nullptr);
      clCheckErr(err, "Launching kernel \"averageNetworkUpdateBiasKernel\"");
    }
    
    err = clSetKernelArg(kw, 0, sizeof(cl_mem), &weightGradientBuffer[x]);
    err |= clSetKernelArg(kw, 1, sizeof(cl_mem), &layers[x].weightsBuffer);
    if (useOptimizer) {
      err |= clSetKernelArg(kw, 2, sizeof(cl_mem), &layers[x].expAvgW);
      err |= clSetKernelArg(kw, 3, sizeof(cl_mem), &layers[x].expAvgSqW);
    }
    clCheckErr(err, "Setting kernel arguments: \"averageNetworkUpdateWeightKernel\"");

    {
      size_t globalSize[2] = { (size_t) neurons, (size_t) previousNeurons };
      err = clEnqueueNDRangeKernel(queue, kw, 2, nullptr, globalSize, nullptr, 0, nullptr, nullptr);
      clCheckErr(err, "Launching kernel \"averageNetworkUpdateWeightKernel\"");
    }
  }

  err = clFinish(queue);
  clCheckErr(err, "Finishing kernel run");
}

void Network::calculateInputError(cl_mem inputErrorBuffer) {
  if (inputErrorBuffer == nullptr) return;
  
  int batchSize = lastTrainBatchSize;
  int previousNeurons = layers[0].countNeurons();

  auto status = clblast::Gemm(clblast::Layout::kRowMajor, clblast::Transpose::kNo, clblast::Transpose::kYes,
			      batchSize, networkInputs, previousNeurons, 1.0f,
			      lastTrainHiddenDeltas[0], 0, previousNeurons,
			      layers[0].weightsBuffer, 0, networkInputs, 0.0f,
			      inputErrorBuffer, 0, networkInputs, &queue, nullptr);
  
  if (status != clblast::StatusCode::kSuccess) throw std::runtime_error("GEMM error: " + std::to_string(static_cast<int>(status)));
  err = clFinish(queue);
  clCheckErr(err, "Finishing kernel run");
}

void Network::saveToFile(const std::string& path, bool saveAdamMomentums) {
  /* –––––––––––––––– Structure –––––––––––––– //
     - savedAdam?

     - BETA1
     - BETA2
     - EPSILON
     - WEIGHT_DECAY
     - timestepAdam
     
     - learningRate
     - layer count
     - network inputs

     - for each layer:
       - activation function
       - neurons
       - previous neurons
       - biases
       - weights

       - expAvgB
       - expAvgSqB
       - expAvgW
       - expAvgSqW
  */

  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) throw std::runtime_error("Failed to create or open file for writing: " + path);

  int layerCount = layers.size();

  out.write(reinterpret_cast<const char*>(&saveAdamMomentums), sizeof(saveAdamMomentums));

  if (saveAdamMomentums) {
    out.write(reinterpret_cast<const char*>(&BETA1), sizeof(BETA1));
    out.write(reinterpret_cast<const char*>(&BETA2), sizeof(BETA2));
    out.write(reinterpret_cast<const char*>(&EPSILON), sizeof(EPSILON));
    out.write(reinterpret_cast<const char*>(&WEIGHT_DECAY), sizeof(WEIGHT_DECAY));
    out.write(reinterpret_cast<const char*>(&timestepAdam), sizeof(timestepAdam));
  }
  
  out.write(reinterpret_cast<const char*>(&learningRate), sizeof(learningRate));  
  out.write(reinterpret_cast<const char*>(&networkInputs), sizeof(networkInputs));
  out.write(reinterpret_cast<const char*>(&layerCount), sizeof(layerCount));

  for (Layer& l : layers) {
    int neurons = l.neurons;
    int previous = l.previousNeurons;
    Activation function = l.activationFunction;
    
    out.write(reinterpret_cast<const char*>(&function), sizeof(function));    
    out.write(reinterpret_cast<const char*>(&neurons), sizeof(neurons));
    out.write(reinterpret_cast<const char*>(&previous), sizeof(previous));

    cl_mem biases = l.biasesBuffer;
    cl_mem weights = l.weightsBuffer;

    std::vector<float> biasVector(neurons);
    std::vector<float> weightVector(neurons * previous);

    err = clEnqueueReadBuffer(queue, biases, CL_TRUE, 0, sizeof(float) * neurons, biasVector.data(), 0, nullptr, nullptr);
    err |= clEnqueueReadBuffer(queue, weights, CL_TRUE, 0, sizeof(float) * neurons * previous, weightVector.data(), 0, nullptr, nullptr);
    clCheckErr(err, "Reading cl_mem buffers");

    out.write(reinterpret_cast<const char*>(biasVector.data()), sizeof(biasVector[0]) * biasVector.size());
    out.write(reinterpret_cast<const char*>(weightVector.data()), sizeof(weightVector[0]) * weightVector.size());

    if (saveAdamMomentums) {
      cl_mem expAvgB = l.expAvgB;
      cl_mem expAvgSqB = l.expAvgSqB;
      cl_mem expAvgW = l.expAvgW;
      cl_mem expAvgSqW = l.expAvgSqW;

      std::vector<float> expAvgBVector(neurons);
      std::vector<float> expAvgSqBVector(neurons);
      std::vector<float> expAvgWVector(neurons * previous);
      std::vector<float> expAvgSqWVector(neurons * previous);

      err = clEnqueueReadBuffer(queue, expAvgB, CL_TRUE, 0, sizeof(float) * neurons, expAvgBVector.data(), 0, nullptr, nullptr);
      err |= clEnqueueReadBuffer(queue, expAvgSqB, CL_TRUE, 0, sizeof(float) * neurons, expAvgSqBVector.data(), 0, nullptr, nullptr);
      err |= clEnqueueReadBuffer(queue, expAvgW, CL_TRUE, 0, sizeof(float) * neurons * previous, expAvgWVector.data(), 0, nullptr, nullptr);
      err |= clEnqueueReadBuffer(queue, expAvgSqW, CL_TRUE, 0, sizeof(float) * neurons * previous, expAvgSqWVector.data(), 0, nullptr, nullptr);
      clCheckErr(err, "Reading cl_mem buffers");

      out.write(reinterpret_cast<const char*>(expAvgBVector.data()), sizeof(expAvgBVector[0]) * expAvgBVector.size());
      out.write(reinterpret_cast<const char*>(expAvgSqBVector.data()), sizeof(expAvgSqBVector[0]) * expAvgSqBVector.size());
      out.write(reinterpret_cast<const char*>(expAvgWVector.data()), sizeof(expAvgWVector[0]) * expAvgWVector.size());
      out.write(reinterpret_cast<const char*>(expAvgSqWVector.data()), sizeof(expAvgSqWVector[0]) * expAvgSqWVector.size());
    }
  }
  
  if (!out) throw std::runtime_error("Failed to write to path: " + path);
}

void Network::loadFromFile(const std::string& path) {
  /* –––––––––––––––– Structure –––––––––––––– //
     - savedAdam?

     - BETA1
     - BETA2
     - EPSILON
     - WEIGHT_DECAY
     - timestepAdam
     
     - learningRate
     - layer count
     - network inputs

     - for each layer:
       - activation function
       - neurons
       - previous neurons
       - biases
       - weights

       - expAvgB
       - expAvgSqB
       - expAvgW
       - expAvgSqW
  */

  std::ifstream in (path, std::ios::binary);
  if (!in) throw std::runtime_error("Failed to open file for reading the network: " + path);

  bool saveAdamMomentum;
  float lr;
  int lc, ni;

  in.read(reinterpret_cast<char*>(&saveAdamMomentum), sizeof(saveAdamMomentum));
  if (saveAdamMomentum) {
    float b1, b2, e, wd;
    int ta;
    
    in.read(reinterpret_cast<char*>(&b1), sizeof(b1));
    in.read(reinterpret_cast<char*>(&b2), sizeof(b2));
    in.read(reinterpret_cast<char*>(&e), sizeof(e));
    in.read(reinterpret_cast<char*>(&wd), sizeof(wd));
    in.read(reinterpret_cast<char*>(&ta), sizeof(ta));

    BETA1 = b1;
    BETA2 = b2;
    EPSILON = e;
    WEIGHT_DECAY = wd;
    timestepAdam = ta;
  }
  
  in.read(reinterpret_cast<char*>(&lr), sizeof(lr));
  in.read(reinterpret_cast<char*>(&ni), sizeof(ni));
  in.read(reinterpret_cast<char*>(&lc), sizeof(lc));

  learningRate = lr;
  networkInputs = ni;
  
  layers.reserve(lc);
  for (int l = 0; l < lc; l++) {
    int neurons, previous;
    Activation function;
    in.read(reinterpret_cast<char*>(&function), sizeof(function));
    in.read(reinterpret_cast<char*>(&neurons), sizeof(neurons));
    in.read(reinterpret_cast<char*>(&previous), sizeof(previous));

    std::vector<float> biasVector(neurons);
    std::vector<float> weightVector(neurons * previous);
    
    in.read(reinterpret_cast<char*>(biasVector.data()), sizeof(biasVector[0]) * biasVector.size());
    in.read(reinterpret_cast<char*>(weightVector.data()), sizeof(weightVector[0]) * weightVector.size());
    
    cl_mem biases = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * neurons, biasVector.data(), &err);
    clCheckErr(err, "Creation of cl_mem: \"biases\"");
    cl_mem weights = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * neurons * previous, weightVector.data(), &err);
    clCheckErr(err, "Creation of cl_mem: \"weights\"");

    layers.emplace_back(context, err, queue, previous, neurons, function);
    layers[l].setBiases(biases);
    layers[l].setWeights(weights);

    if (saveAdamMomentum) {
      std::vector<float> expAvgBVector(neurons);
      std::vector<float> expAvgSqBVector(neurons);
      std::vector<float> expAvgWVector(neurons * previous);
      std::vector<float> expAvgSqWVector(neurons * previous);

      in.read(reinterpret_cast<char*>(expAvgBVector.data()), sizeof(expAvgBVector[0]) * expAvgBVector.size());
      in.read(reinterpret_cast<char*>(expAvgSqBVector.data()), sizeof(expAvgSqBVector[0]) * expAvgSqBVector.size());
      in.read(reinterpret_cast<char*>(expAvgWVector.data()), sizeof(expAvgWVector[0]) * expAvgWVector.size());
      in.read(reinterpret_cast<char*>(expAvgSqWVector.data()), sizeof(expAvgSqWVector[0]) * expAvgSqWVector.size());

      cl_mem expAvgB = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * neurons, expAvgBVector.data(), &err);
      clCheckErr(err, "Creation of cl_mem: \"expAvgB\"");
      cl_mem expAvgSqB = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * neurons, expAvgSqBVector.data(), &err);
      clCheckErr(err, "Creation of cl_mem: \"expAvgSqB\"");
      cl_mem expAvgW = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * neurons * previous, expAvgWVector.data(), &err);
      clCheckErr(err, "Creation of cl_mem: \"expAvgW\"");
      cl_mem expAvgSqW = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * neurons * previous, expAvgSqWVector.data(), &err);
      clCheckErr(err, "Creation of cl_mem: \"expAvgSqW\"");

      layers[l].setExpAvgB(expAvgB);
      layers[l].setExpAvgSqB(expAvgSqB);
      layers[l].setExpAvgW(expAvgW);
      layers[l].setExpAvgSqW(expAvgSqW);
    }
  }

  if (!in) throw std::runtime_error("Failed to read from file: " + path);
}

void Network::safeReleaseKernel(cl_kernel& k) {
  if (k) clReleaseKernel(k);
  k = nullptr;
}

void Network::safeReleaseMemory(cl_mem& m) {
  if (m) clReleaseMemObject(m);
  m = nullptr;
}

const char* Network::clErrorToString(cl_int err) {
  switch (err) {
  case CL_SUCCESS: return "CL_SUCCESS";
  case CL_DEVICE_NOT_FOUND: return "CL_DEVICE_NOT_FOUND";
  case CL_DEVICE_NOT_AVAILABLE: return "CL_DEVICE_NOT_AVAILABLE";
  case CL_OUT_OF_RESOURCES: return "CL_OUT_OF_RESOURCES";
  case CL_OUT_OF_HOST_MEMORY: return "CL_OUT_OF_HOST_MEMORY";
  case CL_INVALID_VALUE: return "CL_INVALID_VALUE";
  case CL_INVALID_DEVICE: return "CL_INVALID_DEVICE";
  case CL_INVALID_CONTEXT:  return "CL_INVALID_CONTEXT";
  case CL_INVALID_COMMAND_QUEUE: return "CL_INVALID_COMMAND_QUEUE";
  case CL_INVALID_MEM_OBJECT: return "CL_INVALID_MEM_OBJECT";
  case CL_INVALID_KERNEL: return "CL_INVALID_KERNEL";
  case CL_INVALID_KERNEL_ARGS: return "CL_INVALID_KERNEL_ARGS";
  case CL_INVALID_WORK_DIMENSION: return "CL_INVALID_WORK_DIMENSION";
  case CL_INVALID_WORK_GROUP_SIZE: return "CL_INVALID_WORK_GROUP_SIZE";
  case CL_INVALID_ARG_INDEX: return "CL_INVALID_ARG_INDEX";
  case CL_INVALID_ARG_VALUE: return "CL_INVALID_ARG_VALUE";
  case CL_INVALID_BUFFER_SIZE: return "CL_INVALID_BUFFER_SIZE";
  case CL_INVALID_PROGRAM: return "CL_INVALID_PROGRAM";
  case CL_INVALID_PROGRAM_EXECUTABLE: return "CL_INVALID_PROGRAM_EXECUTABLE";
  case CL_BUILD_PROGRAM_FAILURE: return "CL_BUILD_PROGRAM_FAILURE";
  default: return "UNKNOWN_CL_ERROR";
  }
}

void Network::clCheckErr(cl_int err, const char* location) {
  if (err != CL_SUCCESS)
    throw std::runtime_error(std::string(location) + " failed: " + clErrorToString(err));
}
