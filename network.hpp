#pragma once
#include <vector>
#include <random>
#include <iostream>
#include <cstring>
#include <fstream>
#include <stdexcept>

#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <clblast.h>

/*
  TODO:
  - enable disabling optimizer
  - enable returning input error for CPU
  - more efficient softmax
  - write CPU based counterpart
  - write documentation
*/

class Network {
public:
  enum class Activation {
    Linear,
    ReLU,
    LeakyReLU,
    Sigmoid,
    Tanh,
    Softmax
  };
  
  Network(const std::vector<int>& neuronsInLayers, Activation hiddenActivationFunction, Activation outputActivationFunction, float learningRate, bool fastMath = true, bool printDeviceInfo = false, int usedDeviceIndex = 0);
  Network(const std::vector<int>& neuronsInLayers, const std::vector<Activation>& activationFunctions, float learningRate, bool fastMath = true, bool printDeviceInfo = false, int usedDeviceIndex = 0);
  Network(const std::string& path, bool fastMath = true, bool printDeviceInfo = false, int usedDeviceIndex = 0);
  ~Network();

  void setLearningRate(float l);
  void setBETA1(float b);
  void setBETA2(float b);
  void setEpsilon(float e);
  void setWeightDecay(float w);

  float getLearningRate();
  float getBETA1();
  float getBETA2();
  float getEpsilon();
  float getWeightDecay();

  void setDropoutRate(std::vector<float> rates);
  std::vector<float> getDropoutRates();

  std::vector<std::vector<float>> run(const std::vector<std::vector<float>>& inputs);
  std::vector<std::vector<float>> run(cl_mem inputs, int batchSize);
  cl_mem runCL_MEM(cl_mem inputs, int batchSize);
  cl_mem runCL_MEM(const std::vector<std::vector<float>>& inputs);

  void trainSupervised(const std::vector<std::vector<float>>& inputs, const std::vector<std::vector<float>>& expectedOutputs, cl_mem inputErrorBuffer = nullptr);
  void trainSupervised(cl_mem inputs, cl_mem expectedOutputs, int batchSize, cl_mem inputErrorBuffer = nullptr);
  void trainReinforcement(const std::vector<std::vector<float>>& inputs, const std::vector<int>& chosenActions, std::vector<float> rewards, cl_mem inputErrorBuffer = nullptr);
  void trainReinforcement(cl_mem inputs, cl_mem chosenActions, cl_mem rewards, int batchSize, cl_mem inputErrorBuffer = nullptr);
  void trainReinforcement(const std::vector<std::vector<float>>& inputs, const std::vector<std::vector<float>>& actions, std::vector<float> rewards, float sigma, cl_mem inputErrorBuffer = nullptr);
  void trainReinforcement(cl_mem inputs, cl_mem actions, cl_mem rewards, int batchSize, float sigma, cl_mem inputBufferError = nullptr);

  void saveToFile(const std::string& path, bool saveAdamMomentums = false);
  
private:  
  class Layer {
  public:    
    Layer(cl_context context, cl_int err, cl_command_queue queue, int previousNeurons, int neurons, Network::Activation activationFunction);
    ~Layer();

    void setBiases(cl_mem biases);
    void setWeights(cl_mem weights);
    void setExpAvgB(cl_mem buffer);
    void setExpAvgSqB(cl_mem buffer);
    void setExpAvgW(cl_mem buffer);
    void setExpAvgSqW(cl_mem buffer);
    
    int countNeurons();
    void run(cl_kernel kernel, cl_mem inputBuffer, cl_mem outputBuffer, int batchSize);
    void run(cl_kernel kernel, cl_mem inputBuffer, cl_mem outputBuffer, cl_mem activations, cl_mem zs, cl_mem mask, int batchSize);

    void setDropoutRate(float rate);
    float getDropoutRate();
    
    cl_mem biasesBuffer = nullptr;
    cl_mem weightsBuffer = nullptr;

    cl_mem expAvgB = nullptr;
    cl_mem expAvgSqB = nullptr;
    cl_mem expAvgW = nullptr;
    cl_mem expAvgSqW = nullptr;
    
    int previousNeurons;
    int neurons;

    Network::Activation activationFunction;

  private:
    cl_context context = nullptr;
    cl_int err = 0;
    cl_command_queue queue = nullptr;

    float dropoutRate = 0.0f;

    std::mt19937 gen;
    std::uniform_int_distribution<uint32_t> dist;
  };

  // Adam parameters
  float BETA1 = 0.9f;
  float BETA2 = 0.999f;
  float EPSILON = 1e-08;
  float WEIGHT_DECAY = 1e-4;
  int timestepAdam = 0;

  cl_device_id getDevice(bool printDeviceInfo, int usedDeviceIndex);
  std::string getNetworkKernelSource(Activation activationFunction);
  void createKernels();
  
  std::vector<float> flat2DArray(const std::vector<std::vector<float>>& array);
  std::vector<std::vector<float>> construct2DArray(int x, int y, const std::vector<float>& array);
  void getActivationsAndZs();
  void computeHiddenDeltas();

  void computeOutputDeltasSupervised();
  void computeOutputDeltasSingleActionReinforcement();
  void computeOutputDeltasOutputVectorReinforcement(float sigma);

  void updateNetwork();
  void calculateInputError(cl_mem inputErrorBuffer);
  void reinitializeClMemObjects(int maxNeurons);

  static void safeReleaseKernel(cl_kernel& k);
  static void safeReleaseMemory(cl_mem& m);
  static const char* clErrorToString(cl_int err);
  static void clCheckErr(cl_int err, const char* location);
  
  void loadFromFile(const std::string& path);

  std::vector<Layer> layers;
  std::vector<cl_program> programs;

  float learningRate;

  int networkInputs;
  
  cl_context context = nullptr;
  cl_int err = 0;
  cl_command_queue queue = nullptr;
  cl_device_id device = nullptr;

  std::vector<cl_kernel> biasAndActivateKernels;
  std::vector<cl_kernel> zGetKernels;
  std::vector<cl_kernel> deltaKernels;

  cl_kernel averageNetworkUpdateBiasKernel = nullptr;
  cl_kernel averageNetworkUpdateWeightKernel = nullptr;
  cl_kernel softMaxKernel = nullptr;
  cl_kernel copyProbsToActivationKernel = nullptr;
  
  cl_kernel outputDeltaSingleActionReinforcementKernel = nullptr;
  cl_kernel outputDeltaOutputVectorReinforcementKernel = nullptr;

  int lastRunBatchSize = -1;
  cl_mem lastRunInputBuffer = nullptr;
  cl_mem lastRunOutputBuffer = nullptr;

  int lastTrainBatchSize = -1;
  cl_mem lastTrainInputBuffer = nullptr;
  cl_mem lastTrainInputBuffer2 = nullptr;
  cl_mem lastTrainOutputBuffer = nullptr;
  cl_mem lastTrainOutputDeltas = nullptr;
  cl_mem lastTrainExpectedOutputs = nullptr;

  cl_mem onesBuffer = nullptr;
  cl_mem lastTrainRewardBuffer = nullptr;

  std::vector<cl_mem> lastTrainActivations;
  std::vector<cl_mem> lastTrainDropoutMask;
  std::vector<cl_mem> lastTrainZs;
  std::vector<cl_mem> lastTrainHiddenDeltas;
  std::vector<cl_mem> weightGradientBuffer;
  std::vector<cl_mem> biasGradientBuffer;
};
