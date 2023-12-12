//
//  main.swift
//  AI_Assignment
//
//  Created by 이치훈 on 11/24/23.
//

import Foundation

public class ActivationFunction {
  
  // sigmoid ActivationFunction :: differential == true, sigmoid 미분값을 return함
  class func sigmoid(x: Double, differential: Bool = false) -> Double {
    if differential == true {
      // O1 * {1 - O1}
      return x * (1.0 - x)
    } else {
      return 1.0 / (1.0 + exp(-x))
    }
  }
  
}

public class NeuralNetConstants {
  
  public static let learningRate: Double = 0.03
  public static let momentum: Double = 0.6
  public static let iterations: Int = 100000
  
}

public class Layer {
  
  private var output: [Double]
  private var input: [Double]
  private var weights: [Double]
  private var dWeights: [Double]
  
  init(inputSize: Int, outputSize: Int) {
    self.output = [Double](repeating: 0, count: outputSize)
    self.input = [Double](repeating: 0, count: inputSize + 1)
    self.weights = (0..<(1 + inputSize) * outputSize).map { _ in
      return Double.random(in: -0.5...0.5)
    }
    self.dWeights = [Double](repeating: 0, count: weights.count)
  }
  
  // MARK: Forward Passing
  public func activation(inputArray: [Double]) -> [Double] {
    
    for (i, e) in inputArray.enumerated() {
      self.input[i] = e
    }
    input[input.count-1] = 1
    var offSet = 0
    
    for i in 0..<output.count {
      for j in 0..<input.count {
        //Get NET
        output[i] += weights[offSet+j] * input[j]
      }
      
      output[i] = ActivationFunction.sigmoid(x: output[i])
      offSet += input.count
    }
    
    return output
  }
  
  // MARK: BackPropagation
  public func deltaRule(error: [Double], learningRate: Double, momentum: Double) -> [Double] {
    var offset = 0
    var nextError = [Double](repeating: 0, count: input.count)
    
    for i in 0..<output.count {
      
      let delta = error[i] * ActivationFunction.sigmoid(x: output[i], differential: true)
      
      for j in 0..<input.count {
        let weightIndex = offset + j
        nextError[j] = nextError[j] + weights[weightIndex] * delta
        // dw: 연결강도 변화량 (에타(학습률)*델타*input)
        let dw = learningRate * delta * input[j]
        weights[weightIndex] += dWeights[weightIndex] * momentum + dw
        dWeights[weightIndex] = dw
      }
      
      offset += input.count
    }
    
    return nextError
  }
  
}

public class NeuralNetwork {
  
  private var layers: [Layer] = []
  
  public init(inputSize: Int, hiddenSize: Int, outputSize: Int) {
    self.layers.append(Layer(inputSize: inputSize, outputSize: hiddenSize))
    self.layers.append(Layer(inputSize: hiddenSize, outputSize: outputSize))
  }
  
  public func forwardPasing(input: [Double]) -> [Double] {
    var activations = input
    
    for i in 0..<layers.count {
      activations = layers[i].activation(inputArray: activations)
    }
    
    return activations
  }
  
  public func backpropagation(input: [Double], targetOutput: [Double], learningRate: Double, momentum: Double) {
    let calculatedOutput = forwardPasing(input: input)
    var error = [Double](repeating: 0, count: calculatedOutput.count)
    
    for i in 0..<error.count {
      error[i] = targetOutput[i] - calculatedOutput[i]
    }
    
    for i in (0...layers.count-1).reversed() {
      error = layers[i].deltaRule(error: error, learningRate: learningRate, momentum: momentum)
    }
    
  }
  
}

// MARK: - Traning

// 가중치 저장해놓고 testData project애서 고정 가중치로 활용
var traningData: [[Double]] = [
  [6.3,    3.3,    6.0,    2.5],
  [5.8,    2.7,    5.1,    1.9],
  [7.1,    3.0,    5.9,    2.1],
  [6.3,    2.9,    5.6,    1.8],
  [6.5,    3.0,    5.8,    2.2],
  [7.6,    3.0,    6.6,    2.1],
  [4.9,    2.5,    4.5,    1.7],
  [7.3,    2.9,    6.3,    1.8],
  [6.7,    2.5,    5.8,    1.8],
  [7.2,    3.6,    6.1,    2.5],
  [6.5,    3.2,    5.1,    2.0],
  [6.4,    2.7,    5.3,    1.9],
  [6.8,    3.0,    5.5,    2.1],
  [5.7,    2.5,    5.0,    2.0],
  [5.8,    2.8,    5.1,    2.4],
  [6.4,    3.2,    5.3,    2.3],
  [6.5,    3.0,    5.5,    1.8],
  [7.7,    3.8,    6.7,    2.2],
  [7.7,    2.6,    6.9,    2.3],
  [6.0,    2.2,    5.0,    1.5],
  [6.9,    3.2,    5.7,    2.3],
  [5.6,    2.8,    4.9,    2.0],
  [7.7,    2.8,    6.7,    2.0],
  [6.3,    2.7,    4.9,    1.8],
  [6.7,    3.3,    5.7,    2.1],
  [7.0,    3.2,    4.7,    1.4],
  [6.4,    3.2,    4.5,    1.5],
  [6.9,    3.1,    4.9,    1.5],
  [5.5,    2.3,    4.0,    1.3],
  [6.5,    2.8,    4.6,    1.5],
  [5.7,    2.8,    4.5,    1.3],
  [6.3,    3.3,    4.7,    1.6],
  [4.9,    2.4,    3.3,    1.0],
  [6.6,    2.9,    4.6,    1.3],
  [5.2,    2.7,    3.9,    1.4],
  [5.0,    2.0,    3.5,    1.0],
  [5.9,    3.0,    4.2,    1.5],
  [6.0,    2.2,    4.0,    1.0],
  [6.1,    2.9,    4.7,    1.4],
  [5.6,    2.9,    3.6,    1.3],
  [6.7,    3.1,    4.4,    1.4],
  [5.6,    3.0,    4.5,    1.5],
  [5.8,    2.7,    4.1,    1.0],
  [6.2,    2.2,    4.5,    1.5],
  [5.6,    2.5,    3.9,    1.1],
  [5.9,    3.2,    4.8,    1.8],
  [6.1,    2.8,    4.0,    1.3],
  [6.3,    2.5,    4.9,    1.5],
  [6.1,    2.8,    4.7,    1.2],
  [6.4,    2.9,    4.3,    1.3],
  // 여기서 부터 class 3
  [5.1,    3.5,    1.4,    0.2],
  [4.9,    3.0,    1.4,    0.2],
  [4.7,    3.2,    1.3,    0.2],
  [4.6,    3.1,    1.5,    0.2],
  [5.0,    3.6,    1.4,    0.2],
  [5.4,    3.9,    1.7,    0.4],
  [4.6,    3.4,    1.4,    0.3],
  [5.0,    3.4,    1.5,    0.2],
  [4.4,    2.9,    1.4,    0.2],
  [4.9,    3.1,    1.5,    0.1],
  [5.4,    3.7,    1.5,    0.2],
  [4.8,    3.4,    1.6,    0.2],
  [4.8,    3.0,    1.4,    0.1],
  [4.3,    3.0,    1.1,    0.1],
  [5.8,    4.0,    1.2,    0.2],
  [5.7,    4.4,    1.5,    0.4],
  [5.4,    3.9,    1.3,    0.4],
  [5.1,    3.5,    1.4,    0.3],
  [5.7,    3.8,    1.7,    0.3],
  [5.1,    3.8,    1.5,    0.3],
  [5.4,    3.4,    1.7,    0.2],
  [5.1,    3.7,    1.5,    0.4],
  [4.6,    3.6,    1.0,    0.2],
  [5.1,    3.3,    1.7,    0.5],
  [4.8,    3.4,    1.9,    0.2]
]
var traningTargets = [[Double]]()

for i in 1...75 {
  switch i {
  case 1...25:
    traningTargets.append([1, 0, 0])
  case 26...50:
    traningTargets.append([0, 1, 0])
  case 51...75:
    traningTargets.append([0, 0, 1])
  default:
    print("other num!")
  }
}

let neuralN = NeuralNetwork(inputSize: 4, hiddenSize: 3, outputSize: 3)

for i in 0..<NeuralNetConstants.iterations {
  
  for j in 0..<traningTargets.count {
    neuralN.backpropagation(input: traningData[j], targetOutput: traningTargets[j], learningRate: NeuralNetConstants.learningRate, momentum: NeuralNetConstants.momentum)
  }
  
  for j in 0..<traningTargets.count {
    let t = traningData[j]
    let result = neuralN.forwardPasing(input: t)
    print("(traning: \(i+1)) \(t[0]), \(t[1]), \(t[2]), \(t[3])  --  \(result[0]), \(result[1]), \(result[2])")
  }
  
}

// MARK: - Testing

print("========================= <Test> =========================")
var testingData: [[Double]] = [
  [7.2,    3.2,    6.0,    1.8],
  [6.2,    2.8,    4.8,    1.8],
  [6.1,    3.0,    4.9,    1.8],
  [6.4,    2.8,    5.6,    2.1],
  [7.2,    3.0,    5.8,    1.6],
  [7.4,    2.8,    6.1,    1.9],
  [7.9,    3.8,    6.4,    2.0],
  [6.4,    2.8,    5.6,    2.2],
  [6.3,    2.8,    5.1,    1.5],
  [6.1,    2.6,    5.6,    1.4],
  [7.7,    3.0,    6.1,    2.3],
  [6.3,    3.4,    5.6,    2.4],
  [6.4,    3.1,    5.5,    1.8],
  [6.0,    3.0,    4.8,    1.8],
  [6.9,    3.1,    5.4,    2.1],
  [6.7,    3.1,    5.6,    2.4],
  [6.9,    3.1,    5.1,    2.3],
  [5.8,    2.7,    5.1,    1.9],
  [6.8,    3.2,    5.9,    2.3],
  [6.7,    3.3,    5.7,    2.5],
  [6.7,    3.0,    5.2,    2.3],
  [6.3,    2.5,    5.0,    1.9],
  [6.5,    3.0,    5.2,    2.0],
  [6.2,    3.4,    5.4,    2.3],
  [5.9,    3.0,    5.1,    1.8],
  [6.6,    3.0,    4.4,    1.4],
  [6.8,    2.8,    4.8,    1.4],
  [6.7,    3.0,    5.0,    1.7],
  [6.0,    2.9,    4.5,    1.5],
  [5.7,    2.6,    3.5,    1.0],
  [5.5,    2.4,    3.8,    1.1],
  [5.5,    2.4,    3.7,    1.0],
  [5.8,    2.7,    3.9,    1.2],
  [6.0,    2.7,    5.1,    1.6],
  [5.4,    3.0,    4.5,    1.5],
  [6.0,    3.4,    4.5,    1.6],
  [6.7,    3.1,    4.7,    1.5],
  [6.3,    2.3,    4.4,    1.3],
  [5.6,    3.0,    4.1,    1.3],
  [5.5,    2.5,    4.0,    1.3],
  [5.5,    2.6,    4.4,    1.2],
  [6.1,    3.0,    4.6,    1.4],
  [5.8,    2.6,    4.0,    1.2],
  [5.0,    2.3,    3.3,    1.0],
  [5.6,    2.7,    4.2,    1.3],
  [5.7,    3.0,    4.2,    1.2],
  [5.7,    2.9,    4.2,    1.3],
  [6.2,    2.9,    4.3,    1.3],
  [5.1,    2.5,    3.0,    1.1],
  [5.7,    2.8,    4.1,    1.3],
  [5.0,    3.0,    1.6,    0.2],
  [5.0,    3.4,    1.6,    0.4],
  [5.2,    3.5,    1.5,    0.2],
  [5.2,    3.4,    1.4,    0.2],
  [4.7,    3.2,    1.6,    0.2],
  [4.8,    3.1,    1.6,    0.2],
  [5.4,    3.4,    1.5,    0.4],
  [5.2,    4.1,    1.5,    0.1],
  [5.5,    4.2,    1.4,    0.2],
  [4.9,    3.1,    1.5,    0.2],
  [5.0,    3.2,    1.2,    0.2],
  [5.5,    3.5,    1.3,    0.2],
  [4.9,    3.6,    1.4,    0.1],
  [4.4,    3.0,    1.3,    0.2],
  [5.1,    3.4,    1.5,    0.2],
  [5.0,    3.5,    1.3,    0.3],
  [4.5,    2.3,    1.3,    0.3],
  [4.4,    3.2,    1.3,    0.2],
  [5.0,    3.5,    1.6,    0.6],
  [5.1,    3.8,    1.9,    0.4],
  [4.8,    3.0,    1.4,    0.3],
  [5.1,    3.8,    1.6,    0.2],
  [4.6,    3.2,    1.4,    0.2],
  [5.3,    3.7,    1.5,    0.2],
  [5.0,    3.3,    1.4,    0.2]
]

for i in 0..<testingData.count {
  let t = testingData[i]
  let result = neuralN.forwardPasing(input: testingData[i])
  print("(test: \(i+1)) \(t[0]), \(t[1]), \(t[2]), \(t[3])  --  \(result[0]), \(result[1]), \(result[2])")
}
