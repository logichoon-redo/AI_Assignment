//
//  main.swift
//  AI_Assignment
//
//  Created by 이치훈 on 11/24/23.
//

import Foundation

public class ActivationFunction {
  
  // sigmoid ActivationFunction :: differential == true, sigmoid 미분값을 return함
  class func sigmoid(v: Float, differential: Bool = false) -> Float {
//    let x = 1.0 / (1.0 + exp(-v))
    
    if differential == true {
      return v * (1.0 - v)
    } else {
      return 1.0 / (1.0 + exp(-v))
    }
  }
  
}

public class NeuralNetConstants {
  
  public static let learningRate: Float = 0.3
  public static let momentum: Float = 0.03
  public static let iterations: Int = 10000
  
}

public class Layer {
  
  private var output: [Float]
  private var input: [Float]
  var weights: [Float]
  private var dWeights: [Float]
  
  init(inputSize: Int, outputSize: Int) {
    self.output = [Float](repeating: 0, count: outputSize)
    self.input = [Float](repeating: 0, count: inputSize + 1)
    self.weights = (0..<(1 + inputSize) * outputSize).map { _ in
      return (-0.5...0.5).random()
    }
    self.dWeights = [Float](repeating: 0, count: weights.count)
  }
  
  // Forward Passing
  public func run(inputArray: [Float]) -> [Float] {
    
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
      
      output[i] = ActivationFunction.sigmoid(v: output[i])
      offSet += input.count
    }
    
    return output
  }
  
  // BackPropagation
  public func train(error: [Float], learningRate: Float, momentum: Float) -> [Float] {
    
    var offset = 0
    var nextError = [Float](repeating: 0, count: input.count)
    
    for i in 0..<output.count {
      
      let delta = error[i] * ActivationFunction.sigmoid(v: output[i], differential: true)
      
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

public class BackpropNeuralNetwork {
  
  var layers: [Layer] = []
  
  public init(inputSize: Int, hiddenSize: Int, outputSize: Int) {
    self.layers.append(Layer(inputSize: inputSize, outputSize: hiddenSize))
    self.layers.append(Layer(inputSize: hiddenSize, outputSize: outputSize))
  }
  
  public func run(input: [Float]) -> [Float] {
    var activations = input
    
    for i in 0..<layers.count {
      activations = layers[i].run(inputArray: activations)
    }
    
    return activations
  }
  
  public func train(input: [Float], targetOutput: [Float], learningRate: Float, momentum: Float) {
    let calculatedOutput = run(input: input)
    var error = [Float](repeating: 0, count: calculatedOutput.count)
    
    for i in 0..<error.count {
      error[i] = targetOutput[i] - calculatedOutput[i]
    }
    
    for i in (0...layers.count-1).reversed() {
      error = layers[i].train(error: error, learningRate: learningRate, momentum: momentum)
    }
    
  }
  
}

extension ClosedRange where Bound: FloatingPoint {
  public func random() -> Bound {
    let range = self.upperBound - self.lowerBound
    let randomValue = (Bound(arc4random_uniform(UINT32_MAX)) / Bound(UINT32_MAX)) * range + self.lowerBound
    return randomValue
  }
}

// 가중치 저장해놓고 testData project애서 고정 가중치로 활용
var traningData: [[Float]] = [
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
var traningResults = [[Float]]()

for i in 1...75 {
  switch i {
  case 1...25:
    traningResults.append([1, 0, 0])
  case 26...50:
    traningResults.append([0, 1, 0])
  case 51...75:
    traningResults.append([0, 0, 1])
  default:
    print("other num!")
  }
}

let backProp = BackpropNeuralNetwork(inputSize: 4, hiddenSize: 3, outputSize: 3)

for j in 0..<NeuralNetConstants.iterations {
  
  for i in 0..<traningResults.count {
    backProp.train(input: traningData[i], targetOutput: traningResults[i], learningRate: NeuralNetConstants.learningRate, momentum: NeuralNetConstants.momentum)
  }
  
  for i in 0..<traningResults.count {
    let t = traningData[i]
    let result = backProp.run(input: t)
    print("(\(j)) \(t[0]), \(t[1]), \(t[2]), \(t[3])  --  \(result[0]), \(result[1]), \(result[2])")
  }
  
}

_=backProp.layers.map {
  // 연결강도
  print($0.weights)
}

