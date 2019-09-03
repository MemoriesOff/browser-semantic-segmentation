/**
 * @license
 * This file is revised by MemoriesOff (https://github.com/MemoriesOff) 
 * based on the original version at 
 * https://github.com/tensorflow/tfjs-models/blob/master/body-pix/src/mobilenet.ts
 * The author mainly modified the mobileNetArchitecture const and the
 * convToOutput function in order to generate the FCN model.
 * 
 * you can use it based on the Apache License, Version 2.0
 *
 * ===========================original notice===================================
 *
 * Copyright 2018 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */


const mobileNetArchitecture= [
  ['conv2d', 2],
  ['separableConv', 1],
  ['separableConv', 2],
  ['separableConv', 1],
  ['separableConv', 2],
  ['separableConv', 1],
  ['separableConv', 2],
  ['separableConv', 1],
  ['separableConv', 1],
  ['separableConv', 1],
  ['separableConv', 1],
  ['separableConv', 1],
//  ['separableConv', 2],
//  ['separableConv', 1]
];

const VALID_OUTPUT_STRIDES = [8, 16, 32];




/**
 * Takes a mobilenet architectures' convolution definitions and converts them
 * into definitions for convolutional layers that will generate outputs with the
 * desired output stride. It does this by reducing the input stride in certain
 * layers and applying atrous convolution in subsequent layers. Raises an error
 * if the output stride is not possible with the architecture.
 */
function toOutputStridedLayers(
    convolutionDefinition,
    outputStride){
  // The currentStride variable keeps track of the output stride of
  // the activations, i.e., the running product of convolution
  // strides up to the current network layer. This allows us to
  // invoke atrous convolution whenever applying the next
  // convolution would result in the activations having output
  // stride larger than the target outputStride.
  let currentStride = 1;

  // The atrous convolution rate parameter.
  let rate = 1;

  return convolutionDefinition.map(([convType, stride], blockId) => {
    let layerStride, layerRate;

    if (currentStride === outputStride) {
      // If we have reached the target outputStride, then we need to
      // employ atrous convolution with stride=1 and multiply the atrous
      // rate by the current unit's stride for use in subsequent layers.
      layerStride = 1;
      layerRate = rate;
      rate *= stride;
    } else {
      layerStride = stride;
      layerRate = 1;
      currentStride *= stride;
    }
  //  console.log(blockId+":"+stride+":"+rate+":"+outputStride);
    return {
      blockId,
      convType,
      stride: layerStride,
      rate: layerRate,
      outputStride: currentStride
    };
  });
}

class MobileNet {
  //private modelWeights;
  // private model: tf.NamedTensorMap;
 // private convolutionDefinitions;

  //private PREPROCESS_DIVISOR = tf.scalar(255.0 / 2);
  //private ONE = tf.scalar(1.0);

  constructor(modelWeights) {
    this.modelWeights = modelWeights;
    this.convolutionDefinitions = mobileNetArchitecture;
	this.PREPROCESS_DIVISOR = tf.scalar(255.0 / 2);
	this.ONE = tf.scalar(1.0);
  }

  predict(input, outputStride){
    // Normalize the pixels [0, 255] to be between [-1, 1].
    const normalized = tf.div(input.toFloat(), this.PREPROCESS_DIVISOR);

    const preprocessedInput = tf.sub(normalized, this.ONE) 
    const layers =
        toOutputStridedLayers(this.convolutionDefinitions, outputStride);

    return layers.reduce(
        (previousLayer,
         {blockId, stride, convType, rate}) => {
          if (convType === 'conv2d') {
            return this.conv(previousLayer, stride, blockId);
          } else if (convType === 'separableConv') {
            return this.separableConv(previousLayer, stride, blockId, rate);
          } else {
            throw Error(`Unknown conv type of ${convType}`);
          }
        },
        preprocessedInput);
  }

  convToOutput(mobileNetOutput,isOnlyPerson=false,isFcn32=false){	  
    //return this.fcnOutput(mobileNetOutput,"1x1_fcn32_logits",false);
	var fcn16=this.fcnOutput(mobileNetOutput,"1x1_fcn16_logits",isOnlyPerson);
	if(!isFcn32) {return fcn16};
	const sepConv12=this.separableConv(mobileNetOutput, 2, 12,1);
	const sepConv13=this.separableConv(sepConv12, 1, 13,1);
	var fcn32=this.fcnOutput(sepConv13,"1x1_fcn32_logits",false);
	if(isOnlyPerson){
		[,fcn32,]=fcn32.split([15,1,5],2);
		fcn32=fcn32.sigmoid();
		const fcn32_resize=fcn32.resizeBilinear([fcn16.shape[0],fcn16.shape[1]],true);
		return fcn16.sigmoid().add(fcn32_resize);
	}
	const fcn32_resize=fcn32.resizeBilinear([fcn16.shape[0],fcn16.shape[1]],true);
	return fcn16.add(fcn32_resize);
  }
  
  
  fcnOutput(inputLayer, outputLayerName, isOnlyPerson=false){
	var weights=this.weights(outputLayerName+'/weights');
	var biases=this.weights(outputLayerName+'/biases')
	if(isOnlyPerson){
		[,weights,]=weights.split([15,1,5],3);
		[,biases,]=biases.split([15,1,5],0);
	}
    return inputLayer.conv2d(weights,[1,1], 'same')
               .add(biases)
  }

   conv(inputs, stride, blockId) {
    const weights = this.convweights(`Conv2d_${String(blockId)}`);
    const a = inputs.conv2d(weights, stride, 'same');
    const b = a.add(this.convBias(`Conv2d_${String(blockId)}`));
    // relu6
    return b.clipByValue(0, 6) 
  }

  separableConv(
      inputs, stride, blockID,
      dilations = 1){
    const dwLayer = `Conv2d_${String(blockID)}_depthwise`;
    const pwLayer = `Conv2d_${String(blockID)}_pointwise`;

    const x1 = inputs
                   .depthwiseConv2D(
                       this.depthwiseWeights(dwLayer), stride, 'same', 'NHWC',
                       dilations)
                   .mul(this.depthwiseWeightsScaled(dwLayer)).add(this.depthwiseBias(dwLayer))
                   // relu6
                   .clipByValue(0, 6) ;

    const x2 = x1.conv2d(this.convweights(pwLayer), [1, 1], 'same')
                   .add(this.convBias(pwLayer))
                   // relu6
                   .clipByValue(0, 6) ;

    return x2;
  }
  weights(layerName){
    return this.modelWeights.weights(layerName);
  }
  convweights(layerName){
    return this.modelWeights.convweights(layerName);
  }

  convBias(layerName, doublePrefix = true){
    return this.modelWeights.convBias(layerName, doublePrefix);
  }

  depthwiseBias(layerName) {
    return this.modelWeights.depthwiseBias(layerName);
  }

  depthwiseWeights(layerName) {
    return this.modelWeights.depthwiseWeights(layerName);
  }
  depthwiseWeightsScaled(layerName) {
    return this.modelWeights.depthwiseWeightsScaled(layerName);
  }

  dispose() {
    this.modelWeights.dispose();
  }
}