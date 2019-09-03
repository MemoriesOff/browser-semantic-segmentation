/**
 * @license
 * This file is revised by MemoriesOff (https://github.com/MemoriesOff) 
 * based on the original version at 
 * https://github.com/tensorflow/tfjs-models/blob/master/body-pix/src/model_weights.ts
 * The author mainly changed the route of the weights.
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

class ModelWeights {
  //var graphModel;

  constructor(graphModel) {
    this.graphModel = graphModel;
  }
  weights(layerName) {
    return this.getVariable(layerName) 
  }
  convweights(layerName) {
    return this.getVariable(`base_fe_scope/MobilenetV1/MobilenetV1/${layerName}/Conv2D/merged_input`) 
  }

  convBias(layerName, doublePrefix = true) {
    return this.getVariable(`base_fe_scope/MobilenetV1/MobilenetV1/${layerName}/BatchNorm/FusedBatchNorm/Offset`) 
  }

  depthwiseBias(layerName) {
    return this.getVariable(`base_fe_scope/MobilenetV1/MobilenetV1/${layerName}/BatchNorm/FusedBatchNorm/Offset`)
  }

  depthwiseWeights(layerName) {
    return this.getVariable(`base_fe_scope/MobilenetV1/${layerName}/depthwise_weights`)
  }
  
  depthwiseWeightsScaled(layerName){
    return this.getVariable(`base_fe_scope/MobilenetV1/MobilenetV1/${layerName}/BatchNorm/FusedBatchNorm/Scaled`)
  }
  
  

  getVariable(name) {
	//console.log(name);
    return this.graphModel.weights[`${name}`][0];
  }

  dispose() {
    this.graphModel.dispose();
  }
}