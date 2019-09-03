/**
 * @license
 * This file is written by MemoriesOff (https://github.com/MemoriesOff) 
 * you can use it based on the Apache License, Version 2.0
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * =============================================================================
 */


class SemanticSegmentation {
	class_names = ['background', 'aeroplane', 'bicycle', 'bird', 'boat','bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable','dog', 'horse', 'motorbike', 'person', 'potted-plant','sheep', 'sofa', 'train', 'tv/monitor', 'ambigious'];
	constructor(modelAddress,modelPixels=[198, 352]){
		this.modelAddress = modelAddress;
		this.modelPixels=modelPixels;
	}			
	async load(){
		this.modelLoader = await tf.loadGraphModel(this.modelAddress);
		this.modelWeights=new ModelWeights(this.modelLoader);
		this.modelNet=new MobileNet(this.modelWeights);
		this.isLoad=true;
		console.log("loadfinish")
	}
	predict(inputImage,isOnlyPerson,isFcn32){
		return tf.tidy(() => {
			const imageTensor = tf.browser.fromPixels(inputImage);
			const {resizedAndPadded,paddedBy,} =  resizeAndPadTo(imageTensor, this.modelPixels); 
			const mobileNetOutput = this.modelNet.predict(resizedAndPadded, 32);
            const segments =this.modelNet.convToOutput(mobileNetOutput,isOnlyPerson,isFcn32);
			const [resizedHeight, resizedWidth] = resizedAndPadded.shape;
            const [height, width] = imageTensor.shape;
			const scaledSegmentScores = scaleAndCropToInputTensorShape(
            segments, [height, width], [resizedHeight, resizedWidth],
            paddedBy);
            if(isOnlyPerson) return scaledSegmentScores.sigmoid();
			return scaledSegmentScores.argMax(2)
		});
	}
}
