/**
 * @license
 * This file is written by MemoriesOff (https://github.com/MemoriesOff) 
 * you can use it based on the Apache License, Version 2.0
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * =============================================================================
 */




var inputFileAddress = document.getElementById('upload-image');
var imgUrl=document.getElementById('image-url');
var inputImg = document.getElementById('input-image');
var inputImgLarge=document.getElementById('input-image-large');
var stateOutput=document.getElementById('status-message');
var runButton=document.getElementById('run-button');
var fcnType=document.getElementById('fcn-type');
var outputImg=document.getElementById('output-image');
var outputLabel=document.getElementById('output-label');
var inputSizeWidth=document.getElementById('inputSize-width');
var inputSizeHeight=document.getElementById('inputSize-height');				


var network=new SemanticSegmentation("web_model/model.json",[512,512]);
var colorMap=[[155,155,200],[174,199,232],[255,127,14],[255,187,120],[44,160,44],[152,223,138],[214,39,40],[255,152,150],[148,103,189],[197,176,213],[140,86,75],[196,156,148],[227,119,194],[247,182,210],[127,127,127],[31,119,180],[188,189,34],[219,219,141],[23,190,207],[158,218,229],[199,199,199]];
network.load().then(()=>{
	stateOutput.innerHTML="models loading success";
	runButton.classList.remove("is-loading");	
});
imgUrl.oninput=function(){
	var url=imgUrl.value;
	var reg=/http:\/\/.+/;
	var reg2=/\.(gif|jpg|jpeg|png|GIF|JPG|PNG)$/;
	if(!(reg.test(url)&&reg2.test(url))){
		stateOutput.innerHTML="<font color=\"#FF0000\">The URL you entered is not valid.</font>";
	}else{
		inputImg.src = url;
		stateOutput.innerHTML="Loading image...";
		document.getElementById("input-card").classList.remove("is-invisible");
								
	}
};
inputImg.onerror=function(){
	stateOutput.innerHTML="<font color=\"#FF0000\">image loading errow</font>";
}
inputImg.onload=function(){
	stateOutput.innerHTML="image loading success";
}

inputFileAddress.onchange = function () {
	//1.获取选中的文件列表
	var fileList = inputFileAddress.files;
	var file = fileList[0];
	//读取文件内容
	var reader = new FileReader();
	reader.readAsDataURL(file);
	reader.onload = function (e) {
		//将结果显示到文本框
		imgUrl.value=inputFileAddress.value;
		//showCanvas(reader.result);
		inputImg.src = reader.result;
		document.getElementById("input-card").classList.remove("is-invisible");						
	}
}

runButton.onclick=async function(){
	if(stateOutput.innerHTML!="image loading success"&&stateOutput.innerHTML.indexOf('running finish.')==-1){
		stateOutput.innerHTML="<font color=\"#FF0000\">image loading errow</font>";
	}else{
		inputImgLarge.src=inputImg.src;
		stateOutput.innerHTML="start running"
		var start = new Date().getTime();
		const alpha=0.8;	
		var hashmap=new Array(21);
		network.modelPixels=[Number(inputSizeHeight.value),Number(inputSizeWidth.value)];
		var netOutput=await network.predict(inputImgLarge,false,fcnType.value=="fcn32").data();
		//var canvas =document.createElement('canvas');
		var canvas =outputImg;
		var ctx = canvas.getContext('2d');
		canvas.width=inputImgLarge.width;
		canvas.height=inputImgLarge.height;
		ctx.drawImage(inputImgLarge, 0, 0, inputImgLarge.width, inputImgLarge.height);
		var imgData=ctx.getImageData(0, 0,inputImgLarge.width, inputImgLarge.height);
		for(var i=0;i<netOutput.length;i++){
			   hashmap[netOutput[i]]=1;
			   imgData.data[4*i+0]=colorMap[netOutput[i]][0]*alpha+imgData.data[4*i+0]*(1-alpha);
			   imgData.data[4*i+1]=colorMap[netOutput[i]][1]*alpha+imgData.data[4*i+1]*(1-alpha);				   
			   imgData.data[4*i+2]=colorMap[netOutput[i]][2]*alpha+imgData.data[4*i+2]*(1-alpha);
			   imgData.data[4*i+3]=255;									   
		}						
		ctx.putImageData(imgData,0, 0);	
		//outputImg.src="url("+canvas.toDataURL("png")+")";
		var label_ctx=outputLabel.getContext('2d');
		var g=0;
		for(var i=0; i<hashmap.length;i++){
			if(hashmap[i]===1) g++;
		}
		outputLabel.height=g*outputLabel.width/4*3;
		for(var i=0; i<hashmap.length;i++){
			if(hashmap[i]===1){
				
				label_ctx.beginPath()
				label_ctx.rect(0,g,outputLabel.width,outputLabel.width/16*9);
				label_ctx.fillStyle='rgb('+colorMap[i][0]+','+colorMap[i][1]+','+colorMap[i][2]+')';
				label_ctx.fill();
				label_ctx.beginPath()
				label_ctx.font = outputLabel.width/16*2+"px serif";
				label_ctx.fillStyle='#fff';		
				label_ctx.textAlign='center';//文本水平对齐方式
				label_ctx.textBaseline='middle';//文本垂直方向，基线位置 								
				label_ctx.fillText(network.class_names[i], outputLabel.width/2, g+outputLabel.width/32*9);
				g=g+outputLabel.width/4*3;								
							
			}	
		}						
		document.getElementById("output-card").classList.remove("is-invisible");	
		document.getElementById("legend-card").classList.remove("is-invisible");
		var end = new Date().getTime();						
		stateOutput.innerHTML="running finish. time: "+(end-start)+" ms";					
	}
}