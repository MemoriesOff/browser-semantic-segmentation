
function resizeAndPadTo(
    imageTensor, [targetH, targetW],
    flipHorizontal = false){
  const [height, width] = imageTensor.shape;

  const targetAspect = targetW / targetH;
  const aspect = width / height;

  let resizeW;
  let resizeH;
  let padL;
  let padR;
  let padT;
  let padB;

  if (aspect > targetAspect) {
    // resize to have the larger dimension match the shape.
    resizeW = targetW;
    resizeH = Math.ceil(resizeW / aspect);

    const padHeight = targetH - resizeH;
    padL = 0;
    padR = 0;
    padT = Math.floor(padHeight / 2);
    padB = targetH - (resizeH + padT);
  } else {
    resizeH = targetH;
    resizeW = Math.ceil(targetH * aspect);

    const padWidth = targetW - resizeW;
    padL = Math.floor(padWidth / 2);
    padR = targetW - (resizeW + padL);
    padT = 0;
    padB = 0;
  }

  const resizedAndPadded = tf.tidy(() => {
    // resize to have largest dimension match image
    let resized;
    if (flipHorizontal) {
      resized = imageTensor.reverse(1).resizeBilinear([resizeH, resizeW],true);
    } else {
      resized = imageTensor.resizeBilinear([resizeH, resizeW],true);
    }

    const padded = tf.pad3d(resized, [[padT, padB], [padL, padR], [0, 0]]);

    return padded;
  });

  return {resizedAndPadded, paddedBy: [[padT, padB], [padL, padR]]};
}


function scaleAndCropToInputTensorShape(
    tensor,[inputTensorHeight, inputTensorWidth],[resizedAndPaddedHeight, resizedAndPaddedWidth],
    [[padT, padB], [padL, padR]]){
  return tf.tidy(() => {
    const inResizedAndPaddedSize = tensor.resizeBilinear(
        [resizedAndPaddedHeight, resizedAndPaddedWidth], true);

    return removePaddingAndResizeBack(
        inResizedAndPaddedSize, [inputTensorHeight, inputTensorWidth],
        [[padT, padB], [padL, padR]]);
  });
}

function removePaddingAndResizeBack(
    resizedAndPadded,[originalHeight, originalWidth],
    [[padT, padB], [padL, padR]]){
  const [height, width] = resizedAndPadded.shape;
  // remove padding that was added
  const cropH = height - (padT + padB);
  const cropW = width - (padL + padR);

  return tf.tidy(() => {
    const withPaddingRemoved = tf.slice3d(
        resizedAndPadded, [padT, padL, 0],
        [cropH, cropW, resizedAndPadded.shape[2]]);

    const atOriginalSize = withPaddingRemoved.resizeBilinear(
        [originalHeight, originalWidth], true);

    return atOriginalSize;
  });
}
