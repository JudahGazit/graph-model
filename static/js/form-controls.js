const queryString = window.location.search;
const urlParams = new URLSearchParams(queryString);

function setDefaultForController(controller, name) {
  let parameterFromUri = urlParams.get(name);
  controller.value = parameterFromUri != null ? parameterFromUri : controller.value;
  controller.onchange();
}

function setDefaultFormValues() {
  let inputs = {
    '#inputNumberOfLeaves': 'leaves',
    '#inputA': 'A',
    '#inputB': 'B',
    '#inputAlpha': 'alpha',
    '#inputBeta': 'beta',

  }
  for (let inputsKey in inputs) {
    setDefaultForController($(inputsKey)[0], inputs[inputsKey]);
  }
}

function generateFormulaLink() {
  let A = $('#inputA')[0].value;
  let B = $('#inputB')[0].value;
  let alpha = $('#inputAlpha')[0].value;
  let beta = $('#inputBeta')[0].value;
  let link = "https://latex.codecogs.com/gif.latex?\\mathbb{P}(i,&space;j)&space;=&space;{A}\\cdot&space;e^{-{alpha}&space;\\cdot&space;|d_{i,&space;j}|}&space;&plus;&space;{B}&space;\\cdot&space;e^&space;{{beta}&space;|l_{i,&space;j}|}"
  link = link.replace('{A}', A).replace('{B}', B).replace('{alpha}', alpha).replace('{beta}', beta)
  $('#formula-img')[0].src = link;
}

function setBLimit() {
  let A = $('#inputA')[0].value;
  let B = $('#inputB')[0].value;
  let N = Math.pow(2, $('#inputNumberOfLeaves')[0].value);
  let betaMin = Math.log(1 / B) / 2;
  let betaMax = Math.log(1 / B) / Math.log(N);
  let alphaMax = Math.log(A) - Math.log(1 - B * Math.pow(N, betaMin))
  alphaMax /= ((2 * Math.PI) / N)
  $('#inputAlpha')[0].max = alphaMax;
  $('#inputBeta')[0].max = betaMax
}


function handleControllerChanged(controllerName, textName) {
  let value = $('#' + controllerName)[0].value;
  $('#' + textName)[0].value = value;
  generateFormulaLink();
  setBLimit();
}

function handleNumberOfLeavesChange() {
  let value = $('#inputNumberOfLeaves')[0].value;
  value = Math.pow(2, value);
  $('#staticNumberOfLeaves')[0].value = value;
  loadGraphBeforeSimulation(value);
}
