let s = new sigma('graph-container')
const GRAPH_DEGREE = 2


function getCircularCoordinate(perimeterFraction, radius) {
  let degrees = 360 * perimeterFraction
  let radians = degrees * Math.PI / 180
  let xValue = Math.sin(radians) * radius
  let yValue = Math.cos(radians) * radius
  return [xValue, yValue]
}


function createNodesInCircle(numOfNodes, radius) {
  let nodes = []
  for (let i = 0; i < numOfNodes; i ++) {
    let xy = getCircularCoordinate((i + 0.5) / numOfNodes, radius)
    let node = {
      id: 'n' + i + 'r' + radius,
      label: i,
      size: 0.1,
      x: xy[0],
      y: xy[1]
    }
    nodes.push(node)
  }
  return nodes
}

function createEdgeBetweenTiers(numOfNodes, tierRadius, rootTierRadius) {
  let edges = []
  for(let i = 0; i < numOfNodes; i ++) {
    let nodeName = 'n' + i + 'r' + tierRadius
    let rootName = 'n' + Math.floor(i / GRAPH_DEGREE) + 'r' + rootTierRadius
    let edge = {
      id: 'e' + nodeName + rootName,
      source: rootName,
      target: nodeName
    }
    edges.push(edge)
  }
  return edges
}

function createRing(numberOfNodes, radius) {
  let edges = []
  for (let i = 0; i < numberOfNodes; i ++) {
    let nodeName = 'n' + i + 'r' + radius
    let nextName = 'n' + ((i + 1) % numberOfNodes) + 'r' + radius
    let edge = {
      id: 'e' + nodeName + nextName,
      source: nodeName,
      target: nextName,
    }
    edges.push(edge)
  }
  return edges
}

function createTree(numberOfLeaves) {
  let radius = 1
  let nodes = createNodesInCircle(numberOfLeaves, radius)
  let edges = createRing(numberOfLeaves, radius)
  while (Math.floor(numberOfLeaves / GRAPH_DEGREE) > 0) {
    let nextRadius = ((2 / 3) * GRAPH_DEGREE) * radius / GRAPH_DEGREE
    let nextNumberOfLeaves = Math.floor(numberOfLeaves / GRAPH_DEGREE)
    nodes = nodes.concat(createNodesInCircle(nextNumberOfLeaves, nextRadius))
    edges = edges.concat(createEdgeBetweenTiers(numberOfLeaves, radius, nextRadius))
    radius = nextRadius
    numberOfLeaves = nextNumberOfLeaves
  }
  return [nodes, edges]
}



function loadGraphBeforeSimulation(numberOfLeaves) {
  s.graph.clear()
  let tree = createTree(numberOfLeaves)
  s.graph.read({
    nodes: tree[0],
    edges: tree[1]
  })
  s.refresh()
}


function drawChart(x, y) {
  let ctx = $('#chart')[0].getContext('2d')
  let chart = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: x,
        datasets: [{
          label: 'edge distance histogram',
          data: y
        }],
        borderWidth: 1
      },
    }
  )
  chart.render()
  return chart
}

function drawGraph(numberOfLeaves, edges) {
  s.graph.clear()
  let nodes = createNodesInCircle(Math.pow(2, numberOfLeaves), 1)
  let edges_formatted = []
  for (let edgeIndex in edges) {
    edges_formatted.push({
      id: 'e' + edges[edgeIndex].source + 'to' + edges[edgeIndex].target,
      source: 'n' + edges[edgeIndex].source + 'r1',
      target: 'n' + edges[edgeIndex].target + 'r1'
    })
  }
  s.graph.read({
    nodes: nodes,
    edges: edges_formatted
  })
  s.refresh()
}


function updateWithData(data) {
  drawChart(data.chart.x, data.chart.y)
  drawGraph(urlParams.get('leaves'), data.edges)
}

function tryParseResult(response) {
  if (response.status == 200) {
    return response.json()
  }
  else {
    $('#error-message')[0].innerHTML = 'p is larger than 1 !'
  }
}

function simulate() {
  if(urlParams.has('leaves')) {
    fetch('/api?' + urlParams.toString()).then(tryParseResult).then(updateWithData)
  }
}

setDefaultFormValues();
simulate();
