// Initialize canvases
// Original input image
canvas = document.querySelector('#canvas');
ctx = canvas.getContext('2d');
// For drawing downscaled version of input
canvas_flat = document.querySelector('#flat');
ctx_flat = canvas_flat.getContext('2d');
// Loss graph
canvas_graph = document.querySelector('#graph');
ctx_graph = canvas_graph.getContext('2d');

// Arrays to store generated canvases and drawing contexts
canvases_convis = [];
ctx_convis = [];
// Add canvases to page
for (var i = 0; i < 3; i ++) {
	convis.innerHTML += '<canvas id="conv-'+i+'" width="100" height="800"></canvas>';
}
// Get contexts and store in array
for (var i = 0; i < 3; i ++) {
	canvases_convis.push(document.querySelector('#conv-' + i));
	ctx_convis.push(canvases_convis[i].getContext('2d'));
}

num_data = 250;
resolution = [30, 30];
conv_filters = 8;
layer_vis_size = 100;
training_delay = 0.01;
train_percent = 70;
test_percent = 30;
optimizer = tf.train.adam(0.0001);

res = resolution;
lvs = layer_vis_size;

inputs = [];
outputs = [];
for (var i = 0; i < num_data; i ++) {
	ctx.fillStyle = 'white';
	ctx.fillRect(0, 0, canvas.width, canvas.height);
	//r = Math.floor(Math.random() * 2);
	r = Math.floor(Math.random() * 10);
	for (var j = 0; j < r; j ++) {
		x = Math.random() * canvas.width;
		y = Math.random() * canvas.height;

		ctx.beginPath();
		ctx.arc(x, y, 25, 0, 2 * Math.PI, false);
		ctx.fillStyle = 'black';
		ctx.fill();
	}
	// do this (resize) as soon as possible
	// use mapping function instead
	imageData = tf.browser.fromPixels(canvas, 1).resizeBilinear(res).div(tf.scalar(255));
	inputs.push(imageData)
	outputs.push(Math.round(r));
	console.log(r);
}
tf.browser.toPixels(
	imageData.resizeNearestNeighbor([canvas.width, canvas.height]),
	canvas_flat
);

train_split = Math.floor(num_data * train_percent * 0.01);
test_split = Math.floor(num_data * test_percent * 0.01);
tr_s = train_split;
te_s = test_split;

inputs = tf.stack(inputs);
training_in = inputs.slice([0], [tr_s]);
testing_in = inputs.slice([tr_s], [te_s]);

outputs = tf.tensor(outputs).expandDims(1);
training_out = outputs.slice([0], [tr_s]);
testing_out = outputs.slice([tr_s], [te_s]);

// ctx.getImageData(0, 0, canvas.width, canvas.height)

// Loss function (mean squared error)
const loss = (pred, label) => pred.sub(label).square().mean();
// Initialize model
const model = tf.sequential();
// Number of data points in input (depreciated)
const units = res[0] * res[1] * 1;

// Convolutional layers
// First conv2d layer
model.add(tf.layers.conv2d(
	{
		inputShape: [res[0], res[1], 1],
		filters: conv_filters,
		kernelSize: 3,
		strides: 1,
		activation: 'relu'
	}
));
// First maxPooling2d layer
model.add(tf.layers.maxPooling2d(
	{
		poolSize: 2,
		strides: 2
	}
));
// Second conv2d layer
model.add(tf.layers.conv2d(
	{
		filters: conv_filters,
		kernelSize: 3,
		stride: 1,
		activation: 'relu'
	}
));
// Second maxPooling2d layer
model.add(tf.layers.maxPooling2d(
	{
		poolSize: 2,
		strides: 2
	}
));
// Third conv2d layer
model.add(tf.layers.conv2d(
	{
		filters: conv_filters,
		kernelSize: 3,
		stride: 1,
		activation: 'relu'
	}
));
// Third maxPooling2d layer
model.add(tf.layers.maxPooling2d(
	{
		poolSize: 2,
		strides: 2
	}
));

// Flatten output of conv layer series into one 1-dimensional tensor
model.add(tf.layers.flatten({}));
// Add a dense layer to perform final calculations on data processed by conv layers
model.add(tf.layers.dense({units: 32}));
// Leaky ReLU to introduce more nonlinearity
model.add(tf.layers.leakyReLU());
// Dropout layer to prevent overfitting
model.add(tf.layers.dropout(0.8, {rate: 0.8}));
model.add(tf.layers.dense({units: 1}));

// Display model summary in console
model.summary();

e = 0;
epoch = [];
train_loss = [];
test_loss = [];
//out = tf.tensor([0])

// Create variable to store layer output so it isn't disposed, but we're not creating a new tensor with each call of the function
out = tf.variable(tf.zeros([lvs, lvs, 1]))

// Render activations of neural network layers
function renderConvLayers() {
	console.log('---------------------------------')
	console.log(tf.memory())
	tf.tidy(
		() => {
			// Loop through each layer of the network
			for (let j = 0; j < 1; j ++) {
				console.log('j __________')
				// Generate layer output
				val = model.layers[0]
					.apply(
						// Slice one image from the input tensor
						inputs.slice(
							[num_data - 1],
							[1]
						)
					)
					// Limit to correct value range for float32 tensor
					.clipByValue(0, 1)
				// Loop through each filter in the conv layer, if applicable
				for (let w = 0; w < conv_filters; w ++) {
					b = val
						// Get rid of batch dimension
						.squeeze()
						// Slice one filter from the layer output
						.slice(
							[0, 0, w - 1],
							[28, 28, 1]
						)
						// Resize to match canvas size
						.resizeNearestNeighbor([lvs, lvs]);
					// Assign to variable so all this ^ can be disposed
					out.assign(b);
					console.log(tf.memory())
					
					console.log(tf.memory())
					//console.log(out)
					tf.browser.toPixels(
						out,
						//out.squeeze().reshape([28*4, 28*2, 1]).clipByValue(0, 1).resizeNearestNeighbor([200, 400]),
						//tf.randomUniform([28, 28, 1]).resizeNearestNeighbor([100, 100]),
					).then(
						(d) => {
							//console.log(j)
							//console.log(d)
							
							// Render filter output to canvas
							ctx_convis[j].putImageData(
								// Create ImageData object
								new ImageData(d, lvs, lvs),
								0,
								// Top is filter number * layer_vis_size
								w * lvs
							)
							
							//out.dispose();
							//console.log(out)
							//tf.disposeVariables();
						}
					)
					console.log(tf.memory())
					
					//.resolve()
					//console.log(data);
				}
			}
		}
	)
	console.log(tf.memory())
}

function train() {
	if (e % 10 == 0 || e == 0) {
		renderConvLayers();
	}
	
	tf.tidy(
		() => {
			for (var i = 0; i < 1; i ++) {
				train_prediction = model.predict(training_in);
				test_prediction = model.predict(testing_in);
				
				if (e % 10 == 0) {
					epoch.push(e);
					train_loss.push(
						loss(model.predict(training_in), training_out).dataSync()
					);
					test_loss.push(
						loss(test_prediction, testing_out).dataSync()
					);
					graph.update();
				}
				
				e ++;
				
				optimizer.minimize(() => loss(model.predict(training_in), training_out));
				//console.log(loss(prediction, outputs));
				//loss(test_prediction, testing_out).print();
			}
			//console.log(tf.memory());
		}
	)
}

// Draw graph of loss over time
// Most of this is self-explanatory
const graph = new Chart(ctx_graph, {
	// Line chart
	type: 'line',
	// Data ~
	data: {
		labels: epoch,
		datasets: [
			{
				label: 'Training Loss',
				borderColor: 'rgb(255, 99, 132)',
				data: train_loss
			},
			{
				label: 'Testing Loss',
				borderColor: 'rgb(96, 157, 255)',
				data: test_loss
			}
		]
	},
	// Graph options
	options: {
		// Title
		title: {
            display: true,
            text: 'Loss'
        },
		// Axis labels
		scales: {
			xAxes: [{
				scaleLabel: {
					display: true,
					labelString: 'Epoch'
				}
			}],
			yAxes: [{
				scaleLabel: {
					display: true,
					labelString: 'Loss'
				}
			}]
		},
		// Hide graph points
		elements: {
			point:{
				radius: 0
			}
		},
		// Don't animate graph (for optimization)
		animation: {
			duration: 0
		},
		hover: {
			animationDuration: 0
		},
		responsiveAnimationDuration: 0
	}
});

// Run training function at pre-set intervals
window.setInterval(train, training_delay / 1000);