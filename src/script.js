// Settings
num_data = 250;
max_circles = 10;
resolution = [30, 30];
conv_filters = 8;
layer_vis_size = 100;
training_delay = 0.01;
train_percent = 70;
test_percent = 30;
optimizer = tf.train.adam(0.0001);
num_vis_layers = 5;
use_one_hot = false;

// Aliases (for convenience)
res = resolution;
lvs = layer_vis_size;


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
for (var i = 0; i < num_vis_layers; i ++) {
	convis.innerHTML += '<canvas data-tilt id="conv-'+i+'" width="100" height="800"></canvas>';
}
// Get contexts and store in array
for (var i = 0; i < num_vis_layers; i ++) {
	canvases_convis.push(document.querySelector('#conv-' + i));
	ctx_convis.push(canvases_convis[i].getContext('2d'));
}


// Arrays of complete sets of inputs and outputs
inputs = [];
outputs = [];
// Generate random circles on canvas
for (var i = 0; i < num_data; i ++) {
	// Reset canvas
	ctx.fillStyle = 'white';
	ctx.fillRect(0, 0, canvas.width, canvas.height);
	//r = Math.floor(Math.random() * 2);
	// Select a random number of circles to draw
	r = Math.floor(Math.random() * max_circles);
	for (var j = 0; j < r; j ++) {
		// Generate random coordinates within canvas
		x = Math.random() * canvas.width;
		y = Math.random() * canvas.height;

		// Draw circle
		ctx.beginPath();
		ctx.arc(x, y, 25, 0, 2 * Math.PI, false);
		ctx.fillStyle = 'black';
		ctx.fill();
	}
	// do this (resize) as soon as possible
	// use mapping function instead
	// Save canvas to a tensor
	imageData = tf.browser.fromPixels(canvas, 1)
		.resizeBilinear(res)
		.div(tf.scalar(255));
	// Add image data to list of inputs
	inputs.push(imageData)
	// Add output (number of generated circles) to output array
	outputs.push(Math.round(r));
	console.log(r);
}
// Draw downscaled version of input image to canvas
tf.browser.toPixels(
	imageData
	.resizeNearestNeighbor([canvas.width, canvas.height]),
	canvas_flat
);

// Calculate split between training data and testing data
train_split = Math.floor(num_data * train_percent * 0.01);
test_split = Math.floor(num_data * test_percent * 0.01);
// More aliases
tr_s = train_split;
te_s = test_split;

// Merge inputs into one big tensor
inputs = tf.stack(inputs);
// Split input tensor into training/testing data
training_in = inputs.slice([0], [tr_s]);
testing_in = inputs.slice([tr_s], [te_s]);

// Generate a tensor from the output data
if (use_one_hot) {
	outputs = tf.oneHot(outputs, max_circles);
} else {
	outputs = tf.tensor(outputs).expandDims(1);	
}
// Split output tensor into training/testing data
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

if (use_one_hot) {
	model.add(tf.layers.dense({units: max_circles, activation: 'softmax'}));
} else {
	model.add(tf.layers.dense({units: 1}));
}

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
	tf.tidy(
		() => {
			// Slice one image from the input tensor
			out_val = inputs.slice(
					[num_data - 1],
					[1]
				);
			
			// Loop through each layer of the network
			for (let j = 0; j < num_vis_layers; j ++) {
				console.log('j __________')
				// Generate layer output
				out_val = model.layers[j]
					.apply(out_val)
					// Limit to correct value range for float32 tensor
					.clipByValue(0, 1)
				// Loop through each filter in the conv layer, if applicable
				for (let w = 0; w < conv_filters; w ++) {
					b = out_val
						// Get rid of batch dimension
						.squeeze()
						// Slice one filter from the layer output
						.slice(
							[0, 0, w - 1],
							[-1, -1, 1]
						)
						// Resize to match canvas size
						.resizeNearestNeighbor([lvs, lvs]);
					// Assign to variable so all this ^ can be disposed
					out.assign(b);
					
					tf.browser.toPixels(
						out,
						//out.squeeze().reshape([28*4, 28*2, 1]).clipByValue(0, 1).resizeNearestNeighbor([200, 400]),
						//tf.randomUniform([28, 28, 1]).resizeNearestNeighbor([100, 100]),
					).then(
						(d) => {
							// Render filter output to canvas
							ctx_convis[j].putImageData(
								// Create ImageData object
								new ImageData(d, lvs, lvs),
								0,
								// Top is filter number * layer_vis_size
								w * lvs
							)
						}
					);
				}
			}
		}
	)
}

function train() {
	if (e % 10 == 0 || e == 0) {
		renderConvLayers();
		console.log(tf.memory());
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
				
				if (use_one_hot) {
					optimizer.minimize(() => tf.losses.softmaxCrossEntropy(training_out, model.predict(training_in)));
				} else {
					optimizer.minimize(() => loss(model.predict(training_in), training_out));
				}
				//console.log(loss(prediction, outputs));
				//loss(test_prediction, testing_out).print();
			}
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