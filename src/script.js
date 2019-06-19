canvas = document.querySelector('#canvas');
ctx = canvas.getContext('2d');
canvas_flat = document.querySelector('#flat');
ctx_flat = canvas_flat.getContext('2d');
canvas_graph = document.querySelector('#graph');
ctx_graph = canvas_graph.getContext('2d');

canvases_convis = [];
ctx_convis = [];
for (var i = 0; i < 3; i ++) {
	convis.innerHTML += '<canvas id="conv-'+i+'" width="100" height="800"></canvas>';
}
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

const loss = (pred, label) => pred.sub(label).square().mean();
const model = tf.sequential();
const units = res[0] * res[1] * 1;

model.add(tf.layers.conv2d(
	{
		inputShape: [res[0], res[1], 1],
		filters: conv_filters,
		kernelSize: 3,
		strides: 1,
		activation: 'relu'
	}
));
model.add(tf.layers.maxPooling2d(
	{
		poolSize: 2,
		strides: 2
	}
));
model.add(tf.layers.conv2d(
	{
		filters: conv_filters,
		kernelSize: 3,
		stride: 1,
		activation: 'relu'
	}
));
model.add(tf.layers.maxPooling2d(
	{
		poolSize: 2,
		strides: 2
	}
));
model.add(tf.layers.conv2d(
	{
		filters: conv_filters,
		kernelSize: 3,
		stride: 1,
		activation: 'relu'
	}
));
model.add(tf.layers.maxPooling2d(
	{
		poolSize: 2,
		strides: 2
	}
));

model.add(tf.layers.flatten({}));
model.add(tf.layers.dense({units: 32}));
model.add(tf.layers.leakyReLU());
model.add(tf.layers.dropout(0.8, {rate: 0.8}));
model.add(tf.layers.dense({units: 1}));

model.summary();

e = 0;
epoch = [];
train_loss = [];
test_loss = [];
//out = tf.tensor([0])
out = tf.variable(tf.zeros([lvs, lvs, 1]))

function renderConvLayers () {
	console.log('---------------------------------')
	console.log(tf.memory())
	for (let j = 0; j < 1; j ++) {
		console.log('j __________')
		for (let w = 0; w < conv_filters; w ++) {
			tf.tidy(
				() => {
					console.log(tf.memory())
					val = model.layers[0]
						.apply(
							inputs.slice([num_data - 1], [1])
						)
						.squeeze()
						.slice(
							[0, 0, w - 1],
							[28, 28, 1]
						)
						.clipByValue(0, 1)
						.resizeNearestNeighbor([lvs, lvs])
					out.assign(val);
					
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
							ctx_convis[j].putImageData(
								new ImageData(d, lvs, lvs),
								0,
								w * lvs
							)
							//out.dispose();
							//console.log(out)
							//tf.disposeVariables();
						}
					)
					console.log(tf.memory())
				}
			)
			
			//.resolve()
			//console.log(data);
		}
	}
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

const graph = new Chart(ctx_graph, {
	type: 'line',
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

window.setInterval(train, training_delay / 1000);