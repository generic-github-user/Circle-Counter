canvas = document.querySelector('#canvas');
ctx = canvas.getContext('2d');
canvas_flat = document.querySelector('#flat');
ctx_flat = canvas_flat.getContext('2d');
canvas_graph = document.querySelector('#graph');
ctx_graph = canvas_graph.getContext('2d');

inputs = [];
outputs = [];
for (var i = 0; i < 100; i ++) {
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
	imageData = tf.browser.fromPixels(canvas, 1).resizeBilinear([10, 10]).div(tf.scalar(255));
	inputs.push(imageData)
	outputs.push(Math.round(r));
	console.log(r);
}
tf.browser.toPixels(
	imageData.resizeNearestNeighbor([canvas.width, canvas.height]),
	canvas_flat
);
inputs = tf.stack(inputs);
training_in = inputs.slice([0], [70]);
testing_in = inputs.slice([70], [30]);

outputs = tf.tensor(outputs).expandDims(1);
training_out = outputs.slice([0], [70]);
testing_out = outputs.slice([70], [30]);

// ctx.getImageData(0, 0, canvas.width, canvas.height)

const loss = (pred, label) => pred.sub(label).square().mean();
const optimizer = tf.train.adam(0.001);
const model = tf.sequential();
const units = 10 * 10 * 1;
model.add(tf.layers.flatten({inputShape: [10, 10, 1]}));
model.add(tf.layers.dense({units: units * 0.75, activation: 'tanh'}));
model.add(tf.layers.dense({units: units * 0.5, activation: 'tanh'}));
model.add(tf.layers.dense({units: units * 0.25, activation: 'tanh'}));
model.add(tf.layers.dense({units: 1}));

model.summary();

e = 0;
epoch = [];
train_loss = [];
test_loss = [];

function train() {
	tf.tidy(
		() => {
			for (var i = 0; i < 1; i ++) {
				train_prediction = model.predict(training_in);
				test_prediction = model.predict(testing_in);
				
				optimizer.minimize(() => loss(model.predict(training_in), training_out));
				//console.log(loss(prediction, outputs));
				//loss(test_prediction, testing_out).print();
				
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

window.setInterval(train, 10);