canvas = document.querySelector('#canvas');
ctx = canvas.getContext('2d');
canvas_flat = document.querySelector('#flat');
ctx_flat = canvas_flat.getContext('2d');

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
outputs = tf.tensor(outputs).expandDims(1);

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

function train() {
	tf.tidy(
		() => {
			for (var i = 0; i < 1; i ++) {
				prediction = model.predict(inputs);
				optimizer.minimize(() => loss(model.predict(inputs), outputs));
				//console.log(loss(prediction, outputs));
				loss(model.predict(inputs), outputs).print();
			}
			//console.log(tf.memory());
		}
	)
}

window.setInterval(train, 10);