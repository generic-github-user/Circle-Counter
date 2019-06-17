canvas = document.querySelector('#canvas');
ctx = canvas.getContext('2d');
canvas_flat = document.querySelector('#flat');
ctx_flat = canvas_flat.getContext('2d');

inputs = [];
outputs = [];
for (var i = 0; i < 10; i ++) {
	ctx.fillStyle = 'white';
	ctx.fillRect(0, 0, canvas.width, canvas.height);
	r = Math.random() * 10;
	for (var j = 0; j < r; j ++) {
		x = Math.random() * canvas.width;
		y = Math.random() * canvas.height;

		ctx.beginPath();
		ctx.arc(x, y, 25, 0, 2 * Math.PI, false);
		ctx.fillStyle = 'black';
		ctx.fill();
	}
	// do this (resize) as soon as possible
	imageData = tf.browser.fromPixels(canvas, 1).resizeBilinear([10, 10]);
	inputs.push(imageData)
	outputs.push(Math.round(r));
	console.log(r);
}
tf.browser.toPixels(
	imageData.resizeNearestNeighbor([canvas.width, canvas.height]),
	canvas_flat
);
inputs = tf.stack(inputs);
outputs = tf.tensor(outputs);

// ctx.getImageData(0, 0, canvas.width, canvas.height)

const loss = (pred, label) => pred.sub(label).square().mean();
const optimizer = tf.train.sgd(0.00000001);
const model = tf.sequential();
const units = 10 * 10 * 1;
model.add(tf.layers.flatten({inputShape: [10, 10, 1]}));
model.add(tf.layers.dense({units: units * 0.75}));
model.add(tf.layers.dense({units: units * 0.5}));
model.add(tf.layers.dense({units: units * 0.25}));
model.add(tf.layers.dense({units: 1}));

model.summary();
for (var i = 0; i < 1000; i ++) {
	prediction = model.predict(inputs);
	optimizer.minimize(() => loss(model.predict(inputs), outputs));
	//console.log(loss(prediction, outputs));
	loss(model.predict(inputs), outputs).print();
}