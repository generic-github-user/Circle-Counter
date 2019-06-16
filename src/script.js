canvas = document.querySelector('#canvas');
ctx = canvas.getContext('2d');

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
	inputs.push(tf.browser.fromPixels(canvas).resizeBilinear([50, 50]))
	outputs.push(Math.round(r));
	console.log(r);
}
inputs = tf.stack(inputs);
outputs = tf.tensor(outputs);

// ctx.getImageData(0, 0, canvas.width, canvas.height)