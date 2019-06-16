canvas = document.querySelector('#canvas');
ctx = canvas.getContext('2d');

inputs = [];
r = Math.random() * 10;
for (var i = 0; i < r; i ++) {
	x = Math.random() * canvas.width;
	y = Math.random() * canvas.height;

	ctx.beginPath();
	ctx.arc(x, y, 25, 0, 2 * Math.PI, false);
	ctx.fillStyle = 'black';
	ctx.fill();
	
	// do this (resize) as soon as possible
	inputs.push(tf.browser.fromPixels(canvas).resizeBilinear([50, 50]))
}
console.log(r);
inputs = tf.stack(inputs);
