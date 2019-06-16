canvas = document.querySelector('#canvas');
ctx = canvas.getContext('2d');

r = Math.random() * 10;
for (var i = 0; i < r; i ++) {
	x = Math.random() * canvas.width;
	y = Math.random() * canvas.height;

	context.beginPath();
	context.arc(x, y, 25, 0, 2 * Math.PI, false);
	context.fillStyle = 'black';
	context.fill();
}
console.log(r);