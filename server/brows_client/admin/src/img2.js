// Получаем элемент canvas
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
canvas.width = window.innerWidth;
canvas.height = window.innerHeight;

const POINTS_QUEUE = [
  'left_top',
  'right_top',
  'right_bottom',
  'left_bottom'
]

const CUR_KEY = POINTS_QUEUE.pop()

const positions = [
  {x: 20, y: 20},
  {x: canvas.width - 20, y: 20},
  {x: canvas.width - 20, y: canvas.height - 20},
  {x: 20, y: canvas.height - 20},
];
let currentPos = 0;

function drawCircle(x, y) {
  ctx.beginPath();
  ctx.arc(x, y, 50, 0, 2 * Math.PI);
  ctx.fillStyle = "red";
  ctx.fill();
}

function drawNextCircle() {
  CUR_KEY = POINTS_QUEUE.pop()
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (currentPos < positions.length) {
    const pos = positions[currentPos];
    drawCircle(pos.x, pos.y);
  }
  currentPos++;
}

canvas.addEventListener("click", drawNextCircle);
drawNextCircle();