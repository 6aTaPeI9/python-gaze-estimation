var EVENT_DISP = document.getElementById('event_disp')
var main_box = document.getElementById("main");
var currentQuarter = 0;

function toggle_circle(top, right, bottom, left, hide, ev_key) {
  var circle = document.getElementsByClassName('circle')[0]
  circle.classList.add('loader')
  var c_title = document.getElementById("c_title");
  c_title.textContent = 'Watch'

  collect_img(ev_key).then(() => {
    main_box.removeAttribute('style');
    main_box.style.top = top;
    main_box.style.right = right;
    main_box.style.bottom = bottom;
    main_box.style.left = left;
    c_title.textContent = 'Click';
    circle.classList.remove('loader');

    if (hide){
      main_box.style.visibility = 'hidden'
      document.getElementById("process").style.visibility = null;
      process_img()
    }
  })
}

function draw_circle() {
    currentQuarter = currentQuarter + 1;
    switch(currentQuarter) {
      case 1: // Верхний правый угол
        toggle_circle(0, 0, null, null, false, 'topLeft')
        break;
      case 2: // Нижний правый угол
        toggle_circle(null, 0, 0, null, false, 'topRight')
        break;
      case 3: // Нижний левый угол
        toggle_circle(null, null, 0, 0, false, 'botRight')
        break;
      case 4:
        toggle_circle(null, null, 0, 0, true, 'botLeft')
        this.onclick = undefined;
        break;
      default:
        break;
    }
  };

fetch('http://localhost:8484/status/', {
  method: 'POST',
  headers: {
      'Accept': 'application/json',
      'Content-Type': 'application/json'
  },
  body: '{}'
  })
  .then(response => response.json())
  .then((response) => {
      console.log(response)
      if (response['track_runned']){
        main_box.style.visibility = 'hidden'
        document.getElementById("process").style.visibility = null;
        run_status_check()
      }
      else{
          navigator.mediaDevices
          .getUserMedia({ video: true, audio: false })
          .then((stream) => {
              video.srcObject = stream;
          })
          .catch((err) => {
              console.error(`An error occurred: ${err}`);
          });
          main_box.onclick = draw_circle
      }
  })