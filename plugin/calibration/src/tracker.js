var EVENT_DISP = document.getElementById('event_disp')
var CALIB_MAP = {}
var RESULT_CALIB = {}

let canvas = document.querySelector("#canvas")
let video = document.querySelector("#video");
var width = 620


function run_status_check(){
    setInterval(() => {
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
        var parsed = response
        var title = document.querySelector(".calib_title");
        var status = parsed['status']

        if (status == 'do_not_watch'){
            title.textContent = 'Do not watch';
            title.style.color = 'brown';
        }

        if (['watch_left', 'watch_right'].includes(status)){
            title.textContent = 'Watching';
            title.style.color = 'greenyellow';
        }
    })
    }, 1000);
}

function process_img(){
    console.log('Start proces')
    var socket = new WebSocket("ws://192.168.1.111:8486");
    var send_cnt = 0
    var ended = false

    socket.onmessage = (event) => {
        var data = JSON.parse(event.data)
        console.log('Recived: ', event.data)

        if (!RESULT_CALIB[data['calib_key']]){
            RESULT_CALIB[data['calib_key']] = []
        }

        if (data.pwc1) {
            RESULT_CALIB[data['calib_key']].push(data)
        }

        send_cnt -= 1

        if (ended && send_cnt == 0){
            fetch('http://localhost:8484/run_monitor/', {
                method: 'POST',
                headers: {
                    'Accept': 'application/json',
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({"frame_data": RESULT_CALIB})
            })
            video.srcObject.getTracks()[0].stop()
            run_status_check()
        }
    }

    socket.onopen = () => {
        for (let [key, value] of Object.entries(CALIB_MAP)) {
            for (let obj_img of value){
                console.log('sended', send_cnt)
                socket.send(JSON.stringify({
                    'frame': obj_img,
                    'custom': {'calib_key': key}
                }))
                send_cnt += 1;
            }
        }
        ended = true
    }
}

function collect_img(stash_key) {
    var inrt_id = setInterval(() => {
        height = video.videoHeight / (video.videoWidth / width);
        video.setAttribute("width", width);
        video.setAttribute("height", height);
        canvas.setAttribute("width", width);
        canvas.setAttribute("height", height);

        context = canvas.getContext("2d");
        context.drawImage(video, 0, 0, canvas.width, canvas.height);

        base64 = canvas.toDataURL("image/jpg");
        base64 = base64.replace(/^data:image\/?[A-z]*;base64,/, '');

        if (!CALIB_MAP[stash_key]){
            CALIB_MAP[stash_key] = []
        }
        if (CALIB_MAP[stash_key].length >= 5){
            CALIB_MAP[stash_key].shift()
        }

        CALIB_MAP[stash_key].push(base64)
        console.log(CALIB_MAP)
    }, 100)

    return new Promise((resolve, reject) => {
        setTimeout(() => {
            clearInterval(inrt_id);
            resolve();
        }, 500);

    })
}
