STATUS_BIND = {
  'do_not_watch': 'Do not working',
  'watch_left': 'Working',
  'watch_right': 'Working',
}

// Создаем класс Card
class Card {
    constructor(container) {
      this.container = container;
    }

    is_not_working(status) {
      // Определение класса который требуется применить к карточке
      if (status == 'Do not working'){
        return 'red_state'
      }
      return null
    }
    // Метод для отображения карточки
    append(device, status, photo) {
      // Проверяем существует ли уже такая карточка

      var strip_device = device.replace(/[\s-]+/g, '');
      var cur_card = document.querySelector(`[data-uuid='${strip_device}']`)
      if (!cur_card){
        // Делаем новую
        // Создаем элемент div с классом card
        const cardDiv = document.createElement('div');
        cardDiv.classList.add('card');
        if (this.is_not_working(status)){
          cardDiv.classList.add('red_state');
        }
        cardDiv.setAttribute('data-uuid', strip_device)

        // Создаем элемент img с атрибутом src равным фото сотрудника
        const img = document.createElement('img');
        img.src = photo;
        cardDiv.appendChild(img);

        // Создаем элемент p с именем сотрудника
        const nameParagraph = document.createElement('p');
        nameParagraph.textContent = `Device: ${device}`;
        cardDiv.appendChild(nameParagraph);

        // Создаем элемент p со статусом сотрудника
        const statusParagraph = document.createElement('p');
        statusParagraph.textContent = `Status: ${status}`;
        cardDiv.appendChild(statusParagraph);
        this.container.appendChild(cardDiv)
      }
      else{
        if (this.is_not_working(status)){
          cur_card.classList.add('red_state')
        }
        else{
          cur_card.classList.remove('red_state')
        }

        var col = cur_card.querySelector('img');
        col.src = photo;

        var col = cur_card.querySelectorAll('p')[0];
        col.textContent = `Device: ${device}`;

        var col = cur_card.querySelectorAll('p')[1];
        col.textContent = `Status: ${status}`;
      }
    }
  }


// отладочное
// var employees = [
// {
//     device: '737John Doe',
//     status: 'Working',
//     photo: 'staff.png'
// },
// {
//     device: 'Jane Smith',
//     status: 'Do not working',
//     photo: 'staff.png'
// },
// {
//     device: 'Mark Johnson',
//     status: 'Do not working',
//     photo: 'staff.png'
// },
// ];

// Находим элемент-контейнер, где будут располагаться карточки
const container = document.getElementById('employee-container');
var CARD_OBJ = new Card(container);
// for (let card of employees){
//   CARD_OBJ.append(card.device, card.status, card.photo)
// }


setInterval(() => {
  fetch('http://localhost:8483/read_devices/', {
  method: 'POST',
  headers: {
      'Accept': 'application/json',
      'Content-Type': 'application/json'
  },
  body: '{}'
  })
  .then(response => response.json())
  .then((response) => {
    response = JSON.parse(response)
    console.log(response)
    for (let [key, card] of Object.entries(response)) {
        console.log(card)
        var image = decodeURIComponent(escape(card.photo))
        image = "data:image/jpg;base64," + image;
        card.photo = image
        card.status = STATUS_BIND[card.status]
        CARD_OBJ.append(card.device, card.status, card.photo)
      }
  })
}, 2000)