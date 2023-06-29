
class EmployeeCard {
  constructor(container, employees) {
    this.container = container;
    this.employees = employees;
    this.currentIndex = 0;

    this.renderCard();
    this.addListeners();
  }

  renderCard() {
    const employee = this.employees[this.currentIndex];

    const template = `
      <div class="employee-card">
        <div class="image-wrapper">
          <img src="${employee.image}" alt="${employee.name}">
        </div>
        <div class="info-wrapper">
          <div class="status">${employee.status}</div>
          <div class="name">${employee.name}</div>
          <div class="device">${employee.device}</div>
        </div>
      </div>
    `;

    this.container.innerHTML = template;
  }

addListeners() {
    const card = this.container.querySelector('.employee-card');

    card.addEventListener('click', () => {
      this.currentIndex = (this.currentIndex + 1) % this.employees.length;
      this.renderCard();
    });
  }
}

const employees = [
  {
    image: '/path/to/image1.jpg',
    status: 'Online',
    name: 'John Doe',
    device: 'Macbook Pro'
  },
  {
    image: '/path/to/image2.jpg',
    status: 'Offline',
    name: 'Jane Smith',
    device: 'Lenovo Thinkpad'
  },
  // Add more employees here...
];

const container = document.querySelector('.employee-container');
new EmployeeCard(container, employees);