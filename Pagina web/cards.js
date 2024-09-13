const cards = document.querySelectorAll('.card');
const leftBtn = document.getElementById('left-btn');
const rightBtn = document.getElementById('right-btn');
const buscarAlmasBtn = document.getElementById('btnBuscarAlmas');
const viajeBtn = document.getElementById('btnViaje');
const predContainer = document.querySelector('.container-form_pred.form');
const soulmateContainer = document.querySelector('.card-container.soulmate');
const predictionContainer = document.querySelector('.pred-container.pred');
const cancelarBtn = document.getElementById('cancelar');
const subirBtn = document.getElementById('subir');
let currentCardIndex = 0;

leftBtn.addEventListener('click', () => {
    swipeCard('left');
});

rightBtn.addEventListener('click', () => {
    swipeCard('right');
    match();
    soulmateContainer.classList.add('hide');
    predContainer.classList.remove('hide');
    var elemento1 = document.getElementById("left-btn");
    var elemento2 = document.getElementById("right-btn");
    elemento1.style.display = "none";
    elemento2.style.display = "none";
});

cancelarBtn.addEventListener('click', () => {
    predContainer.classList.add('hide');
    soulmateContainer.classList.remove('hide');
    var elemento1 = document.getElementById("left-btn");
    var elemento2 = document.getElementById("right-btn");
    elemento1.style.display = "block";
    elemento2.style.display = "block";
})

subirBtn.addEventListener('click', () => {
    predictionContainer.classList.remove('hide');
    predContainer.classList.add('hide');
})

buscarAlmasBtn.addEventListener('click', () => {
    location.reload();
})

viajeBtn.addEventListener('click', () => {
    location.reload();
})

function swipeCard(direction) {
    const currentCard = cards[currentCardIndex];
    const feedbackClass = direction === 'left' ? 'show-feedback-left' : 'show-feedback-right';
    const swipeClass = direction === 'left' ? 'swipe-left' : 'swipe-right';

    // Agregar animación de swipe
    currentCard.classList.add(swipeClass);
    currentCard.querySelector(`.${direction}-feedback`).classList.add(feedbackClass);

    // Esperar que la animación termine antes de pasar a la siguiente tarjeta
    setTimeout(() => {
        currentCard.classList.remove(swipeClass);
        currentCard.querySelector(`.${direction}-feedback`).classList.remove(feedbackClass);

        // Ocultar la tarjeta actual
        currentCard.classList.remove('active');

        // Mostrar la siguiente tarjeta
        currentCardIndex = (currentCardIndex + 1) % cards.length;
        cards[currentCardIndex].classList.add('active');
    }, 500);
}

function match(pred){

    if (pred) {
        console.log('match');
        var elemento1 = document.getElementById("abordoB");
        var elemento2 = document.getElementById("emojiB");
        var elemento3 = document.getElementById("mensajeB");
        var elemento4 = document.getElementById("btnBuscarAlmas");
        var elemento5 = document.getElementById("btnViaje");
        var elemento6 = document.getElementById("abordoM");
        var elemento7 = document.getElementById("emojiM");
        var elemento8 = document.getElementById("mensajeM");
        var elemento9 = document.getElementById("btnBuscarAlmas");
        elemento1.style.display = "block";
        elemento2.style.display = "block";
        elemento3.style.display = "block";
        elemento4.style.display = "block";
        elemento5.style.display = "block";
        elemento6.style.display = "none";
        elemento7.style.display = "none";
        elemento8.style.display = "none";
        elemento9.style.display = "none";
    } else {
        console.log('no match');
        var elemento1 = document.getElementById("abordoM");
        var elemento2 = document.getElementById("emojiM");
        var elemento3 = document.getElementById("mensajeM");
        var elemento4 = document.getElementById("btnBuscarAlmas");
        elemento1.style.display = "block";
        elemento2.style.display = "block";
        elemento3.style.display = "block";
        elemento4.style.display = "block";
    }
}

// Guardarc el asiento seleccionado
let selectedSeat = null;

document.querySelectorAll('.seat').forEach(button => {
    button.addEventListener('click', function() {
        selectedSeat = this.getAttribute('data-seat');
        console.log('Asiento seleccionado:', selectedSeat);
    });
});

// Función que manda a llamar al modelo para hacer la predicción
function modelo(){
    var xhr = new XMLHttpRequest();
    var url = 'http://172.24.65.198:8080/predictjson';
    const form = document.querySelector('.information_form');
    const formData = new FormData(form);

    const dataArray = Array.from(formData.entries());

    // Obtenemos datos del formulario
    function getValue(key) {
        const entry = dataArray.find(([field]) => field === key);
        return entry ? entry[1] : null;
    }

    const hasFamily = getValue('yesNoOption') === 'yes';

    // json que se manda
    var data = JSON.stringify({
        "home_planet": getValue('userOrigin'),
        "destination": getValue('userDestiny'),
        "birth_date": getValue('userbirthday'),
        "name": getValue('userName'),
        "has_family": hasFamily,
        "photo": "/9j/4AAQSkZJRgABAQEAAAAAAAD/4QAYRXhpZgAATU0AKgAAAAgAAwESAAMAAAABAAEAAAEaAAUAAAABAAAAYgEbAAUAAAABAAAAagEoAAMAAAABAAIAAAExAAIAAAAeAAAAcgEyAAIAAAAUAAAAkIdpAAQAAAABAAAApAAAAABHZX",
        "real_data": true,
        "cabin": selectedSeat
    });

    xhr.open('POST', url, true);
    xhr.setRequestHeader('Content-Type', 'application/json');

    xhr.onload = function() {
        if (xhr.status >= 200 && xhr.status < 300) {
            var response = JSON.parse(xhr.responseText);
            console.log('Éxito:', response);
            var response = JSON.parse(xhr.responseText);
            console.log('Éxito:', response);

            const prediction = response.Prediction === "True";
            match(prediction);

        } else {
            console.error('Error:', xhr.statusText);
        }
    };

    xhr.onerror = function() {
        console.error('Error en la petición');
    };

    xhr.send(data);
}

// Función para obtener la lista para las cartas de la bd
let globalUsers = [];
function getUsers() {
    var xhr = new XMLHttpRequest();
    var url = 'http://172.24.65.198:8080/users';

    xhr.open('GET', url, true);

    xhr.onload = function() {
        if (xhr.status >= 200 && xhr.status < 300) {
            globalUsers = JSON.parse(xhr.responseText);
            console.log('Usuarios:', globalUsers);
        } else {
            console.error('Error:', xhr.statusText);
        }
    };

    xhr.onerror = function() {
        console.error('Error en la petición');
    };

    xhr.send();
}

getUsers();
