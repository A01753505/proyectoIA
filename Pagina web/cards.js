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

function match(){
    const randomNumber = Math.floor(Math.random() * 2) + 1;

    if (randomNumber === 1) {
        console.log('match');
        var elemento1 = document.getElementById("abordoB");
        var elemento2 = document.getElementById("emojiB");
        var elemento3 = document.getElementById("mensajeB");
        var elemento4 = document.getElementById("btnBuscarAlmas");
        var elemento5 = document.getElementById("btnViaje");
        elemento1.style.display = "block";
        elemento2.style.display = "block";
        elemento3.style.display = "block";
        elemento4.style.display = "block";
        elemento5.style.display = "block";
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