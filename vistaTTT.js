const divTicTacToe = document.getElementById('tictactoe');


const estadoTablero = {
    inicial: [
        0, 0, 0,
        0, 0, 0,
        0, 0, 0
    ],
    estadoActual: [
        0, 0, 0,
        0, 0, 0,
        0, 0, 0
    ]
}


const pintarTablero = () => {
    divTicTacToe.innerHTML = "";
    let textoPintado = ""
    for (let i = 0; i < estadoTablero.estadoActual.length; i++) {
        if (i == 0 || i == 3 || i == 6) {
            textoPintado += `<div id="fila-${i}" class="flex gap-2 mb-2">`;
            for (let j = 0; j < estadoTablero.estadoActual.length / 3; j++) {
                const valorOX = estadoTablero.estadoActual[i + j] == 0 ? "" : estadoTablero.estadoActual[i + j] == 1 ? "O" : "X"
                textoPintado += `<span id="celda-${i}-${j}" class="bg-indigo-100 rounded-xl px-2 text-center w-1/3 h-20 items-center flex" onclick="ponerMarca(${i + j})"><p class="mx-auto text-7xl">${valorOX}</p></span>`
            }
            textoPintado += `</div>`
        }
    }
    divTicTacToe.innerHTML = textoPintado

}