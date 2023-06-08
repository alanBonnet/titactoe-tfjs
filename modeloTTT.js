const modeloTTT = tf.sequential();

modeloTTT.add(tf.layers.dense({ units: 128, activation: 'relu', inputShape: [9] }));
modeloTTT.add(tf.layers.dense({ units: 64, activation: 'relu' }));
modeloTTT.add(tf.layers.dense({ units: 64, activation: 'relu' }));
modeloTTT.add(tf.layers.dense({ units: 32, activation: 'relu' }));
modeloTTT.add(tf.layers.dense({ units: 32, activation: 'relu' }));
modeloTTT.add(tf.layers.dense({ units: 9, activation: 'softmax' }));

modeloTTT.compile({ optimizer: 'adam', loss: 'categoricalCrossentropy' });


const movimientosGanados = [
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8],
    [0, 3, 6],
    [1, 4, 7],
    [2, 5, 8],
    [0, 4, 8],
    [2, 4, 6]
]

const pointsRole = {//segun el punto es el role
    'ai': 1,
    'opponent': -1,
    'empty': 0
}

const rolePoints = {// según el role es el punto
    "1": "ai",
    "-1": "opponent",
    "0": "empty"
}


const tableroATensor = (estadoTablero) => {
    return tf.tensor2d([estadoTablero], [1, 9]);
}
const reiniciarPartida = () => {
    estadoTablero.estadoActual = estadoTablero.inicial.map(e => e);
    pintarTablero();
}
function generarMovimientoGanador(indice) {
    const movimientosGanadores = [
        { entrada: [1, 1, 0, 0, 0, 0, 0, 0, 0], proximoMovimiento: [0, 0, 1, 0, 0, 0, 0, 0, 0] }, // Ejemplo de movimiento ganador en la primera fila
        { entrada: [0, 0, 0, 1, 1, 0, 0, 0, 0], proximoMovimiento: [0, 0, 0, 0, 0, 1, 0, 0, 0] }, // Ejemplo de movimiento ganador en la segunda fila
        { entrada: [0, 0, 0, 0, 0, 0, 1, 1, 0], proximoMovimiento: [0, 0, 0, 0, 0, 0, 0, 0, 1] }, // Ejemplo de movimiento ganador en la tercera fila
        { entrada: [1, 0, 0, 1, 0, 0, 0, 0, 0], proximoMovimiento: [0, 0, 0, 0, 0, 0, 1, 0, 0] }, // Ejemplo de movimiento ganador en la primera columna
        { entrada: [0, 1, 0, 0, 1, 0, 0, 0, 0], proximoMovimiento: [0, 0, 0, 0, 0, 0, 0, 1, 0] }, // Ejemplo de movimiento ganador en la segunda columna
        { entrada: [0, 0, 1, 0, 0, 1, 0, 0, 0], proximoMovimiento: [0, 0, 0, 0, 0, 0, 0, 0, 1] }, // Ejemplo de movimiento ganador en la tercera columna
        { entrada: [1, 0, 0, 0, 1, 0, 0, 0, 0], proximoMovimiento: [0, 0, 0, 0, 0, 0, 0, 0, 1] }, // Ejemplo de movimiento ganador en diagonal descendente
        { entrada: [0, 0, 0, 0, 1, 0, 1, 0, 0], proximoMovimiento: [0, 0, 1, 0, 0, 0, 0, 0, 0] }, // Ejemplo de movimiento ganador en diagonal ascendente

    ];

    return movimientosGanadores[indice];
}

const generarDatosEntrenamiento = (iteraciones = 100) => {
    const datosEntrenamiento = [];
    tf.tidy(() => {

        const datosEntrada = tableroATensor(estadoTablero.estadoActual).flatten().dataSync();
        for (let i = 0; i < iteraciones; i++) {
            const indiceAleatorio = Math.floor(Math.random() * 9);
            const datosProximoMovimiento = Array.from({ length: 9 }).fill(0)
            datosProximoMovimiento[indiceAleatorio] = 1;

            if (i % 5 === 0) {// movimientos ganadores adicionales
                const indiceMovimientoGanador = Math.floor(Math.random() * 3);
                const movimientoGanador = generarMovimientoGanador(indiceMovimientoGanador);
                datosEntrenamiento.push({ entrada: movimientoGanador.entrada, proximoMovimiento: movimientoGanador.proximoMovimiento })
            }
            datosEntrenamiento.push({ entrada: datosEntrada, proximoMovimiento: datosProximoMovimiento })

        }
    })
    return datosEntrenamiento;
}

const datosAEntrenarModeloTTT = {
    numeroDeVecesAEntrenar: 100,
    datosEntradaYEtiqueta: generarDatosEntrenamiento,
    modelo: modeloTTT
}

const entrenarModelo = async ({ numeroDeVecesAEntrenar, datosEntradaYEtiqueta, modelo }) => {
    const datosEntrenamiento = datosEntradaYEtiqueta(200);
    const entradas = tf.tensor2d(datosEntrenamiento.map(datos => datos.entrada));
    const proximoMovimientos = tf.tensor2d(datosEntrenamiento.map(datos => datos.proximoMovimiento));

    await modelo.fit(entradas, proximoMovimientos, {
        epochs: numeroDeVecesAEntrenar,
        shuffle: true,
        // callbacks: {
        //     onEpochEnd: (epoch, logs) => {
        //         console.log(`Epoca ${epoch + 1} - perdida: ${logs.loss}`)
        //     }
        // }
    })
    entradas.dispose()
    proximoMovimientos.dispose()
    pintarTablero()
    console.log("Entrenamiento completado")
}

const realizarMovimientoIA = async () => {
    const entrada = tableroATensor(estadoTablero.estadoActual);
    const predicciones = await modeloTTT.predict(entrada);
    const movimientoPredicho = await tf.tidy(() => {
        const valoresValidos = predicciones.mul(tf.tensor1d(estadoTablero.estadoActual.map(cell => cell === 0 ? 1 : 0)));
        return tf.argMax(valoresValidos, 1).dataSync();
    });
    ponerMarca(movimientoPredicho, "ai")
}
const verificarEstadoGanador = () => {
    for (let posicionGanador of movimientosGanados) {
        const [a, b, c] = posicionGanador;
        if (
            estadoTablero.estadoActual[a] !== 0 &&
            estadoTablero.estadoActual[a] === estadoTablero.estadoActual[b] &&
            estadoTablero.estadoActual[a] === estadoTablero.estadoActual[c]
        ) {
            return true;
        }
    }
    return false;
}
const ponerMarca = (posicion, usuario = "opponent") => {
    if (estadoTablero.estadoActual[posicion] !== 0) return false;

    if (usuario != "opponent") {
        estadoTablero.estadoActual[posicion] = 1;
        pintarTablero();
        if (verificarEstadoGanador()) {
            pintarTablero();
            alert("perdiste!")
            return "ia ganador";
        }
        return true;
    }
    estadoTablero.estadoActual[posicion] = -1;
    if (verificarEstadoGanador()) {
        pintarTablero();
        alert("ganaste!")
        return "usuario ganador";
    }
    //verificación empate
    if(!estadoTablero.estadoActual.includes(0)){
        pintarTablero()
        alert("empate")
        return "empate";
    }
    realizarMovimientoIA()
    pintarTablero();
    return true;
}


window.onload = async () => { await entrenarModelo(datosAEntrenarModeloTTT) }