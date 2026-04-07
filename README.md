# Tarea #1: Tensor++
Autores:
1. Walter Sebastian Aquino Pachas
2. Guillermo Arturo Heredia Cadenas

Una libreria en C++ llamada Tensor++ inspirada en bibliotecas cientificas como Numpy y PyTorch.

## Compilación y Ejecución

### Requisitos
- CMake versión 3.10 o superior
- Compilador C++ que soporte C++20 (como g++)

### Instrucciones de Compilación
1. Asegúrate de tener CMake instalado:
   ```bash
   sudo apt update
   sudo apt install cmake
   ```

2. Crea un directorio de build:
   ```bash
   mkdir build
   cd build
   ```

3. Configura el proyecto con CMake:
   ```bash
   cmake ..
   ```

4. Compila el proyecto:
   ```bash
   make
   ```

### Ejecución
Después de compilar, ejecuta el programa:
```bash
./Tarea1
```

El programa implementa una red neuronal simple utilizando la clase Tensor, siguiendo los pasos especificados en la tarea.

## Descripción de la Red Neuronal
La red neuronal implementada sigue el flujo completo de procesamiento de datos como se detalla en la siguiente tabla:

| Paso | Operación | Dimensión resultante |
|------|-----------|----------------------|
| 1 | Tensor de entrada (datos crudos) | 1000 × 20 × 20 |
| 2 | view | 1000 × 400 |
| 3 | matmul con pesos W1 | 1000 × 100 |
| 4 | Suma con bias b1 (1 × 100) | 1000 × 100 |
| 5 | Activación ReLU | 1000 × 100 |
| 6 | matmul con pesos W2 | 1000 × 10 |
| 7 | Suma con bias b2 (1 × 10) | 1000 × 10 |
| 8 | Activación Sigmoid | 1000 × 10 |

El código en `main.cpp` demuestra el funcionamiento completo de la red neuronal.
