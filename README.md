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
La red neuronal implementada consta de las siguientes capas:

1. **Entrada**: Tensor de 1000 × 20 × 20
2. **View**: Transformación a 1000 × 400
3. **Capa Lineal 1**: Multiplicación por matriz 400 × 100 + bias 1 × 100
4. **ReLU**: Activación no lineal
5. **Capa Lineal 2**: Multiplicación por matriz 100 × 10 + bias 1 × 10
6. **Sigmoid**: Activación final

El código en `main.cpp` demuestra el funcionamiento completo de la red neuronal.
