#include <stdlib.h>

#include "Board.h"


// Populate a board with initial values and allocate needed memory
void InitBoard(Board* B) {

    // Define initial values (all set to 0 as placeholders)
    //TODO define specific values here
    B->ones_left = 0;
    B->x_max = 0;
    B->x_min = 0;
    B->y_max = 0;
    B->y_min = 0;
    B->last_num = 0;

    // Allocate element and position arrays
    B->elements = (int*) malloc(MAX_ELEMENTS * sizeof(int));
    B->pos_x = (int*) malloc(MAX_POSITIONS * sizeof(int));
    B->pos_y = (int*) malloc(MAX_POSITIONS * sizeof(int));
}

// Create and populate a device board with identical values ot input board
Board InitDeviceBoard(const Board Bhost) {

    // Shallow copy of scalar values
    Board Bdevice = Bhost;

    // Allocate the element and position arrays on the device
    cudaMalloc(&Bdevice.elements, MAX_ELEMENTS * sizeof(int));
    cudaMalloc(&Bdevice.pos_x, MAX_POSITIONS * sizeof(int));
    cudaMalloc(&Bdevice.pos_y, MAX_POSITIONS * sizeof(int));

    // Copy element and positions arrays
    cudaMemcpy(Bdevice.elements, Bhost.elements, MAX_ELEMENTS * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(Bdevice.pos_x, Bhost.pos_x, MAX_POSITIONS * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(Bdevice.pos_y, Bhost.pos_y, MAX_POSITIONS * sizeof(int), cudaMemcpyHostToDevice);

    // Return the board
    return Bdevice;
}

// Copy a host board to a device board
void CopyToDeviceBoard(Board* Bdevice, const Board* Bhost) {

    // Copy scalar values
    Bdevice->ones_left = Bhost->ones_left;
    Bdevice->x_max = Bhost->x_max;
    Bdevice->y_max = Bhost->y_max;
    Bdevice->y_min = Bhost->y_min;
    Bdevice->last_num = Bhost->last_num;

    // Copy element and position arrays
    cudaMemcpy(Bdevice->elements, Bhost->elements, MAX_ELEMENTS * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(Bdevice->pos_x, Bhost->pos_x, MAX_POSITIONS * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(Bdevice->pos_y, Bhost->pos_y, MAX_POSITIONS * sizeof(int), cudaMemcpyHostToDevice);
}

// Copy a device board to a host board
void CopyFromDeviceBoard(Board* Bhost, const Board* Bdevice) {

    // Copy scalar values
    Bhost->ones_left = Bdevice->ones_left;
    Bhost->x_max = Bdevice->x_max;
    Bhost->y_max = Bdevice->y_max;
    Bhost->y_min = Bdevice->y_min;
    Bhost->last_num = Bdevice->last_num;

    // Copy element and position arrays
    cudaMemcpy(Bhost->elements, Bdevice->elements, MAX_ELEMENTS * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(Bhost->pos_x, Bdevice->pos_x, MAX_POSITIONS * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(Bhost->pos_y, Bdevice->pos_y, MAX_POSITIONS * sizeof(int), cudaMemcpyDeviceToHost);
}

// Free a device board
void FreeDeviceBoard(Board* B) {

    // Free arrays
    cudaFree(B->elements);
    cudaFree(B->pos_x);
    cudaFree(B->pos_y);
}

// Free a host board
void FreeBoard(Board* B) {

    // Free arrays
    free(B->elements);
    free(B->pos_x);
    free(B->pos_y);
}