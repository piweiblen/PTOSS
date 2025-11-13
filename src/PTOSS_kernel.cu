#include <stdio.h>
#include <cuda.h>

#include "Board.h"
#include "PTOSS_kernel.h"


// Matrix multiplication kernel thread specification
__global__ void SearchKernel(Board* Boards, int cur_max) {
    //TODO assign each thread one board to completely search the state space of
}


// Kernel calling function specification
void SearchOnDevice(Board* Boards) {  //TODO will need some way to return results eventually
    // TODO 
    // take list of host boards, convert to device boards
    // setup the execution configuration
    // launch the device computation threads
    // get results from device
    // free device matrices
}
