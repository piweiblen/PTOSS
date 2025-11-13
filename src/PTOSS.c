#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "Board.h"
#include "PTOSS_kernel.h"


int main(int argc, char** argv) {
    if (argc == 1) {
        // For no args we'll start with n=1 and compute as many terms as possible
        //TODO a while loop, I imagine
        for (int i=1;i<10;i++) {
            printf("%d -> ?????\n", i);
        }
    } else if (argc == 2) {
        // For one arg we'll compute only the specified term
        int n = atoi(argv[1]);
        printf("%d -> ?????\n", n);
    }
}

void computeTerm(int n) {
    //TODO write
    // we'll start with a predefined set of boards with 2 placed already
    // (we can hard code/ignore trivial n=1 case)
    // and branch breadth first until a sufficient number of boards has been generated
    // (number of GPU threads we wish to run, probably #defined in PTOSS_kernel)
    // call SearchOnDevice with this list of boards
    // pretty print result and at least on example grid
    // for fun we could eventually save all maximizing grids to a file
}