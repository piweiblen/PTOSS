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
        

        // for now let's just do a crude search of N=2 for testing purposes
        Board search_board;
        int max_found = -1;
        Board max_board;
        int count = 0;
        // prep board for searching
        InitBoard(&search_board, 2);
        InitBoard(&max_board, 2);
        insert_element(&search_board, 0, 0, 2);
        insert_one(&search_board, -1, -1);
        insert_one(&search_board, 1, 1);
        insert_element(&search_board, -1, 0, 0);
        printf("initial board:\n");
        pretty_print(&search_board);
        while ((search_board.pos_x[1] == -1) && (search_board.pos_y[1] == 0)) {
            next_board_state(&search_board);
            // pretty_print(&search_board);
            if (max_found < search_board.last_num) {
                max_found = search_board.last_num;
                CopyHostBoard(&max_board, &search_board);
            }
            count += 1;
        }
        printf("final board:\n");
        pretty_print(&max_board);
        printf("checked %d unique states:\n", count);
        FreeBoard(&search_board);
        FreeBoard(&max_board);

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