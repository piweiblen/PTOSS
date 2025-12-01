#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "Board.h"
#include "PTOSS_kernel.h"


int main(int argc, char** argv) {
    int N=4;
    if (argc == 1) {
        // TODO: For no args we'll start with n=1 and compute as many terms as possible
        

        // for now let's just do an incomplete search for testing purposes
        Board search_board;
        int max_found = -1;
        Board max_board;
        unsigned int count = 0;
        // prep board for searching
        InitBoard(&search_board, N);
        insert_element(&search_board, 0, 0, 2);
        insert_one(&search_board, -1, -1);
        insert_one(&search_board, 1, 1);
        insert_element(&search_board, -1, 0, 0);
        printf("initial board:\n");
        pretty_print(&search_board);
        while ((search_board.pos_x[1] == -1) && (search_board.pos_y[1] == 0)) {
            next_board_state(&search_board);
            // if (count%100000 == 0) pretty_print(&search_board);
            // if (search_board.last_num+1 == 7) {
            //     printf("board index %d:\n", count);
            //     printf("last num %d:\n", search_board.last_num+1);
            //     printf("last info %d:\n", search_board.info[search_board.last_num-1]);
            //     pretty_print(&search_board);
            // }
            if (max_found < search_board.last_num) {
                max_found = search_board.last_num;
                CopyHostBoard(&max_board, &search_board);
            }
            count += 1;
            // if (count > 1300000) break;
        }
        printf("final board (%d):\n", max_board.last_num+1);
        pretty_print(&max_board);
        printf("checked %d unique states\n", count);

    } else if (argc == 2) {
        // TODO: For one arg we'll compute only the specified term


        // for now we'll ignor the arg and do a crude search of N=2 on the GPU for testing purposes
        Board search_boards[6];
        Board max_board;
        // prep boards for searching
        for (int i=0; i<6; i++) {
            InitBoard(&search_boards[i], N);
            insert_element(&search_boards[i], 0, 0, 2);
        }
        insert_one(&search_boards[0], -1, -1);
        insert_one(&search_boards[0], 1, 1);
        insert_element(&search_boards[0], -1, 0, 0);

        insert_one(&search_boards[1], -1, -1);
        insert_one(&search_boards[1], 1, 0);
        insert_element(&search_boards[1], -1, 0, 0);

        insert_one(&search_boards[2], -1, -1);
        insert_one(&search_boards[2], 1, -1);
        insert_element(&search_boards[2], -1, 0, 0);

        insert_one(&search_boards[3], 0, -1);
        insert_one(&search_boards[3], 1, 1);
        insert_element(&search_boards[3], -1, -1, 0);

        insert_one(&search_boards[4], 0, -1);
        insert_one(&search_boards[4], 1, 0);
        insert_element(&search_boards[4], -1, -1, 0);

        insert_one(&search_boards[5], 0, -1);
        insert_one(&search_boards[5], 1, -1);
        insert_element(&search_boards[5], -1, -1, 0);

        printf("search boards:\n");
        for (int i=0; i<6; i++) {
            pretty_print(&search_boards[i]);
        }

        clock_t tic = clock();
        SearchOnDevice(search_boards, &max_board, 6);
        clock_t toc = clock();
        printf("GPU time: %f seconds\n", (double)(toc - tic) / CLOCKS_PER_SEC);

        printf("final board (%d):\n", max_board.last_num+1);
        pretty_print(&max_board);
    }
}

void computeTerm(int n) {
    //TODO: write
    // we'll start with a predefined set of boards with 2 placed already
    // (we can hard code/ignore trivial n=1 case)
    // and branch breadth first until a sufficient number of boards has been generated
    // (number of GPU threads we wish to run, probably #defined in PTOSS_kernel)
    // call SearchOnDevice with this list of boards
    // pretty print result and at least one example grid
    // for fun we could eventually save all maximizing grids to a file
}