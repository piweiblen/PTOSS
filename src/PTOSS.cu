#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "Board.h"
#include "PTOSS_kernel.h"


Board computeTermCPU(int N) {

    Board max_board;
    InitBoard(&max_board, N);
    // hardcode result for n=1
    if (N == 1) {
        insert_one(&max_board, C_OFF, C_OFF);
        return max_board;
    }

    // set up starting boards
    int num_b = 6;
    Board* search_boards = (Board *) malloc(num_b * sizeof(Board));
    // here is all possible ways to place the first 2 hardcoded
    for (int i=0; i<num_b; i++) {
        InitBoard(&search_boards[i], N);
        insert_element(&search_boards[i], C_OFF, C_OFF, 2);
    }
    insert_one(&search_boards[0], C_OFF-1, C_OFF-1);
    insert_one(&search_boards[0], C_OFF+1, C_OFF+1);
    insert_one(&search_boards[1], C_OFF-1, C_OFF);
    insert_one(&search_boards[1], C_OFF+1, C_OFF+1);
    insert_one(&search_boards[2], C_OFF-1, C_OFF+1);
    insert_one(&search_boards[2], C_OFF+1, C_OFF+1);
    insert_one(&search_boards[3], C_OFF-1, C_OFF);
    insert_one(&search_boards[3], C_OFF+1, C_OFF);
    insert_one(&search_boards[4], C_OFF, C_OFF+1);
    insert_one(&search_boards[4], C_OFF+1, C_OFF);
    insert_one(&search_boards[5], C_OFF+1, C_OFF+1);
    insert_one(&search_boards[5], C_OFF+1, C_OFF);

    // get the 3s placed so we have convenient anchor points to check against
    const int depth_chart[8] = {1, 4, 4, 4, 3, 3, 3, 3};
    for (int depth=2; depth<depth_chart[N-1]; depth++) {
        gen_all_next_boards(&search_boards, &num_b);
        printf("generated %d boards at depth %d...\n", num_b, depth+1);
    }
    printf("searching %d boards...\n", num_b);
    // for (int i=0; i<num_b; i++) {
    //     pretty_print(&search_boards[i]);
    // }

    // search all boards sequentially on the CPU
    clock_t tic = clock();
    Board* B;
    int max_found = 0;
    int count = 0;
    uint32_t packed_int;
    int anchor_idx, anchor_x, anchor_y;
    for (int i=0; i<num_b; i++) {
        B = search_boards + i;
        anchor_idx = B->last_num - 1;
        packed_int = B->packed_array[anchor_idx];
        anchor_x = unpack_pos_x(packed_int);
        anchor_y = unpack_pos_y(packed_int);
        while ((anchor_x == unpack_pos_x(B->packed_array[anchor_idx])) && (anchor_y == unpack_pos_y(B->packed_array[anchor_idx]))) {
            next_board_state(B);
            if (max_found < B->last_num) {
                max_found = B->last_num;
                CopyHostBoard(&max_board, B);
            }
            count += 1;
        }
    }
    clock_t toc = clock();
    double split = (double)(toc - tic) / CLOCKS_PER_SEC;
    printf("CPU checked %d unique states in %.3f seconds\n", count, split);

    free(search_boards);

    return max_board;
}

Board computeTermGPU(int N) {

    Board max_board;
    InitBoard(&max_board, N);
    // hardcode result for n=1
    if (N == 1) {
        insert_one(&max_board, C_OFF, C_OFF);
        return max_board;
    }

    // set up starting boards
    int num_b = 6;
    Board* search_boards = (Board *) malloc(num_b * sizeof(Board));
    // here is all possible ways to place the first 2 hardcoded
    for (int i=0; i<num_b; i++) {
        InitBoard(&search_boards[i], N);
        insert_element(&search_boards[i], C_OFF, C_OFF, 2);
    }
    insert_one(&search_boards[0], C_OFF-1, C_OFF-1);
    insert_one(&search_boards[0], C_OFF+1, C_OFF+1);
    insert_one(&search_boards[1], C_OFF-1, C_OFF);
    insert_one(&search_boards[1], C_OFF+1, C_OFF+1);
    insert_one(&search_boards[2], C_OFF-1, C_OFF+1);
    insert_one(&search_boards[2], C_OFF+1, C_OFF+1);
    insert_one(&search_boards[3], C_OFF-1, C_OFF);
    insert_one(&search_boards[3], C_OFF+1, C_OFF);
    insert_one(&search_boards[4], C_OFF, C_OFF+1);
    insert_one(&search_boards[4], C_OFF+1, C_OFF);
    insert_one(&search_boards[5], C_OFF+1, C_OFF+1);
    insert_one(&search_boards[5], C_OFF+1, C_OFF);

    // do a breadth first search of boards until we have sufficiently many
    const int depth_chart[8] = {1, 6, 9, 9, 5, 6, 7, 8};
    for (int depth=2; depth<depth_chart[N-1]; depth++) {
        gen_all_next_boards(&search_boards, &num_b);
        printf("generated %d boards at depth %d...\n", num_b, depth+1);
    }
    printf("searching %d boards on the GPU...\n", num_b);
    // printf("search boards:\n");
    // for (int i=0; i<num_b; i++) {
    //     pretty_print(&search_boards[i]);
    // }

    // search all boards in parallel on the GPU
    clock_t tic = clock();
    SearchOnDevice(search_boards, &max_board, num_b);
    clock_t toc = clock();
    printf("GPU time: %f seconds\n", (double)(toc - tic) / CLOCKS_PER_SEC);

    free(search_boards);

    return max_board;
}

int main(int argc, char** argv) {
    Board max_board;
    if (argc == 2) {
        max_board = computeTermGPU(atoi(argv[1]));
    } else if ((argc == 3) && (strcmp(argv[1], "CPU") == 0)) {
        max_board = computeTermCPU(atoi(argv[2]));
    } else {
        printf("Invalid command arguments\n");
        printf("\n");
        printf("Usage:\n");
        printf("  PTOSS [N<=4]          find optimal board for N ones using multithreading on the GPU\n");
        printf("  PTOSS CPU [N<=4]      find optimal board for N ones using the CPU\n");
        return 0;
    }
    printf("final board (%d):\n", max_board.last_num+1);
    pretty_print(&max_board);
    return 0;
}