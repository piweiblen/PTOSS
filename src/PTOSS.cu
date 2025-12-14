#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "Board.h"
#include "PTOSS_kernel.h"

#define MAX_BOARD_LIST 3000000

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))


// set up starting boards, all possible ways to place the first 2
// inputs should be empty pointers for storing results
void gen_first_boards(Board** boards, int* num_b, const int N) {

    Board* B;
    *num_b = 6;
    *boards = (Board *) malloc((*num_b) * sizeof(Board));
    for (int i=0; i<*num_b; i++) {
        B = (*boards) + i;
        InitBoard(B, N);
    }
    B = (*boards) + 0;
    insert_one(B, C_OFF-1, C_OFF-1);
    insert_one(B, C_OFF+1, C_OFF+1);
    B = (*boards) + 1;
    insert_one(B, C_OFF-1, C_OFF);
    insert_one(B, C_OFF+1, C_OFF+1);
    B = (*boards) + 2;
    insert_one(B, C_OFF-1, C_OFF+1);
    insert_one(B, C_OFF+1, C_OFF+1);
    B = (*boards) + 3;
    insert_one(B, C_OFF-1, C_OFF);
    insert_one(B, C_OFF+1, C_OFF);
    B = (*boards) + 4;
    insert_one(B, C_OFF, C_OFF+1);
    insert_one(B, C_OFF+1, C_OFF);
    B = (*boards) + 5;
    insert_one(B, C_OFF+1, C_OFF+1);
    insert_one(B, C_OFF+1, C_OFF);
    for (int i=0; i<*num_b; i++) {
        B = (*boards) + i;
        insert_element(B, 2, C_OFF, C_OFF, 0);
    }
}

// given an array of boards generate all possible boards which can be made by adding an element
void gen_all_next_boards(Board** boards, int* num_b, Board** multiboards, int* num_mub, bool* any_dedup) {
    
    int default_nb = 16;
    int cur_idx = 0;
    Board* B;
    int dedup_idx, dedup_num;
    bool new_any_dedup = false;
    int start_H, start_P;

    // start with boards -> boards
    int new_num_bb = default_nb;
    Board* new_bds_bb = (Board *) malloc(new_num_bb * sizeof(Board));
    cur_idx = 0;
    for (int i=0; i<*num_b; i++) {  // loop over all boards
        B = (*boards) + i;
        dedup_idx = cur_idx;
        for (int j=0; j<B->next_idx; j++) {  // loop over all possible anchors for the next number
            // filter out impossible anchors
            if (((B->next_idx - B->ones_down + 2) / 2 < unpack_value(B->packed_array[j])) &&
                (unpack_value(B->packed_array[j]) < (B->next_idx - B->ones_down + 2) - B->ones_num)) {
                continue;
            }
            start_H = 0;
            start_P = 0;
            while (true) {  // loop over all possible positions/one placements
                // set up next board in the array
                if (cur_idx >= new_num_bb) {
                    new_num_bb = MIN(2 * new_num_bb, new_num_bb + 1000000);
                    new_bds_bb = (Board *) realloc(new_bds_bb, new_num_bb * sizeof(Board));
                }
                CopyHostBoard(&new_bds_bb[cur_idx], B);

                if (look_around(&new_bds_bb[cur_idx], j, &start_H, &start_P)) {
                    start_P++;
                    cur_idx++;
                } else {
                    break;
                }
            }
        }
        if (any_dedup) {
            // deduplicate generated boards
            dedup_num = cur_idx - dedup_idx;
            new_any_dedup |= remove_duplicates(new_bds_bb + dedup_idx, &dedup_num);
            cur_idx = dedup_idx + dedup_num;
        }
    }
    new_num_bb = cur_idx;

    // next boards -> muliboards
    int new_num_bm = default_nb;
    Board* new_bds_bm = (Board *) malloc(new_num_bm * sizeof(Board));
    cur_idx = 0;
    for (int i=0; i<*num_b; i++) {  // loop over all boards
        B = (*boards) + i;
        dedup_idx = cur_idx;
        for (int j=0; j<B->next_idx; j++) {  // loop over all possible anchors for the next number
            // filter out impossible anchors
            if (((B->next_idx - B->ones_down + 2) / 2 < unpack_value(B->packed_array[j])) &&
                (unpack_value(B->packed_array[j]) < (B->next_idx - B->ones_down + 2) - B->ones_num)) {
                continue;
            }
            start_P = 0;
            while (true) {  // loop over all possible one placements
                // set up next board in the array
                if (cur_idx >= new_num_bm) {
                    new_num_bm = MIN(2 * new_num_bm, new_num_bm + 1000000);
                    new_bds_bm = (Board *) realloc(new_bds_bm, new_num_bm * sizeof(Board));
                }
                CopyHostBoard(&new_bds_bm[cur_idx], B);

                if (split_board(&new_bds_bm[cur_idx], &start_P)) {
                    start_P++;
                    cur_idx++;
                } else {
                    break;
                }
            }
        }
        if (any_dedup) {
            // deduplicate generated boards
            dedup_num = cur_idx - dedup_idx;
            new_any_dedup |= remove_duplicates(new_bds_bm + dedup_idx, &dedup_num);
            cur_idx = dedup_idx + dedup_num;
        }
    }
    new_num_bm = cur_idx;

    // next muliboards -> boards
    int start_tr, anc_0, anc_1, start_H0, start_H1;
    int new_num_mb = default_nb;
    Board* new_bds_mb = (Board *) malloc(new_num_mb * sizeof(Board));
    cur_idx = 0;
    for (int i=0; i<*num_mub; i++) {  // loop over all boards
        B = (*multiboards) + i;
        dedup_idx = cur_idx;
        for (int j=0; j<B->next_idx; j++) {  // loop over all possible anchors for the next number
            // filter out impossible anchors
            if (((B->next_idx - B->ones_down + 2) / 2 < unpack_value(B->packed_array[j])) &&
                (unpack_value(B->packed_array[j]) < (B->next_idx - B->ones_down + 2) - B->ones_num)) {
                continue;
            }
            start_tr = anc_0 = anc_1 = start_H0 = start_H1 = start_P = 0;
            while (true) {  // loop over all possible one placements
                // set up next board in the array
                if (cur_idx >= new_num_mb) {
                    new_num_mb = MIN(2 * new_num_mb, new_num_mb + 1000000);
                    new_bds_mb = (Board *) realloc(new_bds_mb, new_num_mb * sizeof(Board));
                }
                CopyHostBoard(&new_bds_mb[cur_idx], B);

                if (merge_board(&new_bds_mb[cur_idx], &start_tr, &anc_0, &anc_1, &start_H0, &start_H1, &start_P)) {;
                    start_P++;
                    cur_idx++;
                } else {
                    break;
                }
            }
        }
        if (any_dedup) {
            // deduplicate generated boards
            dedup_num = cur_idx - dedup_idx;
            new_any_dedup |= remove_duplicates(new_bds_mb + dedup_idx, &dedup_num);
            cur_idx = dedup_idx + dedup_num;
        }
    }
    new_num_mb = cur_idx;

    // finally muliboards -> muliboards
    int new_num_mm = default_nb;
    Board* new_bds_mm = (Board *) malloc(new_num_mm * sizeof(Board));
    cur_idx = 0;
    for (int i=0; i<*num_mub; i++) {  // loop over all boards
        B = (*multiboards) + i;
        dedup_idx = cur_idx;
        for (int j=0; j<B->next_idx; j++) {  // loop over all possible anchors for the next number
            // filter out impossible anchors
            if (((B->next_idx - B->ones_down + 2) / 2 < unpack_value(B->packed_array[j])) &&
                (unpack_value(B->packed_array[j]) < (B->next_idx - B->ones_down + 2) - B->ones_num)) {
                continue;
            }
            start_H = 0;
            start_P = 0;
            while (true) {  // loop over all possible positions/one placements
                // set up next board in the array
                if (cur_idx >= new_num_mm) {
                    new_num_mm = MIN(2 * new_num_mm, new_num_mm + 1000000);
                    new_bds_mm = (Board *) realloc(new_bds_mm, new_num_mm * sizeof(Board));
                }
                CopyHostBoard(&new_bds_mm[cur_idx], B);

                if (look_around(&new_bds_mm[cur_idx], j, &start_H, &start_P)) {
                    start_P++;
                    cur_idx++;
                } else {
                    break;
                }
            }
        }
        if (any_dedup) {
            // deduplicate generated boards
            dedup_num = cur_idx - dedup_idx;
            new_any_dedup |= remove_duplicates(new_bds_mm + dedup_idx, &dedup_num);
            cur_idx = dedup_idx + dedup_num;
        }
    }
    new_num_mm = cur_idx;

    // transfer generated boards to the output
    *any_dedup = new_any_dedup;
    *num_b = new_num_bb + new_num_mb;
    *boards = (Board *) realloc(*boards, MAX(1,*num_b) * sizeof(Board));
    *num_mub = new_num_bm + new_num_mm;
    *multiboards = (Board *) realloc(*multiboards, MAX(1,*num_mub) * sizeof(Board));
    memcpy(*boards, new_bds_bb, new_num_bb * sizeof(Board));
    memcpy((*boards)+new_num_bb, new_bds_mb, new_num_mb * sizeof(Board));
    memcpy(*multiboards, new_bds_bm, new_num_bm * sizeof(Board));
    memcpy((*multiboards)+new_num_bm, new_bds_mm, new_num_mm * sizeof(Board));
    // clean up
    free(new_bds_bb);
    free(new_bds_bm);
    free(new_bds_mb);
    free(new_bds_mm);
}

// find all boards up to a certain depth, and merge any remaining multiboards
// first two inputs should be empty pointers for storing results
void gen_boards_to_depth(Board** boards, int* num_b, const int N, const int depth) {

    // start with our hardcoded depth two
    int cur_depth = 2;
    gen_first_boards(boards, num_b, N);
    bool any_dedup = true;

    // then generate up to the desired depth
    int num_mub = 0;
    Board* new_mubs = (Board *) malloc(sizeof(Board));
    while (cur_depth < depth) {
        cur_depth++;
        gen_all_next_boards(boards, num_b, &new_mubs, &num_mub, &any_dedup);
        if (num_mub) {
            printf("generated %d boards and %d multiboards at depth %d...\n",
                   *num_b, num_mub, cur_depth);
        } else {
            printf("generated %d boards at depth %d...\n", *num_b, cur_depth);
        }
        if ((cur_depth == 4) && (N >= 5)) {
            *num_b = 1;
            printf("reducing to 1...\n");
        }
    }

    // then keep iterating on the multiboards until they are all merged or gone
    if (num_mub) {
        printf("cleaning up %d multiboards...\n", num_mub);
        int new_num = 0;
        Board* new_boards = (Board *) malloc(sizeof(Board));
        while (num_mub) {
            cur_depth++;
            gen_all_next_boards(&new_boards, &new_num, &new_mubs, &num_mub, &any_dedup);
            *boards = (Board *) realloc(*boards, (*num_b + new_num) * sizeof(Board));
            for (int i=0; i<new_num; i++) {
                CopyHostBoard((*boards) + (*num_b) + i, new_boards + i);
            }
            printf("added %d boards, now %d multiboards remaining at depth %d...\n",
                   new_num, num_mub, cur_depth);
            *num_b += new_num;
            new_num = 0;
        }
        free(new_boards);
    }
    free(new_mubs);
}


Board computeTermCPU(int N) {

    clock_t tic = clock();
    Board max_board;
    int max_found = 0;
    InitBoard(&max_board, N);
    // hardcode result for n=1
    if (N == 1) {
        insert_one(&max_board, C_OFF, C_OFF);
        return max_board;
    }

    // do a breadth first search of boards to generate a pool of boards to
    // then do another breadth first search on
    const int depth_chart[8] = {1, 4, 4, 4, 4, 4, 4, 4};
    // start with our hardcoded depth two
    int num_b;
    Board* search_boards;
    int cur_depth = 2;
    gen_first_boards(&search_boards, &num_b, N);
    int count = num_b;
    bool any_dedup = true;
    // generate up to the first depth
    int num_mub = 0;
    Board* search_mubs = (Board *) malloc(sizeof(Board));
    while (cur_depth < depth_chart[N-1]) {
        cur_depth++;
        gen_all_next_boards(&search_boards, &num_b, &search_mubs, &num_mub, &any_dedup);
        if (num_mub) {
            printf("generated %d boards and %d multiboards at depth %d...\n",
                   num_b, num_mub, cur_depth);
        } else {
            printf("generated %d boards at depth %d...\n", num_b, cur_depth);
        }
        count += num_b + num_mub;
    }
    // begin second tier depth first search
    int new_num_b;
    Board* new_search_boards;
    int new_num_mub;
    Board* new_search_mubs = (Board *) malloc(sizeof(Board));
    for (int i=0; i<num_b; i++) {
        bool new_any_dedup = any_dedup;
        new_num_b = 1;
        new_num_mub = 0;
        new_search_boards = (Board *) malloc(sizeof(Board));
        CopyHostBoard(new_search_boards, search_boards + i);
        while ((new_num_b > 0) || (new_num_mub > 0)) {
            if (new_num_b > 0) {
                Board* B = new_search_boards;
                if (max_found < B->next_idx) {
                    max_found = B->next_idx;
                    CopyHostBoard(&max_board, B);
                }
            }
            gen_all_next_boards(&new_search_boards, &new_num_b, &new_search_mubs, &new_num_mub, &new_any_dedup);
            count += new_num_b + new_num_mub;
        }
        free(new_search_boards);
    }
    free(new_search_mubs);
    new_search_boards = (Board *) malloc(sizeof(Board));
    // and the same for the multiboards
    for (int i=0; i<num_mub; i++) {
        bool new_any_dedup = any_dedup;
        new_num_b = 0;
        new_num_mub = 1;
        new_search_mubs = (Board *) malloc(sizeof(Board));
        CopyHostBoard(new_search_mubs, search_mubs + i);
        while ((new_num_b > 0) || (new_num_mub > 0)) {
            if (new_num_b > 0) {
                Board* B = new_search_boards;
                if (max_found < B->next_idx) {
                    max_found = B->next_idx;
                    CopyHostBoard(&max_board, B);
                }
            }
            gen_all_next_boards(&new_search_boards, &new_num_b, &new_search_mubs, &new_num_mub, &new_any_dedup);
            count += new_num_b + new_num_mub;
        }
        free(new_search_mubs);
    }
    free(new_search_boards);

    free(search_boards);
    free(search_mubs);

    clock_t toc = clock();
    printf("CPU checked %d unique states\n", count);
    printf("total time: %f seconds\n", (double)(toc - tic) / CLOCKS_PER_SEC);

    return max_board;
}

Board computeTermGPU(int N) {

    clock_t tict = clock();
    Board max_board;
    InitBoard(&max_board, N);
    // hardcode result for n=1
    if (N == 1) {
        insert_one(&max_board, C_OFF, C_OFF);
        return max_board;
    }

    // do a breadth first search of boards to generate a pool of boards to search
    // to ensure all boards are searched, depth should be at least N-2 for all N>4 
    const int depth_chart[8] = {1, 6, 9, 12, 6, 5, 5, 6};
    int num_b;
    Board* search_boards;
    gen_boards_to_depth(&search_boards, &num_b, N, depth_chart[N-1]);
    printf("searching %d boards on the GPU...\n", num_b);
    // printf("search boards:\n");
    // for (int i=0; i<num_b; i++) {
    //     pretty_print(&search_boards[i]);
    // }

    // search all boards in parallel on the GPU
    clock_t tic = clock();
    uint32_t* dvc_boards;
    int* dvc_max_val;
    LaunchSearchKernel(search_boards, &dvc_boards, &dvc_max_val, num_b);
    max_board = GetKernelResult(&dvc_boards, &dvc_max_val, num_b);
    clock_t toc = clock();
    printf("GPU time: %f seconds\n", (double)(toc - tic) / CLOCKS_PER_SEC);
    printf("total time: %f seconds\n", (double)(toc - tict) / CLOCKS_PER_SEC);

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
        printf("  PTOSS [N]          find optimal board for N ones using multithreading on the GPU\n");
        printf("  PTOSS CPU [N]      find optimal board for N ones using the CPU\n");
        return 0;
    }
    printf("final board (%d):\n", max_board.next_idx - max_board.ones_down + 1);
    pretty_print(&max_board);
    return 0;
}