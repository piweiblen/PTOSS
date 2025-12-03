#include <math.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "Board.h"

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

const int x_around[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
const int y_around[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
#define AR_IDX(x,y) ((((x)+(3*(y))+5)*4)/5)

const int P_rngs[10] = {0, 1, 9, 37, 93, 163, 219, 247, 255, 256};
const int P_bits[256] = {0, 1, 2, 4, 8, 16, 32, 64, 128, 3, 5, 9, 17, 33, 65, 129, 6, 10, 18, 34, 66, 130, 12, 20, 36, 68, 132, 24, 40, 72, 136, 48, 80, 144, 96, 160, 192, 7, 11, 19, 35, 67, 131, 13, 21, 37, 69, 133, 25, 41, 73, 137, 49, 81, 145, 97, 161, 193, 14, 22, 38, 70, 134, 26, 42, 74, 138, 50, 82, 146, 98, 162, 194, 28, 44, 76, 140, 52, 84, 148, 100, 164, 196, 56, 88, 152, 104, 168, 200, 112, 176, 208, 224, 15, 23, 39, 71, 135, 27, 43, 75, 139, 51, 83, 147, 99, 163, 195, 29, 45, 77, 141, 53, 85, 149, 101, 165, 197, 57, 89, 153, 105, 169, 201, 113, 177, 209, 225, 30, 46, 78, 142, 54, 86, 150, 102, 166, 198, 58, 90, 154, 106, 170, 202, 114, 178, 210, 226, 60, 92, 156, 108, 172, 204, 116, 180, 212, 228, 120, 184, 216, 232, 240, 31, 47, 79, 143, 55, 87, 151, 103, 167, 199, 59, 91, 155, 107, 171, 203, 115, 179, 211, 227, 61, 93, 157, 109, 173, 205, 117, 181, 213, 229, 121, 185, 217, 233, 241, 62, 94, 158, 110, 174, 206, 118, 182, 214, 230, 122, 186, 218, 234, 242, 124, 188, 220, 236, 244, 248, 63, 95, 159, 111, 175, 207, 119, 183, 215, 231, 123, 187, 219, 235, 243, 125, 189, 221, 237, 245, 249, 126, 190, 222, 238, 246, 250, 252, 127, 191, 223, 239, 247, 251, 253, 254, 255};
const int P_idxs[256] = {0, 1, 2, 9, 3, 10, 16, 37, 4, 11, 17, 38, 22, 43, 58, 93, 5, 12, 18, 39, 23, 44, 59, 94, 27, 48, 63, 98, 73, 108, 128, 163, 6, 13, 19, 40, 24, 45, 60, 95, 28, 49, 64, 99, 74, 109, 129, 164, 31, 52, 67, 102, 77, 112, 132, 167, 83, 118, 138, 173, 148, 183, 198, 219, 7, 14, 20, 41, 25, 46, 61, 96, 29, 50, 65, 100, 75, 110, 130, 165, 32, 53, 68, 103, 78, 113, 133, 168, 84, 119, 139, 174, 149, 184, 199, 220, 34, 55, 70, 105, 80, 115, 135, 170, 86, 121, 141, 176, 151, 186, 201, 222, 89, 124, 144, 179, 154, 189, 204, 225, 158, 193, 208, 229, 213, 234, 240, 247, 8, 15, 21, 42, 26, 47, 62, 97, 30, 51, 66, 101, 76, 111, 131, 166, 33, 54, 69, 104, 79, 114, 134, 169, 85, 120, 140, 175, 150, 185, 200, 221, 35, 56, 71, 106, 81, 116, 136, 171, 87, 122, 142, 177, 152, 187, 202, 223, 90, 125, 145, 180, 155, 190, 205, 226, 159, 194, 209, 230, 214, 235, 241, 248, 36, 57, 72, 107, 82, 117, 137, 172, 88, 123, 143, 178, 153, 188, 203, 224, 91, 126, 146, 181, 156, 191, 206, 227, 160, 195, 210, 231, 215, 236, 242, 249, 92, 127, 147, 182, 157, 192, 207, 228, 161, 196, 211, 232, 216, 237, 243, 250, 162, 197, 212, 233, 217, 238, 244, 251, 218, 239, 245, 252, 246, 253, 254, 255};
const int P_wieght[256] = {0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 8};


// Populate a board with initial values and allocate needed memory
void InitBoard(Board* B, int N) {

    // Define initial values
    B->ones_num = N;
    B->ones_left = N;
    B->last_num = 0;  // maybe rename this var. this is the next available index in our pos arrays
}

// Copy a host board to a host board
void CopyHostBoard(Board* Bdest, const Board* Bsrc) {

    // Copy scalar values
    Bdest->ones_num = Bsrc->ones_num;
    Bdest->ones_left = Bsrc->ones_left;
    Bdest->last_num = Bsrc->last_num;

    // Copy position arrays
    memcpy(Bdest->packed_array, Bsrc->packed_array, MAX_HEIGHT * sizeof(uint32_t));
}

void flatten_board_list(uint32_t* flat_boards, Board* Boards, int num) {

    // put list of boards into a flat 1D array of ints
    int elem_idx;
    for (int b=0; b<num; b++) {
        elem_idx = 0;
        flat_boards[elem_idx * num + b] = Boards[b].ones_num;
        elem_idx++;
        flat_boards[elem_idx * num + b] = Boards[b].ones_left;
        elem_idx++;
        flat_boards[elem_idx * num + b] = Boards[b].last_num;
        elem_idx++;
        for (int i=0; i<MAX_HEIGHT; i++) {
            flat_boards[elem_idx * num + b] = Boards[b].packed_array[i];
            elem_idx++;
        }
    }
}

void unflatten_board_list(Board* Boards, uint32_t* flat_boards, int num) {

    // put list of boards into a flat 1D array of ints
    int elem_idx;
    for (int b=0; b<num; b++) {
        elem_idx = 0;
        Boards[b].ones_num = flat_boards[elem_idx * num + b];
        elem_idx++;
        Boards[b].ones_left = flat_boards[elem_idx * num + b];
        elem_idx++;
        Boards[b].last_num = flat_boards[elem_idx * num + b];
        elem_idx++;
        for (int i=0; i<MAX_HEIGHT; i++) {
            Boards[b].packed_array[i] = flat_boards[elem_idx * num + b];
            elem_idx++;
        }
    }
}

void pretty_print(Board* B) {

    // output our board in a nice grid format
    int width = floor(log10(MAX(1, abs(B->last_num+1)))) + 2;
    int min_x = INT_MAX;
    int max_x = INT_MIN;
    int min_y = INT_MAX;
    int max_y = INT_MIN;
    for (int i=0; i<B->last_num; i++) {
        uint32_t packed_int = B->packed_array[i];
        int pos_x = unpack_pos_x(packed_int);
        int pos_y = unpack_pos_y(packed_int);
        min_x = MIN(min_x, pos_x);
        max_x = MAX(max_x, pos_x);
        min_y = MIN(min_y, pos_y);
        max_y = MAX(max_y, pos_y);
    }
    for (int i=MAX_HEIGHT-B->ones_num; i<MAX_HEIGHT-B->ones_left; i++) {
        uint32_t packed_int = B->packed_array[i];
        int pos_x = unpack_pos_x(packed_int);
        int pos_y = unpack_pos_y(packed_int);
        min_x = MIN(min_x, pos_x);
        max_x = MAX(max_x, pos_x);
        min_y = MIN(min_y, pos_y);
        max_y = MAX(max_y, pos_y);
    }
    printf("\n");
    for (int y=min_y; y<=max_y; y++) {
        for (int x=min_x; x<=max_x; x++) {
            bool found = false;
            for (int i=0; i<B->last_num; i++) {
                uint32_t packed_int = B->packed_array[i];
                if ((unpack_pos_x(packed_int) == x) && (unpack_pos_y(packed_int) == y)) {
                    if (found) {  // for debug purposes, make the print messy for overlapping nums
                        printf("r%d", i+2);
                    } else {
                        printf("%*d", width, i+2);
                        found = true;
                    }
                }
            }
            for (int i=MAX_HEIGHT-B->ones_num; i<MAX_HEIGHT-B->ones_left; i++) {
                uint32_t packed_int = B->packed_array[i];
                if ((unpack_pos_x(packed_int) == x) && (unpack_pos_y(packed_int) == y)) {
                    if (found) {  // for debug purposes, make the print messy for overlapping nums
                        printf("r1");
                    } else {
                        printf("%*d", width, 1);
                        found = true;
                    }
                }
            }
            if (!found) {
                printf("%*d", width, 0);
            }
        }
        printf("\n");
    }
    printf("\n");
}

void insert_element(Board* B, int x, int y, int ones_added) {

    // insert an element onto our board
    B->packed_array[B->last_num] = pack(x, y, ones_added);
    B->last_num += 1;
}

void insert_one(Board* B, int x, int y) {

    // insert a one onto our board
    B->packed_array[MAX_HEIGHT-B->ones_left] = pack_pos(B->packed_array[B->last_num], x, y);
    B->ones_left -= 1;
}

void remove_element(Board* B, int* one_positions) {

    // remove an element from our board
    uint32_t packed_int = B->packed_array[B->last_num - 1];
    int otr = unpack_info(packed_int);  // ones to remove
    int x = unpack_pos_x(packed_int);
    int y = unpack_pos_y(packed_int);
    *one_positions = 0;
    for (int i=MAX_HEIGHT-B->ones_left-otr; i<MAX_HEIGHT-B->ones_left; i++) {
        packed_int = B->packed_array[i];
        int xi = unpack_pos_x(packed_int);
        int yi = unpack_pos_y(packed_int);

        *one_positions |= (1 << AR_IDX(xi - x, yi - y));
    }
    B->ones_left += otr;
    B->last_num -= 1;
}

int get_sum(Board* B, int x, int y, int* least_neighbor, int* open_neighbors) {  //TODO: lots of wasted time here, in the end we'll want to stop when we know (sum > target)

    // get the sum of elements surrounding position
    // returns -1 if position already populated
    int sum = 0;
    *least_neighbor = MAX_HEIGHT;
    *open_neighbors = 255;
    for (int i=0; i<B->last_num; i++) {

        // Unpack xi and yi
        uint32_t packed_int = B->packed_array[i];
        int xi = unpack_pos_x(packed_int);
        int yi = unpack_pos_y(packed_int);

        if ((xi == x) && (yi== y)) {
            return -1;
        }
        if ((x-1 <= xi) && (xi <= x+1)) {
            if ((y-1 <= yi) && (yi <= y+1)) {
                sum += i+2;
                *least_neighbor = MIN(*least_neighbor, i);
                *open_neighbors ^= (1 << AR_IDX(xi - x, yi - y));
            }
        }
    }
    for (int i=MAX_HEIGHT-B->ones_num; i<MAX_HEIGHT-B->ones_left; i++) {
        
        // Unpack xi and yi
        uint32_t packed_int = B->packed_array[i];
        int xi = unpack_pos_x(packed_int);
        int yi = unpack_pos_y(packed_int);

        if ((xi == x) && (yi == y)) {
            return -1;
        }
        if ((x-1 <= xi) && (xi <= x+1)) {
            if ((y-1 <= yi) && (yi <= y+1)) {
                sum += 1;
                *least_neighbor = MIN(*least_neighbor, i);
                *open_neighbors ^= (1 << AR_IDX(xi - x, yi - y));
            }
        }
    }
    return sum;
}

bool look_around(Board* B, int index, int* start_H, int* start_P) {

    // check around the location of a particular index for a spot to place the next element
    uint32_t packed_int = B->packed_array[index];
    int xi = unpack_pos_x(packed_int);
    int yi = unpack_pos_y(packed_int);
    int new_x, new_y;
    int cur_sum, min_nb, open_nb, ones_needed;
    for (int H=*start_H; H<8; H++) {  // iterate over all spots around (i+2)
        new_x = xi + x_around[H];
        new_y = yi + y_around[H];
        cur_sum = get_sum(B, new_x, new_y, &min_nb, &open_nb);
        if (min_nb == index) {  // don't go in spots with a lower anchor
            if (cur_sum <= B->last_num+2) {
                ones_needed = B->last_num + 2 - cur_sum;
                if (ones_needed <= MIN(B->ones_left, P_wieght[P_idxs[open_nb]])) {
                    for (int P=MAX(*start_P, P_rngs[ones_needed]); P<P_rngs[ones_needed+1]; P++) {
                        if ((P_bits[P] & open_nb) != P_bits[P]) continue;  // one positions must be open
                        insert_element(B, new_x, new_y, ones_needed);
                        for (int b=0; b<8; b++) {
                            if (P_bits[P] & (1<<b)) insert_one(B, new_x+x_around[b], new_y+y_around[b]);
                        }
                        *start_H = H;
                        *start_P = P;
                        return true;
                    }
                }
            }
        }
        *start_P = 0;
    }
    return false;
}

bool look_around(Board* B, int index, int start_H, int start_P) {

    // overload for when we don't care about capturing the start values
    int idc_H = start_H;
    int idc_P = start_P;
    return look_around(B, index, &idc_H, &idc_P);
}

void next_board_state(Board* B) {

    // iterate a board to its next state i.e. the next position in the search
    // this function assumes that "2" has already been placed

    // first try to add the next number
    for (int i=0; i<B->last_num / 2; i++) {  // choose a num (i+2) to try to place around
        if (look_around(B, i, 0, 0)) return;
    }
    for (int i=B->last_num - B->ones_num; i<B->last_num; i++) {  // we need to also look around high numbers
        if (look_around(B, i, 0, 0)) return;
    }
    if (B->last_num+2 <= 8) { // we need to also look around ones for small numbers
        for (int i=MAX_HEIGHT-B->ones_num; i<MAX_HEIGHT-B->ones_left; i++) { 
            if (look_around(B, i, 0, 0)) return;
        }
    }

    fprintf(stderr, "Past initial look_arounds\n");

    // failing to add a number, we'll attempt to move the current highest to a new position
    // continuing to remove elements until we succeed at moving one
    int old_x, old_y;
    int old_nb, last_P, last_H;
    while (B->last_num - 1) {  // abort if "3" is removed
        // first find where we left off
        uint32_t packed_int = B->packed_array[B->last_num - 1];
        old_x = unpack_pos_x(packed_int);
        old_y = unpack_pos_y(packed_int);
        get_sum(B, old_x, old_y, &old_nb, &last_P); // this line is only to get old_nb, last_P is garbage data here
                                                    // TODO: stop being lazy here, should write a seperate func that stops when old_nb is found
        uint32_t old_nb_packed = B->packed_array[old_nb];
        last_H = AR_IDX(old_x - unpack_pos_x(old_nb_packed), old_y - unpack_pos_y(old_nb_packed));
        // remove the element and search for new spot
        remove_element(B, &last_P);

        // Start with the element it was already around
        if (look_around(B, old_nb, last_H, P_idxs[last_P] + 1)) return;

        for (int i=old_nb+1; i<B->last_num / 2; i++) {  // choose a num (i+2) to try to place around
            if (look_around(B, i, 0, 0)) return;
        }
        for (int i=MAX(old_nb+1, B->last_num - B->ones_num); i<B->last_num; i++) {  // we need to also look around high numbers
            if (look_around(B, i, 0, 0)) return;
        }
        if (B->last_num+2 <= 8) { // we need to also look around ones for small numbers
            for (int i=MAX(old_nb+1, MAX_HEIGHT - B->ones_num); i<MAX_HEIGHT-B->ones_left; i++) { 
                if (look_around(B, i, 0, 0)) return;
            }
        }
    }
}

bool equivalent(Board* B1, Board* B2) {

    // determine if two boards are equivalent up to reflection/rotation
    // TODO: write
    return false;
}

void remove_duplicates(Board** boards, int* num_b) {
    
    // remove all duplicate boards from an array of boards
    // TODO: write
}

void gen_all_next_boards(Board** boards, int* num_b) {
    
    // given an array of boards generate all possible boards which can be made by adding an element
    int new_num_b = 16;
    Board* new_boards = (Board *) malloc(new_num_b * sizeof(Board));
    int cur_idx = 0;
    int start_H, start_P;
    Board* B;
    for (int i=0; i<*num_b; i++) {  // loop over all boards
        B = (*boards) + i;
        for (int j=0; j<MAX_HEIGHT-B->ones_left; j++) {  // loop over all possible anchors for the next number
            // filter out impossible anchors
            if ((j >= B->last_num / 2) && (j < B->last_num - B->ones_num)) continue;
            if ((j >= B->last_num) && (j < MAX_HEIGHT-B->ones_num)) continue;
            if ((B->last_num+2 > 8) && (j >= B->last_num)) continue;
            start_H = 0;
            start_P = 0;
            while (true) {  // loop over all possible positions/one placements
                // set up next board in the array
                if (cur_idx >= new_num_b) {
                    new_num_b *= 2;
                    new_boards = (Board *) realloc(new_boards, new_num_b * sizeof(Board));
                }
                CopyHostBoard(&new_boards[cur_idx], B);

                if (look_around(&new_boards[cur_idx], j, &start_H, &start_P)) {
                    start_P++;
                    cur_idx++;
                } else {
                    break;
                }
            }
        }
    }
    *boards = (Board *) realloc(*boards, cur_idx * sizeof(Board));
    memcpy(*boards, new_boards, cur_idx * sizeof(Board));
    free(new_boards);
    *num_b = cur_idx;
}

void pos_x_add() {

}