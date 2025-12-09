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

// ----- helper functions -----

// Transform a coordinate
void transform(int* x, int* y, int transform) {

    // swap x and y coordinates or not
    int temp;
    if (transform & 0b1) {
        temp = *x;
        *x = *y;
        *y = temp;
    }

    // Make x positive or negative
    if (transform & 0b10)
        (*x) *= -1;

    // Make y positive or negative
    if (transform & 0b100)
        (*y) *= -1;
}

// ----- board functions -----

// Populate a board with initial values and allocate needed memory
void InitBoard(Board* B, int N) {

    // Define initial values
    B->ones_num = N;
    B->ones_down = 0;
    B->next_idx = 0;
}

// Copy a host board to a host board
void CopyHostBoard(Board* Bdest, const Board* Bsrc) {

    // Copy scalar values
    Bdest->ones_num = Bsrc->ones_num;
    Bdest->ones_down = Bsrc->ones_down;
    Bdest->next_idx = Bsrc->next_idx;

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
        flat_boards[elem_idx * num + b] = Boards[b].ones_down;
        elem_idx++;
        flat_boards[elem_idx * num + b] = Boards[b].next_idx;
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
        Boards[b].ones_down = flat_boards[elem_idx * num + b];
        elem_idx++;
        Boards[b].next_idx = flat_boards[elem_idx * num + b];
        elem_idx++;
        for (int i=0; i<MAX_HEIGHT; i++) {
            Boards[b].packed_array[i] = flat_boards[elem_idx * num + b];
            elem_idx++;
        }
    }
}

// return whether or not B is a multiboard
bool is_mub(Board* B) {
    for (int i=0; i<B->next_idx; i++) {
        if (unpack_mub(B->packed_array[i])) {
            return true;
        }
        if (unpack_value(B->packed_array[i]) > 8) {
            return false;
        }
    }
    return false;
}

// output one board in a nice grid format
void pretty_print(Board* B, int mub) {

    int width = floor(log10(MAX(1, abs(B->next_idx - B->ones_down + 1)))) + 2;
    int min_x = INT_MAX;
    int max_x = INT_MIN;
    int min_y = INT_MAX;
    int max_y = INT_MIN;
    for (int i=0; i<B->next_idx; i++) {
        uint32_t packed_int = B->packed_array[i];
        if (unpack_mub(packed_int) != mub) continue;
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
            for (int i=0; i<B->next_idx; i++) {
                uint32_t packed_int = B->packed_array[i];
                if (unpack_mub(packed_int) != mub) continue;
                if ((unpack_pos_x(packed_int) == x) && (unpack_pos_y(packed_int) == y)) {
                    if (found) {  // for debug purposes, make the print messy for overlapping nums
                        printf("r%d", unpack_value(packed_int));
                    } else {
                        printf("%*d", width, unpack_value(packed_int));
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

// overload to handle multiboards
void pretty_print(Board* B) {
    pretty_print(B, 0);
    if (is_mub(B)) {
        printf("+ secondary board:\n");
        pretty_print(B, 1);
    }
}

// insert an element onto our board
void insert_element(Board* B, int value, int x, int y) {

    B->packed_array[B->next_idx] = pack(x, y, value, 0);
    B->next_idx += 1;
}

// overload for multiboard
void insert_element(Board* B, int value, int x, int y, int mub) {

    B->packed_array[B->next_idx] = pack(x, y, value, mub);
    B->next_idx += 1;
}

// insert a one onto our board
void insert_one(Board* B, int x, int y) {

    B->packed_array[B->next_idx] = pack(x, y, 1, 0);
    B->next_idx += 1;
    B->ones_down += 1;
}

// overload for multiboard
void insert_one(Board* B, int x, int y, int mub) {

    B->packed_array[B->next_idx] = pack(x, y, 1, mub);
    B->next_idx += 1;
    B->ones_down += 1;
}

// remove an element from our board
void remove_element(Board* B, int* one_positions) {

    uint32_t packed_int = B->packed_array[B->next_idx - 1];
    int x = unpack_pos_x(packed_int);
    int y = unpack_pos_y(packed_int);
    B->next_idx -= 1;
    packed_int = B->packed_array[B->next_idx - 1];
    *one_positions = 0;
    while (unpack_value(packed_int) == 1) {
        int xi = unpack_pos_x(packed_int);
        int yi = unpack_pos_y(packed_int);
        *one_positions |= (1 << AR_IDX(xi - x, yi - y));
        B->ones_down -= 1;
        B->next_idx -= 1;
        packed_int = B->packed_array[B->next_idx - 1];
    }
}

// get the lowest index of neighboring cells of the given position
int get_anchor(Board* B, int mub, int x, int y) {

    int pos_x, pos_y;
    for (int i=0; i<B->next_idx; i++) {
        uint32_t packed_int = B->packed_array[i];
        if (unpack_mub(packed_int) != mub) continue;
        pos_x = unpack_pos_x(packed_int);
        pos_y = unpack_pos_y(packed_int);
        if ((pos_x == x) && (pos_y == y)) {
            continue;
        }
        if ((x-1 <= pos_x) && (pos_x <= x+1)) {
            if ((y-1 <= pos_y) && (pos_y <= y+1)) {
                return i;
            }
        }
    }
    return MAX_HEIGHT;
}

// get the sum of elements surrounding position
// returns INT_MAX if position already populated
int get_sum(Board* B, int mub, int x, int y, int target, int* anchor, int* open_neighbors) {

    uint32_t packed_int;
    int xi, yi, xii, yii;
    *anchor = MAX_HEIGHT;
    *open_neighbors = 255;
    int sum = 0;
    for (int i=0; i<B->next_idx; i++) {
        // Unpack xi and yi
        packed_int = B->packed_array[i];
        if (unpack_mub(packed_int) != mub) continue;
        xi = unpack_pos_x(packed_int);
        yi = unpack_pos_y(packed_int);
        if ((xi == x) && (yi== y)) {
            return INT_MAX;
        }
        if ((x-1 <= xi) && (xi <= x+1) && (y-1 <= yi) && (yi <= y+1)) {
            sum += unpack_value(packed_int);
            *anchor = MIN(*anchor, i);
            *open_neighbors &= ~(1 << AR_IDX(xi - x, yi - y));
            if (sum > target) return INT_MAX;
        }
        if ((x-2 <= xi) && (xi <= x+2) && (y-2 <= yi) && (yi <= y+2)) {
            if (1 < unpack_value(packed_int)) {  // new ones can't be adjacent to old nums
                for (int H=0; H<8; H++) { 
                    xii = xi + x_around[H];
                    yii = yi + y_around[H];
                    if ((x == xii) && (y == yii)) continue;
                    if ((x-1 <= xii) && (xii <= x+1) && (y-1 <= yii) && (yii <= y+1)) {
                        *open_neighbors &= ~(1 << AR_IDX(xii - x, yii - y));
                    }
                }
            }
        }
    }
    return sum;
}

// check around the location of a particular index for a spot to place the next element
bool look_around(Board* B, int index, int* start_H, int* start_P) {

    uint32_t packed_int = B->packed_array[index];
    int xi = unpack_pos_x(packed_int);
    int yi = unpack_pos_y(packed_int);
    int mub = unpack_mub(packed_int);
    int next_num = B->next_idx - B->ones_down + 2;
    int new_x, new_y;
    int cur_sum, min_nb, open_nb, ones_needed;
    for (int H=*start_H; H<8; H++) {  // iterate over all spots around (i+2)
        new_x = xi + x_around[H];
        new_y = yi + y_around[H];
        cur_sum = get_sum(B, mub, new_x, new_y, next_num, &min_nb, &open_nb);
        if (min_nb == index) {  // don't go in spots with a lower anchor
            if (cur_sum <= next_num) {
                ones_needed = next_num - cur_sum;
                if (ones_needed <= MIN(B->ones_num - B->ones_down, P_wieght[P_idxs[open_nb]])) {
                    for (int P=MAX(*start_P, P_rngs[ones_needed]); P<P_rngs[ones_needed+1]; P++) {
                        if ((P_bits[P] & open_nb) != P_bits[P]) continue;  // one positions must be open
                        for (int b=0; b<8; b++) {
                            if (P_bits[P] & (1<<b))
                                insert_one(B, new_x+x_around[b], new_y+y_around[b], mub);
                        }
                        insert_element(B, next_num, new_x, new_y, mub);
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

// overload for when we don't care about capturing the start values or about multiboards
bool look_around(Board* B, int index, int start_H, int start_P) {

    int idc_H = start_H;
    int idc_P = start_P;
    return look_around(B, index, &idc_H, &idc_P);
}

// iterate a board to its next state i.e. the next position in the search
// this function assumes that "2" has already been placed
void next_board_state(Board* B) {

    // first try to add the next number
    for (int i=0; i<B->next_idx; i++) {
        if (((B->next_idx - B->ones_down + 2) / 2 < unpack_value(B->packed_array[i])) &&
            (unpack_value(B->packed_array[i]) < (B->next_idx - B->ones_down + 2) - B->ones_num)) {
            continue;
        }
        if (look_around(B, i, 0, 0)) return;
    }

    // failing to add a number, we'll attempt to move the current highest to a new position
    // continuing to remove elements until we succeed at moving one
    int old_x, old_y;
    int old_nb, last_P, last_H;
    while (B->next_idx > 3) {  // abort if "3" is removed
        // first find where we left off
        uint32_t packed_int = B->packed_array[B->next_idx - 1];
        old_x = unpack_pos_x(packed_int);
        old_y = unpack_pos_y(packed_int);
        old_nb = get_anchor(B, 0, old_x, old_y);
        uint32_t old_nb_packed = B->packed_array[old_nb];
        last_H = AR_IDX(old_x - unpack_pos_x(old_nb_packed), old_y - unpack_pos_y(old_nb_packed));
        // remove the element and search for new spot
        remove_element(B, &last_P);

        // Start with the element it was already around
        if (look_around(B, old_nb, last_H, P_idxs[last_P] + 1)) return;

        for (int i=old_nb+1; i<B->next_idx; i++) {
            if (((B->next_idx - B->ones_down + 2) / 2 < unpack_value(B->packed_array[i])) &&
                (unpack_value(B->packed_array[i]) < (B->next_idx - B->ones_down + 2) - B->ones_num)) {
                continue;
            }
            if (look_around(B, i, 0, 0)) return;
        }
    }
}

// ----- multiboard madness -----

// attempt to split board into a multiboard
bool split_board(Board* B, int* start_P) {

    int ones_needed = B->next_idx - B->ones_down + 2;
    if (B->ones_num - B->ones_down < ones_needed) return false;
    // if we have enough ones, we can split
    for (int P=MAX(*start_P, P_rngs[ones_needed]); P<P_rngs[ones_needed+1]; P++) {
        for (int b=0; b<8; b++) {
            if (P_bits[P] & (1<<b)) 
                insert_one(B, C_OFF + x_around[b], C_OFF + y_around[b], 1);
        }
        insert_element(B, ones_needed, C_OFF, C_OFF, 1);
        *start_P = P;
        return true;
    }
    return false;
}

// reorient one board of a multiboard
void reorient_mub(Board* B, int mub, int t) {
    uint32_t packed_int;
    int x, y, xt, yt, x_off, y_off;
    bool first_coord = true;
    for (int i=0; i<B->next_idx; i++) {
        packed_int = B->packed_array[i];
        if (unpack_mub(packed_int) != mub) continue;
        x = xt = unpack_pos_x(packed_int);
        y = yt = unpack_pos_y(packed_int);
        transform(&xt, &yt, t);
        if (first_coord) {
            x_off = x - xt;
            y_off = y - yt;
            first_coord = false;
            continue;
        }
        B->packed_array[i] = pack_pos(packed_int, xt + x_off, yt + y_off);
    }
}

// bring multiboard back down to one board while aligning the given coordinates
bool overlay_mub(Board* B, int x_off, int y_off) {

    // confirm no new adjacencies will be born
    for (int i=0; i<B->next_idx; i++) {
        uint32_t packed_int0 = B->packed_array[i];
        if (unpack_mub(packed_int0) != 1) continue;
        int xi = unpack_pos_x(packed_int0) + x_off;
        int yi = unpack_pos_y(packed_int0) + y_off;
        for (int j=0; j<B->next_idx; j++) {
            uint32_t packed_int1 = B->packed_array[j];
            if (unpack_mub(packed_int1) != 0) continue;
            int xj = unpack_pos_x(packed_int1);
            int yj = unpack_pos_y(packed_int1);
            if ((unpack_value(packed_int0) == 1) && (unpack_value(packed_int1) == 1))
                continue;
            if ((-1 <= xi - xj) && (xi - xj <= 1) && (-1 <= yi - yj) && (yi - yj <= 1))
                return false;
        }
    }
    // all clear to drop it down
    for (int i=0; i<B->next_idx; i++) {
        uint32_t packed_int = B->packed_array[i];
        if (unpack_mub(packed_int) != 1) continue;
        int xi = unpack_pos_x(packed_int) + x_off;
        int yi = unpack_pos_y(packed_int) + y_off;
        B->packed_array[i] = pack(xi, yi, unpack_value(packed_int), 0);
    }
    return true;
}

// attempt to merge a multiboard back into a regular board
bool merge_board(Board* Bin, int* start_tr, int* anc_0, int* anc_1, int* start_H0, int* start_H1, int* start_P) {
    
    Board* B = (Board*) malloc(sizeof(Board));
    int target = Bin->next_idx - Bin->ones_down + 2;
    uint32_t packed_int0, packed_int1;
    int xi, yi, xj, yj;
    int new_xi, new_yi, new_xj, new_yj;
    int cur_sum0, min_nb0, cur_sum1, min_nb1;
    int open_nb, open_nb0, open_nb1;
    int ones_needed;
    for (int t=*start_tr; t<8; t++) {  // set the orientation of the second board
        memcpy(B, Bin, sizeof(Board));
        reorient_mub(B, 1, t);
        // find our spot in mub 0
        for (int i=*anc_0; i<B->next_idx; i++) {  // anchor index for mub 0
            *anc_0 = 0;
            packed_int0 = B->packed_array[i];
            if (unpack_mub(packed_int0) != 0) continue;
            xi = unpack_pos_x(packed_int0);
            yi = unpack_pos_y(packed_int0);
            for (int H0=*start_H0; H0<8; H0++) {
                *start_H0 = 0;
                new_xi = xi + x_around[H0];
                new_yi = yi + y_around[H0];
                cur_sum0 = get_sum(B, 0, new_xi, new_yi, target, &min_nb0, &open_nb0);
                if (cur_sum0 > target) continue;
                if (min_nb0 != i) continue;  // don't go in spots with a lower anchor
                // find our spot in mub 1
                for (int j=*anc_1; j<B->next_idx; j++) {  // anchor index for mub 1
                    *anc_1 = 0;
                    packed_int1 = B->packed_array[j];
                    if (unpack_mub(packed_int1) != 1) continue;
                    if (unpack_value(packed_int1) + cur_sum0 > target) continue;
                    xj = unpack_pos_x(packed_int1);
                    yj = unpack_pos_y(packed_int1);
                    for (int H1=*start_H1; H1<8; H1++) {
                        *start_H1 = 0;
                        new_xj = xj + x_around[H1];
                        new_yj = yj + y_around[H1];
                        cur_sum1 = get_sum(B, 1, new_xj, new_yj, target-cur_sum0, &min_nb1, &open_nb1);
                        if (cur_sum1 > target-cur_sum0) continue;
                        if (min_nb1 != j) continue;  // don't go in spots with a lower anchor
                        // if we get this far, the sums work out
                        ones_needed = target - cur_sum0 - cur_sum1;
                        open_nb = open_nb0 & open_nb1;
                        if (ones_needed > B->ones_num - B->ones_down) continue;
                        if (ones_needed > P_wieght[P_idxs[open_nb]]) continue;
                        // if we get this far, the ones work out
                        if (!overlay_mub(B, new_xi - new_xj, new_yi - new_yj)) continue;
                        // with the boards overlayed, the rest is the same as in look_around
                        // validate that the one positions aren't adjacent to other numbers
                        for (int P=MAX(*start_P, P_rngs[ones_needed]); P<P_rngs[ones_needed+1]; P++) {
                            if ((P_bits[P] & open_nb) != P_bits[P]) continue;  // one positions must be open
                            insert_element(B, target, new_xi, new_yi);
                            for (int b=0; b<8; b++) {
                                if (P_bits[P] & (1<<b))
                                    insert_one(B, new_xi+x_around[b], new_yi+y_around[b]);
                            }
                            memcpy(Bin, B, sizeof(Board));
                            *start_tr = t;
                            *anc_0 = i;
                            *anc_1 = j;
                            *start_H0 = H0;
                            *start_H1 = H1;
                            *start_P = P;
                            free(B);
                            return true;
                        }
                        // if we get here there were no valid Ps left, but we already overlayed
                        memcpy(B, Bin, sizeof(Board));
                        reorient_mub(B, 1, t);
                        *start_P = 0;
                    }
                }
            }
        }
    }
    free(B);
    return false;
}

// ----- breadth first board generation -----

// Determine if two boards are equivalent up to reflection/rotation
bool equivalent(Board* B1, Board* B2, int mub) {

    // Check if the parameters of the boards are equal
    if (B1->ones_num != B2->ones_num) return false;
    if (B1->ones_down != B2->ones_down) return false;
    if (B1->next_idx != B2->next_idx) return false;

    // Check all transforms of the arrays to see if any match up
    bool valid, first_coord;
    int x1, y1, x2, y2;
    int x_off, y_off;
    uint32_t B1_packed_int, B2_packed_int;
    for (int t=0; t<8; t++) {
        valid = true;
        first_coord = true;
        // check that all numbers align
        for (int i = 0; i < B1->next_idx; i++) {
            B1_packed_int = B1->packed_array[i];
            B2_packed_int = B2->packed_array[i];
            if (unpack_value(B1_packed_int) != unpack_value(B2_packed_int))
                return false;
            if (unpack_mub(B1_packed_int) != unpack_mub(B2_packed_int))
                return false;
            if (unpack_value(B1_packed_int) == 1)
                continue;
            if (unpack_mub(B1_packed_int) != mub)
                continue;
            x1 = unpack_pos_x(B1_packed_int);
            y1 = unpack_pos_y(B1_packed_int);
            x2 = unpack_pos_x(B2_packed_int);
            y2 = unpack_pos_y(B2_packed_int);
            transform(&x2, &y2, t);
            if (first_coord) {
                x_off = x1 - x2;
                y_off = y1 - y2;
                first_coord = false;
                continue;
            }
            if ((x1 != x2 + x_off) || (y1 != y2 + y_off)) {
                valid = false;
                break;
            }
        }
        if (!valid) continue;
        // check that all ones align
        for (int i = 0; i < B1->next_idx; i++) {
            B1_packed_int = B1->packed_array[i];
            if (unpack_value(B1_packed_int) != 1) continue;
            if (unpack_mub(B1_packed_int) != mub) continue;
            x1 = unpack_pos_x(B1_packed_int);
            y1 = unpack_pos_y(B1_packed_int);
            valid = false;
            for (int j = 0; j < B1->next_idx; j++) {
                B2_packed_int = B2->packed_array[j];
                if (unpack_value(B2_packed_int) != 1) continue;
                if (unpack_mub(B2_packed_int) != mub) continue;
                x2 = unpack_pos_x(B2_packed_int);
                y2 = unpack_pos_y(B2_packed_int);
                transform(&x2, &y2, t);
                if ((x1 == x2 + x_off) && (y1 == y2 + y_off)) {
                    valid = true;
                    break;
                }
            }
            if (!valid) break;
        }
        if (valid) return true;
    }

    // Tried all transformations - must not be equivalent
    return false;
}

// overload to handle multiboards
bool equivalent(Board* B1, Board* B2) {

    if (!equivalent(B1, B2, 0)) {
        return false;
    }
    if (is_mub(B1)) {
        if (!equivalent(B1, B2, 1)) {
            return false;
        }
    }
    return true;
}

// Give an array of boards, remove all boards which are duplicates
bool remove_duplicates(Board** boards, int* num_b, bool realloc_arr) {
    
    bool any = false;
    // Loop through all pairs of boards
    for (int b1 = 0; b1 < *num_b-1; b1++) {
        for (int b2 = b1+1; b2 < *num_b; b2++) {

            // If the two boards are equivalent, remove the second board
            if (equivalent(*boards + b1, *boards + b2)) {
                any = true;
                // Loop through all following boards and shift them left
                for (int b = b2; b < *num_b-1; b++) {
                    CopyHostBoard(*boards + b, *boards + b + 1);
                }

                // Lower the amount of boards
                (*num_b)--;
                b2--;
            }
        }
    }
    if (realloc_arr) {
        *boards = (Board *) realloc(*boards, (*num_b) * sizeof(Board));
    }
    return any;
}

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
        insert_element(B, 2, C_OFF, C_OFF);
    }
}

// given an array of boards generate all possible boards which can be made by adding an element
void gen_all_next_boards(Board** boards, int* num_b, Board** multiboards, int* num_mub, bool* any_dedup) {
    
    int default_nb = 16;
    int cur_idx = 0;
    Board* B;
    int dedup_idx, dedup_num;
    bool new_any_dedup = false;
    Board** dedup_ptr = (Board **) malloc(sizeof(*dedup_ptr));
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
            *dedup_ptr = new_bds_bb + dedup_idx;
            new_any_dedup |= remove_duplicates(dedup_ptr, &dedup_num, false);
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
            *dedup_ptr = new_bds_bm + dedup_idx;
            new_any_dedup |= remove_duplicates(dedup_ptr, &dedup_num, false);
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
            *dedup_ptr = new_bds_mb + dedup_idx;
            new_any_dedup |= remove_duplicates(dedup_ptr, &dedup_num, false);
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
            *dedup_ptr = new_bds_mm + dedup_idx;
            new_any_dedup |= remove_duplicates(dedup_ptr, &dedup_num, false);
            cur_idx = dedup_idx + dedup_num;
        }
    }
    new_num_mm = cur_idx;

    // transfer generated boards to the output
    *any_dedup = new_any_dedup;
    *num_b = new_num_bb + new_num_mb;
    *boards = (Board *) realloc(*boards, (*num_b) * sizeof(Board));
    *num_mub = new_num_bm + new_num_mm;
    *multiboards = (Board *) realloc(*multiboards, (*num_mub) * sizeof(Board));
    memcpy(*boards, new_bds_bb, new_num_bb * sizeof(Board));
    memcpy((*boards)+new_num_bb, new_bds_mb, new_num_mb * sizeof(Board));
    memcpy(*multiboards, new_bds_bm, new_num_bm * sizeof(Board));
    memcpy((*multiboards)+new_num_bm, new_bds_mm, new_num_mm * sizeof(Board));
    // clean up
    free(dedup_ptr);
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