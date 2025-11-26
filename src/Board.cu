#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "Board.h"

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

const int x_around[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
const int y_around[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
#define AR_IDX(x,y) ((((x)+(3*(y))+5)*4)/5)


// Populate a board with initial values and allocate needed memory
void InitBoard(Board* B, int N) {

    // Define initial values (all set to 0 as placeholders)
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
    memcpy(Bdest->pos_x, Bsrc->pos_x, MAX_HEIGHT * sizeof(int));
    memcpy(Bdest->pos_y, Bsrc->pos_y, MAX_HEIGHT * sizeof(int));
    memcpy(Bdest->info, Bsrc->info, MAX_HEIGHT * sizeof(int));
}

void flatten_board_list(int* flat_boards, Board* Boards, int num) {

    // put list of boards into a flat 1D array of ints
    int elem_idx;
    for (int b=0; b<num; b++) {
        elem_idx = 0;
        for (int i=0; i<MAX_HEIGHT; i++) {
            flat_boards[elem_idx * num + b] = Boards[b].pos_x[i];
            elem_idx++;
            flat_boards[elem_idx * num + b] = Boards[b].pos_y[i];
            elem_idx++;
            flat_boards[elem_idx * num + b] = Boards[b].info[i];
            elem_idx++;
        }
        flat_boards[elem_idx * num + b] = Boards[b].ones_num;
        elem_idx++;
        flat_boards[elem_idx * num + b] = Boards[b].ones_left;
        elem_idx++;
        flat_boards[elem_idx * num + b] = Boards[b].last_num;
        elem_idx++;
    }
}

void unflatten_board_list(Board* Boards, int* flat_boards, int num) {

    // put list of boards into a flat 1D array of ints
    int elem_idx;
    for (int b=0; b<num; b++) {
        elem_idx = 0;
        for (int i=0; i<MAX_HEIGHT; i++) {
            Boards[b].pos_x[i] = flat_boards[elem_idx * num + b];
            elem_idx++;
            Boards[b].pos_y[i] = flat_boards[elem_idx * num + b];
            elem_idx++;
            Boards[b].info[i] = flat_boards[elem_idx * num + b];
            elem_idx++;
        }
        Boards[b].ones_num = flat_boards[elem_idx * num + b];
        elem_idx++;
        Boards[b].ones_left = flat_boards[elem_idx * num + b];
        elem_idx++;
        Boards[b].last_num = flat_boards[elem_idx * num + b];
        elem_idx++;
    }
}

void pretty_print(Board* B) {

    // TODO output our board in a nice grid format
    int width = floor(log10(MAX(1, abs(B->last_num+1)))) + 2;
    int min_x = 0;
    int max_x = 0;
    int min_y = 0;
    int max_y = 0;
    for (int i=0; i<B->last_num; i++) {
        min_x = MIN(min_x, B->pos_x[i]);
        max_x = MAX(max_x, B->pos_x[i]);
        min_y = MIN(min_y, B->pos_y[i]);
        max_y = MAX(max_y, B->pos_y[i]);
    }
    for (int i=MAX_HEIGHT-B->ones_num; i<MAX_HEIGHT-B->ones_left; i++) {
        min_x = MIN(min_x, B->pos_x[i]);
        max_x = MAX(max_x, B->pos_x[i]);
        min_y = MIN(min_y, B->pos_y[i]);
        max_y = MAX(max_y, B->pos_y[i]);
    }
    printf("\n");
    for (int y=min_y; y<=max_y; y++) {
        for (int x=min_x; x<=max_x; x++) {
            bool found = false;
            for (int i=0; i<B->last_num; i++) {
                if ((B->pos_x[i] == x) && (B->pos_y[i] == y)) {
                    printf("%*d", width, i+2);
                    found = true;
                    break;
                }
            }
            if (!found) {
                for (int i=MAX_HEIGHT-B->ones_num; i<MAX_HEIGHT-B->ones_left; i++) {
                    if ((B->pos_x[i] == x) && (B->pos_y[i] == y)) {
                        printf("%*d", width, 1);
                        found = true;
                        break;
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
    B->pos_x[B->last_num] = x;
    B->pos_y[B->last_num] = y;
    B->info[B->last_num] = ones_added;
    B->last_num += 1;
}

void insert_one(Board* B, int x, int y) {

    // insert a one onto our board
    B->pos_x[MAX_HEIGHT-B->ones_left] = x;
    B->pos_y[MAX_HEIGHT-B->ones_left] = y;
    B->ones_left -= 1;
}

void remove_element(Board* B) {

    // remove an element from our board
    B->ones_left += B->info[B->last_num - 1];
    B->last_num -= 1;
}

int get_sum(Board* B, int x, int y, int* least_neighbor) {  //TODO lots of wasted time here, in the end we'll want to stop when we know (sum > target)

    // get the sum of elements surrounding position
    // returns -1 if position already populated
    int sum = 0;
    *least_neighbor = MAX_HEIGHT;
    for (int i=0; i<B->last_num; i++) {
        if ((B->pos_x[i] == x) && (B->pos_y[i] == y)) {
            return -1;
        }
        if ((x-1 <= B->pos_x[i]) && (B->pos_x[i] <= x+1)) {
            if ((y-1 <= B->pos_y[i]) && (B->pos_y[i] <= y+1)) {
                sum += i+2;
                *least_neighbor = MIN(*least_neighbor, i);
            }
        }
    }
    for (int i=MAX_HEIGHT-B->ones_num; i<MAX_HEIGHT-B->ones_left; i++) {
        if ((B->pos_x[i] == x) && (B->pos_y[i] == y)) {
            return -1;
        }
        if ((x-1 <= B->pos_x[i]) && (B->pos_x[i] <= x+1)) {
            if ((y-1 <= B->pos_y[i]) && (B->pos_y[i] <= y+1)) {
                sum += 1;
                *least_neighbor = MIN(*least_neighbor, i);
            }
        }
    }
    return sum;
}

bool look_around(Board* B, int index, int start_H) {

    // check around the location of a particular index for a spot to place the next element
    int new_x, new_y;
    int min_nb;
    for (int H=start_H; H<8; H++) {  // iterate over all spots around (i+2)
        new_x = B->pos_x[index] + x_around[H];
        new_y = B->pos_y[index] + y_around[H];
        if (get_sum(B, new_x, new_y, &min_nb) == B->last_num+2) {
            if (min_nb == index) {  // don't go in spots we've already checked
                insert_element(B, new_x, new_y, 0);
                return true;
            }
        }
    }
    return false;
}

void next_board_state(Board* B) {

    // iterate a board to its next state i.e. the next position in the search
    // this function assumes that "2" has already been placed

    // first try to add the next number
    for (int i=0; i<B->last_num / 2; i++) {  // choose a num (i+2) to try to place around
        if (look_around(B, i, 0)) return;
    }
    for (int i=B->last_num - B->ones_num; i<B->last_num; i++) {  // we need to also look around high numbers
        if (look_around(B, i, 0)) return;
    }
    if (B->last_num+2 <= 8) { // we need to also look around ones for small numbers
        for (int i=MAX_HEIGHT-B->ones_num; i<MAX_HEIGHT-B->ones_left; i++) { 
            if (look_around(B, i, 0)) return;
        }
    }

    // failing to add a number, we'll attempt to move the current highest to a new position
    // continuing to remove elements until we succeed at moving one
    int old_x, old_y;
    int old_nb, last_H;
    while (true) {
        // first find where we left off
        old_x = B->pos_x[B->last_num - 1];
        old_y = B->pos_y[B->last_num - 1];
        get_sum(B, old_x, old_y, &old_nb);  // TODO being lazy here, should write a new func that stops 
        last_H = AR_IDX(old_x - B->pos_x[old_nb], old_y - B->pos_y[old_nb]);
        // remove the element and search for new spot
        remove_element(B);
        // TODO dry out this WET code
        // start with element it was already around
        look_around(B, old_nb, last_H+1);
        for (int i=old_nb+1; i<B->last_num / 2; i++) {  // choose a num (i+2) to try to place around
            if (look_around(B, i, 0)) return;
        }
        for (int i=MAX(old_nb+1, B->last_num - B->ones_num); i<B->last_num; i++) {  // we need to also look around high numbers
            if (look_around(B, i, 0)) return;
        }
        if (B->last_num+2 <= 8) { // we need to also look around ones for small numbers
            for (int i=MAX(old_nb+1, MAX_HEIGHT-B->ones_num); i<MAX_HEIGHT-B->ones_left; i++) { 
                if (look_around(B, i, 0)) return;
            }
        }
    }

    // big TODO here still: need to be able to insert ones along with elems lol

}