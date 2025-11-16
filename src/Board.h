#ifndef _BOARD_
#define _BOARD_

#define MAX_BOARD_SIZE 64
#define MAX_HEIGHT 128
#define MAX_ELEMENTS 128 // Placeholder value
#define MAX_POSITIONS 128 // Placeholder value

// Board Structure declaration
typedef struct {
    int ones_left;
    int* elements;
    // we treat elements as a rolling 2D buffer, these variables let us index easily
    int x_max; 
    int x_min;
    int y_max;
    int y_min;
    // keep track of the position of each number within the grid
    int* pos_x;
    int* pos_y;
    int last_num;
} Board;

void InitBoard(Board* B);
Board InitDeviceBoard(const Board B);
void CopyToDeviceBoard(Board* Bdevice, const Board* Bhost);
void CopyFromDeviceBoard(Board* Bhost, const Board* Bdevice);
void FreeDeviceBoard(Board* BM);
void FreeBoard(Board* B);

#endif // _BOARD_
