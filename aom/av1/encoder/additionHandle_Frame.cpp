#include "additionHandle_Frame.h"
#include <iostream>

using namespace std;

extern uint8_t**  callTensorflow(uint8_t* ppp, int height, int width, int stride, FRAME_TYPE frame_type);
extern uint8_t**  blockCallTensorflow(uint8_t* ppp, int cur_buf_height, int cur_buf_width, int stride, FRAME_TYPE frame_type);
extern uint16_t** callTensorflow_hbd(uint16_t* ppp, int height, int width, int stride, FRAME_TYPE frame_type);
extern uint16_t** blockCallTensorflow_hbd(uint16_t* ppp, int cur_buf_height, int cur_buf_width, int stride, FRAME_TYPE frame_type);

/*Feed the whole frame image into the network*/
void additionHandle_frame(AV1_COMP *cpi, AV1_COMMON *cm, FRAME_TYPE frame_type) {
	YV12_BUFFER_CONFIG *pcPicYuvRec = &cm->cur_frame->buf;

	if(!cm->seq_params.use_highbitdepth) {
		uint8_t* py = pcPicYuvRec->y_buffer;
		uint8_t* bkuPy = py;

		int height = pcPicYuvRec->y_height;
		int width = pcPicYuvRec->y_width;
		int stride = pcPicYuvRec->y_stride;


		uint8_t **buf = new uint8_t*[height];
		for (int i = 0; i < height; i++) {
			buf[i] = new uint8_t[width];
		}

		buf = callTensorflow(py, height, width, stride, frame_type);

		//FILE *ff = fopen("end.yuv", "wb");
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				*(bkuPy + j) = buf[i][j]; //Fill in the luma buffer again
				//fwrite(bkuPy + j, 1, 1, ff);
			}
			bkuPy += stride;
		}
		//fclose(ff);
	}
	else {
		uint16_t* py = CONVERT_TO_SHORTPTR(pcPicYuvRec->y_buffer);
		uint16_t* bkuPy = py;

		int height = pcPicYuvRec->y_height;
		int width = pcPicYuvRec->y_width;
		int stride = pcPicYuvRec->y_stride;

		uint16_t **buf = new uint16_t*[height];
		for (int i = 0; i < height; i++) {
			buf[i] = new uint16_t[width];
		}

		buf = callTensorflow_hbd(py, height, width, stride, frame_type);

		//FILE *ff = fopen("end.yuv", "wb");
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				*(bkuPy + j) = buf[i][j]; //Fill in the luma buffer again
				//fwrite(bkuPy + j, 1, 1, ff);
			}
			bkuPy += stride;
		}
	}

}

/*Split into 700x700 blocks into the network*/
void additionHandle_blocks(AV1_COMP *cpi, AV1_COMMON *cm, FRAME_TYPE frame_type) {
	YV12_BUFFER_CONFIG *pcPicYuvRec = &cm->cur_frame->buf;
	
	int height = pcPicYuvRec->y_height;
	int width = pcPicYuvRec->y_width;
	int buf_width = 700;
	int buf_height = 700;
	int bufx_count = width / buf_width;
	int bufy_count = height / buf_height;
	int stride = pcPicYuvRec->y_stride;
	int cur_buf_width;
	int cur_buf_height;

	if (!cm->seq_params.use_highbitdepth) {
		uint8_t* py = pcPicYuvRec->y_buffer;
		uint8_t* bkuPy = py;
		uint8_t* bkuPyTemp = bkuPy;


		for (int y = 0; y < bufy_count+1; y++) {
			for (int x = 0; x < bufx_count+1; x++) {

				if (x == bufx_count)
					cur_buf_width = width - buf_width * bufx_count;
				else
					cur_buf_width = buf_width;
				if (y == bufy_count)
					cur_buf_height = height - buf_height* bufy_count;
				else
					cur_buf_height = buf_height;


				if (cur_buf_width != 0 && cur_buf_height != 0) {

					uint8_t **buf = new uint8_t*[cur_buf_height];
					for (int i = 0; i < cur_buf_height; i++) {
						buf[i] = new uint8_t[cur_buf_width];
					}

					buf = blockCallTensorflow(py + x*buf_width + stride*buf_height*y, cur_buf_height, cur_buf_width, stride, frame_type);

					bkuPy = bkuPyTemp;
					//FILE *ff = fopen("end.yuv", "wb");
					for (int i = 0; i < cur_buf_height; i++) {
						for (int j = 0; j < cur_buf_width; j++) {
							*(bkuPy + j + x * buf_width + y * buf_height * stride) = buf[i][j]; //Fill in the luma buffer again
							//fwrite(bkuPy + j, 1, 1, ff);
						}
						bkuPy += stride;
					}
					//fclose(ff);

					for (int i = 0; i < cur_buf_height; i++) {
						delete[] buf[i];
					}
					delete buf;
				}
			}
		}
	}
	else {
		uint16_t* py = CONVERT_TO_SHORTPTR(pcPicYuvRec->y_buffer);
		uint16_t* bkuPy = py;
		uint16_t* bkuPyTemp = bkuPy;

		for (int y = 0; y < bufy_count+1; y++) {
			for (int x = 0; x < bufx_count+1; x++) {
				if (x == bufx_count)
					cur_buf_width = width - buf_width * bufx_count;
				else
					cur_buf_width = buf_width;
				if (y == bufy_count)
					cur_buf_height = height - buf_height* bufy_count;
				else
					cur_buf_height = buf_height;


				if (cur_buf_width != 0 && cur_buf_height != 0) {

					uint16_t **buf = new uint16_t*[cur_buf_height];
					for (int i = 0; i < cur_buf_height; i++) {
						buf[i] = new uint16_t[cur_buf_width];
					}

					buf = blockCallTensorflow_hbd(py + x*buf_width + stride*buf_height*y, cur_buf_height, cur_buf_width, stride, frame_type);

					bkuPy = bkuPyTemp;
					//FILE *ff = fopen("end.yuv", "wb");
					for (int i = 0; i < cur_buf_height; i++) {
						for (int j = 0; j < cur_buf_width; j++) {
							*(bkuPy + j + x * buf_width + y * buf_height * stride) = buf[i][j]; 
							//fwrite(bkuPy + j, 1, 1, ff);
						}
						bkuPy += stride;
					}
					//fclose(ff);

					for (int i = 0; i < cur_buf_height; i++) {
						delete[] buf[i];
					}
					delete buf;
				}
			}
	    }
	}

}


/*frame_type determines what kind of network blocks need to be fed into, not exactly equivalent to what frame the current frame is.*/
/*Low bitdepth*/
uint8_t **blocks_to_cnn_secondly(uint8_t *pBuffer_y, int height, int width, int stride, FRAME_TYPE frame_type) {

	uint8_t **dst = new uint8_t *[height];
	for (int i = 0; i < height; i++) {
		dst[i] = new uint8_t[width];
	}

	if (frame_type == FRAME_TYPES) { 
		for (int r = 0; r < height; ++r) {
			for (int c = 0; c < width; ++c) {
				dst[r][c] = (uint8_t)(*(pBuffer_y + c));
			}
			pBuffer_y += stride;
		}
		return dst;
	}
	else if (frame_type != FRAME_TYPES) {
		int buf_width = 1000;
		int buf_height = 1000;
		int bufx_count = width / buf_width;
		int bufy_count = height / buf_height; /**/
		int cur_buf_width;
		int cur_buf_height;

		
		for (int y = 0; y < bufy_count + 1; y++) {
			for (int x = 0; x < bufx_count + 1; x++) {
				if (x == bufx_count)
					cur_buf_width = width - buf_width * bufx_count;
				else
					cur_buf_width = buf_width;
				if (y == bufy_count)
					cur_buf_height = height - buf_height* bufy_count;
				else
					cur_buf_height = buf_height;


				if (cur_buf_width != 0 && cur_buf_height != 0) {

					uint8_t **buf = new uint8_t*[cur_buf_height];
					for (int i = 0; i < cur_buf_height; i++) {
						buf[i] = new uint8_t[cur_buf_width];
					}
					buf = blockCallTensorflow(pBuffer_y + x*buf_width + stride*buf_height*y, cur_buf_height, cur_buf_width, stride, frame_type);

					for (int i = 0; i < cur_buf_height; i++) {
						for (int j = 0; j < cur_buf_width; j++) {
							dst[y * buf_height + i][x * buf_width + j] = buf[i][j]; 
						}
					}

					for (int i = 0; i < cur_buf_height; i++) {
						delete[] buf[i];
					}
					delete buf;
				}
			}
		}

	}
	/*
	FILE *ff = fopen("end.yuv", "wb");
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			fwrite(*(dst+i)+j, 1, 1, ff);
		}
	}
	fclose(ff);
	*/
	return dst;

}
