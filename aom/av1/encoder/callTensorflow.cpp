/*
 * Copyright (c) 2019, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */

#include "stdio.h"
#include <Python.h>
#include <sstream>

#include <limits.h>
#include <math.h>
#include <stdio.h>

#include "config/aom_config.h"
#include "config/aom_dsp_rtcd.h"
#include "config/aom_scale_rtcd.h"
#include "config/av1_rtcd.h"

#include "aom_dsp/aom_dsp_common.h"
#include "aom_dsp/aom_filter.h"
#if CONFIG_DENOISE
#include "aom_dsp/grain_table.h"
#include "aom_dsp/noise_util.h"
#include "aom_dsp/noise_model.h"
#endif
#include "aom_dsp/psnr.h"
#if CONFIG_INTERNAL_STATS
#include "aom_dsp/ssim.h"
#endif
#include "aom_ports/aom_timer.h"
#include "aom_ports/mem.h"
#include "aom_ports/system_state.h"
#include "aom_scale/aom_scale.h"
#if CONFIG_BITSTREAM_DEBUG || CONFIG_MISMATCH_DEBUG
#include "aom_util/debug_util.h"
#endif  // CONFIG_BITSTREAM_DEBUG || CONFIG_MISMATCH_DEBUG

#include "av1/common/alloccommon.h"
#include "av1/common/cdef.h"
#include "av1/common/filter.h"
#include "av1/common/idct.h"
#include "av1/common/reconinter.h"
#include "av1/common/reconintra.h"
#include "av1/common/resize.h"
#include "av1/common/tile_common.h"

#include "av1/encoder/av1_multi_thread.h"
#include "av1/encoder/aq_complexity.h"
#include "av1/encoder/aq_cyclicrefresh.h"
#include "av1/encoder/aq_variance.h"
#include "av1/encoder/bitstream.h"
#include "av1/encoder/context_tree.h"
#include "av1/encoder/encodeframe.h"
#include "av1/encoder/encodemv.h"
#include "av1/encoder/encoder.h"
#include "av1/encoder/encodetxb.h"
#include "av1/encoder/ethread.h"
#include "av1/encoder/firstpass.h"
#include "av1/encoder/grain_test_vectors.h"
#include "av1/encoder/hash_motion.h"
#include "av1/encoder/mbgraph.h"
#include "av1/encoder/picklpf.h"
#include "av1/encoder/pickrst.h"
#include "av1/encoder/random.h"
#include "av1/encoder/ratectrl.h"
#include "av1/encoder/rd.h"
#include "av1/encoder/rdopt.h"
#include "av1/encoder/segmentation.h"
#include "av1/encoder/speed_features.h"
#include "av1/encoder/temporal_filter.h"
#include "av1/encoder/reconinter_enc.h"

using namespace std;

uint8_t** callTensorflow(uint8_t* ppp, int height, int width, int stride, FRAME_TYPE frame_type){

	Py_SetPath(L"/home/chenjs/a5/aom_190109/aom/av1/encoder:"
	  "/home/chenjs/.conda/envs/tf2/lib:"
	  "/home/chenjs/.conda/envs/tf2/lib/python3.6:"
	  "/home/chenjs/.conda/envs/tf2/lib/python3.6/site-packages:"
	  "/home/chenjs/.conda/envs/tf2/lib/python3.6/lib-dynload");

	PyObject * pModule = NULL;
	PyObject * pFuncI = NULL;
	PyObject * pFuncB = NULL;
	PyObject * pArgs = NULL;

	Py_Initialize();

	if (!Py_IsInitialized()) {
		printf("Python init failed!\n");
		return NULL;
	}
	//char *path = NULL;
	//path = getcwd(NULL, 0);
	//printf("current working directory : %s\n", path);
	//free(path);

	// import python
	pModule = PyImport_ImportModule("TEST");

	//PyEval_InitThreads();
	if (!pModule) {
		printf("don't load Pmodule\n");
		Py_Finalize();
		return NULL;
	}

	pFuncI = PyObject_GetAttrString(pModule, "entranceI");
	if (!pFuncI) {
		printf("don't get I function!");
		Py_Finalize();
		return NULL;
	}
	pFuncB = PyObject_GetAttrString(pModule, "entranceB");
	if (!pFuncB) {
		printf("don't get B function!");
		Py_Finalize();
		return NULL;
	}
	
	PyObject* list = PyList_New(height); 
	pArgs = PyTuple_New(1);                 
	PyObject** lists = new PyObject*[height];
	//stringstream ss;
	for (int i = 0; i < height; i++)
	{
		lists[i] = PyList_New(0);
		for (int j = 0; j < width; j++)
		{
			PyList_Append(lists[i], Py_BuildValue("i", *(ppp + j)));
		}
		PyList_SetItem(list, i, lists[i]);
		ppp += stride;
		//PyList_Append(list, lists[i]);
	}
	PyTuple_SetItem(pArgs, 0, list);    //"list" is the input image

	PyObject *presult = NULL;

	//printf("\nstart tensorflow!\n");
	if (frame_type == KEY_FRAME){
		presult = PyEval_CallObject(pFuncI, pArgs);
	}
	else{
		presult = PyEval_CallObject(pFuncB, pArgs);
	}

    /*
	Py_ssize_t q = PyList_Size(presult);
	printf("%d", q);
	*/

	uint8_t **rePic = new uint8_t*[height];
	for (int i = 0; i < height; i++){
		rePic[i] = new uint8_t[width];
	}
	uint8_t s;

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			PyArg_Parse(PyList_GetItem(PyList_GetItem(presult, i), j), "B", &s); 
			rePic[i][j] = s;
		}

	}
	//fclose(fp);

	//Py_Finalize();
	return rePic;
}


uint16_t** callTensorflow_hbd(uint16_t* ppp, int height, int width, int stride, FRAME_TYPE frame_type){

	Py_SetPath(L"/home/chenjs/a5/aom_190109/aom/av1/encoder:"
	  "/home/chenjs/.conda/envs/tf2/lib:"
	  "/home/chenjs/.conda/envs/tf2/lib/python3.6:"
	  "/home/chenjs/.conda/envs/tf2/lib/python3.6/site-packages:"
	  "/home/chenjs/.conda/envs/tf2/lib/python3.6/lib-dynload");

	PyObject * pModule = NULL;
	PyObject * pFuncI = NULL;
	PyObject * pFuncB = NULL;
	PyObject * pArgs = NULL;

	// Initialize the python environment
	Py_Initialize();

	if (!Py_IsInitialized()) {
		printf("Python init failed!\n");
		return NULL;
	}
	//char *path = NULL;
	//path = getcwd(NULL, 0);
	//printf("current working directory : %s\n", path);
	//free(path);

	pModule = PyImport_ImportModule("TEST");

	if (!pModule) {
		printf("don't load Pmodule\n");
		Py_Finalize();
		return NULL;
	}
	//printf("succeed acquire python !\n");
	// Get the function pointer that needs to be called
	pFuncI = PyObject_GetAttrString(pModule, "entranceI");
	if (!pFuncI) {
		printf("don't get I function!");
		Py_Finalize();
		return NULL;
	}
	//printf("succeed acquire entranceFunc !\n");
	pFuncB = PyObject_GetAttrString(pModule, "entranceB");
	if (!pFuncB) {
		printf("don't get B function!");
		Py_Finalize();
		return NULL;
	}
	
	PyObject* list = PyList_New(height); 
	pArgs = PyTuple_New(1);          //a new tuple object of size 1   
	PyObject** lists = new PyObject*[height];
	//stringstream ss;
	//Read the data from y buffer into the list
	for (int i = 0; i < height; i++)
	{
		lists[i] = PyList_New(0);
		for (int j = 0; j < width; j++)
		{
			PyList_Append(lists[i], Py_BuildValue("i", *(ppp + j)));//Convert to Python objects
		}
		PyList_SetItem(list, i, lists[i]);
		ppp += stride;
		//PyList_Append(list, lists[i]);
	}
	PyTuple_SetItem(pArgs, 0, list);   

	PyObject *presult = NULL;

	//printf("\nstart tensorflow!\n");
	if (frame_type == KEY_FRAME){
		presult = PyEval_CallObject(pFuncI, pArgs);//Call the function in the python script
	}
	else{
		presult = PyEval_CallObject(pFuncB, pArgs);
	}

	uint16_t **rePic = new uint16_t*[height];
	for (int i = 0; i < height; i++){
		rePic[i] = new uint16_t[width];
	}
	uint16_t s;

	//FILE *fp = fopen("CPython.yuv", "wb");
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			/*PyList_GetItem(PyList_GetItem(presult, i), j) get the object*/
			/*at position (i,j) in the tuple pointed to by presult*/
			PyArg_Parse(PyList_GetItem(PyList_GetItem(presult, i), j), "H", &s); 
			rePic[i][j] = s;
			//unsigned char uc = (unsigned char)s;
			//fwrite(&uc, 1, 1, fp);
		}

	}
	//fclose(fp);

	//Py_Finalize();
	return rePic;
}


uint8_t** blockCallTensorflow(uint8_t* ppp, int cur_buf_height, int cur_buf_width, int stride, FRAME_TYPE frame_type) {

	
	Py_SetPath(L"/home/chenjs/a5/aom_190109/aom/av1/encoder:"
	  "/home/chenjs/.conda/envs/tf2/lib:"
	  "/home/chenjs/.conda/envs/tf2/lib/python3.6:"
	  "/home/chenjs/.conda/envs/tf2/lib/python3.6/site-packages:"
	  "/home/chenjs/.conda/envs/tf2/lib/python3.6/lib-dynload");

	PyObject * pModule = NULL;
	PyObject * pFuncI = NULL;
	PyObject * pFuncB = NULL;
	PyObject * pArgs = NULL;

	Py_Initialize();

	if (!Py_IsInitialized()) {
		printf("Python init failed!\n");
		return NULL;
	}

	//char *path = NULL;
	//path = getcwd(NULL, 0);
	//printf("current working directory : %s\n", path);
	//free(path);

	pModule = PyImport_ImportModule("TEST_qp43_B");

	//PyEval_InitThreads();
	if (!pModule) {
		printf("don't load Pmodule\n");
		Py_Finalize();
		return NULL;
	}

	pFuncI = PyObject_GetAttrString(pModule, "entranceI");
	if (!pFuncI) {
		printf("don't get I function!");
		Py_Finalize();
		return NULL;
	}
	//printf("succeed acquire entranceFunc !\n");
	pFuncB = PyObject_GetAttrString(pModule, "entranceB");
	if (!pFuncB) {
		printf("don't get B function!");
		Py_Finalize();
		return NULL;
	}
	PyObject* list = PyList_New(cur_buf_height);
	pArgs = PyTuple_New(1);                
	PyObject** lists = new PyObject*[cur_buf_height];

	for (int i = 0; i < cur_buf_height; i++)
	{
		lists[i] = PyList_New(0);
		for (int j = 0; j < cur_buf_width; j++)
		{
			PyList_Append(lists[i], Py_BuildValue("i", *(ppp + j)));
		}
		PyList_SetItem(list, i, lists[i]);
		ppp += stride;
		//PyList_Append(list, lists[i]);
	}
	PyTuple_SetItem(pArgs, 0, list);    
	PyObject *presult = NULL;
	if (frame_type == KEY_FRAME){
		presult = PyEval_CallObject(pFuncI, pArgs);
	}
	else{
		presult = PyEval_CallObject(pFuncB, pArgs);
	}

	uint8_t **rePic = new uint8_t*[cur_buf_height];
	for (int i = 0; i < cur_buf_height; i++) {
		rePic[i] = new uint8_t[cur_buf_width];
	}
	uint8_t s;

	//FILE *fp = fopen("CPython.yuv", "wb");
	for (int i = 0; i < cur_buf_height; i++)
	{
		for (int j = 0; j < cur_buf_width; j++)
		{
			PyArg_Parse(PyList_GetItem(PyList_GetItem(presult, i), j), "B", &s);
			rePic[i][j] = s;
		}

	}
	//fclose(fp);

	//Py_Finalize();
	return rePic;
}


uint16_t** blockCallTensorflow_hbd(uint16_t* ppp, int cur_buf_height, int cur_buf_width, int stride, FRAME_TYPE frame_type) {
	
	Py_SetPath(L"/home/chenjs/a5/aom_190109/aom/av1/encoder:"
	  "/home/chenjs/.conda/envs/tf2/lib:"
	  "/home/chenjs/.conda/envs/tf2/lib/python3.6:"
	  "/home/chenjs/.conda/envs/tf2/lib/python3.6/site-packages:"
	  "/home/chenjs/.conda/envs/tf2/lib/python3.6/lib-dynload");

	PyObject * pModule = NULL;
	PyObject * pFuncI = NULL;
	PyObject * pFuncB = NULL;
	PyObject * pArgs = NULL;

	Py_Initialize();

	if (!Py_IsInitialized()) {
		printf("Python init failed!\n");
		return NULL;
	}

	//char *path = NULL;
	//path = getcwd(NULL, 0);
	//printf("current working directory : %s\n", path);
	//free(path);

	pModule = PyImport_ImportModule("TEST");

	//PyEval_InitThreads();
	if (!pModule) {
		printf("don't load Pmodule\n");
		Py_Finalize();
		return NULL;
	}

	pFuncI = PyObject_GetAttrString(pModule, "entranceI");
	if (!pFuncI) {
		printf("don't get I function!");
		Py_Finalize();
		return NULL;
	}
	pFuncB = PyObject_GetAttrString(pModule, "entranceB");
	if (!pFuncB) {
		printf("don't get B function!");
		Py_Finalize();
		return NULL;
	}
	PyObject* list = PyList_New(cur_buf_height);
	pArgs = PyTuple_New(1);                
	PyObject** lists = new PyObject*[cur_buf_height];

	for (int i = 0; i < cur_buf_height; i++)
	{
		lists[i] = PyList_New(0);
		for (int j = 0; j < cur_buf_width; j++)
		{
			PyList_Append(lists[i], Py_BuildValue("i", *(ppp + j)));
		}
		PyList_SetItem(list, i, lists[i]);
		ppp += stride;
		//PyList_Append(list, lists[i]);
	}
	PyTuple_SetItem(pArgs, 0, list);    
	PyObject *presult = NULL;
	if (frame_type == KEY_FRAME){
		presult = PyEval_CallObject(pFuncI, pArgs);
	}
	else{
		presult = PyEval_CallObject(pFuncB, pArgs);
	}

	uint16_t **rePic = new uint16_t*[cur_buf_height];
	for (int i = 0; i < cur_buf_height; i++) {
		rePic[i] = new uint16_t[cur_buf_width];
	}

	uint16_t s;
	for (int i = 0; i < cur_buf_height; i++)
	{
		for (int j = 0; j < cur_buf_width; j++)
		{
			PyArg_Parse(PyList_GetItem(PyList_GetItem(presult, i), j), "H", &s);
			rePic[i][j] = s;

		}

	}
	//Py_Finalize();
	return rePic;
}
