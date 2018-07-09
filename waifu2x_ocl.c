//---------------------------------------------------------
//	Cat's eye
//
//		©2016-2018 Yuichiro Nakada
//---------------------------------------------------------

// clang -Os waifu2x_ocl.c -o waifu2x_ocl `pkg-config --libs --cflags OpenCL`
#define _CRT_SECURE_NO_WARNINGS 1
#include <stdlib.h>
#include <stdint.h>

//#define _DEBUG
#ifdef _DEBUG
#define debug_s(x)	{x;}
#else
#define debug_s(x)
#endif

#include "ocl.h"
#include "clock.h"

#define PARG_IMPLEMENTATION
#include "parg.h"
#define PARSON_IMPLEMENTATION
#include "parson.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_STATIC
#include "stb_image_resize.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

typedef struct {
	int type;	// MLP, CONV, MAXPOOL
	int act;	// activation function type
	int in;	// input channel
	int out;	// output channel
	int size;	// input size (ch * x * y)
	int width;	// input width
	int height;	// input height
	int ksize;	// kernel size
	int stride;
} CatsEye_Layer;

#define real		float
typedef struct {
	// number of each layer
	int layers;
	CatsEye_Layer *u;

	// input layers
	real *xdata;
	int xsize;
	// output layers [o = f(z)]
	real **z, **o, *odata;
	int osize;
	// error value
	real **d, *ddata;
	int dsize;
	// weights
	real **w, *wdata;
	int *ws, wsize;
	// bias
	real **b, *bdata;
	int *bs, bsize;
} CatsEye;

int CatsEye_loadJson(CatsEye *this, char *name)
{
	JSON_Value *root_value = json_parse_file(name);
	if (json_value_get_type(root_value) != JSONArray) return 1;
	JSON_Array *a = json_value_get_array(root_value);

	this->layers = json_array_get_count(a);
	this->u = malloc(sizeof(CatsEye_Layer)*this->layers);
	this->b = malloc(sizeof(real*)*this->layers);
	this->bs = malloc(sizeof(int)*this->layers);
	this->w = malloc(sizeof(real*)*this->layers);
	this->ws = malloc(sizeof(int)*this->layers);

	this->bsize = 0;
	for (int i=0; i<this->layers; i++) {
		JSON_Object *o = json_array_get_object(a, i);
		this->bs[i] = this->bsize;
		this->bsize += json_object_get_number(o, "nOutputPlane");
	}
	this->bdata = malloc(sizeof(real)*this->bsize);
	for (int i=0; i<this->layers; i++) {
		JSON_Object *o = json_array_get_object(a, i);
		JSON_Array *aa = json_object_get_array(o, "bias");
		for (int j=0; j<json_array_get_count(aa); j++) {
			this->bdata[this->bs[i]+j] = json_array_get_number(aa, j);
		}
	}

	this->wsize = 0;
	for (int i=0; i<this->layers; i++) {
		JSON_Object *o = json_array_get_object(a, i);
		this->ws[i] = this->wsize;
		this->wsize += json_object_get_number(o, "nInputPlane")*json_object_get_number(o, "nOutputPlane")
			*json_object_get_number(o, "kW")*json_object_get_number(o, "kH");
	}
	this->wdata = malloc(sizeof(real)*this->wsize);
	for (int i=0; i<this->layers; i++) {
		JSON_Object *o = json_array_get_object(a, i);
		JSON_Array *aa = json_object_get_array(o, "weight");
		int kW = json_object_get_number(o, "kW");
		int kH = json_object_get_number(o, "kH");
		int in = json_object_get_number(o, "nInputPlane");
		int out = json_object_get_number(o, "nOutputPlane");
		this->u[i].ksize = kW;
		this->u[i].in = in;
		this->u[i].out = out;

		for (int j=0; j<out; j++) {
			for (int k=0; k<in; k++) {
				JSON_Array *aaa = json_array_get_array(json_array_get_array(aa, j), k);

				for (int m=0; m<kH; m++) {
					JSON_Array *aaaa = json_array_get_array(aaa, m);
					for (int n=0; n<kW; n++) {
						this->wdata[this->ws[i] +(j*in+k)*kW*kH +m*kW +n] = json_array_get_number(aaaa, n);
					}
				}
			}
		}
	}
	printf("wsize:%d\n", this->wsize);

	json_value_free(root_value);
	return 0;
}

#define XSIZE		256
#define YSIZE		256
#define DATA_XSIZE	4096
#define DATA_YSIZE	2048
#define KERNEL_W	256
#define KERNEL_H	281	// 287136/4/256

char convolution[] = OCLSTRINGIFY(

kernel void convolution(global float4 *X/*256*256*/, int swap, /*constant*/global const float4 *W/*3*3*/, int wpos, constant float4 *bias, int bpos, int INPUTPLANE/*/4*/)
{
	int gid = get_global_id(0) +get_global_id(1)*XSIZE; // 0 - (256*256-1)
	int op = get_global_id(2); // output plane
//	if (gid>65535) return;
//	if (op>32) return;

	global float4 *Z;
	if (swap) {
		Z = X +gid +XSIZE*YSIZE*op;
		X += DATA_XSIZE*DATA_YSIZE;
	} else {
		Z = X +DATA_XSIZE*DATA_YSIZE +gid +XSIZE*YSIZE*op;
	}
	//Z[0] = gid/(256.0*256.0);

	global const/*constant*/ float4 *w = W +wpos +INPUTPLANE*3*3*4*op;
	if (INPUTPLANE==1) w = W +wpos +3*3*op;
	global const/*constant*/ float4 *w2 = w +INPUTPLANE*3*3;
	global const/*constant*/ float4 *w3 = w +INPUTPLANE*3*3*2;
	global const/*constant*/ float4 *w4 = w +INPUTPLANE*3*3*3;
	bias += bpos +op;

	float4 p[9];
	float4 a[9];
	float4 z = *bias;
	for (int i=0; i<INPUTPLANE; i++) {
		p[0] = X[clamp(gid + -1 + -1*256, 0, 256*256)];
		p[1] = X[clamp(gid +  0 + -1*256, 0, 256*256)];
		p[2] = X[clamp(gid +  1 + -1*256, 0, 256*256)];
		p[3] = X[clamp(gid + -1 +  0*256, 0, 256*256)];
		p[4] = X[/*clamp(gid +  0 +  0*256, 0, 256*256)*/gid];
		p[5] = X[clamp(gid +  1 +  0*256, 0, 256*256)];
		p[6] = X[clamp(gid + -1 +  1*256, 0, 256*256)];
		p[7] = X[clamp(gid +  0 +  1*256, 0, 256*256)];
		p[8] = X[clamp(gid +  1 +  1*256, 0, 256*256)];
		X += 256*256;

		a[0] = *w++;
		a[1] = *w++;
		a[2] = *w++;
		a[3] = *w++;
		a[4] = *w++;
		a[5] = *w++;
		a[6] = *w++;
		a[7] = *w++;
		a[8] = *w++;

		z.x += dot((float3)(p[0].x, p[1].x, p[2].x), a[0].xyz);
		z.x += dot((float3)(p[3].x, p[4].x, p[5].x), (float3)(a[0].w, a[1].x, a[1].y));
		z.x += dot((float3)(p[6].x, p[7].x, p[8].x), (float3)(a[1].z, a[1].w, a[2].x));

		if (INPUTPLANE!=1) {
			z.x += dot((float3)(p[0].y, p[1].y, p[2].y), a[2].yzw);
			z.x += dot((float3)(p[3].y, p[4].y, p[5].y), a[3].xyz);
			z.x += dot((float3)(p[6].y, p[7].y, p[8].y), (float3)(a[3].w, a[4].x, a[4].y));

			z.x += dot((float3)(p[0].z, p[1].z, p[2].z), (float3)(a[4].z, a[4].w, a[5].x));
			z.x += dot((float3)(p[3].z, p[4].z, p[5].z), a[5].yzw);
			z.x += dot((float3)(p[6].z, p[7].z, p[8].z), a[6].xyz);

			z.x += dot((float3)(p[0].w, p[1].w, p[2].w), (float3)(a[6].w, a[7].x, a[7].y));
			z.x += dot((float3)(p[3].w, p[4].w, p[5].w), (float3)(a[7].z, a[7].w, a[8].x));
			z.x += dot((float3)(p[6].w, p[7].w, p[8].w), a[8].yzw);

			a[0] = *w2++;
			a[1] = *w2++;
			a[2] = *w2++;
			a[3] = *w2++;
			a[4] = *w2++;
			a[5] = *w2++;
			a[6] = *w2++;
			a[7] = *w2++;
			a[8] = *w2++;

			z.y += dot((float3)(p[0].x, p[1].x, p[2].x), a[0].xyz);
			z.y += dot((float3)(p[3].x, p[4].x, p[5].x), (float3)(a[0].w, a[1].x, a[1].y));
			z.y += dot((float3)(p[6].x, p[7].x, p[8].x), (float3)(a[1].z, a[1].w, a[2].x));
		}

		z.y += dot((float3)(p[0].y, p[1].y, p[2].y), a[2].yzw);
		z.y += dot((float3)(p[3].y, p[4].y, p[5].y), a[3].xyz);
		z.y += dot((float3)(p[6].y, p[7].y, p[8].y), (float3)(a[3].w, a[4].x, a[4].y));

		if (INPUTPLANE!=1) {
			z.y += dot((float3)(p[0].z, p[1].z, p[2].z), (float3)(a[4].z, a[4].w, a[5].x));
			z.y += dot((float3)(p[3].z, p[4].z, p[5].z), a[5].yzw);
			z.y += dot((float3)(p[6].z, p[7].z, p[8].z), a[6].xyz);

			z.y += dot((float3)(p[0].w, p[1].w, p[2].w), (float3)(a[6].w, a[7].x, a[7].y));
			z.y += dot((float3)(p[3].w, p[4].w, p[5].w), (float3)(a[7].z, a[7].w, a[8].x));
			z.y += dot((float3)(p[6].w, p[7].w, p[8].w), a[8].yzw);

			a[0] = *w3++;
			a[1] = *w3++;
			a[2] = *w3++;
			a[3] = *w3++;
			a[4] = *w3++;
			a[5] = *w3++;
			a[6] = *w3++;
			a[7] = *w3++;
			a[8] = *w3++;

			z.z += dot((float3)(p[0].x, p[1].x, p[2].x), a[0].xyz);
			z.z += dot((float3)(p[3].x, p[4].x, p[5].x), (float3)(a[0].w, a[1].x, a[1].y));
			z.z += dot((float3)(p[6].x, p[7].x, p[8].x), (float3)(a[1].z, a[1].w, a[2].x));

			z.z += dot((float3)(p[0].y, p[1].y, p[2].y), a[2].yzw);
			z.z += dot((float3)(p[3].y, p[4].y, p[5].y), a[3].xyz);
			z.z += dot((float3)(p[6].y, p[7].y, p[8].y), (float3)(a[3].w, a[4].x, a[4].y));
		}

		z.z += dot((float3)(p[0].z, p[1].z, p[2].z), (float3)(a[4].z, a[4].w, a[5].x));
		z.z += dot((float3)(p[3].z, p[4].z, p[5].z), a[5].yzw);
		z.z += dot((float3)(p[6].z, p[7].z, p[8].z), a[6].xyz);

		if (INPUTPLANE!=1) {
			z.z += dot((float3)(p[0].w, p[1].w, p[2].w), (float3)(a[6].w, a[7].x, a[7].y));
			z.z += dot((float3)(p[3].w, p[4].w, p[5].w), (float3)(a[7].z, a[7].w, a[8].x));
			z.z += dot((float3)(p[6].w, p[7].w, p[8].w), a[8].yzw);

			a[0] = *w4++;
			a[1] = *w4++;
			a[2] = *w4++;
			a[3] = *w4++;
			a[4] = *w4++;
			a[5] = *w4++;
			a[6] = *w4++;
			a[7] = *w4++;
			a[8] = *w4++;

			z.w += dot((float3)(p[0].x, p[1].x, p[2].x), a[0].xyz);
			z.w += dot((float3)(p[3].x, p[4].x, p[5].x), (float3)(a[0].w, a[1].x, a[1].y));
			z.w += dot((float3)(p[6].x, p[7].x, p[8].x), (float3)(a[1].z, a[1].w, a[2].x));

			z.w += dot((float3)(p[0].y, p[1].y, p[2].y), a[2].yzw);
			z.w += dot((float3)(p[3].y, p[4].y, p[5].y), a[3].xyz);
			z.w += dot((float3)(p[6].y, p[7].y, p[8].y), (float3)(a[3].w, a[4].x, a[4].y));

			z.w += dot((float3)(p[0].z, p[1].z, p[2].z), (float3)(a[4].z, a[4].w, a[5].x));
			z.w += dot((float3)(p[3].z, p[4].z, p[5].z), a[5].yzw);
			z.w += dot((float3)(p[6].z, p[7].z, p[8].z), a[6].xyz);
		}

		z.w += dot((float3)(p[0].w, p[1].w, p[2].w), (float3)(a[6].w, a[7].x, a[7].y));
		z.w += dot((float3)(p[3].w, p[4].w, p[5].w), (float3)(a[7].z, a[7].w, a[8].x));
		z.w += dot((float3)(p[6].w, p[7].w, p[8].w), a[8].yzw);
	}

	// Leaky ReLU
//	z += *bias;
	z = (float4)max(z, (float4)0.0) + (float4)min(z, (float4)0.0) * (float4)0.1;
	*Z = z;
}

);
float X[4*DATA_XSIZE*DATA_YSIZE*2];
int swap, wpos, bpos, INPUTPLANE;
args_t args[] = {
	{ CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR, sizeof(float)*4*DATA_XSIZE*DATA_YSIZE*2, 0, X, OCL_WRITE|OCL_READ }, // X
	{ 0, sizeof(int), 0, &swap, 0 },
	{ CL_MEM_READ_ONLY|CL_MEM_ALLOC_HOST_PTR, /*sizeof(float)*4*3*3*/0, 0, 0, OCL_WRITE }, // W
	{ 0, sizeof(int), 0, &wpos, 0 },
	{ CL_MEM_READ_ONLY|CL_MEM_ALLOC_HOST_PTR, /*sizeof(float)*4*3*3*/0, 0, 0, OCL_WRITE }, // bias
	{ 0, sizeof(int), 0, &bpos, 0 },
	{ 0, sizeof(int), 0, &INPUTPLANE, 0 },
	{ 0, 0, 0, 0, 0 },
};
ocl_t kernel[] = {
//	{ "convolution", 0, 3/*dim*/,{256,256,/*output plane*/1,},{1,1,1,}, args },
	{ "convolution", 0, 3/*dim*/,{256,256,/*output plane*/1,},{128,1,1,}, args },
};
int ksz = sizeof(kernel)/sizeof(kernel[0]);

void *recalloc(void *p, int s, int ss)
{
	void *r = calloc(1, ss);
//	printf("%x %x %d %d\n", r, p, s, ss);
	if (!r) return 0;
	memcpy(r, p, s);
	free(p);
	return r;
}

void result(char *name, int w, int h)
{
	oclKernelArgsRead(args);
	float *d = !swap ? X : X +DATA_XSIZE*DATA_YSIZE*4;
#ifdef _DEBUG
	for (int i=0; i<8/*h*/; i++) {
//		for (int j=0; j<8/*w*/; j++) printf("%2.3f ", d[(i*w+j)*4]);
		for (int j=0; j<8/*w*/; j++) printf("%2.3f ", d[(i*/*w*/256+j)*4]);
		printf("\n");
	}
	printf("\n");
#endif

	uint8_t *o = calloc(w*h, 1);
	for (int y=0; y<h; y++) {
		for (int x=0; x<w; x++) {
//			o[y*w+x] = d[(y*w+x)*4]*256;
			//o[y*w+x] = d[(y*w+x)*4+1]*256;
			o[y*w+x] = d[((y/256*w/256+x/256)*256*256 +(y%256*256+x%256))*4]*256;
		}
	}
	stbi_write_png(name, w, h, 1, o, 0);
	free(o);
}

void waifu2x_ocl_run(CatsEye *cat, float *yuv, uint8_t *s, int sx, int sy, uint8_t *p, int wx)
{
	float *u = yuv + 256*256*4;
	float *v = yuv + 256*256*5;
	int width = XSIZE;
	int height = YSIZE;
	if (sx<XSIZE) width = sx;	// small size <256
	if (sy<YSIZE) height = sy;
	for (int y=0; y<height; y++) {
		for (int x=0; x<width; x++) {
			uint8_t r = s[(y*sx+x)*3];
			uint8_t g = s[(y*sx+x)*3+1];
			uint8_t b = s[(y*sx+x)*3+2];

			yuv[(y*256+x)*4] = (0.298912*r +0.586611*g +0.114478*b)/256.0;	// CCIR Rec.601
			u[y*256+x] = -0.1687*r -0.3313*g +0.500 *b;
			v[y*256+x] =  0.500 *r -0.4187*g -0.0813*b;
//			yuv[(y*256+x)*4] = 0.299*r +0.587*g +0.114*b;	// CCIR Rec.601
//			u[y*256+x] = -0.147*r -0.289*g +0.436*b;
//			v[y*256+x] = 0.615*r -0.515*g -0.100*b;

			X[(y*256+x)*4  ] = yuv[(y*256+x)*4];
			X[(y*256+x)*4+1] = yuv[(y*256+x)*4];
			X[(y*256+x)*4+2] = yuv[(y*256+x)*4];
			X[(y*256+x)*4+3] = yuv[(y*256+x)*4];
		}
	}
//	debug_s(stbi_write_png("output_256.png", 256, 256, 3, p, 0));
//	debug_s(stbi_write_png("output_y.png", 256, 256, 1, yuv, 0));

	debug_s(clock_start());
	oclKernelArgsWrite(args);
	swap = 0;
	for (int i=0; i<cat->layers; i++) {
		int a = (cat->u[i].out+3)/4;
		int w = a>16 ? 16 : a;
		int h = (a+15)/16;
		debug_s(printf("%d %d %dx%d %d %d %2.4f %2.4f\n", cat->u[i].in, cat->u[i].out, w, h, (cat->u[i].in+3)/4, cat->ws[i], cat->wdata[cat->ws[i]], cat->bdata[cat->bs[i]]));

		INPUTPLANE = (cat->u[i].in+3)/4;
		bpos = cat->bs[i]/4;
		wpos = cat->ws[i]/4;
		kernel[0].global_size[2] = a;

		oclRun(&kernel[0]);
		swap ^= 1;
#ifdef _DEBUG
		char buff[256];
		sprintf(buff, "output2x_%02d.png", i+1);
		result(buff, XSIZE*w, YSIZE*h);
#endif
	}
	oclKernelArgsRead(args);
	debug_s(clock_end());

	float *d = !swap ? X : X +DATA_XSIZE*DATA_YSIZE*4;
	for (int y=8; y<YSIZE-8; y++) {
		for (int x=8; x<XSIZE-8; x++) {
	//for (int y=0; y<YSIZE; y++) {
		//for (int x=0; x<XSIZE; x++) {
//			float yy = yuv[(y*256+x)*4];
			float yy = d[(y*256+x)*4]*256.0;
			int r = yy                     +1.402  *v[y*256+x];
			int g = yy -0.34414*u[y*256+x] -0.71414*v[y*256+x];
			int b = yy +1.772  *u[y*256+x];
			uint8_t *pix = &p[(y*wx+x)*3];
			if (!pix[0] || !pix[1] || !pix[2]) {
				pix[0] = r>255 ? 255 : r<0 ? 0 : r;
				pix[1] = g>255 ? 255 : g<0 ? 0 : g;
				pix[2] = b>255 ? 255 : b<0 ? 0 : b;
			}

//			p[(y*XSIZE+x)*3]   = 256*(yy                   +1.140*v[y*256+x]);
//			p[(y*XSIZE+x)*3+1] = 256*(yy -0.395*u[y*256+x] -0.580*v[y*256+x]);
//			p[(y*XSIZE+x)*3+2] = 256*(yy +2.032*u[y*256+x]);
		}
	}
}

int waifu2x_ocl(char *name, char *output, char *model, float scale)
{
	uint8_t *pixels;
	int w, h, bpp;
	pixels = stbi_load(name, &w, &h, &bpp, 3);
	assert(pixels);
	printf("%s %dx%d %d\n", name, w, h, bpp);
	bpp = 3;

	// resize
	int sx = w * scale;
	int sy = h * scale;
	uint8_t *pix = malloc(sx*sy*bpp);
	stbir_resize_uint8_srgb(pixels, w, h, 0, pix, sx, sy, 0, bpp, -1, 0);
	stbi_image_free(pixels);
	debug_s(stbi_write_jpg("output.jpg", sx, sy, bpp, pix, 0));

	// expand edge by +16
	sx += 16;
	sy += 16;
	pixels = calloc(sx*sy*bpp, 1);
	for (int y=8; y<sy-8; y++) {
		memcpy(pixels +(8+(y*sx))*bpp, pix +((y-8)*(sx-16))*bpp, (sx-16)*bpp);
	}
	free(pix);
	pix = pixels;

	CatsEye cat;
	int r = CatsEye_loadJson(&cat, model);
	assert(!r);
	cat.wdata = recalloc(cat.wdata, sizeof(real)*cat.wsize, sizeof(real)*KERNEL_W*KERNEL_H*4); // 256*281
	cat.bdata = recalloc(cat.bdata, sizeof(real)*cat.bsize, sizeof(real)*(cat.bsize+3));
	assert(cat.wdata);
	assert(cat.bdata);

	args[2].size = sizeof(real)*KERNEL_W*KERNEL_H*4;
	args[2].s = cat.wdata;
	args[4].size = sizeof(real)*(cat.bsize+3);
	args[4].s = cat.bdata;

	oclSetup(0, 0);
	oclKernel(kernel, ksz, "-cl-denorms-are-zero -cl-finite-math-only -cl-fast-relaxed-math -Werror", convolution);
	oclKernelArgs(kernel, ksz);

//#ifndef _DEBUG
//	args[0].size = sizeof(float)*4*256*256; // for speed up
//#endif

	float *yuv = calloc(256*256*(4+2), sizeof(float));
//	uint8_t *o = calloc(XSIZE*YSIZE, 3);
//	waifu2x_ocl_run(&cat, yuv, pix, sx, sy, o, 256);
//	stbi_write_png("output2x.png", XSIZE, YSIZE, 3, o, 0);
	printf("%d %d -> %d %d *%f\n", w, h, sx, sy, scale);
	uint8_t *o = calloc(sx*sy, 3);
	for (int y=0; y<sy-1; y+=256-16) {
		for (int x=0; x<sx-1; x+=256-16) {
			int ox = x+256 > sx ? sx-(256+1) : x;
			int oy = y+256 > sy ? sy-(256+1) : y;
			printf("%d %d\n", ox, oy);
			waifu2x_ocl_run(&cat, yuv, pix+(ox+oy*sx)*3, sx, sy, o+(ox+oy*sx)*3, sx);
		}
	}
//	stbi_write_png(output, sx, sy, 3, o, 0);
	free(yuv);
	free(pix);

	// shrink edge by -16
	sx -= 16;
	sy -= 16;
	pix = calloc(sx*sy*bpp, 1);
	for (int y=0; y<sy; y++) {
		memcpy(pix +(y*sx)*bpp, o +(8+(y+8)*(sx+16))*bpp, sx*bpp);
	}
	free(o);
	char *ext = strrchr(output, '.');
	if (!strcmp(ext, ".jpg")) stbi_write_jpg(output, sx, sy, 3, pix, 0);
	else stbi_write_png(output, sx, sy, 3, pix, 0);
	free(pix);

	free(cat.bdata);
	free(cat.wdata);

	oclReleaseKernel(kernel, ksz);
	oclFinish();
	return 0;
}

void usage(FILE* fp, char** argv)
{
	fprintf(fp,
		"Usage: %s [options] file\n\n"
		"Options:\n"
		"-h                 Print this message\n"
		"-m <model name>    waifu2x model name [noise2_model.json...]\n"
		"-s <scale>         Magnification [1.0, 1.6, 2.0...]\n"
		"-o <output name>   output file name\n"
		"\n",
		argv[0]);
}

int main(int argc, char* argv[])
{
	char *name = 0;
	char *model = "noise1_model.json";
	char *output = "output2x.png";
	float scale = 2.0;

	struct parg_state ps;
	int c;
	parg_init(&ps);
	while ((c = parg_getopt(&ps, argc, argv, "hm:s:o:")) != -1) {
		switch (c) {
		case 1:
			name = (char*)ps.optarg;
			break;
		case 'o':
			output = (char*)ps.optarg;
			break;
		case 'm':
			model = (char*)ps.optarg;
			break;
		case 's':
			scale = atof(ps.optarg);
			break;
		case 'h':
		default:
			usage(stderr, argv);
			return 1;
		}
	}
	if (!name) {
		usage(stderr, argv);
		return 1;
	}
	waifu2x_ocl(name, output, model, scale);

	return 0;
}
