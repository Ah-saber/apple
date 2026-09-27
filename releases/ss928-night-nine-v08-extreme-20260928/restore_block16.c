/* Diagnostic compact row16 output restoration. No SDK stride guessing.
   ./restore_block16 input.bin output.bin 2|4 [height width]
   Time covers the restoration copy; SDK inference/report time is separate.
   This diagnostic's timing alone is not a complete inference result. */
#define _POSIX_C_SOURCE 200809L
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static double now_ms(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1000.0+t.tv_nsec/1e6;}
static void restore(unsigned char *dst,const unsigned char *src,size_t h,size_t w,size_t bytes){
 size_t row=w*bytes;
 for(size_t y=0;y<h;y++)memcpy(dst+y*row,src+((y%16)*(h/16)+y/16)*row,row);
}
int main(int argc,char **argv){
 if(argc!=4&&argc!=6){fprintf(stderr,"usage: %s input.bin output.bin bytes(2|4) [height width]\n",argv[0]);return 2;}
 size_t bytes=(size_t)strtoul(argv[3],NULL,10),h=argc==6?(size_t)strtoul(argv[4],NULL,10):3072,w=argc==6?(size_t)strtoul(argv[5],NULL,10):3840;
 if((bytes!=2&&bytes!=4)||!h||h%16||!w||w>SIZE_MAX/bytes||h>SIZE_MAX/(w*bytes))return 2;
 size_t n=h*w*bytes;unsigned char *src=malloc(n),*dst=malloc(n);FILE *f=fopen(argv[1],"rb");
 if(!src||!dst||!f){fprintf(stderr,"allocation/input failed\n");return 2;}
 if(fread(src,1,n,f)!=n||fgetc(f)!=EOF){fprintf(stderr,"input must be compact NCHW [1,16,H/16,W] with exact bytes\n");return 2;}fclose(f);
 for(int i=0;i<30;i++)restore(dst,src,h,w,bytes);
 volatile unsigned long checksum=0;
 printf("sample,restore_ms,diagnostic_only\n");
 for(int i=0;i<200;i++){double a=now_ms();restore(dst,src,h,w,bytes);double b=now_ms();checksum+=dst[(size_t)i%n];printf("%d,%.9f,true\n",i,b-a);}
 f=fopen(argv[2],"wb");if(!f||fwrite(dst,1,n,f)!=n)return 2;fclose(f);fprintf(stderr,"checksum=%lu; output bytes=%zu; NPU plus restoration must be measured synchronously for total\n",checksum,n);free(src);free(dst);return 0;
}
