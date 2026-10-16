#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "svp_acl.h"
#include "svp_acl_mdl.h"
#include "svp_acl_ext.h"

#ifdef IR_GLIBC_217
extern int ir_old_start(int (*)(int,char**,char**),int,char**,void (*)(void),void (*)(void),void (*)(void),void*);
__asm__(".symver ir_old_start,__libc_start_main@GLIBC_2.17");
int __wrap___libc_start_main(int (*fn)(int,char**,char**),int ac,char**av,void(*in)(void),void(*fi)(void),void(*rt)(void),void *sp) {
    return ir_old_start(fn,ac,av,in,fi,rt,sp);
}
#endif

#define LIMIT 64
#define CHECK(call) do { int code=(call); if(code) { fprintf(stderr,"FAIL %s => %d line %d\n",#call,code,__LINE__); goto cleanup; } } while(0)
#define NEED(value) do { if(!(value)) { fprintf(stderr,"FAIL %s line %d\n",#value,__LINE__); goto cleanup; } } while(0)

typedef struct {
    svp_acl_mdl_dataset *set;
    void *pointer[LIMIT];
    size_t bytes[LIMIT], stride[LIMIT], count;
} buffers;

typedef struct {
    char name[256], dtype[32];
    size_t item_bytes, rows, width;
} logical_tensor;

static double now(void) {
    struct timespec value;
    clock_gettime(CLOCK_MONOTONIC, &value);
    return value.tv_sec+value.tv_nsec*1e-9;
}

static int finish_file(FILE **stream) {
    int status=fclose(*stream);
    *stream=NULL;
    return status;
}

static int specs(const char *folder, const char *side, logical_tensor *values, size_t *count) {
    char path[2048];
    snprintf(path,sizeof(path),"%s/%s_spec.txt",folder,side);
    FILE *stream=fopen(path,"r");
    if(!stream) return -1;
    int ok=fscanf(stream,"%zu",count)==1 && *count>0 && *count<=LIMIT;
    for(size_t i=0;ok && i<*count;i++) {
        ok=fscanf(stream,"%255s %31s %zu %zu %zu",values[i].name,values[i].dtype,&values[i].item_bytes,&values[i].rows,&values[i].width)==5;
        ok=ok && values[i].rows>0 && values[i].width>0 && (values[i].item_bytes==2 || values[i].item_bytes==4);
    }
    fclose(stream);
    return ok?0:-1;
}

static int allocate(buffers *buffer, svp_acl_mdl_desc *desc, int output) {
    buffer->set=svp_acl_mdl_create_dataset();
    if(!buffer->set) return -1;
    buffer->count=output?svp_acl_mdl_get_num_outputs(desc):svp_acl_mdl_get_num_inputs(desc);
    if(buffer->count>LIMIT) return -1;
    for(size_t i=0;i<buffer->count;i++) {
        buffer->bytes[i]=output?svp_acl_mdl_get_output_size_by_index(desc,i):svp_acl_mdl_get_input_size_by_index(desc,i);
        buffer->stride[i]=output?svp_acl_mdl_get_output_default_stride(desc,i):svp_acl_mdl_get_input_default_stride(desc,i);
        if(!buffer->bytes[i] || !buffer->stride[i]) return -1;
        if(svp_acl_rt_malloc_cached(&buffer->pointer[i],buffer->bytes[i],SVP_ACL_MEM_MALLOC_NORMAL_ONLY)) return -1;
        memset(buffer->pointer[i],0,buffer->bytes[i]);
        if(svp_acl_rt_mem_flush(buffer->pointer[i],buffer->bytes[i])) return -1;
        svp_acl_data_buffer *data=svp_acl_create_data_buffer(buffer->pointer[i],buffer->bytes[i],buffer->stride[i]);
        if(!data) return -1;
        if(svp_acl_mdl_add_dataset_buffer(buffer->set,data)) {
            svp_acl_destroy_data_buffer(data);
            return -1;
        }
    }
    return 0;
}

static void release(buffers *buffer) {
    if(buffer->set) {
        for(size_t i=0;i<svp_acl_mdl_get_dataset_num_buffers(buffer->set);i++)
            svp_acl_destroy_data_buffer(svp_acl_mdl_get_dataset_buffer(buffer->set,i));
        svp_acl_mdl_destroy_dataset(buffer->set);
    }
    for(size_t i=0;i<LIMIT;i++) if(buffer->pointer[i]) svp_acl_rt_free(buffer->pointer[i]);
}

static int type_matches(const logical_tensor *tensor, int type) {
    return (!strcmp(tensor->dtype,"FLOAT") && type==SVP_ACL_FLOAT) ||
        (!strcmp(tensor->dtype,"FLOAT16") && type==SVP_ACL_FLOAT16);
}

int main(int argc, char **argv) {
    if(argc!=6) {
        fprintf(stderr,"Usage: %s model.om case_dir existing_result_dir warmup runs\n",argv[0]);
        return 1;
    }
    setbuf(stdout,NULL);
    int warmup=atoi(argv[4]), runs=atoi(argv[5]);
    if(warmup<0 || runs<1 || runs>10000) return 1;
    int status=2, initialized=0, loaded=0;
    uint32_t model_id=0, aicpu=0;
    void *model_data=NULL;
    svp_acl_rt_context context=NULL;
    svp_acl_mdl_desc *desc=NULL;
    buffers input={0}, output={0};
    logical_tensor ins[LIMIT], outs[LIMIT];
    size_t ni=0,no=0,output_indices[LIMIT]={0};
    FILE *stream=NULL, *csv=NULL;
    char path[2048];
    double begin=now(), init_ms=0, load_ms=0, setup_ms=0;

    NEED(!specs(argv[2],"inputs",ins,&ni));
    NEED(!specs(argv[2],"outputs",outs,&no));
    CHECK(svp_acl_init(getenv("IR_SVP_CONFIG")));
    initialized=1;
    CHECK(svp_acl_rt_set_device(0));
    CHECK(svp_acl_rt_create_context(&context,0));
    init_ms=(now()-begin)*1000;
    double t=now();
    stream=fopen(argv[1],"rb");
    NEED(stream && !fseek(stream,0,SEEK_END));
    long length=ftell(stream);
    NEED(length>0);
    rewind(stream);
    CHECK(svp_acl_rt_malloc_cached(&model_data,(size_t)length,SVP_ACL_MEM_MALLOC_NORMAL_ONLY));
    NEED(fread(model_data,1,(size_t)length,stream)==(size_t)length);
    fclose(stream);stream=NULL;
    CHECK(svp_acl_rt_mem_flush(model_data,(size_t)length));
    CHECK(svp_acl_mdl_load_from_mem(model_data,(size_t)length,&model_id));
    loaded=1;
    desc=svp_acl_mdl_create_desc();NEED(desc);
    CHECK(svp_acl_mdl_get_desc(desc,model_id));
    CHECK(svp_acl_ext_get_mdl_aicpu_task_num(model_id,&aicpu));
    NEED(aicpu==0);
    load_ms=(now()-t)*1000;t=now();
    NEED(!allocate(&input,desc,0));
    NEED(!allocate(&output,desc,1));
    NEED(output.count==no);
    for(size_t n=0;n<ni;n++) {
        size_t index=0;
        CHECK(svp_acl_mdl_get_input_index_by_name(desc,ins[n].name,&index));
        NEED(index<input.count);
        NEED(type_matches(&ins[n],svp_acl_mdl_get_input_data_type(desc,index)));
        size_t row_bytes=ins[n].width*ins[n].item_bytes;
        NEED(input.stride[index]>=row_bytes && input.bytes[index]>=ins[n].rows*input.stride[index]);
        snprintf(path,sizeof(path),"%s/%s.bin",argv[2],ins[n].name);
        stream=fopen(path,"rb");NEED(stream);
        for(size_t row=0;row<ins[n].rows;row++)
            NEED(fread((char*)input.pointer[index]+row*input.stride[index],1,row_bytes,stream)==row_bytes);
        NEED(fgetc(stream)==EOF);
        fclose(stream);stream=NULL;
        CHECK(svp_acl_rt_mem_flush(input.pointer[index],input.bytes[index]));
    }
    for(size_t n=0;n<no;n++) {
        size_t index=output.count;
        for(size_t candidate=0;candidate<output.count;candidate++) {
            const char *name=svp_acl_mdl_get_output_name_by_index(desc,candidate);
            if(name && !strcmp(name,outs[n].name)) { index=candidate;break; }
        }
        NEED(index<output.count);
        output_indices[n]=index;
        NEED(type_matches(&outs[n],svp_acl_mdl_get_output_data_type(desc,index)));
        NEED(output.stride[index]>=outs[n].width*outs[n].item_bytes && output.bytes[index]>=outs[n].rows*output.stride[index]);
    }
    for(size_t index=0;index<input.count;index++) {
        const char *name=svp_acl_mdl_get_input_name_by_index(desc,index);
        if(name && !strcmp(name,SVP_ACL_DYNAMIC_TENSOR_NAME)) {
            CHECK(svp_acl_mdl_set_dynamic_batch_size(model_id,input.set,index,1));
        } else {
            int found=0;
            for(size_t n=0;n<ni;n++) if(name && !strcmp(name,ins[n].name)) found=1;
            NEED(found);
        }
    }
    setup_ms=(now()-t)*1000;
    snprintf(path,sizeof(path),"%s/timing.csv",argv[3]);
    csv=fopen(path,"w");NEED(csv);
    fprintf(csv,"iteration,phase,host_execute_ms\n");
    for(int i=-warmup;i<runs;i++) {
        double start=now();
        CHECK(svp_acl_mdl_execute(model_id,input.set,output.set));
        double elapsed=(now()-start)*1000;
        fprintf(csv,"%d,%s,%.9f\n",i,i<0?"warmup":"measured",elapsed);
        fflush(csv);
    }
    fclose(csv);csv=NULL;
    for(size_t n=0;n<no;n++) {
        size_t index=output_indices[n], row_bytes=outs[n].width*outs[n].item_bytes;
        CHECK(svp_acl_rt_mem_flush(output.pointer[index],output.bytes[index]));
        snprintf(path,sizeof(path),"%s/output_%03zu.bin",argv[3],n);
        stream=fopen(path,"wb");NEED(stream);
        for(size_t row=0;row<outs[n].rows;row++)
            NEED(fwrite((char*)output.pointer[index]+row*output.stride[index],1,row_bytes,stream)==row_bytes);
        NEED(!finish_file(&stream));
    }
    snprintf(path,sizeof(path),"%s/run.json",argv[3]);
    stream=fopen(path,"w");NEED(stream);
    fprintf(stream,"{\"boundary\":\"synchronous SDK module execute\",\"pure_NPU_profile\":false,\"full_pipeline\":false,\"aicpu_tasks\":%u,\"warmup\":%d,\"runs\":%d,\"init_ms\":%.6f,\"load_ms\":%.6f,\"setup_ms\":%.6f}\n",aicpu,warmup,runs,init_ms,load_ms,setup_ms);
    NEED(!finish_file(&stream));
    status=0;
cleanup:
    if(stream) fclose(stream);
    if(csv) fclose(csv);
    release(&input);release(&output);
    if(desc) svp_acl_mdl_destroy_desc(desc);
    if(loaded) svp_acl_mdl_unload(model_id);
    if(model_data) svp_acl_rt_free(model_data);
    if(context) svp_acl_rt_destroy_context(context);
    if(initialized) svp_acl_finalize();
    printf("module_bench_status=%d\n",status);
    return status;
}
