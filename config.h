#pragma once
#include <stdint.h>

using namespace std;

#define SHARE_FILE0 "/home/gs/jmh/share_memory"
#define SHARE_FILE1 "/home/gs/jmh/share_memory_pred"

#define _PUSH_
#define _PULL_



#ifdef _INTVALUE_
typedef int Vtype;
#else
typedef float Vtype;
#endif

#ifdef _INT64_
typedef int64_t IntV;
#elif defined(_UNSIGNEDINT)
typedef uint32_t IntV;
#else
typedef int IntV;
#endif


struct col_type
{
#ifdef DCOL_
    int rk;
#endif
    int id;
#ifdef WEIGHTS_
    Vtype val;
#endif // 

};

struct halfpacked_edge
{
    int v0;
#ifdef WEIGHTS_
    Vtype val;
#endif // 

};



#ifndef DIRECTED_
#define outrowstarts rowstarts
#define outcolumn column
#define outremote_rowstarts remote_rowstarts
#define outremote_column remote_column
#endif

int ranks,num_procs;
int64_t num_vertice,totaledge,myrowoffset,myrowend;
IntV local_vert,numv_per_proc;

IntV Local_V(IntV v)
{       
        return v%numv_per_proc;
}   

int Owner(IntV v)
{
        return v/numv_per_proc;
}

#ifdef DCOL_
int Globalid(col_type a)
{
        return a.rk*numv_per_proc + a.id;
}
#endif
int Globalid(int a, int b)
{
        return a*numv_per_proc + b;
}

#ifdef DCOL_
void write_col(col_type & a, int b)
{
        a.rk = Owner(b);
        a.id = Local_V(b);
}

#ifdef WEIGHTS_
void write_col(col_type& a, int b, Vtype c)
{
    a.rk = Owner(b);
    a.id = Local_V(b);
    a.val = c;
}
#endif
#else
void write_col(col_type& a, int b)
{
    a.id = b;
}

#ifdef WEIGHTS_
void write_col(col_type& a, int b, Vtype c)
{
    a.id = b;
    a.val = c;
}
#endif
#endif