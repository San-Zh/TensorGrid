/**
 * @file timer.h
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2023-05-18
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#pragma once

#include <sys/time.h>
#include <cstdlib>
// #include "utils.h"

#define FLOAT double
#define _INLINE_ inline

class timer {
  private:
    struct timeval t_begin, t_end;
    FLOAT timeuse;

  public:
    _INLINE_ timer() : timeuse(0.0)
    {
        gettimeofday(&t_begin, NULL);
        // gettimeofday(&t_end, NULL);
    };

    _INLINE_ ~timer(){};

  public:
    _INLINE_ void reset(void)
    {
        timeuse += use();
        gettimeofday(&t_begin, NULL);
        // gettimeofday(&t_end, NULL);
    }

    // void end(void) { gettimeofday(&t_end, NULL); }

    _INLINE_ FLOAT use()
    {
        gettimeofday(&t_end, NULL);
        return (FLOAT)(t_end.tv_sec - t_begin.tv_sec) + (FLOAT)(t_end.tv_usec - t_begin.tv_usec) / 1000000.0;
    }

    _INLINE_ FLOAT use_usec()
    {
        gettimeofday(&t_end, NULL);
        return (FLOAT)(t_end.tv_sec - t_begin.tv_sec) * 1000000.0 + (FLOAT)(t_end.tv_usec - t_begin.tv_usec);
    }

    _INLINE_ FLOAT total() { return timeuse + use(); }
};
