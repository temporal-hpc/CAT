#pragma once

#include <cstdio>
#include "Memory/CADataDomain.cuh"

class CADataPrinter {
   public:
    static void printCADataWithHalo(CADataDomain<int>* data) {
        for (int i = 0; i < data->getSideLength(); i++) {
            for (int j = 0; j < data->getSideLength(); j++) {
                size_t index = i * data->getSideLength() + j;

                int element = data->getElementAt(index);
                if (element == 0) {
                    printf("- ");
                } else {
                    printf("%d ", element);
                }
            }
            printf("\n");
        }
        printf("\n");
    }
    static void printCADataWithoutHalo(CADataDomain<int>* data) {
        for (int i = 0; i < data->getSideLengthWithoutHalo(); i++) {
            for (int j = 0; j < data->getSideLengthWithoutHalo(); j++) {
                int element = data->getInnerElementAt(i, j);
                if (element == 0) {
                    printf("  ");
                } else {
                    printf("%d ", element);
                }
            }
            printf("\n");
        }
        printf("\n");
    }
};