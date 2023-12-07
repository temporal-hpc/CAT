#pragma once

#include <iostream>
#include "Memory/CADataDomain.cuh"
template <typename T>
class CADataPrinter {
   public:
    static void printCADataWithHalo(CADataDomain<T>* data) {
        for (int i = 0; i < data->getSideLength(); i++) {
            for (int j = 0; j < data->getSideLength(); j++) {
                size_t index = i * data->getSideLength() + j;

                T element = data->getElementAt(index);
                if (element == 0) {
                    std::cout << "- ";
                } else {
                    std::cout << element << " ";
                }
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
    static void printCADataWithoutHalo(CADataDomain<T>* data) {
        for (int i = 0; i < data->getSideLengthWithoutHalo(); i++) {
            for (int j = 0; j < data->getSideLengthWithoutHalo(); j++) {
                T element = data->getInnerElementAt(i, j);
                if (element == 0) {
                    std::cout << "  ";
                } else {
                    std::cout << element << " ";
                }
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
};
