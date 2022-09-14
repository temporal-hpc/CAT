#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>

#include "Debug.h"

int debugLevel = 0;

FILE* debugFile = 0;
FILE* pStdout = stdout;
/** This routine will initialize the variables from the arguments
 *  passed to <tt>main()</tt>. If the <b>first</b> argument is, or starts with
 *  <tt>-debug</tt> the class is initialized and the first parameter is
 *  removed from the <tt>args</tt>. The first parameter may have one of
 *  three forms:
 *  <ol>
 *  <li><tt>-debug</tt> - turn on debugging at level 1</li>
 *  <li><tt>-debugValue</tt> - set debugging level to <tt>Value</tt>
 *  (e.g. <tt>-debug5</tt>)</li>
 *  <li><tt>-debugValue@fileName</tt> set debugging level to <tt>Value</tt>
 *  and send debugging output to the file <tt>fileName</tt>. If you use
 *  this option, the file must b closed using <tt>debugClose()</tt>.</li>
 *  </ol>
 *  On return, <tt>argc</tt> and <tt>argv[]</tt> may be modified.
 *  @param  argc - the number of parameters passed to <tt>main</tt>
 *  @param  args the array of arguments passed to <tt>main</tt>
 */

void debugInit(int level, std::string outputFile) {
    debugFile = stderr;

    if (level > 0) {
        debugLevel = 1;

        if (outputFile.length() != 0) {
            debugToFile(outputFile);
        }
    }
}

/** Send debugging output to a file.
 *  @param fileName name of file to send output to
 */

void debugToFile(std::string fileName) {
    debugClose();

    FILE* f = fopen(fileName.c_str(), "w"); // "w+" ?

    if (f)
        debugFile = f;
}

/** Close the output file if it was set in <tt>toFile()</tt> */

void debugClose(void) {
    if (debugFile && (debugFile != stderr)) {
        fclose(debugFile);
        debugFile = stderr;
    }
}
