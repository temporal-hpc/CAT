reset
set   autoscale                        # scale axes automatically
set term postscript eps color blacktext "Times" 24
set output 'plots/3s-test1-write-speedup-comp.eps'
set title '3s-write Kernel Speedup with {{/Symbol l}}({/Symbol w})_{{/Symbol r}=8}'
set ytics nomirror
unset ytics
set xtics (0, 256, 512, 768, 1024, 1408)
set xrange [0:1408]
set yrange [0.6:2.5]
set y2tics 0.2
set link y2
set y2label 'S_{{/Symbol l}}' rotate by 0 offset -1.5
set xlabel 'N'
set font "Times, 20"
set key Left left top reverse samplen 1.0 font "Times,20" spacing 1

set style line 1 dashtype 3 pt 8 lw 1.0 lc rgb "black"
set style line 2 dashtype 3 pt 4 lw 1.0 lc rgb "magenta"
set style line 3 dashtype 3 pt 10 lw 1.0 lc rgb "red"
set style line 4 dashtype 3 pt 81 lw 1.0 lc rgb "forest-green"

fbb(x)      =1
set pointsize   0.7
plot    fbb(x) notitle dashtype 2 lc rgb "black",\
        'data/3s-test1-write-teslav100_B8.dat' u 1:($3/$7) title "Tesla V100"    with linespoints ls 4,\
        'data/3s-test1-write-titanx_B8.dat' u 1:($3/$7) title "Titan X"       with linespoints ls 1,\
        'data/3s-test1-write-gtx1050ti_B8.dat' u 1:($3/$7) title "GTX 1050 Ti"   with linespoints ls 2,\
        'data/3s-test1-write-teslak40_B8.dat' u 1:($3/$7) title "Tesla K40"     with linespoints ls 3
