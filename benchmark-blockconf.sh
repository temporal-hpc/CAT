#!/bin/bash
if [ "$#" -ne 10 ]; then
    echo "run as ./benchmark-blockconf.sh DEV GPU_ID ARCH    STARTB ENDB DB    STARTN ENDN DN     OUTTER_REPEATS"
    exit;
fi
DEV=$1
GPUID=$2
ARCH=$3
STARTB=$4
ENDB=$5
DB=$6
STARTN=$7
ENDN=$8
DN=$9
R=${10}

METHODS=("BBox" "Hadouken" "DP")
NM=$((${#METHODS[@]}))

echo "REPEATS=${R}"
GPUPROG=./bin/prog
DATE=$(exec date +"%T-%m-%d-%Y (%:z %Z)")
echo "DATE = ${DATE}"


for BP in `seq ${STARTB} ${DB} ${ENDB}`;
do
    echo "Benchmarking for BPOWER=${BP} METHODS=${NM}"
    BS=$((2**${BP}))
    LB=$((${BS} * ${BS}))
	OUTPUT=data/${GPUID}-REP${R}-BS${BP}.dat
    cont=0
    for N in `seq ${STARTN} ${DN} ${ENDN}`;
    do
        REP=$R
        echo "GPU=${GPUID}  N=${N} B=${BP} R=${N}"
        echo -n "${N}   ${BP}    " >> ${OUTPUT}
        for q in `seq 1 ${NM}`;
        do
            q=$(($q-1))

            # Chosen MAP
            if [ $q -eq 2 ]
            then
                echo "make BSIZE3DX=${BS} BSIZE3DY=${BS} BSIZE3DZ=${BS} ARCH=${ARCH} DP=SI"
                COMPILE=`make BSIZE3DX=${BS} BSIZE3DY=${BS} BSIZE3DZ=${BS} ARCH=${ARCH} DP=SI`
                echo "Compiling with BSIZE2D=$BS RLEVEL=$N"
                echo ${COMPILE}
            else
                echo "make BSIZE3DX=${BS} BSIZE3DY=${BS} BSIZE3DZ=${BS} ARCH=${ARCH}"
                COMPILE=`make BSIZE3DX=${BS} BSIZE3DY=${BS} BSIZE3DZ=${BS} ARCH=${ARCH}`
                echo "Compiling with BSIZE2D=$BS RLEVEL=$N"
                echo ${COMPILE}
            fi

            echo "${GPUPROG} ${DEV} ${N} ${REP} ${q}"
            #echo "${GPUPROG} ${DEV} ${REP} 2 0.5 1"
            echo -n "[BENCHMARK] ${METHODS[$(($q-1))]} ($q) map (${SAMPLES} Samples)............."
			x=`${GPUPROG} ${DEV} ${N} ${REP} ${q}`
            echo "done"
            echo "---> BPOWER=${BP} RLEVEL=${N} --> (MEAN, VAR, STDEV, STERR) -> (${x})"
			echo "Saving in ${OUTPUT}..."
            echo -n "${x}    " >> "${OUTPUT}"
			echo "done"
            echo `make clean`
        done
        echo " " >> "${OUTPUT}"
        echo ""
        echo ""
        echo ""
        cont=${cont}+1
    done
    echo " "
done
echo "END TIME"
timedatectl
echo "*******************************"




