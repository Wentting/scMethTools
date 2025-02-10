rule bismark:
    input:
        data = 
    output:
        peak = "results/{sample}/{sample}_peaks.narrowPeak",
    params:
        option = config["macs"]["param"],
        chromsizefile = config["macs"]["chromsize"],
        name = "{sample}",
        treat = lambda w: getTreatBam(w.sample, config['samples']),
        input = lambda w: getInputBam(w.sample, config['samples']),
        type = lambda w: getBamType(w.sample, config['samples']),
     log:
        "logs/{sample}_macs3.log"
    threads: 1
    shell:
        """
        OUTDIR=./results/{params.name}
        TREAT_PILEUP=$OUTDIR/{params.name}_treat_pileup.bdg
        CONTROL_LAMBDA=$OUTDIR/{params.name}_control_lambda.bdg
        LOGP_BDG=$OUTDIR/{params.name}.logPvalue.bedGraph
        FE_BDG=$OUTDIR/{params.name}.FE.bedGraph
        LOGP_BW=$OUTDIR/{params.name}.logPvalue.bw
        FE_BW=$OUTDIR/{params.name}.FE.bw

        macs3 callpeak -t {params.treat} -c {params.input} -n {params.name} {params.option} -f {params.type} --outdir $OUTDIR > {log} 2>&1

        macs3 bdgcmp -t $TREAT_PILEUP -c $CONTROL_LAMBDA -m ppois -p 1.0 -S 1.0 -o $LOGP_BDG >> {log} 2>&1
        macs3 bdgcmp -t $TREAT_PILEUP -c $CONTROL_LAMBDA -m FE -p 1.0 -S 1.0 -o $FE_BDG >> {log} 2>&1

        bedGraphToBigWig $LOGP_BDG {params.chromsizefile} $LOGP_BW
        bedGraphToBigWig $FE_BDG {params.chromsizefile} $FE_BW
        """