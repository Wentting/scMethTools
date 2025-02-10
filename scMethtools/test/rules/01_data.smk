rule trim_galore_se:
    input:
        rules.extract_se.output
    output:
        trimmed_fastq = "trimgalore_result/{srr}_trimmed.fq.gz"
    params:
        option = config["trim"]["param"]
    log:
        "logs/{srr}_trimgalore.log"
    threads: config["trim"]["threads"]
    shell:
        """
        trim_galore {params.option} \
            --cores {threads} {input} -o trimgalore_result/ > {log} 2>&1
        """


rule trim_galore_pe:
    input:
        rules.extract_pe.output
    output:
        trimmed_fastq = ["trimgalore_result/{srr}_1_val_1.fq.gz", "trimgalore_result/{srr}_2_val_2.fq.gz"]
    params:
        option = config["trim"]["param"]
    log:
        "logs/{srr}_trimgalore.log"
    threads: config["trim"]["threads"]
    shell:
        """
        trim_galore --paired {params.option} \
            --cores {threads} {input[0]} {input[1]} -o trimgalore_result/ > {log} 2>&1
        """

