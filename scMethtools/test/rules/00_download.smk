rule download:
    output:
        srafile = "{output_dir}/{srr}/0-data/raw/{srr}.sra"
    log:
        "logs/{srr}_download_prefetch.log"
    params:
        srrid = "{srr}"
    threads: 1
    retries: config["download"]["retry"]
    resources:
        download_slots = 1 
    shell:
        """
        if config["download"]["enabled"]:
        
            if [ -e {output.srafile}.lock ]; then
                rm {output.srafile}.lock
                echo "{output.srafile} lock found but no output file! Deleting..." >> {log}
            fi

            prefetch --max-size 100000000 --progress --output-directory rawfastq {params.srrid} >> {log} 2>&1
            if [ -e {output.srafile} ]; then
                echo "{output.srafile} download finished!" >> {log}
            else
                mv {output.srafile}* {output.srafile}
                echo "{output.srafile} not find! May end with .sralite. Renaming..." >> {log}
            fi
        else
            echo "print("Skipping download step ...") >> {log}
        """