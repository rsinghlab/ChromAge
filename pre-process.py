import subprocess
import pandas as pd
import os
import json

export_path = "export PATH=$PATH:/gpfs/home/masif/data/masif/sratoolkit.2.11.1-centos_linux64/bin"

def generate_fastq(sra_numbers, output_path):

    # this will download the .sra files to ~/ncbi/public/sra/ (will create directory if not present)
    # for sra_id in sra_numbers:
    #     print ("Currently downloading: " + sra_id)
    #     prefetch = "prefetch " + sra_id
    #     print ("The command used was: " + prefetch)
    #     subprocess.call(prefetch, shell=True)

    # this will extract the .sra files from above into a folder named 'fastq'
    for sra_id in sra_numbers:
        print ("Generating fastq for: " + sra_id)
        fastq_dump = "fastq-dump --outdir /gpfs/home/masif/data/masif/chip-seq-pipeline2/example_input_json/fastq --skip-technical --readids --read-filter pass --dumpbase --split-3 --clip " + sra_id
        print ("The command used was: " + fastq_dump)
        subprocess.call(fastq_dump, shell=True)

def process_srr_val(srr_val):
    cleaned_srr_arr = []
    if not(pd.isna(srr_val)) and srr_val!="2":
        split_srr = srr_val.split(";")
        for x in split_srr:
            x = x.strip()
            cleaned_srr_arr.append(x)
    return cleaned_srr_arr

def create_json_1(path = "/gpfs/home/masif/data/masif/ChromAge/GEO_metadata.csv"):
    with open(path, 'rb') as f:
        df = pd.read_csv(f)
        df = df.sort_values(by="Age", ascending=False)
    print("Finished reading data!")
    for i in range(df.shape[0]):
        control_srr_1 = process_srr_val(df["Control SRR list 1"][i])
        control_srr_2 = process_srr_val(df["Control SRR list 2"][i])
        h3k4me3_srr = process_srr_val(df["H3K4me3 SRR list"][i]) ## h3k4me3, h3k27ac are more important
        h3k27ac_srr = process_srr_val(df["H3K27ac SRR list"][i])
        paired_end = df["SE or PE"][i]
        if (paired_end == "SE"):
            paired_end = False
            
        h3k4me3_json = dict()
        h3k27ac_json = dict()
        generic_json = {
            "chip.pipeline_type" : "histone",
            "chip.genome_tsv" : "/gpfs/home/masif/data/masif/chip-seq-pipeline2/genome/hg38.tsv",
        }

        generate_fastq(control_srr_1)
        generate_fastq(control_srr_2)
        generate_fastq(h3k27ac_srr)
        generate_fastq(h3k4me3_srr)

        h3k4me3_counter = 0
        if (len(h3k4me3_srr) > 0):
            h3k4me3_json = generic_json
            h3k4me3_json["chip.paired_end"] = paired_end
            h3k4me3_json["chip.title"] = "h3k4me3_json_"+i
            h3k4me3_json["chip.description"] = "Example_" + i + "h3k4me3_json"
            h3k4me3_json["chip.fastqs_rep1_R1"] = []
            for x in range in len(h3k4me3_srr):
                h3k4me3_json["chip.fastqs_rep1_R1"] = h3k4me3_json.get("chip.fastqs_rep1_R1").append(
                    "/gpfs/home/masif/data/masif/chip-seq-pipeline2/example_input_json/h3k4me3/" + h3k4me3_srr[x] + "_pass.fastq")

            h3k4me3_json["chip.ctl_fastqs_rep1_R1"] = []
            for x in range in len(control_srr_1):
                h3k4me3_json["chip.ctl_fastqs_rep1_R1"] = h3k4me3_json.get("chip.ctl_fastqs_rep1_R1").append(
                    "/gpfs/home/masif/data/masif/chip-seq-pipeline2/example_input_json/control/" + control_srr_1[x] + "_pass.fastq")
            
            h3k4me3_json["chip.ctl_fastqs_rep2_R1"] = []
            if len(control_srr_2) > 0:
                for x in range in len(control_srr_2):
                    h3k4me3_json["chip.ctl_fastqs_rep2_R1"] = h3k4me3_json.get("chip.ctl_fastqs_rep2_R1").append(
                    "/gpfs/home/masif/data/masif/chip-seq-pipeline2/example_input_json/control/" + control_srr_2[x] + "_pass.fastq")
            
            h3k4me3_json = json.dumps(h3k4me3_json)
            jsonFile = open("/gpfs/home/masif/data/masif/chip-seq-pipeline2/example_input_json/h3k4me3_" + h3k4me3_counter + ".json", "w")
            jsonFile.write(h3k4me3_json)
            jsonFile.close()
            h3k4me3_counter += 1
        
        h3k27ac_counter = 0 
        if (len(h3k27ac_srr) > 0):
            h3k27ac_json = generic_json
            h3k27ac_json["chip.paired_end"] = paired_end
            h3k27ac_json["chip.title"] = "h3k27ac_json_"+i
            h3k27ac_json["chip.description"] = "Example_" + i + "h3k27ac_json"
            h3k27ac_json["chip.fastqs_rep1_R1"] = []
            for x in range in len(h3k27ac_srr):
                h3k27ac_json["chip.fastqs_rep1_R1"] = h3k27ac_json.get("chip.fastqs_rep1_R1").append(
                    "/gpfs/home/masif/data/masif/chip-seq-pipeline2/example_input_json/h3k27ac/" + h3k27ac_srr[x] + "_pass.fastq")

            h3k27ac_json["chip.ctl_fastqs_rep1_R1"] = []
            for x in range in len(control_srr_1):
                h3k27ac_json["chip.ctl_fastqs_rep1_R1"] = h3k27ac_json.get("chip.ctl_fastqs_rep1_R1").append(
                    "/gpfs/home/masif/data/masif/chip-seq-pipeline2/example_input_json/control/" + control_srr_1[x] + "_pass.fastq")
            
            h3k27ac_json["chip.ctl_fastqs_rep2_R1"] = []
            if len(control_srr_2) > 0:
                for x in range in len(control_srr_2):
                    h3k27ac_json["chip.ctl_fastqs_rep2_R1"] = h3k27ac_json.get("chip.ctl_fastqs_rep2_R1").append(
                    "/gpfs/home/masif/data/masif/chip-seq-pipeline2/example_input_json/control/" + control_srr_2[x] + "_pass.fastq")
            
            h3k27ac_json = json.dumps(h3k27ac_json)
            jsonFile = open("/gpfs/home/masif/data/masif/chip-seq-pipeline2/example_input_json/h3k27ac_" + h3k27ac_counter + ".json", "w")
            jsonFile.write(h3k27ac_json)
            jsonFile.close()
            h3k27ac_counter += 1
            break

create_json_1()

# jsons = {
#     "chip.pipeline_type" : "histone",
#     "chip.genome_tsv" : "/gpfs/home/masif/data/masif/chip-seq-pipeline2/genome/hg38.tsv",
#     "chip.fastqs_rep1_R1" : ["/gpfs/home/masif/data/masif/chip-seq-pipeline2/example_input_j"],
#     "chip.ctl_fastqs_rep1_R1" : ["/gpfs/home/masif/data/masif/chip-seq-pipeline2/example_inp"],
#     "chip.paired_end" : False,
#     "chip.title" : "CHROMAGE001",
#     "chip.description" : "Example1 ChromAge"
#     }   
