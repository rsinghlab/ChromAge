import subprocess
import pandas as pd
from pathlib import Path
import json
import numpy as np

export_path = "export PATH=$PATH:/gpfs/data/rsingh47/masif/sratoolkit.2.11.1-centos_linux64/bin"

call_pipeline_base = "caper run /gpfs/data/rsingh47/masif/chip-seq-pipeline2/chip.wdl -i /gpfs/data/rsingh47/masif/pipeline_raw_data"

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
        fastq_dump = "fastq-dump --outdir " + output_path + " --dumpbase --clip " + sra_id
        print ("The command used was: " + fastq_dump)
        subprocess.call(fastq_dump, shell=True)

def process_srr_val(srr_val):
    cleaned_srr_arr = []
    if not(pd.isna(srr_val)):
        split_srr = srr_val.split(";")
        for x in split_srr:
            x = x.strip()
            cleaned_srr_arr.append(x)
    print(cleaned_srr_arr)
    return cleaned_srr_arr

def run_pipeline(path = "/gpfs/data/rsingh47/masif/ChromAge/GEO_metadata.csv", priority=True, H3K4me3 = True, check = False):
    with open(path, 'rb') as f:
        df = pd.read_csv(f)
        df = df.sort_values(by="Age", ascending=False)
        # arr = df["H3K27ac SRR list"]
        # arr1 = df["H3K4me3 SRR list"]
        # arr2 = df["H3K4me1 SRR list"]
        # arr3 = df["H3K27me3 SRR list"]
        # arr4 = df["H3K9me3 SRR list"]
        # arr5 = df["H3K36me3 SRR list"]
        # print(len(arr[~pd.isna(arr)]))
        # print(len(arr1[~pd.isna(arr1)]))
        # print(len(arr2[~pd.isna(arr2)]))
        # print(len(arr3[~pd.isna(arr3)]))
        # print(len(arr4[~pd.isna(arr4)]))
        # print(len(arr5[~pd.isna(arr5)]))
        # return
    print("Finished reading data!")

    for i in range(df.shape[0]):
        control_srr_1 = process_srr_val(df["Control SRR list 1"][i])
        control_srr_2 = process_srr_val(df["Control SRR list 2"][i])

        h3k4me3_srr = process_srr_val(df["H3K4me3 SRR list"][i]) ## h3k4me3, h3k27ac are more important
        h3k4me3_GEO = process_srr_val(df["H3K4me3 GEO"][i])

        h3k27ac_srr = process_srr_val(df["H3K27ac SRR list"][i])
        h3k27ac_GEO = process_srr_val(df["H3K27ac GEO"][i])

        h3k4me1_srr = process_srr_val(df["H3K4me1 SRR list"][i])
        h3k4me1_GEO = process_srr_val(df["H3K4me1 GEO"][i])

        h3k27me3_srr = process_srr_val(df["H3K27me3 SRR list"][i])
        h3k27me3_GEO = process_srr_val(df["H3K27me3 GEO"][i])

        h3k9me3_srr = process_srr_val(df["H3K9me3 SRR list"][i])
        h3k9me3_GEO = process_srr_val(df["H3K9me3 GEO"][i])

        h3k36me3_srr = process_srr_val(df["H3K36me3 SRR list"][i])
        h3k36me3_GEO = process_srr_val(df["H3K36me3 GEO"][i])

        if (len(h3k4me3_GEO) > 0):
            if (h3k4me3_GEO[0] == 'GSM4106272'):
                check = True

        if check:
            paired_end = df["SE or PE"][i]
            if (paired_end == "SE"):
                paired_end = False
            else:
                paired_end = True
                
            h3k4me3_json = dict()
            h3k27ac_json = dict()
            h3k4me1_json = dict()
            h3k27me3_json = dict()
            h3k9me3_json = dict()
            h3k36me3_json = dict()

            generic_json = {
                "chip.pipeline_type" : "histone",
                "chip.genome_tsv" : "/gpfs/data/rsingh47/masif/chip-seq-pipeline2/genome/hg38.tsv",
            }

            CONTROL_DIR = "/gpfs/data/rsingh47/masif/pipeline_raw_data/control/"
            H3K4me3_DIR = "/gpfs/data/rsingh47/masif/pipeline_raw_data/h3k4me3/"
            H3K27ac_DIR = "/gpfs/data/rsingh47/masif/pipeline_raw_data/h3k27ac/"
            H3K4me1_DIR = "/gpfs/data/rsingh47/masif/pipeline_raw_data/h3k4me1/"
            H3K27me3_DIR = "/gpfs/data/rsingh47/masif/pipeline_raw_data/h3k27me3/"
            H3K9me3_DIR = "/gpfs/data/rsingh47/masif/pipeline_raw_data/h3k9me3/"
            H3K36me3_DIR = "/gpfs/data/rsingh47/masif/pipeline_raw_data/h3k36me3/"

            if (priority):
                if (H3K4me3):
                    if (len(h3k4me3_srr) > 0):
                        h3k4me3_json = generic_json
                        h3k4me3_json["chip.paired_end"] = paired_end
                        h3k4me3_json["chip.title"] = "h3k4me3_json_" + h3k4me3_GEO[0]
                        h3k4me3_json["chip.description"] = "Example_" + h3k4me3_GEO[0] + " h3k4me3_json"
                        h3k4me3_json["chip.fastqs_rep1_R1"] = []
                        for x in range (len(h3k4me3_srr)):
                            if not(Path(H3K4me3_DIR + h3k4me3_srr[x] + ".fastq").is_file()):
                                generate_fastq([h3k4me3_srr[x]], H3K4me3_DIR)
                            h3k4me3_json["chip.fastqs_rep1_R1"].append(H3K4me3_DIR + h3k4me3_srr[x] + ".fastq")

                        h3k4me3_json["chip.ctl_fastqs_rep1_R1"] = []
                        for x in range (len(control_srr_1)):
                            if not(Path(CONTROL_DIR + control_srr_1[x] + ".fastq").is_file()):
                                generate_fastq([control_srr_1[x]], CONTROL_DIR)
                            h3k4me3_json["chip.ctl_fastqs_rep1_R1"].append(CONTROL_DIR + control_srr_1[x] + ".fastq")
                        
                        if len(control_srr_2) > 0:
                            h3k4me3_json["chip.ctl_fastqs_rep2_R1"] = []
                            for x in range (len(control_srr_2)):
                                if not(Path(CONTROL_DIR + control_srr_2[x] + ".fastq").is_file()):
                                    generate_fastq([control_srr_2[x]], CONTROL_DIR)
                                h3k4me3_json["chip.ctl_fastqs_rep2_R1"].append(CONTROL_DIR + control_srr_2[x] + ".fastq")
                        
                        print(h3k4me3_json)
                        jsonFile = open(H3K4me3_DIR + "h3k4me3_" + h3k4me3_GEO[0] + ".json", "w")
                        jsonFile.write(json.dumps(h3k4me3_json))
                        jsonFile.close()
                else:     
                    if (len(h3k27ac_srr) > 0):
                        h3k27ac_json = generic_json
                        h3k27ac_json["chip.paired_end"] = paired_end
                        h3k27ac_json["chip.title"] = "h3k27ac_json_" + h3k27ac_GEO[0]
                        h3k27ac_json["chip.description"] = "Example_" + h3k27ac_GEO[0] + " h3k27ac_json"
                        h3k27ac_json["chip.fastqs_rep1_R1"] = []
                        for x in range (len(h3k27ac_srr)):
                            if not(Path(H3K27ac_DIR + h3k27ac_srr[x] + ".fastq").is_file()):
                                generate_fastq([h3k27ac_srr[x]], H3K27ac_DIR)
                            h3k27ac_json["chip.fastqs_rep1_R1"].append(H3K27ac_DIR + h3k27ac_srr[x] + ".fastq")

                        h3k27ac_json["chip.ctl_fastqs_rep1_R1"] = []
                        for x in range (len(control_srr_1)):
                            if not(Path(CONTROL_DIR + control_srr_1[x] + ".fastq").is_file()):
                                generate_fastq([control_srr_1[x]], CONTROL_DIR)
                            h3k27ac_json["chip.ctl_fastqs_rep1_R1"].append(CONTROL_DIR + control_srr_1[x] + ".fastq")
                        
                        if len(control_srr_2) > 0:
                            h3k27ac_json["chip.ctl_fastqs_rep2_R1"] = []
                            for x in range (len(control_srr_2)):
                                if not(Path(CONTROL_DIR + control_srr_2[x] + ".fastq").is_file()):
                                    generate_fastq([control_srr_2[x]], CONTROL_DIR)
                                h3k27ac_json["chip.ctl_fastqs_rep2_R1"].append(CONTROL_DIR + control_srr_2[x] + ".fastq")
                        
                        print(h3k27ac_json)
                        jsonFile = open(H3K27ac_DIR + "h3k27ac_" + h3k27ac_GEO[0] + ".json", "w")
                        jsonFile.write(json.dumps(h3k27ac_json))
                        jsonFile.close()
            
            if not(priority):
                if (len(h3k4me1_srr) > 0):
                    h3k4me1_json = generic_json
                    h3k4me1_json["chip.paired_end"] = paired_end
                    h3k4me1_json["chip.title"] = "h3k4me1_json_" + h3k4me1_GEO[0]
                    h3k4me1_json["chip.description"] = "Example_" + h3k4me1_GEO[0] + " h3k4me1_json"
                    h3k4me1_json["chip.fastqs_rep1_R1"] = []
                    for x in range (len(h3k4me1_srr)):
                        if not(Path(H3K4me1_DIR + h3k4me1_srr[x] + ".fastq").is_file()):
                            generate_fastq([h3k4me1_srr[x]], H3K4me1_DIR)
                        h3k4me1_json["chip.fastqs_rep1_R1"].append(H3K4me1_DIR + h3k4me1_srr[x] + ".fastq")

                    h3k4me1_json["chip.ctl_fastqs_rep1_R1"] = []
                    for x in range (len(control_srr_1)):
                        if not(Path(CONTROL_DIR + control_srr_1[x] + ".fastq").is_file()):
                            generate_fastq([control_srr_1[x]], CONTROL_DIR)
                        h3k4me1_json["chip.ctl_fastqs_rep1_R1"].append(CONTROL_DIR + control_srr_1[x] + ".fastq")
                    
                    if len(control_srr_2) > 0:
                        h3k4me1_json["chip.ctl_fastqs_rep2_R1"] = []
                        for x in range (len(control_srr_2)):
                            if not(Path(CONTROL_DIR + control_srr_2[x] + ".fastq").is_file()):
                                generate_fastq([control_srr_2[x]], CONTROL_DIR)
                            h3k4me1_json["chip.ctl_fastqs_rep2_R1"].append(CONTROL_DIR + control_srr_2[x] + ".fastq")
                    
                    print(h3k4me1_json)
                    jsonFile = open(H3K4me1_DIR + "h3k4me1_" + h3k4me1_GEO[0] + ".json", "w")
                    jsonFile.write(json.dumps(h3k4me1_json))
                    jsonFile.close()

                if (len(h3k27me3_srr) > 0):
                    h3k27me3_json = generic_json
                    h3k27me3_json["chip.paired_end"] = paired_end
                    h3k27me3_json["chip.title"] = "h3k27me3_json_" + h3k27me3_GEO[0]
                    h3k27me3_json["chip.description"] = "Example_" + h3k27me3_GEO[0] + "h3k27me3_json"
                    h3k27me3_json["chip.fastqs_rep1_R1"] = []
                    for x in range (len(h3k27me3_srr)):
                        if not(Path(H3K27me3_DIR + h3k27me3_srr[x] + ".fastq").is_file()):
                            generate_fastq([h3k27me3_srr[x]], H3K27me3_DIR)
                        h3k27me3_json["chip.fastqs_rep1_R1"].append(H3K27me3_DIR + h3k27me3_srr[x] + ".fastq")

                    h3k27me3_json["chip.ctl_fastqs_rep1_R1"] = []
                    for x in range (len(control_srr_1)):
                        if not(Path(CONTROL_DIR + control_srr_1[x] + ".fastq").is_file()):
                            generate_fastq([control_srr_1[x]], CONTROL_DIR)
                        h3k27me3_json["chip.ctl_fastqs_rep1_R1"].append(CONTROL_DIR + control_srr_1[x] + ".fastq")
                    
                    if len(control_srr_2) > 0:
                        h3k27me3_json["chip.ctl_fastqs_rep2_R1"] = []
                        for x in range (len(control_srr_2)):
                            if not(Path(CONTROL_DIR + control_srr_2[x] + ".fastq").is_file()):
                                generate_fastq([control_srr_2[x]], CONTROL_DIR)
                            h3k27me3_json["chip.ctl_fastqs_rep2_R1"].append(CONTROL_DIR + control_srr_2[x] + ".fastq")
                    
                    print(h3k27me3_json)
                    jsonFile = open(H3K27me3_DIR + "h3k27me3_" + h3k27me3_GEO[0] + ".json", "w")
                    jsonFile.write(json.dumps(h3k27me3_json))
                    jsonFile.close()

                if (len(h3k9me3_srr) > 0):
                    h3k9me3_json = generic_json
                    h3k9me3_json["chip.paired_end"] = paired_end
                    h3k9me3_json["chip.title"] = "h3k9me3_json_" + h3k9me3_GEO[0]
                    h3k9me3_json["chip.description"] = "Example_" + h3k9me3_GEO[0] + "h3k9me3_json"
                    h3k9me3_json["chip.fastqs_rep1_R1"] = []
                    for x in range (len(h3k9me3_srr)):
                        if not(Path(H3K9me3_DIR + h3k9me3_srr[x] + ".fastq").is_file()):
                            generate_fastq([h3k9me3_srr[x]], H3K9me3_DIR)
                        h3k9me3_json["chip.fastqs_rep1_R1"].append(H3K9me3_DIR + h3k9me3_srr[x] + ".fastq")

                    h3k9me3_json["chip.ctl_fastqs_rep1_R1"] = []
                    for x in range (len(control_srr_1)):
                        if not(Path(CONTROL_DIR + control_srr_1[x] + ".fastq").is_file()):
                            generate_fastq([control_srr_1[x]], CONTROL_DIR)
                        h3k9me3_json["chip.ctl_fastqs_rep1_R1"].append(CONTROL_DIR + control_srr_1[x] + ".fastq")
                    
                    if len(control_srr_2) > 0:
                        h3k9me3_json["chip.ctl_fastqs_rep2_R1"] = []
                        for x in range (len(control_srr_2)):
                            if not(Path(CONTROL_DIR + control_srr_2[x] + ".fastq").is_file()):
                                generate_fastq([control_srr_2[x]], CONTROL_DIR)
                            h3k9me3_json["chip.ctl_fastqs_rep2_R1"].append(CONTROL_DIR + control_srr_2[x] + ".fastq")
                    
                    print(h3k9me3_json)
                    jsonFile = open(H3K9me3_DIR + "h3k9me3_" + h3k9me3_GEO[0] + ".json", "w")
                    jsonFile.write(json.dumps(h3k9me3_json))
                    jsonFile.close()

                if (len(h3k36me3_srr) > 0):
                    h3k36me3_json = generic_json
                    h3k36me3_json["chip.paired_end"] = paired_end
                    h3k36me3_json["chip.title"] = "h3k36me3_json_" + h3k36me3_GEO[0]
                    h3k36me3_json["chip.description"] = "Example_" + h3k36me3_GEO[0] + "h3k36me3_json"
                    h3k36me3_json["chip.fastqs_rep1_R1"] = []
                    for x in range (len(h3k36me3_srr)):
                        if not(Path(H3K36me3_DIR + h3k36me3_srr[x] + ".fastq").is_file()):
                            generate_fastq([h3k36me3_srr[x]], H3K36me3_DIR)
                        h3k36me3_json["chip.fastqs_rep1_R1"].append(H3K36me3_DIR + h3k36me3_srr[x] + ".fastq")

                    h3k36me3_json["chip.ctl_fastqs_rep1_R1"] = []
                    for x in range (len(control_srr_1)):
                        if not(Path(CONTROL_DIR + control_srr_1[x] + ".fastq").is_file()):
                            generate_fastq([control_srr_1[x]], CONTROL_DIR)
                        h3k36me3_json["chip.ctl_fastqs_rep1_R1"].append(CONTROL_DIR + control_srr_1[x] + ".fastq")
                    
                    if len(control_srr_2) > 0:
                        h3k36me3_json["chip.ctl_fastqs_rep2_R1"] = []
                        for x in range (len(control_srr_2)):
                            if not(Path(CONTROL_DIR + control_srr_2[x] + ".fastq").is_file()):
                                generate_fastq([control_srr_2[x]], CONTROL_DIR)
                            h3k36me3_json["chip.ctl_fastqs_rep2_R1"].append(CONTROL_DIR + control_srr_2[x] + ".fastq")
                    
                    print(h3k36me3_json)
                    jsonFile = open(H3K36me3_DIR + "h3k36me3_" + h3k36me3_GEO[0] + ".json", "w")
                    jsonFile.write(json.dumps(h3k36me3_json))
                    jsonFile.close()

            if (priority):
                if (H3K4me3):
                    if (len(h3k4me3_json) != 0):
                        h3k4me3_pipeline_call = call_pipeline_base + "/h3k4me3/" + "h3k4me3_" + h3k4me3_GEO[0] + ".json"
                        print ("Running the encode pipeline for" + str(h3k4me3_json))
                        print ("The command used was: " + h3k4me3_pipeline_call)
                        subprocess.call(h3k4me3_pipeline_call, shell=True)
                        subprocess.call("sh extract_output.sh", shell=True)
                else:
                    if (len(h3k27ac_json) != 0):
                        h3k27ac_pipeline_call = call_pipeline_base + "/h3k27ac/" + "h3k27ac_" + h3k27ac_GEO[0] + ".json"
                        print ("Running the encode pipeline for" + str(h3k27ac_json))
                        print ("The command used was: " + h3k27ac_pipeline_call)
                        subprocess.call(h3k27ac_pipeline_call, shell=True)
                        subprocess.call("sh extract_output.sh", shell=True)

            if not(priority):
                if (len(h3k4me1_json) != 0):
                    h3k4me1_pipeline_call = call_pipeline_base + "/h3k4me1/" + "h3k4me1_" + h3k4me1_GEO[0] + ".json"
                    print ("Running the encode pipeline for" + str(h3k4me1_json))
                    print ("The command used was: " + h3k4me1_pipeline_call)
                    subprocess.call(h3k4me1_pipeline_call, shell=True)
                    subprocess.call("sh extract_output.sh", shell=True)

                if (len(h3k27me3_json) != 0):
                    h3k27me3_pipeline_call = call_pipeline_base + "/h3k27me3/" + "h3k27me3_" + h3k27me3_GEO[0] + ".json"
                    print ("Running the encode pipeline for" + str(h3k27me3_json))
                    print ("The command used was: " + h3k27me3_pipeline_call)
                    subprocess.call(h3k27me3_pipeline_call, shell=True)
                    subprocess.call("sh extract_output.sh", shell=True)

                if (len(h3k9me3_json) != 0):
                    h3k9me3_pipeline_call = call_pipeline_base + "/h3k9me3/" + "h3k9me3_" + h3k9me3_GEO[0] + ".json"
                    print ("Running the encode pipeline for" + str(h3k9me3_json))
                    print ("The command used was: " + h3k9me3_pipeline_call)
                    subprocess.call(h3k9me3_pipeline_call, shell=True)
                    subprocess.call("sh extract_output.sh", shell=True)

                if (len(h3k36me3_json) != 0):
                    h3k36me3_pipeline_call = call_pipeline_base + "/h3k36me3/" + "h3k36me3_" + h3k36me3_GEO[0] + ".json"
                    print ("Running the encode pipeline for" + str(h3k36me3_json))
                    print ("The command used was: " + h3k36me3_pipeline_call)
                    subprocess.call(h3k36me3_pipeline_call, shell=True)
                    subprocess.call("sh extract_output.sh", shell=True)
        print("ROW " + str(i) + " finished")

local_path = "/Users/haider/Documents/Fall-2021/ChromAge/GEO_metadata.csv"

run_pipeline()
# run_pipeline(H3K4me3=False)
# run_pipeline(priority=False)


# json_example = {
#     "chip.pipeline_type" : "histone",
#     "chip.genome_tsv" : "/gpfs/home/masif/data/masif/chip-seq-pipeline2/genome/hg38.tsv",
#     "chip.fastqs_rep1_R1" : ["/gpfs/home/masif/data/masif/chip-seq-pipeline2/example_input_j"],
#     "chip.ctl_fastqs_rep1_R1" : ["/gpfs/home/masif/data/masif/chip-seq-pipeline2/example_inp"],
#     "chip.paired_end" : False,
#     "chip.title" : "CHROMAGE001",
#     "chip.description" : "Example1 ChromAge"
#     }   
