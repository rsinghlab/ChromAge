import pyBigWig
import sys

#hg38 numbers
chrom_info_hg38 = {
    'chr1':248956422,
    'chr2':242193529,
    'chr3':198295559,
    'chr4':190214555,
    'chr5':181538259,
    'chr6':170805979,
    'chr7':159345973,
    'chr8':145138636,
    'chr9':138394717,
    'chr10':133797422,
    'chr11':135086622,
    'chr12':133275309,
    'chr13':114364328,
    'chr14':107043718,
    'chr15':101991189,
    'chr16':90338345,
    'chr17':83257441,
    'chr18':80373285,
    'chr19':58617616,
    'chr20':64444167,
    'chr21':46709983,
    'chr22':50818468,
    'chrX':156040895
}

def check_files(file_path, verbose = True): #makes sure that files are not corrupted, and if so, removes from file_names  
    try: 
        bw = pyBigWig.open(file_path)
        #loop through all chromosomes to check if each one can be opened
        for chrom in list(chrom_info_hg38.keys()):
            chrom_bases = bw.values(chrom, 0, 42, numpy = True) 
    
    except: 
        if verbose == True:
            print("file corrupted:", file_path)

    bw.close()

def main():
    if len(sys.argv) != 2 or sys.argv[1][-7:] != ".bigwig":
        print("USAGE: python assignment.py <Big Wig File Name>")
        print("<Big Wig File Name>: [_.bigwig]")
        exit()
    
    check_files(sys.argv[1])


if __name__ == '__main__':
    main()

