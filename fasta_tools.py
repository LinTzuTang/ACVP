from Bio import SeqIO
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.patches as mpl_patches
import seaborn as sns
import random
import os


# read fasta as lists
def read_fasta_list(fasta_name, input_path=None):
    input_path = input_path or './' 
    r = []
    for record in SeqIO.parse(os.path.join(input_path, fasta_name), 'fasta'):
        r.append(record)
    return r

def read_fasta(fasta_path):
    r = {}
    for record in SeqIO.parse(fasta_path, 'fasta'):
        idtag = str(record.id)
        seq = str(record.seq)
        r[idtag] = seq
    return r

def save_fasta(fasta_list, fasta_name, output_path=None):
    output_path = output_path or './'
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
    SeqIO.write(fasta_list, output_path+fasta_name, "fasta")
    return output_path
    
def fasta_remove_duplicate(fasta_name, input_path=None, output_path=None):
    input_path = input_path or 'Data/Fasta'
    output_path = output_path or 'Output/Fasta'
    path = os.path.join(input_path, fasta_name)
    r = []
    for record in SeqIO.parse(path, 'fasta'):
        idtag = str(record.id)
        seq = str(record.seq)
        r.append(record)
        for index,item in enumerate(r[:-1]):
            if seq == str(r[index].seq):
                r.pop()

    print('Remain:%s'%len(r))
    SeqIO.write(r, os.path.join(output_path, fasta_name.split('.')[0])+ '_independent.fasta', "fasta")
    return output_path
    
def fasta_length_filter(fasta_name, min_len=10, max_len=200, input_path = None):
    input_path = input_path or './'
    length_filter_count = 0
    orign_fasta_list=[]
    fasta_list=[]
    for record in SeqIO.parse(path+ fasta_name,'fasta'):
        orign_fasta_list.append(record)
        if len(record.seq) < int(min_len) or len(record.seq) > int(max_len) :
            length_filter_count = length_filter_count+1 
        else:
            fasta_list.append(record)
    print('orign:%s'%len(orign_fasta_list)+"\n")
    print('length not in range:%s'%length_filter_count+"\n")
    print('Remain:%s'%len(fasta_list))
    return fasta_list

def fasta_unusual_filter(fasta_name, unusual_chars=('B', 'Z', 'U','X','i'), input_path = None):
    input_path = input_path or './'
    unusual_peptide_count = 0
    orign_fasta_list=[]
    fasta_list=[]
    for record in SeqIO.parse(path+ fasta_name,'fasta'):
        orign_fasta_list.append(record)
        if any(c in record.seq for c in ('B', 'Z', 'U','X','i')):
            unusual_peptide_count = unusual_peptide_count+1
        else:
            fasta_list.append(record)
    print('orign:%s'%len(orign_fasta_list)+"\n")
    print('contain unusual chars:%s'% unusual_peptide_count+"\n")
    print('Remain:%s'%len(fasta_list))
    return fasta_list

def fasta_rename(fasta_name,file_name='renamed'):   
    path = 'Data/Fasta/'+ fasta_name
    r = []
    for record in SeqIO.parse(path, 'fasta'):
        seq = str(record.seq)
        r.append(seq)    
    #name and write to fasta
    name = []
    for i in range(1,len(r)+1):
        name.append('>non_amp_'+str(i))
    print('sequence_count:%s'%len(r))
    path = 'Output/Fasta/'
    with open(path+'%s.fasta'%file_name, 'w') as fasta:
        for index, sequence in enumerate(r):
            fasta.write(name[index]+'\n')
            fasta.write(sequence+'\n')

def random_sample(fasta_path, numbers, output_path=None):
    orign_fasta_list=[]
    fasta_list=[]
    for record in SeqIO.parse(fasta_path,'fasta'):
        orign_fasta_list.append(record)
    fasta_list = random.sample(orign_fasta_list, numbers)
    print('orign:%s'%len(orign_fasta_list)+"\n")
    print('random sample:%s'%len(fasta_list))
    output_path=output_path or './'
    save_fasta(fasta_list, os.path.basename(fasta_path).split('.')[0]+'_%s'%numbers+'.fasta', output_path= output_path)
    
def random_generator(records, length):
    file_name = 'random_generate'
    alphabet = 'ACDEFGHIKLMNPQRSTVWY'
    with open('%s_%s.fasta'%(file_name,records), 'w') as fasta:
        for i in range(1,records+1):            
            fasta.write('>random_peptide_'+str(i)+'\n')
            fasta.write(''.join(random.choices(alphabet, k=length))+'\n')
        print('sequence_count:%s'%records)
    return path

#random generator sequences length based on a dataset
def random_generator_based(fasta_path):
    # get all sequence length 
    r = []
    for record in SeqIO.parse(fasta_path, 'fasta'):
        length = len(str(record.seq))
        r.append(length)
    file_name = 'random_generate'
    path = './'
    alphabet = 'ACDEFGHIKLMNPQRSTVWY'
    with open(path+'%s_%s.fasta'%(file_name,len(r)), 'w') as fasta:
        for index, length in enumerate(r):            
            fasta.write('>random_peptide_'+str(index+1)+'\n')
            fasta.write(''.join(random.choices(alphabet, k=length))+'\n')
        print('sequence_count:%s'%len(r))
    return path

def AMP_filter(fasta_name, min_len=10, max_len=200,fasta_path = None):    
    aa_list = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']
    length_filter_count = 0
    unusual_peptide_count = 0
    orign_fasta_list=[]
    fasta_list=[]
    # filter and count length not in range
    for record in SeqIO.parse(fasta_name,'fasta'):
        orign_fasta_list.append(record)
        if len(record.seq) < int(min_len) or len(record.seq) > int(max_len) :
            length_filter_count = length_filter_count+1
        elif set(record.seq) <= set(aa_list):
            fasta_list.append(record)
    totally_filter_out = len(orign_fasta_list)-len(fasta_list)     
    
    # count contain unusual amino acid
    for record in SeqIO.parse(fasta_name,'fasta'):
        if not set(record.seq) <= set(aa_list):
            unusual_peptide_count = unusual_peptide_count+1
    
    # save fasta
    fasta_path = fasta_path or './'
    if not os.path.isdir(fasta_path):
        os.mkdir(fasta_path)
    SeqIO.write(fasta_list, fasta_path+os.path.basename(fasta_name).split('.')[0]+ '_filtered.fasta', "fasta")
    
    print('orign:%s'%len(orign_fasta_list)+"\n")
    print('length not in range:%s'%length_filter_count+"\n")
    print('contain unusual amino acid:%s'% unusual_peptide_count+"\n")
    print('Totally filter out:%s'% totally_filter_out+"\n")
    print('Remain:%s'%len(fasta_list))
    return fasta_path

def fasta_traintest_split(fasta_name, test_size = 0.1, input_path=None, output_path=None):
    r = read_fasta(fasta_name, input_path)
    x_train ,x_test = train_test_split(r,test_size = test_size, random_state=10)   
    save_fasta(x_train, fasta_name.split('.')[0]+'_%s'%len(x_train)+'.fasta',output_path)
    save_fasta(x_test, fasta_name.split('.')[0]+'_%s'%len(x_test)+'.fasta',output_path)
    return output_path
    
# filter out fasta_fname_2 within fasta_fname_1
def fasta_comparison(fasta_fname_1, fasta_fname_2):
    path = './'
    r = []
    s = []
    for record in SeqIO.parse(fasta_fname_1, 'fasta'):
        idtag = str(record.id)
        seq = str(record.seq)
        for record2 in SeqIO.parse(fasta_fname_2, 'fasta'):
            seq2 = str(record2.seq)
            if seq == seq2:
                s.append(record)
                break
        else:
            r.append(record)
            continue
    print('Same records :%s'%len(s))
    print('Remain:%s'%len(r))
    
    SeqIO.write(r, fasta_fname_1.split('.')[0]+ '_independent.fasta', "fasta")
    SeqIO.write(s, fasta_fname_1.split('.')[0]+ '_same.fasta', "fasta")
    return path

#combine two fasta and rename
def combine_fasta(fasta_name_1, fasta_name_2):
    path = './'
    r = []
    for record in SeqIO.parse(path+fasta_name_1, 'fasta'):
        seq = str(record.seq)
        r.append(seq)
    for record in SeqIO.parse(path+fasta_name_2, 'fasta'):
        seq = str(record.seq)
        r.append(seq)
    #name and write to fasta
    name = []
    for i in range(1,len(r)+1):
        name.append('>non_amp_'+str(i))
    print('sequence_count:%s'%len(r))
    file_name = 'combined'
    with open(path+'%s_%s.fasta'%(file_name,len(r)), 'w') as fasta:
        for index, sequence in enumerate(r):
            fasta.write(name[index]+'\n')
            fasta.write(sequence+'\n')
    return path

#plot length distribution
def length_distribution(fasta_name):
    path = 'Data/Fasta/'
    length_dict = {}
    for record in SeqIO.parse(fasta_name, 'fasta'):
        idtag = str(record.id)
        seq = str(record.seq)
        length_dict[idtag] = len(seq)
    # dict -> list
    length_list = []
    for value in length_dict.values():
        length_list.append(value)
    print('Total Records:%s'%len(length_list))
    print('Max length:%s'%max(length_list))
    text = ('Total Records:%s'%len(length_list)+'\n'+'Max length:%s'%max(length_list))
    sns.set()
    sns.distplot(length_list,bins=15, kde=False)
    plt.title( 'Length Distribution', fontfamily = 'serif', fontsize = '14' )
    plt.xlabel('Sequence Length', fontfamily = 'serif', fontsize = '12' )
    plt.ylabel('Count', fontfamily = 'serif', fontsize = '12' )
    
    # create a list with two empty handles (or more if needed)
    handles = [mpl_patches.Rectangle((0, 0), 1, 1, fc="white", ec="white", 
                                     lw=0, alpha=0)] * 2

    # create the corresponding number of labels (= the text you want to display)
    labels = []
    labels.append('Total Records : %s'%len(length_list))
    labels.append('Max length : %s'%max(length_list))

    # create the legend, supressing the blank space of the empty line symbol and the
    # padding between symbol and label by setting handlelenght and handletextpad
    plt.legend(handles, labels, loc='best', fontsize='small', 
              fancybox=True, framealpha=0.7, 
              handlelength=0, handletextpad=0)
    
    plot_path = 'Output/Plot/'
    if not os.path.isdir(plot_path):
        os.mkdir(plot_path)
    plt.savefig(plot_path+fasta_name.split('.')[0]+'_length_distribution.png')

