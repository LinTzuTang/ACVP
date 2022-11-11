from fasta_tools import read_fasta, save_fasta
from protein_encoding import PC_6
import numpy as np
import argparse
import pandas as pd
from tensorflow.keras.models import load_model


def main(input_fasta_name,output_csv_name):
    # translate fasta file to PC_6 encode np.array format
    g_encoded = PC_6(input_fasta_name)
    g_array = np.array(list(g_encoded.values())).astype('float32')
    # load model
    model = load_model('ACVP_transfer.h5')
    # predict
    score = model.predict(g_array)
    classifier = score>0.5

    # make dataframe
    df = pd.DataFrame(score)
    df.insert(0,'Peptide' ,g_encoded.keys())
    df.insert(2,'Prediction results', classifier)
    df['Prediction results'] = df['Prediction results'].replace({True: 'Yes', False: 'No'})
    df = df.rename({0:'Score'}, axis=1)
    # output csv
    df.to_csv(output_csv_name)

#arg
if __name__ == '__main__':   
    parser = argparse.ArgumentParser(description='ACVP predictor')
    parser.add_argument('-f','--fasta_name',help='input fasta name',required=True)
    parser.add_argument('-o','--output_csv',help='output csv name',required=True)
    args = parser.parse_args()
    input_fasta_name = args.fasta_name
    output_csv_name =  args.output_csv
    main(input_fasta_name,output_csv_name)