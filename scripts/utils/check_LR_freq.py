import pandas as pd
import os

def main():
    # load CSV
    df = pd.read_csv('data/real/nnunet_faf_v0_dataset_v2_local.csv')

    # count frequencies of laterality
    laterality_counts = df['laterality'].value_counts()
    print("Laterality frequencies:")
    print(laterality_counts)

if __name__ == '__main__':
    main()