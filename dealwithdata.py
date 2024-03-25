import numpy as np
import gensim
from gensim.models import Word2Vec
from keras.preprocessing.sequence import pad_sequences
import collections
from sklearn.model_selection import train_test_split

def get_1_trids():
    nucle_com = []
    chars = ['A', 'C', 'G', 'U']
    base = len(chars)
    end = len(chars) ** 1
    for i in range(0, end):
        n = i
        ch0 = chars[n % base]
        nucle_com.append(ch0)
    word_index = dict((w, i) for i, w in enumerate(nucle_com))
    return word_index

def get_2_trids():
    nucle_com = []
    chars = ['A', 'C', 'G', 'U']
    base = len(chars)
    end = len(chars) ** 2
    for i in range(0, end):
        n = i
        ch0 = chars[n % base]
        n = n // base
        ch1 = chars[n % base]
        nucle_com.append(ch0 + ch1)
    word_index = dict((w, i) for i, w in enumerate(nucle_com))
    return word_index

def frequency(seq, kmer, coden_dict):
    Value = []
    k = kmer
    coden_dict = coden_dict
    for i in range(len(seq) - int(k) + 1):
        kmer = seq[i:i + k]
        kmer_value = coden_dict[kmer.replace('T', 'U')]
        Value.append(kmer_value)
    freq_dict = dict(collections.Counter(Value))
    return freq_dict

def coden(seq,kmer,tris):
    coden_dict = tris
    freq_dict = frequency(seq,kmer,coden_dict)
    vectors = np.zeros((101, len(coden_dict.keys())))
    for i in range(len(seq) - int(kmer) + 1):
        vectors[i][coden_dict[seq[i:i + kmer].replace('T', 'U')]] = 1
    return vectors.tolist()

coden_dict1 = {'GCU': 0, 'GCC': 0, 'GCA': 0, 'GCG': 0,  # alanine<A>
              'UGU': 1, 'UGC': 1,  # systeine<C>
              'GAU': 2, 'GAC': 2,  # aspartic acid<D>
              'GAA': 3, 'GAG': 3,  # glutamic acid<E>
              'UUU': 4, 'UUC': 4,  # phenylanaline<F>
              'GGU': 5, 'GGC': 5, 'GGA': 5, 'GGG': 5,  # glycine<G>
              'CAU': 6, 'CAC': 6,  # histidine<H>
              'AUU': 7, 'AUC': 7, 'AUA': 7,  # isoleucine<I>
              'AAA': 8, 'AAG': 8,  # lycine<K>
              'UUA': 9, 'UUG': 9, 'CUU': 9, 'CUC': 9, 'CUA': 9, 'CUG': 9,  # leucine<L>
              'AUG': 10,  # methionine<M>
              'AAU': 11, 'AAC': 11,  # asparagine<N>
              'CCU': 12, 'CCC': 12, 'CCA': 12, 'CCG': 12,  # proline<P>
              'CAA': 13, 'CAG': 13,  # glutamine<Q>
              'CGU': 14, 'CGC': 14, 'CGA': 14, 'CGG': 14, 'AGA': 14, 'AGG': 14,  # arginine<R>
              'UCU': 15, 'UCC': 15, 'UCA': 15, 'UCG': 15, 'AGU': 15, 'AGC': 15,  # serine<S>
              'ACU': 16, 'ACC': 16, 'ACA': 16, 'ACG': 16,  # threonine<T>
              'GUU': 17, 'GUC': 17, 'GUA': 17, 'GUG': 17,  # valine<V>
              'UGG': 18,  # tryptophan<W>
              'UAU': 19, 'UAC': 19,  # tyrosine(Y)
              'UAA': 20, 'UAG': 20, 'UGA': 20,  # STOP code
              }

def coden1(seq):
    vectors = np.zeros((len(seq), 21))
    for i in range(len(seq) - 2):
        vectors[i][coden_dict1[seq[i:i + 3].replace('T', 'U')]] = 1
    return vectors.tolist()#矩阵转换为列表

def NCP(seq):
    phys_dic = {
        'A': [1, 1, 1],
        'U': [0, 0, 1],
        'C': [0, 1, 0],
        'G': [1, 0, 0]}
    seqLength = len(seq)
    sequence_vector = np.zeros([101, 3])
    for i in range(0, seqLength):
        sequence_vector[i, 0:3] = phys_dic[seq[i]]
    for i in range(seqLength, 101):
        sequence_vector[i, -1] = 1
    return sequence_vector


def nd(seq):
    seq = seq.strip()
    nd_list = [None] * 101
    for j in range(101):
        #print(seq[0:j])
        if seq[j] == 'A':
            nd_list[j] = round(seq[0:j+1].count('A') / (j + 1), 3)
        elif seq[j] == 'U':
            nd_list[j] = round(seq[0:j+1].count('U') / (j + 1), 3)
        elif seq[j] == 'C':
            nd_list[j] = round(seq[0:j+1].count('C') / (j + 1), 3)
        elif seq[j] == 'G':
            nd_list[j] = round(seq[0:j+1].count('G') / (j + 1), 3)
    return np.array(nd_list)

def dpcp(seq):
    phys_dic = {
        # Shift Slide Rise Tilt Roll Twist Stacking_energy Enthalpy Entropy Free_energy Hydrophilicity
        'AA': [-0.08, -1.27, 3.18, -0.8, 7, 31, -13.7, -6.6, -18.4, -0.93, 0.04],
        'AU': [-0.06, -1.36, 3.24, 1.1, 7.1, 33, -15.4, -5.7, -15.5, -1.1, 0.14],
        'AC': [0.23, -1.43, 3.24, 0.8, 4.8, 32, -13.8, -10.2, -26.2, -2.24, 0.14, ],
        'AG': [-0.04, -1.5, 3.3, 0.5, 8.5, 30, -14, -7.6, -19.2, -2.08, 0.08],
        'UA': [0.07, -1.7, 3.38, 1.3, 9.4, 32, -14.2, -13.3, -35.5, -2.35, 0.1],
        'UU': [0.23, -1.43, 3.24, 0.8, 4.8, 32, -13.8, -10.2, -26.2, -2.24, 0.27],
        'UC': [0.07, -1.39, 3.22, 0, 6.1, 35, -16.9, -14.2, -34.9, -3.42, 0.26],
        'UG': [-0.01, -1.78, 3.32, 0.3, 12.1, 32, -11.1, -12.2, -29.7, -3.26, 0.17],
        'CA': [0.11, -1.46, 3.09, 1, 9.9, 31, -14.4, -10.5, -27.8, -2.11, 0.21],
        'CU': [-0.04, -1.5, 3.3, 0.5, 8.5, 30, -14, -7.6, -19.2, -2.08, 0.52],
        'CC': [-0.01, -1.78, 3.32, 0.3, 8.7, 32, -11.1, -12.2, -29.7, -3.26, 0.49],
        'CG': [0.3, -1.89, 3.3, -0.1, 12.1, 27, -15.6, -8, -19.4, -2.36, 0.35],
        'GA': [-0.02, -1.45, 3.26, -0.2, 10.7, 32, -16, -8.1, -22.6, -1.33, 0.21],
        'GU': [-0.08, -1.27, 3.18, -0.8, 7, 31, -13.7, -6.6, -18.4, -0.93, 0.44],
        'GC': [0.07, -1.7, 3.38, 1.3, 9.4, 32, -14.2, -10.2, -26.2, -2.35, 0.48],
        'GG': [0.11, -1.46, 3.09, 1, 9.9, 31, -14.4, -7.6, -19.2, -2.11, 0.34]}

    seqLength = len(seq)
    sequence_vector = np.zeros([101, 11])
    k = 2
    for i in range(0, seqLength - 1):
        sequence_vector[i, 0:11] = phys_dic[seq[i:i + k]]
    return sequence_vector / 101



# getCircRNA2Vec ( 输入 x 个序列，序列长为y，则输出为 x * y 以及 x * 30)
def seq2ngram(seqs, k, s, wv):
    list22 = []
    # print('need to n-gram %d lines' % len(seqs))
    for num, line in enumerate(seqs):
        if num < 3000000:
            line = line.strip()
            l = len(line)
            list2 = []
            for i in range(0, l, s):
                if i + k >= l + 1:
                    break
                list2.append(line[i:i + k])
            list22.append(convert_data_to_index(list2, wv))
    return list22

def convert_data_to_index(string_data, wv):
    index_data = []
    for word in string_data:
        if word in wv:
            index_data.append(wv.vocab[word].index)
    return index_data

def dealwithCircRNA2Vec(seq):
    dataX = []
    dataX.append(seq)
    dataX = np.array(dataX)

    k = 10
    s = 1
    vector_dim = 30
    MAX_LEN = 101
    model1 = gensim.models.Doc2Vec.load('../circRNA2Vec/circRNA2Vec_model')
    seqs = seq2ngram(dataX, k, s, model1.wv)
    dataX = pad_sequences(seqs, maxlen=MAX_LEN, padding='post')
    dataX = np.array(dataX)
    return dataX[-1]


#secondary structu with seq
coden_dict2 = {'AF': 0, 'AT': 1, 'AI': 2, 'AH': 3, 'AM': 4, 'AS': 5, 'CF': 6, 'CT': 7, 'CI': 8, 'CH': 9, 'CM': 10,
               'CS': 11, 'GF': 12, 'GT': 13, 'GI': 14, 'GH': 15, 'GM': 16, 'GS': 17, 'UF': 18, 'UT': 19, 'UI': 20,
               'UH': 21, 'UM': 22, 'US': 23, }


def coden2(useful, ignore):
    vectors = np.zeros((len(useful), 24))
    for i in range(len(useful)):
        vectors[i][coden_dict2[useful[i] + ignore[i]]] = 1
    return vectors.tolist()

def dealwithdata(protein):
    tris1 = get_1_trids()
    tris2 = get_2_trids()
    dataX = []
    dataY = []
    count = 0
    with open('../Datasets/circRNA-RBP/'+ protein + '/positive_sec') as f:
        for line in f:
            if '>' not in line:
                count += 1
                if count == 1:
                    useful = line.strip()
                if count == 2:
                    ignore = line.strip()
                    probStr = coden2(useful, ignore) #SecStr
                    line = useful.replace('T', 'U').strip()
                    prob_Vec = dealwithCircRNA2Vec(line)
                    probStr_Vec = np.column_stack((probStr,prob_Vec))
                    prob_NCP = NCP(line) #NCP
                    probStr_VecNCP = np.column_stack((probStr_Vec,prob_NCP))
                    prob_ND = nd(line) #ND
                    probStr_VecNDCP = np.column_stack((probStr_VecNCP, prob_ND))
                    prob_DPCP = dpcp(line)  #DPCP
                    prob_VecNDPCP = np.column_stack((probStr_VecNDCP, prob_DPCP))
                    kmer = coden1(line.strip())
                    Feature_Encoding = np.column_stack((prob_VecNDPCP, Kmer))
                    dataX.append(Feature_Encoding.tolist())
                    dataY.append([0,1])
                    useful = ''
                    ignore = ''
                    count = 0
    with open('../Datasets/circRNA-RBP/' +protein + '/negative_sec') as f:
        for line in f:
            if '>' not in line:
                count += 1
                if count == 1:
                    useful = line.strip()
                if count == 2:
                    ignore = line.strip()
                    probStr = coden2(useful, ignore) #SecStr
                    line = useful.replace('T', 'U').strip()
                    prob_Vec = dealwithCircRNA2Vec(line)
                    probStr_Vec = np.column_stack((probStr,prob_Vec))
                    prob_NCP = NCP(line) #NCP
                    probStr_VecNCP = np.column_stack((probStr_Vec,prob_NCP))
                    prob_ND = nd(line) #ND
                    probStr_VecNDCP = np.column_stack((probStr_VecNCP, prob_ND))
                    prob_DPCP = dpcp(line)  #DPCP
                    prob_VecNDPCP = np.column_stack((probStr_VecNDCP, prob_DPCP))
                    kmer = coden1(line.strip())
                    Feature_Encoding = np.column_stack((prob_VecNDPCP, Kmer))
                    dataX.append(Feature_Encoding.tolist())
                    dataY.append([0,1])
                    useful = ''
                    ignore = ''
                    count = 0
    indexes = np.random.choice(len(dataY), len(dataY), replace=False)
    dataX = np.array(dataX)[indexes]
    dataY = np.array(dataY)[indexes]
    train_X, test_X, train_y, test_y = train_test_split(dataX, dataY, test_size=0.2)
    return train_X, test_X, train_y, test_y



# train_X, test_X, train_y, test_y=dealwithdata('test')
# print(train_X.shape)






