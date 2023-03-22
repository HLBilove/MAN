# Code reused from  https://github.com/arghosh/AKT
import numpy as np
import math
class DATA(object):
    def __init__(self, n_skill, seqlen, separate_char, name="data"):
        self.separate_char = separate_char
        self.seqlen = seqlen
        self.n_skill = n_skill

    def load_data(self, path):
        file_data = open(path, 'r')
        q_data = []
        qa_data = []
        p_data = []
        for lineID, line in enumerate(file_data):
            line = line.strip()
            # lineID starts from 0
            if lineID % 4 == 0:
                learner_id = lineID//3
            if lineID % 4 == 1:
                S = line.split(self.separate_char)
                if len(S[len(S)-1]) == 0:
                    S = S[:-1]
            if lineID % 4 == 2:
                E = line.split(self.separate_char)
                if len(E[len(E)-1]) == 0:
                    E = E[:-1]
            if lineID % 4 == 3:
                A = line.split(self.separate_char)
                if len(A[len(A)-1]) == 0:
                    A = A[:-1]

                # start split the data
                n_split = 1
                if len(S) > self.seqlen:
                    n_split = math.floor(len(S) / self.seqlen)
                    if len(S) % self.seqlen:
                        n_split = n_split + 1           
                for k in range(n_split):
                    q_seq = []
                    p_seq = []
                    a_seq = []
                    if k == n_split - 1:
                        endINdex = len(A)
                    else:
                        endINdex = (k+1) * self.seqlen
                    for i in range(k * self.seqlen, endINdex):
                        if len(S[i]) > 0:
                            Xindex = int(S[i]) + round(float(A[i])) * self.n_skill
                            q_seq.append(int(S[i]))
                            p_seq.append(int(E[i]))
                            a_seq.append(Xindex)
                        else:
                            print(S[i])
                    q_data.append(q_seq)
                    qa_data.append(a_seq)
                    p_data.append(p_seq)

        file_data.close()
        ### data: [[],[],[],...] <-- set_max_seqlen is used
        # convert data into ndarrays for better speed during training
        q_dataArray = np.zeros((len(q_data), self.seqlen))
        for j in range(len(q_data)):
            dat = q_data[j]
            q_dataArray[j, :len(dat)] = dat

        qa_dataArray = np.zeros((len(qa_data), self.seqlen))
        for j in range(len(qa_data)):
            dat = qa_data[j]
            qa_dataArray[j, :len(dat)] = dat

        p_dataArray = np.zeros((len(p_data), self.seqlen))
        for j in range(len(p_data)):
            dat = p_data[j]
            p_dataArray[j, :len(dat)] = dat

        return q_dataArray, qa_dataArray, p_dataArray

    def load_test_data(self, path):
        file_data = open(path, 'r')
        q_data = []
        qa_data = []
        p_data = []
        test_e_num = 0
        for lineID, line in enumerate(file_data):
            line = line.strip()
            # lineID starts from 0
            if lineID % 4 == 0:
                learner_id = lineID//3
            if lineID % 4 == 1:
                S = line.split(self.separate_char)
                if len(S[len(S)-1]) == 0:
                    S = S[:-1]
            if lineID % 4 == 2:
                E = line.split(self.separate_char)
                if len(E[len(E)-1]) == 0:
                    E = E[:-1]
                test_e_num += len(E)
            if lineID % 4 == 3:
                A = line.split(self.separate_char)
                if len(A[len(A)-1]) == 0:
                    A = A[:-1]

                # start split the data
                n_split = 1
                if len(S) > self.seqlen:
                    n_split = math.floor(len(S) / self.seqlen)
                    if len(S) % self.seqlen:
                        n_split = n_split + 1           
                for k in range(n_split):
                    q_seq = []
                    p_seq = []
                    a_seq = []
                    if k == n_split - 1:
                        endINdex = len(A)
                    else:
                        endINdex = (k+1) * self.seqlen
                    for i in range(k * self.seqlen, endINdex):
                        if len(S[i]) > 0:
                            Xindex = int(S[i]) + round(float(A[i])) * self.n_skill
                            q_seq.append(int(S[i]))
                            p_seq.append(int(E[i]))
                            a_seq.append(Xindex)
                        else:
                            print(S[i])
                    q_data.append(q_seq)
                    qa_data.append(a_seq)
                    p_data.append(p_seq)

        file_data.close()
        ### data: [[],[],[],...] <-- set_max_seqlen is used
        # convert data into ndarrays for better speed during training
        q_dataArray = np.zeros((len(q_data), self.seqlen))
        for j in range(len(q_data)):
            dat = q_data[j]
            q_dataArray[j, :len(dat)] = dat

        qa_dataArray = np.zeros((len(qa_data), self.seqlen))
        for j in range(len(qa_data)):
            dat = qa_data[j]
            qa_dataArray[j, :len(dat)] = dat

        p_dataArray = np.zeros((len(p_data), self.seqlen))
        for j in range(len(p_data)):
            dat = p_data[j]
            p_dataArray[j, :len(dat)] = dat

        return q_dataArray, qa_dataArray, p_dataArray, test_e_num