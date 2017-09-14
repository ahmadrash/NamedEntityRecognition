import string
import data_process
import sys
import subprocess

TRAINING_FILENAME = 'crf_training.txt'
TEST_FILENAME = 'crf_test.txt'
MODEL_FILENAME= 'model'
TEMPLATE_FILENAME = 'template.txt'
RESULT_FILENAME = 'result.txt'
QUERY_FILENAME = 'query.txt'

filename = data_process.TAGGED_FILENAME
training_size = data_process.TRAINING_DATA

def getAccuracy():

    inFile = open(RESULT_FILENAME, 'r')
    sentenceList = inFile.read().split('\n\n')

    if sentenceList[-1] == "":
        del sentenceList[-1]

    count_1 = 0.0
    count_2 = 0.0

    for i in range(len(sentenceList)):

        word = sentenceList[i].split()
        count_1 = count_1 + len(word)/3
        for j in range(0,len(word)-2,3):

            if word[j+1] != word[j+2]:

                count_2 = count_2 + 1

    return (count_1-count_2)/count_1 * 100

sentence_list = data_process.loadData()

num_setences = len(sentence_list)

training_size = int(data_process.TRAINING_DATA *num_setences)

f = open(TRAINING_FILENAME,'w')

for i in range(training_size):

    f.write(sentence_list[i] + '\n\n')
   
f.close


f = open(TEST_FILENAME,'w')

for i in range(training_size - 1,num_setences):

    f.write(sentence_list[i] + '\n\n')
    
f.flush()
f.close

log = open(RESULT_FILENAME, "w")
p = subprocess.Popen(["crf_learn",'-t', TEMPLATE_FILENAME, TRAINING_FILENAME, MODEL_FILENAME])
q = subprocess.Popen(["crf_test", '-m',MODEL_FILENAME, TEST_FILENAME], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

for line in q.stdout:
    sys.stdout.write(line)
    log.write(line)
q.wait()
log.close

print('The accurary is ' + str(getAccuracy()) + ' percent.')

while True:

    response = (raw_input('Type a query(type e to exit): '))

    if response == 'e':
        sys.exit()

    file_name = QUERY_FILENAME+str(count)+'.txt'
            
    word_list = response.split()

    f = open(QUERY_FILENAME,'w')

    for i in range(len(word_list)):

        f.write(word_list[i] + '\n')
    
    f.flush()
    f.close

    q = subprocess.Popen(["crf_test", '-m',MODEL_FILENAME, QUERY_FILENAME], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    for line in q.stdout:
        sys.stdout.write(line)
    q.wait()
    count = count + 1


    



