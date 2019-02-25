
# Importing Essential Files
import numpy as np
import pandas as pd
import  _pickle as cpickle
import matplotlib.pyplot as plt
import seaborn as sb
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing import sequence
from keras.utils import to_categorical
import os
from keras.models import Sequential, Model, load_model
from keras.layers import Dense,Flatten,LSTM,Conv1D,GlobalMaxPool1D,Dropout, Input, concatenate
from keras.layers.embeddings import Embedding
from keras import optimizers
from keras.callbacks import TensorBoard, CSVLogger, ModelCheckpoint
from keras.utils.vis_utils import plot_model
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.layers import Input


# New Coloumn Names to the Given tsv Format
cols = ['Sno','Label','Statement','Subject','Speaker','job title','State_info','Party_aff','barely_true_count','false_counts','half_true_counts','mostly_true_counts','pants_on_fire_counts','venue']

# Reading train.tsv File Along with the Columns
train_df = pd.read_csv('Dataset/train.tsv', sep='\t',names = cols)

# Reading test.tsv File Along with the Columns
test_df = pd.read_csv('Dataset/test.tsv', sep='\t',names = cols)

# Reading valid.tsv File Along with the Columns
valid_df = pd.read_csv('Dataset/valid.tsv', sep='\t',names = cols)

''' Uncomment to Diplay Original Data
    train_df.head(2) '''


''' Dataset - Filtering
    1. Drop Useless Columns,
        i. Sno
        ii. barely_true_count
        iii. false_counts
        iv. half_true_counts
        v. mostly_true_counts
        vi. pants_on_fire_counts
    2. Converting Multinary Classification to Binary Classification
        The Given Dataset Contains 6 Classes as Labels - True, Mostly-True, 
        Half-True, Barely-True, False, Pants-Fire. So converting into Binary-Classification such that:
    
         # Previous-Label    :    Converted-Label
             True            --        True
             Mostly-true     --        True
             Half-true       --        True
             Barely-true     --        False
             False           --        False
             Pants-fire      --        False '''

# Dropping Some Useless Columns from the Dataset
def drop_useless(data):
    data = data.drop(['Sno'],axis=1)
    data = data.drop(['barely_true_count'],axis=1)
    data = data.drop(['false_counts'],axis=1)
    data = data.drop(['half_true_counts'],axis=1)
    data = data.drop(['mostly_true_counts'],axis=1)
    data = data.drop(['pants_on_fire_counts'],axis=1)
    return data

# Converting Six Class Classification to Binary Classification 
def change_Label(data):
    data.Label[data.Label == 'false'] = False
    data.Label[data.Label == 'half-true'] = True
    data.Label[data.Label == 'true'] = True
    data.Label[data.Label == 'mostly-true'] = True
    data.Label[data.Label == 'barely-true'] = True
    data.Label[data.Label == 'pants-fire'] = False
    return data

# Call the Function to Drop Colomns
# Note: Run this code only once Otherwise it will give Error.
# Solution : If error is Flashing, then re-Run the Upper Cells again.
train_df = drop_useless(train_df)
test_df = drop_useless(test_df)
valid_df = drop_useless(valid_df)

# Call the Function to Change the Labels
train_df = change_Label(train_df)
test_df = change_Label(test_df)
valid_df = change_Label(valid_df)

''' Uncomment to dislay Training Dataset - After Changes
train_df.head() '''

''' Uncomment to dislay Testing Dataset - After Changes
    test_df.head() '''

''' Validation Dataset - After Changes 
    valid_df.head()
    print(valid_df.shape) '''

# Saving tsv to csv File in the Directory
train_df.to_csv('train_data.csv', sep='\t', encoding='utf-8',index=False)
test_df.to_csv('test_data.csv', sep='\t', encoding='utf-8',index=False)
valid_df.to_csv('valid_data.csv', sep='\t', encoding='utf-8',index=False)

# Reading the New CSV Files from the Directory
trainData = pd.read_csv('./train_data.csv',sep='\t')
testData = pd.read_csv('./test_data.csv',sep='\t')
validData = pd.read_csv('./valid_data.csv',sep='\t')

# PLotting Count-Plot of the Testing, Training and Validation Dataset
def plot_count(data):
    sb.countplot(x = 'Label',data = data,palette='hls')
    plt.show()

# PLotting train Data
plot_count(trainData)

# PLotting the testing Data
plot_count(testData)

# Plotting Validation Data
plot_count(validData)

''' Topic : One - Hot Encoding of the Given Columns
    The new colomns as explained below are attached in the previous data frame
    and are saved in a new folder named as encoded dataset.
    
    Old Colmns      -  Encoded colmns
    1. job title    -    job_id
    2. Party_aff    -   party_id
    3. State_info   -   state_id
    4. Subject      -   subject_id
    5. venue        -   venue_id
    6. Speaker      -   speaker_id '''

''' Creating Encoded colmn Label_id for the Label colmn '''
labels_dict = {True:1,False:0} # Mapping True and False with Integer Value 
label_reverse_arr = [True,False] # For reverse Mapping

''' Creating Colmn of Label_id for the Train, Test and Valid Dataset '''
trainData['Label_id'] = trainData['Label'].apply(lambda x: labels_dict[x]) 
validData['Label_id'] = validData['Label'].apply(lambda x: labels_dict[x])
testData['Label_id'] = testData['Label'].apply(lambda x: labels_dict[x])


''' Uncomment to see the Label_id Colmn
    validData.head() ''' 

''' Creating Encoded colmn speaker_id for the Label Speaker '''
    
''' Given Dataset contains more then 200 Speakers, hence Shorlisting some from the given pool '''
speakers = ['barack-obama', 'donald-trump', 'hillary-clinton', 'mitt-romney', 
            'scott-walker', 'john-mccain', 'rick-perry', 'chain-email', 
            'marco-rubio', 'rick-scott', 'ted-cruz', 'bernie-s', 'chris-christie', 
            'facebook-posts', 'charlie-crist', 'newt-gingrich', 'jeb-bush', 
            'joe-biden', 'blog-posting','paul-ryan']

''' To store the mapping of each Speaker with an Integer Value'''
speaker_dict = {}
for cnt,speaker in enumerate(speakers):
    speaker = speaker.lower()
    speaker_dict[speaker] = cnt
    
''' Uncomment to print the dictionary 
    print(speaker_dict) '''

''' Function defined to Create Encoded Colomn speaker_id'''
def map_speaker(speaker):
    basestring=(str,bytes)
    if isinstance(speaker,basestring):
        speaker = speaker.lower()
        matches = [s for s in speakers if s in speaker]
        
        if len(matches) > 0:
            return speaker_dict[matches[0]]
        else:
            return len(speakers)
    else:
        return len(speakers)

''' Creating Encoded colmn job_id for the Label job title '''
    
''' Given Dataset contains many jobs type, hence Shorlisting some from the given pool '''
job_list = ['president', 'u.s. senator', 'governor', 'president-elect', 'presidential candidate', 
            'u.s. representative', 'state senator', 'attorney', 'state representative', 'congress']

''' To store the mapping of each Speaker with an Integer Value'''
job_dict = {}
for cnt,job in enumerate(job_list):
    job = job.lower()
    job_dict[job] = cnt

''' Function defined to Create Encoded Colomn job_id'''
def map_job(job):
    basestring=(str,bytes)
    if isinstance(job,basestring):
        job = job.lower()
        matches = [s for s in job_list if s in job]
        
        if len(matches) > 0:
            return job_dict[matches[0]]
        else:
            return 10
    else:
        return 10

''' Creating Encoded colmn party_id for the Label Party_aff '''
    
''' Shortlisting some Parties from the given pool '''
party_list = trainData['Party_aff'].value_counts()
party_dict = {'republican':0,'democrat':1,'none':2,'organization':3,'newsmaker':4}

''' Function defined to Create Encoded Colomn party_id'''
def map_party(party):
    if party in party_dict:
        return party_dict[party]
    else:
        return 5

''' Creating Encoded colmn state_id for the colmn State'''

''' Given Dataset contains many sates type, hence Shorlisting some from the given pool '''
states = ['Alabama','Alaska','Arizona','Arkansas','California','Colorado',
         'Connecticut','Delaware','Florida','Georgia','Hawaii','Idaho', 
         'Illinois','Indiana','Iowa','Kansas','Kentucky','Louisiana',
         'Maine' 'Maryland','Massachusetts','Michigan','Minnesota',
         'Mississippi', 'Missouri','Montana','Nebraska','Nevada',
         'New Hampshire','New Jersey','New Mexico','New York',
         'North Carolina','North Dakota','Ohio',    
         'Oklahoma','Oregon','Pennsylvania','Rhode Island',
         'South  Carolina','South Dakota','Tennessee','Texas','Utah',
         'Vermont','Virginia','Washington','West Virginia',
         'Wisconsin','Wyoming']

''' To store the mapping of each Speaker with an Integer Value'''
states_dict = {}
for cnt,state in enumerate(states):
    state = state.lower()
    states_dict[state] = cnt

# print(states_dict)
''' Function defined to Create Encoded Colomn state_id'''
def map_state(state):
    basestring = (str,bytes)
    if isinstance(state, basestring):
        state = state.lower()
        if state in states_dict:
            return states_dict[state]
        else:
            if 'Washington' in state:
                return states_dict['Washington']
            else:
                return 50 
    else:
        return 50

''' Creating Encoded colmn state_id for the colmn Subject'''

''' Given Dataset contains many subjects type, hence Shorlisting some from the given pool '''
subject_list = ['health','tax','immigration','election','education',
'candidates-biography','economy','gun','jobs','federal-budget','energy','abortion','foreign-policy']

''' To store the mapping of each Speaker with an Integer Value'''
subject_dict = {}
for cnt,subject in enumerate(subject_list):
    subject = subject.lower()
    subject_dict[subject] = cnt

''' Function defined to Create Encoded Colomn state_id'''
def map_subject(subject):
    basestring = (str,bytes)
    if isinstance(subject, basestring):
        subject = subject.lower()
        matches = [s for s in subject_list if s in subject]
        if len(matches) > 0:
            return subject_dict[matches[0]] #Return index of first match
        else:
            return 13 #This maps any other subject to index 13
    else:
        return 13 #Nans or un-string data goes here.

''' Creating Encoded colmn venue_id for the colmn venue'''

''' Given Dataset contains many venues type, hence Shorlisting some from the given pool '''

venue_list = ['news release','interview','tv','radio',
              'campaign','news conference','press conference','press release',
              'tweet','facebook','email']

''' To store the mapping of each venue with an Integer Value'''
venue_dict = {}
for cnt,venue in enumerate(venue_list):
    venue = venue.lower()
    venue_dict[venue] = cnt

''' Function defined to Create Encoded Colomn venue_id'''
def map_venue(venue):
    basestring = (str,bytes)
    if isinstance(venue, basestring):
        venue = venue.lower()
        matches = [s for s in venue_list if s in venue]
        if len(matches) > 0:
            return venue_dict[matches[0]] #Return index of first match
        else:
            return 11 #This maps any other venue to index 11
    else:
        return 11 #Nans or un-string data goes here.

''' Tokenizing each statement from the Colmn Statement and 
    Saving the pickle file in the directory '''

#Tokenize statement and vocab test
vocab_dict = {}

''' Using os module to read if an pickle file already 
    exists otherwise creating a new File'''
if not os.path.exists('./Pickle File/vocab.p'):
    t = Tokenizer()
    t.fit_on_texts(trainData['Statement'])
    vocab_dict = t.word_index
    cpickle.dump( t.word_index, open( "vocab.p", "wb" ))
    print('Vocab dict is created')
    print('Saved vocab dict to pickle file')
else:
    print('Loading vocab dict from pickle file')
    vocab_dict = cpickle.load(open("./Pickle File/vocab.p", "rb" ))

''' Processing the Statement Colomn '''
def prepocess_statement(statement):
    text = text_to_word_sequence(statement)
    val = [0] * 10
    val = [vocab_dict[t] for t in text if t in vocab_dict]
    return val

''' Function for preprocessing for the train,test and validation Dataset '''
def preprocesing(data):
    data['job_id'] = data['job title'].apply(map_job) #Job
    data['party_id'] = data['Party_aff'].apply(map_party) #Party
    data['state_id'] = data['State_info'].apply(map_state) #State
    data['subject_id'] = data['Subject'].apply(map_subject) #Subject
    data['venue_id'] = data['venue'].apply(map_venue) #Venue
    data['speaker_id'] = data['Speaker'].apply(map_speaker) #Speaker
    data['word_ids'] = data['Statement'].apply(prepocess_statement)

''' Get all preprocessing done for the train,test and validation Dataset '''
preprocesing(trainData)
preprocesing(validData)
preprocesing(testData)

# Saving csv File in the Directory
trainData.to_csv('train_data.csv', sep='\t', encoding='utf-8',index=False)
testData.to_csv('test_data.csv', sep='\t', encoding='utf-8',index=False)
validData.to_csv('valid_data.csv', sep='\t', encoding='utf-8',index=False)

'''Initializing the Hyper Parameters'''
vocab_length = len(vocab_dict.keys())
hidden_size = 100 #Has to be same as EMBEDDING_DIM
lstm_size = 100
num_steps = 25
num_epochs = 100
batch_size = 40
kernel_sizes = [2,5,8]
filter_size = 128
#Meta data related hyper params
num_party = 6
num_state = 51
num_venue = 12
num_job = 11
num_sub = 14
num_speaker = 21

''' Filtering the x_train and y_train from the Dataset'''
x_train = trainData['word_ids']
y_train = trainData['Label_id']
x_val = validData['word_ids']
y_val = validData['Label_id']
x_test = testData['word_ids']
y_test = testData['Label_id']

''' Making all of the Vectors to of same size '''
x_train = sequence.pad_sequences(x_train, maxlen=num_steps, padding='post',truncating='post')
y_train = to_categorical(y_train, num_classes=6)
x_val = sequence.pad_sequences(x_val, maxlen=num_steps, padding='post',truncating='post')
y_val = to_categorical(y_val, num_classes=6)
x_test = sequence.pad_sequences(x_test, maxlen=num_steps, padding='post',truncating='post')

''' Making One-Hot Encoding of the Rest of the colomns'''
feature1 = to_categorical(trainData['party_id'],num_classes=num_party)
feature2  = to_categorical(trainData['state_id'],num_classes=num_state)
feature3  = to_categorical(trainData['venue_id'],num_classes=num_venue)
feature4  = to_categorical(trainData['job_id'],num_classes=num_job)
feature5  = to_categorical(trainData['subject_id'],num_classes=num_sub)
feature6  = to_categorical(trainData['speaker_id'],num_classes=num_speaker)
    
''' Stacking the Rest of the converted Colomn and Making a new Train Data'''
X_train_meta = np.hstack((feature1,feature2,feature3,feature4,feature5,feature6))#concat a and b

''' Making One-Hot Encoding of the Rest of the colomns'''
feature1 = to_categorical(testData['party_id'],num_classes=num_party)
feature2  = to_categorical(testData['state_id'],num_classes=num_state)
feature3  = to_categorical(testData['venue_id'],num_classes=num_venue)
feature4  = to_categorical(testData['job_id'],num_classes=num_job)
feature5  = to_categorical(testData['subject_id'],num_classes=num_sub)
feature6  = to_categorical(testData['speaker_id'],num_classes=num_speaker)
    
''' Stacking the Rest of the converted Colomn and Making a new Test Data'''
X_test_meta = np.hstack((feature1,feature2,feature3,feature4,feature5,feature6))#concat a and b

''' Making One-Hot Encoding of the Rest of the colomns'''
feature1 = to_categorical(validData['party_id'],num_classes=num_party)
feature2  = to_categorical(validData['state_id'],num_classes=num_state)
feature3  = to_categorical(validData['venue_id'],num_classes=num_venue)
feature4  = to_categorical(validData['job_id'],num_classes=num_job)
feature5  = to_categorical(validData['subject_id'],num_classes=num_sub)
feature6  = to_categorical(validData['speaker_id'],num_classes=num_speaker)

''' Stacking the Rest of the converted Colomn and Making a new Valid Data'''
X_valid_meta = np.hstack((feature1,feature2,feature3,feature4,feature5,feature6))#concat a and b

''' Using Glove Vector to achieve the Vector for each statement'''
embedding_index = {}

''' Reading the Glove text from the Directory'''
with open("../glove.6B.100d.txt") as fp:
    for line in fp:
        values = line.split()
        vectors = np.asarray(values[1:],dtype='float32')
        embedding_index[values[0].lower()] = vectors
print("Found %s word vectors"%len(embedding_index))

''' Setting Constant for Dimention to 100 '''
EMBEDDING_DIM = 100

'''Creating Embedding Matrix and adding the Vector '''
num_words = len(vocab_dict) + 1
embedding_matrix = np.zeros((num_words,EMBEDDING_DIM))

for word, i in vocab_dict.items():
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

embedding_index = None

''' Checking Converted x_train'''
x_train


# LSTM Model

''' Initializing Object of Sequential Model'''
model = Sequential()
model.add(Embedding(vocab_length+1, hidden_size, input_length=num_steps))
model.add(LSTM(hidden_size))
model.add(Dense(6, activation='softmax'))

''' Preparation and instantiating LSTM Object '''

statement_input = Input(shape=(num_steps,), dtype='int32', name='main_input')
x = Embedding(vocab_length+1,EMBEDDING_DIM,weights=[embedding_matrix],input_length=num_steps,trainable=False)(statement_input) #Preloaded glove embeddings
lstm_in = LSTM(lstm_size,dropout=0.2)(x)
meta_input = Input(shape=(X_train_meta.shape[1],), name='aux_input')
x_meta = Dense(64, activation='relu')(meta_input)
x = concatenate([lstm_in, x_meta])
main_output = Dense(6, activation='softmax', name='main_output')(x)
model = Model(inputs=[statement_input, meta_input], outputs=[main_output])


'''Defining a complex model. Adding meta data features to the model (CNN Based) '''

kernel_arr = []
statement_input = Input(shape=(num_steps,), dtype='int32', name='main_input')
x = Embedding(output_dim=hidden_size, input_dim=vocab_length+1, input_length=num_steps)(statement_input) #Train embeddings from scratch

for kernel in kernel_sizes:
    x_1 = Conv1D(filters=filter_size,kernel_size=kernel)(x)
    x_1 = GlobalMaxPool1D()(x_1)
    kernel_arr.append(x_1)
conv_in = concatenate(kernel_arr)
conv_in = Dropout(0.6)(conv_in)
conv_in = Dense(128, activation='relu')(conv_in)

''' Meta input '''
meta_input = Input(shape=(X_train_meta.shape[1],), name='aux_input')
x_meta = Dense(64, activation='relu')(meta_input)
x = concatenate([conv_in, x_meta])
main_output = Dense(6, activation='softmax', name='main_output')(x)
model = Model(inputs=[statement_input, meta_input], outputs=[main_output])

''' Importing Essetial Filed Required for the Model'''

plot_model(model, to_file='model_lstm.png', show_shapes=True, show_layer_names=True)
SVG(model_to_dot(model,show_shapes=True).create(prog='dot', format='svg'))


'''Compile model and print summary'''
'''Define specific optimizer to counter over-fitting'''
sgd = optimizers.SGD(lr=0.025, clipvalue=0.3, nesterov=True)
adam = optimizers.Adam(lr=0.000075, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
model.compile(optimizer=sgd,loss='categorical_crossentropy',metrics=['categorical_accuracy'])
print(model.summary())


''' The computations you'll use TensorFlow for - like training a massive deep neural 
    network - can be complex and confusing. To make it easier to understand, debug, 
    and optimize TensorFlow programs, we've included a suite of visualization tools 
    called TensorBoard. You can use TensorBoard to visualize your TensorFlow graph, 
    plot quantitative metrics about the execution of your graph, and show additional 
    data like images that pass through it. When TensorBoard is fully configured, '''

tb = TensorBoard()
csv_logger = CSVLogger('./logs/events.out.tfevents.1551048190.rashid-ideapad')
filepath= "weights.best.hdf5"
#Or use val_loss depending on whatever the heck you want
checkpoint = ModelCheckpoint(filepath, monitor='val_categorical_accuracy', verbose=1, save_best_only=True, mode='max')

''' Training of the Model on the Training DataSet '''
model.fit({'main_input': x_train, 'aux_input': X_train_meta},
          {'main_output': y_train},epochs=num_epochs, batch_size=batch_size,
          validation_data=({'main_input': x_val, 'aux_input': X_valid_meta},{'main_output': y_val}),
         callbacks=[tb,csv_logger,checkpoint])


# ##### Results #####
# 
# -  ```Loss``` : 0.0061
# - ```Categorical_accuracy```  :  0.9992
# - ```Val_loss``` - 1.5211
# - ```val_categorical_accuracy```  :  0.6931

'''Load The Pre-Trained Model'''
model1 = load_model('./weights.best.hdf5')

'''Make predictions on test set'''
preds = model1.predict([x_test,X_test_meta], batch_size=batch_size, verbose=1)

''' To Print the Total len of preds ''' 
print(len(preds))
print(label_reverse_arr[np.argmax(preds[0])])

'''Write data to file'''
vf = open('predictions.txt', 'w+')
counter = 0
for pred in preds:  
    line_string = label_reverse_arr[np.argmax(pred)]
    vf.write(str(line_string)+'\n')
    counter += 1
print(counter)
print("Predictions written to file")
vf.close()

