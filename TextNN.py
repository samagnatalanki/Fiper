
# coding: utf-8

# In[7]:


import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
from flask import Flask
from flask import request
app = Flask(__name__)

# In[8]:


from subprocess import check_output
#print(check_output(["ls", "/input"]).decode("utf8"))


# In[9]:


# load dataset
text=pd.read_csv("mbti_1.csv" ,index_col='type')
print(text.shape)
print(text[0:5])
print(text.iloc[2])


# In[10]:


from sklearn.preprocessing import LabelBinarizer

# One hot encode labels
labels=text.index.tolist()
encoder=LabelBinarizer(neg_label=0, pos_label=1, sparse_output=False)
labels=encoder.fit_transform(labels)
labels=np.array(labels)
print(labels[50:55])


# In[11]:


mbti_dict={0:'ENFJ',1:'ENFP',2:'ENTJ',3:'ENTP',4:'ESFJ',5:'ESFP',6:'ESTJ',7:'ESTP',8:'INFJ',9:'INFP',10:'INTJ',11:'INTP',12:'ISFJ',13:'ISFP',14:'ISFP',15:'ISTP'}


# In[12]:


import re

# Function to clean data ... will be useful later
def post_cleaner(post):
    """cleans individual posts`.
    Args:
        post-string
    Returns:
         cleaned up post`.
    """
    # Covert all uppercase characters to lower case
    post = post.lower() 
    
    # Remove |||
    post=post.replace('|||',"") 

    # Remove URLs, links etc
    post = re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''', '', post, flags=re.MULTILINE) 
    # This would have removed most of the links but probably not all 

    # Remove puntuations 
    puncs1=['@','#','$','%','^','&','*','(',')','-','_','+','=','{','}','[',']','|','\\','"',"'",';',':','<','>','/']
    for punc in puncs1:
        post=post.replace(punc,'') 

    puncs2=[',','.','?','!','\n']
    for punc in puncs2:
        post=post.replace(punc,' ') 
    # Remove extra white spaces
    post=re.sub( '\s+', ' ', post ).strip()
    return post


# In[13]:


# Clean up posts
# Covert pandas dataframe object to list. I prefer using lists for prepocessing. 
posts=text.posts.tolist()
posts=[post_cleaner(post) for post in posts]


# In[14]:


# Count total words
from collections import Counter

word_count=Counter()
for post in posts:
    word_count.update(post.split(" ")) #number of times each word occurs



# In[15]:


# Size of the vocabulary available to the RNN
vocab_len=len(word_count)
print(vocab_len)

print(len(posts[0]))


# In[16]:


# Create a look up table 
vocab = sorted(word_count, key=word_count.get, reverse=True)#sorts the words based on number of occurances
# Create your dictionary that maps vocab words to integers here
vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}#numbers each word in vocal from 1 to 172984

posts_ints=[]
for post in posts:
    posts_ints.append([vocab_to_int[word] for word in post.split()])#converts each occurance of word into int

print(posts_ints[0])
print(len(posts_ints[0]))


# In[17]:


posts_lens = Counter([len(x) for x in posts])
print("Zero-length reviews: {}".format(posts_lens[0]))
print("Maximum review length: {}".format(max(posts_lens)))
print("Minimum review length: {}".format(min(posts_lens)))

seq_len = 500
features=np.zeros((len(posts_ints),seq_len),dtype=int) #Return a new array of given shape and type, with zeros.
for i, row in enumerate(posts_ints):
    features[i, -len(row):] = np.array(row)[:seq_len] #taking first 500 words
print(features[:10])
pred1=features[0]
print(pred1)


# In[18]:


# Split data into training, test,training and validation

split_frac = 0.8

num_ele=int(split_frac*len(features))
rem_ele=len(features)-num_ele
train_x, val_x = features[:num_ele],features[num_ele:int(rem_ele/2)+num_ele]
train_y, val_y = labels[:num_ele],labels[num_ele:int(rem_ele/2)+num_ele]

test_x =features[num_ele+int(rem_ele/2):]
test_y = labels[num_ele+int(rem_ele/2):]

print("\t\t\tFeature Shapes:")
print("Train set: \t\t{}".format(train_x.shape), 
      "\nValidation set: \t{}".format(val_x.shape),
      "\nTest set: \t\t{}".format(test_x.shape))
print(test_x[0])
print(test_y[0])


# In[19]:


lstm_size = 256
lstm_layers = 1
batch_size = 256
learning_rate = 0.01
embed_dim=250


# In[20]:


n_words = len(vocab_to_int) + 1 # Adding 1 because we use 0's for padding, dictionary started at 1

# Create the graph object
graph = tf.Graph()
# Add nodes to the graph
with graph.as_default():
    input_data = tf.placeholder(tf.int32, [None, None], name='inputs')
    labels_ = tf.placeholder(tf.int32, [None, None], name='labels')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')


# In[21]:


# Embedding
with graph.as_default():
    embedding= tf.Variable(tf.random_uniform(shape=(n_words,embed_dim),minval=-1,maxval=1))
    embed=tf.nn.embedding_lookup(embedding,input_data)
    print(embed.shape)


# In[22]:


#LSTM cell
with graph.as_default():
    # basic LSTM cell
    lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
    
    # Add dropout to the cell
    drop = tf.contrib.rnn.DropoutWrapper(lstm,output_keep_prob=keep_prob)
    
    # Stack up multiple LSTM layers, for deep learning
    cell = tf.contrib.rnn.MultiRNNCell([drop]* lstm_layers)
    
    # Getting an initial state of all zeros
    initial_state = cell.zero_state(batch_size, tf.float32)


# In[23]:


with graph.as_default():
    outputs,final_state=tf.nn.dynamic_rnn(cell,embed,dtype=tf.float32 )
with graph.as_default():
    
    pre = tf.layers.dense(outputs[:,-1], 16, activation=tf.nn.relu)
    predictions=tf.layers.dense(pre, 16, activation=tf.nn.softmax)
    
    cost = tf.losses.mean_squared_error(labels_, predictions)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)


# In[24]:


with graph.as_default():
    correct_pred = tf.equal(tf.cast(tf.round(predictions), tf.int32), labels_)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# In[25]:


def get_batches(x, y, batch_size=100):
    
    n_batches = len(x)//batch_size
    x, y = x[:n_batches*batch_size], y[:n_batches*batch_size]
    for ii in range(0, len(x), batch_size):
        yield x[ii:ii+batch_size], y[ii:ii+batch_size]


# In[26]:


def get_batches1(x, y, batch_size=100):
    
    #n_batches = len(x)//batch_size
    #x, y = x[:n_batches*batch_size], y[:n_batches*batch_size]
    #for ii in range(0, len(x), batch_size):
    yield x[0], y[0]


# In[27]:


epochs = 3

with graph.as_default():
    saver = tf.train.Saver()

with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())
    iteration = 1
    for e in range(epochs):
        state = sess.run(initial_state)
        
        for ii, (x, y) in enumerate(get_batches(train_x, train_y, batch_size), 1):
            feed = {input_data: x,
                    labels_: y,
                    keep_prob: 1.0,
                    initial_state: state}
            loss, state, _ = sess.run([cost, final_state, optimizer], feed_dict=feed)
            
            if iteration%5==0:
                print("Epoch: {}/{}".format(e, epochs),
                      "Iteration: {}".format(iteration),
                      "Train loss: {:.3f}".format(loss))

            if iteration%25==0:
                val_acc = []
                val_state = sess.run(cell.zero_state(batch_size, tf.float32))
                for x, y in get_batches(val_x, val_y, batch_size):
                    feed = {input_data: x,
                            labels_: y,
                            keep_prob: 1,
                            initial_state: val_state}
                    batch_acc, val_state = sess.run([accuracy, final_state], feed_dict=feed)
                    val_acc.append(batch_acc)
                print("Val acc: {:.3f}".format(np.mean(val_acc)))
            iteration +=1
    saver.save(sess, "checkpoints/mbti.ckpt")


# In[28]:


test_acc = []
vall=[]
with tf.Session(graph=graph) as sess:
    saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))
    test_state = sess.run(cell.zero_state(batch_size, tf.float32))
    for ii, (x, y) in enumerate(get_batches(test_x, test_y, batch_size), 1):
        feed = {input_data: x,
                labels_: y,
                keep_prob: 1,
                initial_state: test_state}
        batch_acc, test_state = sess.run([accuracy, final_state], feed_dict=feed)
        vall, test_state = sess.run([predictions, final_state], feed_dict=feed)
        print(vall[0])
        test_acc.append(batch_acc)    
    print("Test accuracy: {:.3f}".format(np.mean(test_acc)))


def user_post_cleaner(post):
    """cleans individual posts`.
    Args:
        post-string
    Returns:
         cleaned up post`.
    """
    # Covert all uppercase characters to lower case
    post = post.lower() 
    
    # Remove |||
    post=post.replace('|||',"") 

    # Remove URLs, links etc
    post = re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''', '', post, flags=re.MULTILINE) 
    # This would have removed most of the links but probably not all 

    # Remove puntuations 
    puncs1=['@','#','$','%','^','&','*','(',')','-','_','+','=','{','}','[',']','|','\\','"',"'",';',':','<','>','/']
    for punc in puncs1:
        post=post.replace(punc,'') 

    puncs2=[',','.','?','!','\n']
    for punc in puncs2:
        post=post.replace(punc,' ') 
    # Remove extra white spaces
    post=re.sub( '\s+', ' ', post ).strip()
    return post


def predict_personality(text_entered):
    cleaned_text=user_post_cleaner(text_entered)
    inp_int=[]
    inp_int.append([vocab_to_int[word] for word in cleaned_text.split()])#converts each occurance of word into int
    print(len(inp_int[0]))
    x=inp_int[0]
    feat_inp=x[:500]
    print(len(feat_inp))
    inp_int.pop()
    inp_int.append(feat_inp)
    print(inp_int)
    #print(len(inp_int))
    label_inp=[]
    label_inp.append([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    print(label_inp)
    with tf.Session(graph=graph) as sess:
        saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))
        test_state = sess.run(cell.zero_state(batch_size, tf.float32))
        feed = {input_data: inp_int,
                labels_: label_inp,
                keep_prob: 1,
                initial_state: test_state}
        vall, test_state = sess.run([predictions, final_state], feed_dict=feed)
        print(vall[0])
        
    maxindex=np.argmax(vall[0])
    return mbti_dict[maxindex]


maxindex=np.argmax(vall[0])
print(mbti_dict[maxindex])

@app.route('/')
def display1():
    return "Sam......You're the best :)"

@app.route('/face')
def display2():
        str1=request.args.get('Landmarks')
        str1=str1.replace("%20"," ")
        for i in str1.split():
                print("\n")
                for j in i.split(','):
                        if j!='[' and j!=']':
                                s=((j.replace('(','')).replace(')',''))
                                print(float(s))
        return "Looks like it works for face"

@app.route('/text')
def display3():
    str2=request.args.get('Text')
    str2=str2.replace("%20"," ")
    op=predict_personality(str2);
    return op

if __name__=='__main__':
    app.run()

