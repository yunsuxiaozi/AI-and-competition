"""
--requirement 包含依赖项列表的路径文件,这个txt文件就像是购物清单,告诉你要下载哪些库以及对应的版本.
ftfy
tabpfn
iterative-stratification
pytorch-tabnet==4.1.0
lightgbm==3.3.2
-q:减少输出信息
--no-index:不从互联网上下载
--find-links file:../input/offline-package-wheeler/是根据所给的相对路径安装
"""

!pip install -q --requirement ../input/offline-package-wheeler/requirements.txt  \
--no-index --find-links file:../input/offline-package-wheeler/

#necessary
import pandas as pd#导入csv文件的库
import numpy as np#进行矩阵运算的库
import scipy#基于numpy的科学计算库
#Models
import lightgbm as lgb
import xgboost as xgb
import catboost as catbst
#other
import copy#提供了用于复制对象的功能,(浅拷贝copy如果原始对象改变,这个拷贝也会变,深拷贝deepcopy则不会受到影响.)
import os#与操作系统进行交互的库
import re#用于正则表达式提取
import regex#比re库更全面的正则表达式库
import shutil#执行文件的复制、移动、删除、重命名操作.
#用于对一组元素计数,OrderedDict是有序的字典
from collections import Counter,OrderedDict
from glob import glob#在某个文件目录下用类似正则表达式的方法匹配符合条件的文件
import dill#对对象进行序列化和反序列化(例如保存和加载树模型)
import sklearn#开源的机器学习库
#集成学习的分类器和回归器
from sklearn.ensemble import BaggingClassifier,BaggingRegressor
#机器学习的评估指标auc和mse
from sklearn.metrics import roc_auc_score,mean_squared_error
#KFold是直接分成k折,StratifiedKFold还要考虑每种类别的占比
from sklearn.model_selection import StratifiedKFold
#在jupyter notebook里显示进度条的库
from tqdm.notebook import tqdm
#将文本数据转换成tfidf矩阵
from sklearn.feature_extraction.text import TfidfVectorizer
#截断奇异值分解的降维技术
from sklearn.decomposition import TruncatedSVD
import ftfy#“fixes text for you”(自动修复文本),自动修复各种文本编码问题
import warnings#避免一些可以忽略的报错
warnings.filterwarnings('ignore')#filterwarnings()方法是用于设置警告过滤器的方法，它可以控制警告信息的输出方式和级别。
#关闭文本分词器的并行化处理.
os.environ["TOKENIZERS_PARALLELISM"] = "false"

cfg = OrderedDict()#有序字典
cfg.seed = 2024#随机种子,这里选择2024
cfg.nbags = 1#6次改为1次#k折交叉验证的时候换k折里的随机种子
cfg.nfolds = 5#5折交叉验证

cfg.target_col = "score"#目标列的列名
cfg.augment = False#是否是数据增强的数据(额外数据)
cfg.is_classification = False#是否是分类任务
cfg.clip_value = 5.5

#设置阈值为first_margin
cfg.first_margin = 10*60*1000#1000毫秒,60秒,10分钟
cfg.run_name =  "exp_178"
cfg.mydata_dir = "../input/lwprq-private/"

import random#提供了一些用于生成随机数的函数
#设置随机种子,保证模型可以复现
def seed_everything(seed):
    np.random.seed(seed)#numpy的随机种子
    random.seed(seed)#python内置的随机种子
seed_everything(cfg.seed)

#给日志信息排序
def order_log(log_df):
    #根据每个人按键的先后顺序排序
    log_df=log_df.sort_values(by=['id', 'down_time'])
    # 重置索引
    log_df = log_df.reset_index(drop=True)
    # 根据'id'列进行分组，并为每个分组添加一个递增的序列
    log_df['event_id'] = log_df.groupby('id').cumcount() + 1
    return log_df

train_log_df=pd.read_csv("/kaggle/input/linking-writing-processes-to-writing-quality/train_logs.csv")
train_log_df=order_log(train_log_df)
print(f"len(train_log_df):{len(train_log_df)}")

train_scores_df=pd.read_csv("/kaggle/input/linking-writing-processes-to-writing-quality/train_scores.csv")

test_log_df=pd.read_csv("/kaggle/input/linking-writing-processes-to-writing-quality/test_logs.csv")
test_log_df=order_log(test_log_df)
print(f"len(test_log_df):{len(test_log_df)}")
train_log_df.head()

#将类别型变量转换成数值型变量
def label_encode(df,col="id"):
    le = sklearn.preprocessing.LabelEncoder()#创建label_encoder实例
    le.fit(df[col])#对id这列进行拟合
    df[col+"_encode"] = le.transform(df[col])#将transform的结果映射到col_encode
    return df#返回df

#删除(写作前10分钟)以前的数据
def remove_margin(log_df, first_margin=2*60*1000):
    #分析数据发现,down_event和up_event同时为未识别,不会有一个是未识别,一个识别的情况.
    #去掉log_df里面up_event未识别的数据,那些数据就像是缺失值,这里选择的是dropna().
    log_df = log_df[log_df.up_event != 'Unidentified'].reset_index(drop=True)
    
    res_df = []
    #groupby log_df就交glog_df
    glog_df = log_df.groupby('id_encode')#按照id_encode,其实就是id来group
    for df in tqdm(glog_df):#df其实是key-value对,(id_encode数据,对应的df数据)
        df = df[1]#取出value,即每个id的logs数据
        #分析数据发现,activity除了Move from,只有Input,Remove/Cut,Nonproduction,Replace,Paste操作.
        #除了Nonproduction,剩下5个都是已经开始写论文了.
        #Shift和CapsLock是切换大小写,因为英文句子的首字母是大写,所以也可以视为开始写论文.
        #所以,这里就是获取写论文开始后的所有的dowm_time数据
        down_times = df[(df.activity != 'Nonproduction') | (
            df.up_event == 'Shift') | (df.up_event == 'CapsLock')].down_time.values
        #比如写作开始时间是第15分钟,那么保留从5分钟开始的数据,如果写作开始时间是第5分钟,那么就全部保留
        df = df[(df.down_time > down_times.min() - first_margin)]
        #event_id也更新一下
        df['event_id'] = np.arange(len(df))
        res_df.append(df)
    #将所有保留的数据拼接在一起
    res_df = pd.concat(res_df).reset_index(drop=True)
    return res_df

#对log数据的时间进行修正
def correct_time(df, gap_limit=10*60*1000, action_time_limit=5*60*1000):
    #store down_time and up_time after correcting.
    down_times,up_times = [],[]
    prev_idx = -1#'id_encode'已经被映射到[0,1,2,……],这里初始化prev_idx是-1
    temp_df=df[['id_encode', 'down_time', 'up_time']].values#id_encode is idx,剩下是需要修改的两列
    for row in tqdm(temp_df):
        #get a row of data
        idx,down_time,up_time = int(row[0]),int(row[1]),int(row[2])
        if prev_idx != idx:#new person idx
            prev_true_down_time = 0#修正后的数据down_time需要从0开始
            prev_down_time = down_time#修正前的down_time就是down_time
        
        #如果相差的时间<0,设置为0,如果超过限制,也就是两个down_time之间相差10分钟,设置为10分钟
        gap_down_time=np.clip(down_time-prev_down_time,0,gap_limit)
        #action_time的修正
        action_time=np.clip(up_time-down_time,0,action_time_limit)

        #这个时刻真实的down_time=上一时刻真实的数据+gap_time(gap_time限制在[0,10分钟]之间了)
        true_down_time = prev_true_down_time + gap_down_time
        #真实的up_time就是真实的down_time+action_time(action_time被限制在[0,5分钟]了)
        true_up_time = true_down_time + action_time
        #save true data
        down_times.append(true_down_time)
        up_times.append(true_up_time)
        #update new data as prev data
        prev_idx,prev_true_down_time,prev_down_time= idx,true_down_time,down_time
    #save correct data
    df['down_time'],df['up_time'] = down_times,up_times
    #将处理好的df返回
    return df

#就是对df的类别列c判断c是否等于类别v
def onehot_encode(df, columns=["target"]):
    for c in columns:
        for v in df[c].unique():
            df[f"{c}_{v}"] = (df[c] == v).astype(np.int8)
    return df

def add_score(df, score_df=None):
    
    #给df加上‘score_0.5_order‘到‘score_5.5_order’这些列,并赋值为0
    df[[f'score_{x}_order' for x in np.linspace(0.5, 5.5, 11)]] = 0
    #如果是训练集
    if score_df is not None:
        df['score'] = score_df.score.values#训练集的score赋值
        #类似df['score_0.5']=(df['score']==0.5),进行onehotencoder.
        df = onehot_encode(df, columns=["score"])
        #score_x_order就是score是否大于某个值.
        for x in np.linspace(0.5, 5.5, 11):
            df.loc[df.score > x, f'score_{x}_order'] = 1
    else:#test set
        df['score'] = 3.5#将score预测为同一个值
        df[[f'score_{x}' for x in np.linspace(0.5, 5.5, 11)]] = 0#全部初始化为0
    df['score_group'] = (df['score']*2).astype(int)#用来分类列的标签 分数*2,并转成int类型
    return df

#这里在为重构论文做准备
#在text的第pos个字符后加上s
def insert_text(text, s, pos):
    #text[:pos]+s+text[pos:]
    return  "".join((text[:pos],s,text[pos:]))
#在text的第pos个字符后移除掉字符串s
def remove_text(text, s, pos):
    #test[:pos]+test[pos+len(s):]
    return "".join((text[:pos],text[pos+len(s):]))
#text在第pos个字符后由s1替换成s2
def replace_text(text, s1, s2, pos):
    #test[:pos]+s2+test[pos+len(s1):]
    return "".join((text[:pos], s2, text[pos+len(s1):]))
#将text的一段文本s从pos1个字符后移动到pos2个字符后(先remove,再insert)
def move_text(text, s, pos1, pos2):
    text = remove_text(text, s, pos1)
    text = insert_text(text, s, pos2)
    return text
#将一个字符串划分成单词的arr
def split_to_word(s, mode='original'):
    s=s.lower()#先将文本转成小写字母
    if mode == 'original':#如果是普通版本的话,就按照空格划分成单词
        s_arr = re.split(' ', s)
    else:#如果是严格版本的话,先将各种标点符号替换为@,再按照@去分割.
        char_sep = '@'
        #空格,逗号,双引号,单引号,句号,小括号,中括号,大括号,感叹号,问号
        #这里试过加"'" 加上单引号分数会变差
        punctuation=[' ',',','"','.','(',')','[',']','{','}','!','?']
        for punc in punctuation:
            s = s.replace(punc, char_sep)
        s_arr = re.split(char_sep, s)
    #保留下有'q',即字符的字符串,有些可能是空格?
    s_arr = [w for w in s_arr if ('q' in w)]
    return s_arr
#将文本按句子划分
def split_to_sentence(s):
    s = s.lower()#将文本转成小写字母
    char_sep = '@'
    #将句号,感叹号,问号转成@,然后按照@划分句子
    punctuation=['.','!','?']
    for punc in punctuation:
        s = s.replace(punc, char_sep)
    s_arr = re.split(char_sep, s)
    #有'q'出现才是单词,否则是' '.
    s_arr = [w for w in s_arr if ('q' in w)]
    #得到划分的列表
    return s_arr
#将文本按段落划分
def split_to_paragraph(s):
    s = s.lower()#将文本转成小写字母
    char_sep = '@'#换行符已经转换成@符号了
    s_arr = re.split(char_sep, s)#按照@划分句子
    #有'q'出现才是单词,否则可能是' '.
    s_arr = [w for w in s_arr if ('q' in w)]
    return s_arr

#重构论文
def recon_writing(df):
    res_all = []#统计每个id的匿名原文
    len_texts = []#统计每个event结束论文的长度
    sentence_counts = []#统计每个event结束的句子数量
    paragraph_counts = []#统计每个event结束的段落数量
    res=''#初始化论文,result
    prev_idx = ''#这里取出的id是字符串('001519c8'),所以上一个idx设置为空字符串
    #取出用来重构原文的数据
    temp_df=df[['id', 'activity','up_event', 'text_change', 'cursor_position', 'word_count']].values
    for row in tqdm(temp_df):
        #取出一行的数据
        idx= str(row[0])
        activity,up_event,text_change= str(row[1]),str(row[2]),str(row[3])
        cursor_position,word_count = int(row[4]),int(row[5])
        #new idx
        if idx != prev_idx:
            res_all.append(res)#需要将上一篇论文保存下来,第一个id更新会保存空字符串
            #初始化下一个人的论文,论文长度,句子数和段落数
            res,len_text,sentence_count,paragraph_count='', 0,0,0
            prev_up_event = ''#由于是一个新的开始,所以上一个动作没有
            prev_res = ['']*5000#为了ctrl+Z这个撤销而准备的
            prev_idx = idx#new_idx is prev_idx
            CAPSLOCK = False#假设刚开始的时候键盘是小写形式的,按一次这个按键,键盘就是改成大写形式

        #只有当松开按键的时候,大小写转换才会完成
        if up_event == 'CapsLock':
            CAPSLOCK = not CAPSLOCK

        #没产出的时候不用管
        if activity != 'Nonproduction':
            #将@换成/,然后将换行符换成@(这里不知道原因)
            text_change = text_change.replace('@', '/').replace('\n', '@')
            
            #写了或者粘贴了text_change后到达了cursor_position
            #相当于从cursor_postion-len(text_change)的地方写入text_change,input的时候考虑了大小写的修正.
            if (activity == 'Input') | (activity == 'Paste'):
                
                #输入的'q'如果是第一个字母或者句子开头,需要修正为大写
                if (CAPSLOCK == True) & (text_change == 'q'): #如果大写键盘锁定并且输入的是q
                    #输入'q'之前的文本,然后把空格,换行符去掉
                    #temp应该是不会影响到原文res,temp只用于后面的判断.
                    temp=res[:cursor_position - len(text_change)].replace(' ', '').replace('@', '')
                    #如果输入的'q'不是全篇第一个字母,并且上一个字母不是一句话的结束,它应该不是大写,修正成小写
                    if  (temp!= '') and  (temp[-1] not in ['.', '!', '?']):
                            CAPSLOCK = False#text_change没问题,是'q'.
                    else:#如果是的话,text_change应该是大写
                        text_change = 'Q'   
                # 不是大写锁定的时候按shift或者大写锁定的时候不按shift,text_change都会变成大写字母
                if (prev_up_event == 'Shift') & (CAPSLOCK == False) & (text_change == 'q'):
                    text_change = 'Q'
                if (prev_up_event != 'Shift') & (CAPSLOCK == True) & (text_change == 'q'):
                    text_change = 'Q'
                    
                #将text_change添加在相应的位置
                res = insert_text(res, text_change,cursor_position - len(text_change))
            elif activity == 'Remove/Cut':#如果行为是移除
                #就将论文pos位置的text_change移除掉
                res = remove_text(res, text_change,cursor_position)
            elif activity == 'Replace':#a => b
                before,after= text_change.split(' => ')
                #和上面一样的大小写转换
                if (after == 'q'):
                    if ((prev_up_event == 'Shift') and (CAPSLOCK == False)) or (CAPSLOCK == True):
                        after = 'Q'
                #res的pos位置从before替换成after
                res = replace_text(res, before, after, cursor_position - len(after))
                
            elif 'Move' in activity:#如果鼠标从一个位置移动到另一个位置
                #'\d+'使用正则表达式提取一个或多个连续的数字 pos[0],pos[1]是起始点start,end,pos[2],pos[3]是终点的start,end
                pos = [int(s) for s in re.findall(r"\d+", activity)]
                #在res这个文本把text_change从pos[0]移动到pos[2]的位置 
                res = move_text(res, text_change, pos[0], pos[2])
                  
            #这里就是考虑:ctrl+Z,即撤销前一次进行的操作 的执行,包括撤销操作上一个状态到底是哪个状态
            if up_event == 'z':
                #(从prev_res[-2]到prev_res[-1]是上一次的操作,prev-res[-1]到res是撤销操作)
                #照理来说,prev_res[-2]应该=res,word_count和其中哪个更接近我就使用哪个
                if np.abs(word_count-len(split_to_word(prev_res[-2],mode='strict'))) < np.abs(word_count-len(split_to_word(res,mode='strict'))):
                    res = prev_res[-2]
                #退回上一步的结果,也就是这里最新的是prev_res[-2]
                prev_res = prev_res[:-1]
            else:#如果不是ctrl+Z
                #这个应该是考虑撤销机制,也就是撤销上一次的操作
                #一直在输入的话撤销会一次性全部撤销,所以不能保存prev_res
                #up_event不是'q'或者空格,'q'或者空格说明仍然在input,会一次撤销掉
                if (activity != 'Input') | (up_event not in ['q', 'Space']):
                    prev_res += [res] 
                    if len(prev_res) > 5000:#这是为了节省内存
                        prev_res = prev_res[1:]
            #当论文有变化的时候才会重新统计,text的长度,句子的数量和段落的数量
            len_text = len(res)
            sentence_count = len(split_to_sentence(res))
            paragraph_count = len(split_to_paragraph(res))

        prev_up_event = up_event#对上一个up_event的更新
        #每个event结束后保存文本长度,句子和段落的数量
        len_texts.append(len_text)
        sentence_counts.append(sentence_count)
        paragraph_counts.append(paragraph_count)
    #保存重构好的论文
    res_all.append(res)
    print(f"recon_writing ok!")
    #返回所有的论文,每个event的文本长度,句子数量和段落数量 res_all[1:]是因为第一行的时候append
    return res_all[1:], len_texts, sentence_counts, paragraph_counts

def get_counts(df, agg_column='id', column='activity', values=None):
    #将df.groupby (agg_column)的column列聚合成一个列表
    tmp_df = df.groupby(agg_column).agg({column: list}).reset_index()
    ret = list()
    for li in tqdm(tmp_df[column].values):#取出一个agg_column的列表
        items = list(Counter(li).items())#list中每个值出现多少次[(a,3),(b,2),(c,1)]
        #初始化column列的unique_value=0,然后根据items的统计结果赋值
        di = dict()
        for k in values:
            di[k] = 0
        for item in items:
            k, v = item[0], item[1]
            if k in di:
                di[k] = v
        #ret存的是一个字典{'a':3,'b':2,'c':1}
        ret.append(di)
    #转成pd.DataFrame
    ret = pd.DataFrame(ret)
    #转换列名
    ret.columns = [f'{column}_{v}_count' for v in values]
    return ret

#对log_df的up_event(字符串)中的标点符号转换成文字描述(就和emoji类似的操作)
def clean_up_event(s):
    #将标点符号用语言来描述.
    replace_chars = (
        (',', 'comma'),
        ('(', 'parentheses_open'),
        (')', 'parentheses_close'),
        ('[', 'square_bracket_open'),
        (']', 'square_bracket_close'),
        ('{', 'curly_bracket_open'),
        ('}', 'curly_bracket_close'),
        ('"', 'double_quote'),
        (':', 'colon'),
        ('\n', 'linebreak'),
    )
    for _hex, _char in replace_chars:
        s = s.replace(_hex, _char)
    return s

#传入的X是重构出来的论文,n_components是截断奇异值分解的维度,preprocessors应该是2个tfidf模型和一个svd模型.
def vectorize(X, n_components=256, preprocessors=None):
    #如果是测试数据,使用训练数据训练好的2个tfidf模型,如果是训练数据,用文本训练2个tfidf模型.
    if preprocessors is None:
        word_vectorizer = TfidfVectorizer(
            analyzer='word',#文本按照单词来分割
            ngram_range=(1, 3),#考虑单个词到3个词的组合
            lowercase=True,#将文本转换成小写
            sublinear_tf=True#随着词频增加,它对tfidf的贡献将减少
        )
        char_vectorizer = TfidfVectorizer(
            analyzer='char',#文本按照字符来分割
            ngram_range=(2, 5),#考虑2个字符到5个字符的组合
            lowercase=False,#将文本转换成小写
            sublinear_tf=True#随着词频增加,它对tfidf的贡献将减少
        )
        #拟合文本数据
        word_vectorizer.fit(X)
        char_vectorizer.fit(X)
    else:
        word_vectorizer = preprocessors[0]
        char_vectorizer = preprocessors[1]

    #得到tfidf特征
    word_features = word_vectorizer.transform(X)
    char_features = char_vectorizer.transform(X)
    #拼接成新矩阵,包含字符特征和词频特征.
    vectorized = scipy.sparse.hstack([char_features, word_features]).toarray()

    #如果是训练集,就训练一个降维模型,如果是测试集,取出训练好的降维模型
    if preprocessors is None:
        svd = TruncatedSVD(n_components=n_components,
                           n_iter=8, random_state=cfg.seed)
        svd.fit(vectorized)
    else:
        svd = preprocessors[2]
    #对向量特征进行降维操作. 
    vectorized = svd.transform(vectorized)
    #返回构造的特征和2个tfidf模型以及1个svd模型.
    return vectorized, (word_vectorizer, char_vectorizer, svd)

#给df加上tfidf的降维特征. 
def add_tfidf(df, preprocessors, n_components = 64):
    #对重构的文本提取word_vec和char_vec,然后降维得到X
    X, preprocessors = vectorize(
        df["reconstructed"], n_components=n_components, preprocessors=preprocessors)
    #将tfidf降维的特征加入df
    df[[f"tfidf_{i}" for i in range(X.shape[1])]] = X
    return df, preprocessors

#这里没有研究具体参数,看起来就是用tfidf模型拟合数据,然后用svd降维.
def vectorize_event(activities, events, times, n_components=256,preprocessors=None):
    if preprocessors is None:#train set
        activity_vectorizer = TfidfVectorizer(
            analyzer='word',
            ngram_range=(1, 5),
            lowercase=False,
            use_idf=False,
            sublinear_tf=True)
        event_vectorizer = TfidfVectorizer(
            analyzer='word',
            ngram_range=(1, 3),
            lowercase=False,
            sublinear_tf=True)
        time_vectorizer = TfidfVectorizer(
            analyzer='word',
            ngram_range=(1, 4),
            lowercase=False,
            sublinear_tf=True)

        activity_vectorizer.fit(activities)
        event_vectorizer.fit(events)
        time_vectorizer.fit(times)
    else:
        activity_vectorizer = preprocessors[0]
        event_vectorizer = preprocessors[1]
        time_vectorizer = preprocessors[2]

    activities = activity_vectorizer.transform(activities)
    events = event_vectorizer.transform(events)
    times = time_vectorizer.transform(times)
    vectorized = scipy.sparse.hstack([activities, events, times]).toarray()

    if preprocessors is None:
        svd = TruncatedSVD(n_components=n_components,
                           n_iter=8, random_state=cfg.seed)
        svd.fit(vectorized)
    else:
        svd = preprocessors[3]
    vectorized = svd.transform(vectorized)
    return vectorized, (activity_vectorizer, event_vectorizer, time_vectorizer, svd)

#计算array的统计特征
def summarize_arr(arr, nan_value=0):
    if len(arr) > 0:
        res=[len(arr),np.mean(arr),np.max(arr),np.min(arr),np.std(arr)]
        res.extend(list(np.quantile(arr, [0.05, 0.25, 0.5, 0.75, 0.95])))
        return res
    else:
        return [nan_value] * 10

def summarize_df(df,log_df,group_col,target_col,methods=["min", "max"],nan_value=0):
    #对于训练数据是(2471,list)
    groups = log_df[[target_col, group_col]].groupby(group_col).apply(
        lambda r: np.sort(r[target_col].values)).values
    #groupby的相关操作.
    for method in methods:
        if method == "num":#统计列表的长度
            res = [len(x) for x in groups]
        elif method == "nunique":#列表unique_value的个数
            res = [len(np.unique(x)) for x in groups]
        elif method == "mean":#如果全部是缺失值就用nan_value填充,否则就用非nan的均值填充
            res = [nan_value if all(
                np.isnan(x)) else np.nanmean(x) for x in groups]
        elif method == "max":#最大值
            res = [nan_value if all(
                np.isnan(x)) else np.nanmax(x) for x in groups]
        elif method == "min":#最小值
            res = [nan_value if all(
                np.isnan(x)) else np.nanmin(x) for x in groups]
        elif method == "std":#方差
            res = [nan_value if all(
                np.isnan(x)) else np.nanstd(x) for x in groups]
        elif method == "sum":#求和
            res = [nan_value if all(
                np.isnan(x)) else np.nansum(x) for x in groups]
        elif method == "skew":#计算x中非缺失值的偏斜度
            res = [nan_value if all(np.isnan(
                x)) else scipy.stats.skew(x[~np.isnan(x)]) for x in groups]
        elif method == "sem":#均值标准误
            res = [nan_value if len(x[~np.isnan(
                x)]) < 2 else scipy.stats.sem(x[~np.isnan(x)]) for x in groups]
        elif method == "kurtosis":#峰度,omisit:省略
            res = [nan_value if all(
                np.isnan(x)) else scipy.stats.kurtosis(x,nan_policy='omit') for x in groups]
        elif method == "quantile":#百分位数一个id有5个
            res = np.array([[nan_value]*5 if all(np.isnan(x)) else list(
                np.nanquantile(x, [0.05, 0.25, 0.5, 0.75, 0.95])) for x in groups])
        #上面是统计各种统计特征,下面是赋值进df里.
        if len(res) == len(df):
            if method == "quantile":
                for i, x in enumerate(['q05', 'q25', 'q50', 'q75', 'q95']):
                    df[f'{target_col}_{x}'] = res[:, i]
            else:
                df[f'{target_col}_{method}'] = res
        else:#不相等可能是因为从group那里就不相等了,可能有id缺少日志信息?
            #先统计到tmp_df里然后再merge
            tmp_df = pd.DataFrame({group_col:log_df[group_col].unique()})
            if method == "quantile":
                for i, x in enumerate(['q05', 'q25', 'q50', 'q75', 'q95']):
                    tmp_df[f'{target_col}_{x}'] = res[:, i]
            else:
                tmp_df[f'{target_col}_{method}'] = res
            df = df.merge(tmp_df, on=group_col,how='left')
    return df

#比赛页面overview搜索burst可以得到官方的介绍 2*1000应该是2秒
def get_burst(df, log_df, burst_time_limit=2*1000, burst_num_limit=1):
    print("< get p,r burst per id>")
    prev_idx = -1#这里用的是id_encode,所以上一个idx是-1
    p_bursts,r_bursts = [],[]#用来存储每个id的p_burst和r_burst
    p_burst,r_burst = [],[]
    for row in tqdm(log_df[['id_encode', 'activity', 'down_time']].values):
        idx,activity,down_time= int(row[0]),str(row[1]),int(row[2])#读取每行的数据
        if prev_idx != idx:#每次遇到一个新的idx,就先把上一个id的burst(p,r)存起来,然后完成初始化统计下一个burst
            p_bursts.append(p_burst)
            r_bursts.append(r_burst)
            #p_burst是存储一个id的p_burst,在p_burst中有很多p_burst_,p_burst_是存储diff_time的列表.
            p_burst,r_burst = [],[]
            p_burst_,r_burst_=[],[]
            prev_down_time = 0#第一个down_time已经被变成0了
            prev_activity = 'dummy'#哑变量
        diff_time = down_time-prev_down_time#2个down_time的差值
        #我对burst不了解,这里就是p_burst的规则,例如时间限制以及做什么activity
        if (diff_time < burst_time_limit)&((prev_activity == 'Input')|(prev_activity == 'Remove/Cut')) \
                                         & ((activity == 'Input')|(activity == 'Remove/Cut')):
            p_burst_.append(diff_time)
        else:
            if len(p_burst_) > burst_num_limit:#至少要是2,也就是3个连续的acitvity是Input或者Remove/Cut.
                p_burst.append(p_burst_)
            p_burst_ = []

        #r_burst的规则松一点,类似的操作,一个id的(p,r)burst=[[diff_time1,diff_time2],[diff_time4,diff_time5,diff_time6]]
        if (prev_activity == 'Input') & (activity == 'Input'):
            r_burst_.append(diff_time)
        else:
            if len(r_burst_) > burst_num_limit:
                r_burst.append(r_burst_)
            r_burst_ = []

        prev_idx,prev_activity,prev_down_time= idx,activity,down_time
    #又是之前重构论文一样,前面多了个-1的burst(p,r)
    p_bursts.append(p_burst)
    r_bursts.append(r_burst)

    print("< add burst feature to df >")
    burst_event_res = []
    burst_time_res = []
    for burst in p_bursts[1:]:#取出每个id的p_burst
        burst_event = [len(b) for b in burst]#b就是上面的p_burst_,得到的是每个p_burst_的len列表
        burst_time = [np.sum(np.array(b)) for b in burst]#每个p_burst_里的diff_time求和的列表
        #得到burst_event和burst_time的10个统计特征
        burst_event_res.append(summarize_arr(burst_event))
        burst_time_res.append(summarize_arr(burst_time))
    #就是把burst_event和burst_time的10个统计特征命名并汇总到df里.
    for i, x in enumerate(["num", "mean", "max", "min", "std", "q05", "q25", "q50", "q75", "q95"]):
        df[f'p_burst_event_{x}'] = np.array(burst_event_res)[:, i]
        df[f'p_burst_time_{x}'] = np.array(burst_time_res)[:, i]

    #r_burst的操作和p_burst操作类似,这里就跳过了
    burst_event_res = []
    burst_time_res = []
    for burst in r_bursts[1:]:
        burst_event = [len(b) for b in burst]
        burst_time = [np.sum(np.array(b)) for b in burst]
        burst_event_res.append(summarize_arr(burst_event))
        burst_time_res.append(summarize_arr(burst_time))
    for i, x in enumerate(["num", "mean", "max", "min", "std", "q05", "q25", "q50", "q75", "q95"]):
        df[f'r_burst_event_{x}'] = np.array(burst_event_res)[:, i]
        df[f'r_burst_time_{x}'] = np.array(burst_time_res)[:, i]
    return df

#获取重构的论文相关的特征.
def get_reconstruction_related_feats(df):
    res = []#所有id的(单词长度大于多少,句子长度大于多少的统计)
    error_res = []#统计错误出现的情况
    word_length_res = []#每个id的单词长度array的统计特征
    words_per_sentence_res = []#每个id的(每个句子有几个单词的统计特征)
    words_per_paragraph_res = []#每个id的(每个段落有几个单词的统计特征)

    #取出每个id重构的论文.
    for s in tqdm(df.reconstructed.values):
        res_ = [len(s)]#单个id的论文特征汇总,首先是论文的长度
        
        #将论文分割成单词
        words = split_to_word(s, mode='strict')
        #统计每个单词长度的列表
        len_words = np.array([len(w) for w in words])
        #词长度少于5的单词个数
        res_.append((len_words < 5).sum())
        #后缀和(词长度大于word_l的词数求和)
        for word_l in [5, 6, 7, 8, 9, 10, 11, 12]:
            res_.append((len_words >= word_l).sum())
        #单词长度array的统计特征
        word_length_res.append(summarize_arr(len_words))

        #将论文分割成句子
        sentences = split_to_sentence(s)
        #统计每个句子的长度
        len_sentences = np.array([len(w) for w in sentences])
        #长度小于50的句子有几个
        res_.append((len_sentences < 50).sum())
        #句子长度大于sent_l的句子有几个
        for sent_l in [50, 60, 75, 100]:
            res_.append((len_sentences >= sent_l).sum())
        #每个句子有多少个单词的特征
        words_per_sentence = np.array(
            [len(split_to_word(w, mode='strict')) for w in sentences])
        #每个句子有多少个单词的统计特征
        words_per_sentence_res.append(summarize_arr(words_per_sentence))
        res.append(res_)

        #将论文分割成段落
        paragraphs = split_to_paragraph(s)
        #每个段落有几个单词
        words_per_paragraph = np.array(
            [len(split_to_word(w, mode='strict')) for w in paragraphs])
        #每个段落有几个单词的统计特征
        words_per_paragraph_res.append(summarize_arr(words_per_paragraph))
        
        #s_是为了留下原始的s,后面要对s做处理.
        s_ = copy.deepcopy(s)
        #读s去除停用词
        s = s.lower()
        s = s.replace('q.q.q.q.q.q.', '')#U.N.I.C.E.F
        s = s.replace('q.q.q.q.q.', '')#没查到什么缩写是5个点,但是应该也有.
        s = s.replace('q.q.q.q.', '')#'N.A.S.A'
        s = s.replace('q.q.q.', '')#'U.S.A'
        s = s.replace('q.q.', '')#'A.M','P.M',e.g.
        s = s.replace('...', '')#省略号meaningless.
        s = s.replace(' q.q ', '')  #' 0.1 ',但是感觉这样可以有0.11,0.111,列举不完,可能是原作者发现训练数据有?
        s = s.replace('$q.q', '')  # $1.0,1美元
        s = s.replace('$qq.q', '')#$10.0,10美元
        s = s.replace('$qqq.q', '')#$100.0,100美元
        s = s.replace('$qqqq.q', '')#$1000.0,1000美元
        s = s.replace('$q,q', '')  # $1,0,是某些地区的写法,$1,0表示一美元?
        s = s.replace('$qq,q', '')
        s = s.replace('$qqq,q', '')
        s = s.replace('$qqqq,q', '')
        #这里和上面美元的处理方式类似,可能是某些地区用逗号代表小数点,这里作者是想清理数字.
        for stopword in [',qqq,qqq,qqq,qqq ',',qqq,qqq,qqq ',',qqq,qqq ',',qqq ']:
            s = s.replace(stopword, '')

        #这里统计的是去停用词后每种错误出现的次数.
        res_ = []
        res_.append(len(re.findall(' \.', s)))#句号前应该没有空格
        res_.append(len(re.findall('\.q', s)))#句号后直接写下一句话,没有空格是错误的
        res_.append(len(re.findall(' \,', s)))#逗号前应该没有空格
        res_.append(len(re.findall('\,q', s)))#逗号后加下一个单词,没有空格
        #开头加上"AAA ",然后将一句话结束的符号替换成AAA,我不知道它统计'AAA q'是算什么语法错误.
        s = 'AAA '+s_
        s = s.replace('.', 'AAA')
        s = s.replace('!', 'AAA')
        s = s.replace('?', 'AAA')
        res_.append(len(re.findall('AAA q', s)))
        #总共统计了5个文本错误的特征.
        error_res.append(res_)

    #每个id相关的一些特征转成np.array
    res = np.array(res)
    error_res = np.array(error_res)
    word_length_res = np.array(word_length_res)
    words_per_sentence_res = np.array(words_per_sentence_res)
    words_per_paragraph_res = np.array(words_per_paragraph_res)
    
    #后面是将重构的论文的相关的特征加入df里
    print("< part 1 reconstruction features >")
    df['len_text'] = res[:, 0]
    for i, x in enumerate([4, 5, 6, 7, 8, 9, 10, 11, 12]):
        df[f'word_length_{x}_count'] = res[:, i+1]
    for i, x in enumerate([49, 50, 60, 75, 100]):
        df[f'sentence_length_{x}_count'] = res[:, i+10]
    print("< part 2-4 reconstruction features >")
    for i, x in enumerate(["num", "mean", "max", "min", "std", "q05", "q25", "q50", "q75", "q95"]):
        df[f'word_length_{x}'] = word_length_res[:, i]
        df[f'words_per_sentence_{x}'] = words_per_sentence_res[:, i]
        df[f'words_per_paragraph_{x}'] = words_per_paragraph_res[:, i]
    print("< part 5 reconstruction features >")
    i = 0
    for x in ['period', 'comma']:
        for y in ['before', 'after']:
            df[f'{x}_{y}_error_count'] = error_res[:, i]
            i += 1
        df[f'{x}_error_count'] = df[f'{x}_before_error_count'] + \
            df[f'{x}_after_error_count']
    df[f'capitalize_error_count'] = error_res[:, i]
    df[f'error_count'] = error_res.sum(axis=1)
    return df

#应该是统计光标位置的特征.
def add_cursor_position_count(df, log_df, max_count=6):
    #这里只需要activity为Input的数据
    log_df = log_df[log_df.activity == 'Input'].reset_index(drop=True)
    #每个id的cursor_position排序后的列表,shape大概是(id_encode_count,len(list))
    groups = log_df[['cursor_position', 'id_encode']].groupby(
        'id_encode').apply(lambda r: np.sort(r['cursor_position'].values)).values
    
    #统计特征:value count>=i的value的count
    res = []
    for x in tqdm(groups):#取出一个id的cursor_position_list:[0,0,1,2,3,4,4,5,……]
        res_ = []
        #每个unique_value出现的次数
        values, counts = np.unique(x, return_counts=True)
        #count==i的value有多少个,count>max_count的value有多少个.
        for i in range(1, max_count):
            res_.append(len(values[counts == i]))
        res_.append(len(values[counts > i]))
        res.append(res_)

    #将统计的特征赋值到df里.
    res = np.array(res)
    for i in range(1, max_count):
        df[f'cursor_position_count_{i}'] = res[:, i-1]
    df[f'cursor_position_count_{max_count}'] = res[:, i]
    return df

def add_tfidf_act(df, log_df, preprocessors_event, n_components = 64):
    #这里应该是log_df里activity列和up_event列中unique.
    activity_dict = {'Input': 0,'Move': 1,
                     'Nonproduction': 2,'Paste': 3,
                     'Remove/Cut': 4,'Replace': 5}
    up_event_dict = {'!': 0, '"': 1, '#': 2, '$': 3, '%': 4, '&': 5, "'": 6, '(': 7, ')': 8, '*': 9, '+': 10, ',': 11,
                     '-': 12, '.': 13, '/': 14, '0': 15, '1': 16, '2': 17, '5': 18, ':': 19, ';': 20, '<': 21, '=': 22,
                     '>': 23, '?': 24, '@': 25, 'A': 26, 'Alt': 27, 'AltGraph': 28, 'ArrowDown': 29, 'ArrowLeft': 30, 'ArrowRight': 31,
                     'ArrowUp': 32, 'AudioVolumeDown': 33, 'AudioVolumeMute': 34, 'AudioVolumeUp': 35, 'Backspace': 36, 'C': 37,
                     'Cancel': 38, 'CapsLock': 39, 'Clear': 40, 'ContextMenu': 41, 'Control': 42, 'Dead': 43, 'Delete': 44, 'End': 45,
                     'Enter': 46, 'Escape': 47, 'F1': 48, 'F10': 49, 'F11': 50, 'F12': 51, 'F15': 52, 'F2': 53, 'F3': 54, 'F6': 55,
                     'Home': 56, 'Insert': 57, 'Leftclick': 58, 'M': 59, 'MediaPlayPause': 60, 'MediaTrackNext': 61, 'MediaTrackPrevious': 62,
                     'Meta': 63, 'Middleclick': 64, 'ModeChange': 65, 'NumLock': 66, 'OS': 67, 'PageDown': 68, 'PageUp': 69, 'Pause': 70,
                     'Process': 71, 'Rightclick': 72, 'S': 73, 'ScrollLock': 74, 'Shift': 75, 'Space': 76, 'T': 77, 'Tab': 78, 'Unidentified': 79, 'Unknownclick': 80,
                     'V': 81, '[': 82, '\\': 83, ']': 84, '^': 85, '_': 86, '`': 87, 'a': 88, 'b': 89, 'c': 90, 'd': 91, 'e': 92, 'f': 93,
                     'g': 94, 'h': 95, 'i': 96, 'j': 97, 'k': 98, 'l': 99, 'm': 100, 'n': 101, 'o': 102, 'p': 103, 'q': 104, 'r': 105, 's': 106, 't': 107,
                     'u': 108, 'v': 109, 'w': 110, 'x': 111, 'y': 112, 'z': 113, '{': 114, '|': 115, '}': 116, '~': 117, '¡': 118, '´': 119, '¿': 120,
                     'ä': 121, 'ı': 122, 'ş': 123, 'ˆ': 124, '–': 125, '—': 126, '›': 127, '€': 128}

    #将activity进行标准化,'Move from'操作统一成同一个操作.
    log_df['activity'] = [
        s if 'Move' not in s else 'Move' for s in log_df['activity']]
    #将activity和up_event按照字典进行映射,如果是缺失值,映射为最后一种+1.
    log_df["activity_encode"] = log_df["activity"].map(
        activity_dict).fillna(6).astype(int)
    log_df["up_event_encode"] = log_df["up_event"].map(
        up_event_dict).fillna(129).astype(int)

    #down_time减上一个up_time的时间差转换成类别型变量,类别0应该是留给缺失值的.
    log_df['time_cat'] = 0
    for i, t in enumerate([0, 100, 250, 500, 1000, 2000]):
        log_df.loc[log_df.inter_key_latency_gap1 > t, 'time_cat'] = i+1
    
    #保留备份,可能是要做什么操作.
    log_df["tempa"] = log_df["activity_encode"].copy()
    log_df["tempt"] = log_df["time_cat"].copy()
    log_df["tempu"] = log_df["up_event_encode"].copy()
    #26个字母的大小写*n个,从'aa'到'ZZZZZZZZZZ'
    tokens = np.array([[x*n for x in list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')]
                      for n in range(2, 10)]).reshape(-1)
    #将原本的字典的key(原始数据)映射为数字,然后又替换成tokens[i]
    for i in range(np.max([log_df["up_event_encode"].max(), 
                           log_df["activity_encode"].max(), 
                           log_df["time_cat"].max()])):
        log_df.loc[log_df["time_cat"] == i, "tempt"] = tokens[i]
        log_df.loc[log_df["activity_encode"] == i, "tempa"] = tokens[i]
        log_df.loc[log_df["up_event_encode"] == i, "tempu"] = tokens[i]
    #range(max)没有考虑max,也就是缺失值映射的最大值.这里赋值回原始列.
    log_df["activity_encode"] = log_df["tempa"].astype(str)
    log_df["up_event_encode"] = log_df["tempu"].astype(str)
    log_df["time_cat"] = log_df["tempt"].astype(str)
    #将临时变量删除掉
    log_df.drop(columns=["tempa", "tempu", "tempt"], inplace=True)

    #[id,其他是字符串拼接]
    tmp_df = log_df[['id', 'id_encode', "activity_encode", "up_event_encode", "time_cat"]].groupby(['id_encode']).apply(lambda r: (
        (r['id']).values[0], ' '.join(r['activity_encode']), ' '.join(list(r['up_event_encode'])), ' '.join(list(r['time_cat']))))
    #tfidf构造特征,然后用svd模型降维.
    X, preprocessors_event = vectorize_event([x[1] for x in tmp_df], [x[2] for x in tmp_df], [
                                             x[3] for x in tmp_df], n_components=n_components, preprocessors=preprocessors_event)
    #将tfidf降维的特征传入汇总的df表格中.
    df[[f"tfidf_act_{i}" for i in range(X.shape[1])]] = X
    return df, preprocessors_event

#这里是一些离散的统计特征
def add_ratio(df):
    #平均每个时间单位写了多少词
    df['word_time_ratio'] = df['word_count_max'] / df['up_time_max']
    #平均每个event写了多少词
    df['word_event_ratio'] = df['word_count_max'] / df['event_id']
    #平均每个时间单位多少event
    df['event_time_ratio'] = df['event_id'] / df['up_time_max']
    #后面3个暂时没看
    df['idle_time_ratio'] = df['inter_key_latency_gap1_sum'] / df['up_time_max']
    df['DI_ratio'] = df['activity_Input_count'] / (df['activity_Input_count']+df['activity_Remove/Cut_count'])
    df['text_key_ratio'] = df['len_text'] /  (df['activity_Input_count']+df['activity_Remove/Cut_count'])
    return df


def add_counts(df, log_df):
    #activity的6个字符
    activities = ['Nonproduction', 'Input',
                  'Remove/Cut', 'Replace', 'Move', 'Paste']
    #up_event的主要字符?
    up_events = ["'", '.', '?', 'ArrowLeft', 'ArrowRight', 'Backspace',
                 'Delete', 'Enter', 'Leftclick', 'Space', 'comma', 'q']
    #数据标准化,Move from
    log_df['activity'] = [
        s if 'Move' not in s else 'Move' for s in log_df['activity']]
    
    #获取每个id的每个activity出现了几次,然后汇总到df里.
    tmp_df = get_counts(log_df, agg_column='id_encode',
                        column='activity', values=activities)
    df = pd.concat([df, tmp_df], axis=1)
    
    #数据清洗,将标点符号转成标点符号的文字描述.
    log_df["up_event"] = [clean_up_event(s) for s in log_df['up_event']]
    #如果不是主要的up_event就用other表示
    log_df["up_event"] = [
        s if s in up_events else 'other' for s in log_df['up_event']]
    #获取up_event的count特征,并汇总到df表格.
    tmp_df = get_counts(log_df, agg_column='id_encode',
                        column='up_event', values=up_events+['other'])
    df = pd.concat([df, tmp_df], axis=1)
    return df

def add_stats(df, log_df):
    #log_df的每列做哪些特征,为什么选择这些特征也许是作者做实验得出的?
    #sentence_count和paragraph_count是新加的特征,其余4列是原log_df有的特征
    feats_stat = [
        ('up_time', ['min', 'max']),
        ('action_time', ['sum', 'max', 'mean', 'std', 'kurtosis', 'quantile']),
        ('cursor_position', ['max', 'mean', 'skew', 'kurtosis']),
        ('sentence_count', ['max', 'mean', 'kurtosis', 'quantile']),
        ('paragraph_count', ['max', 'mean', 'kurtosis', 'quantile']),
        ('word_count', ['max', 'mean', 'kurtosis', 'quantile'])
    ]
    #这里统计的都是activity=Input的统计特征
    for item in tqdm(feats_stat):
        colname, methods = item[0], item[1]
        log_df_ = log_df[(log_df.activity == 'Input')].reset_index(drop=True)
        df = summarize_df(df, log_df_, 'id_encode', colname, methods)
    #每一个id的上一个activity
    log_df['prev_activity'] = log_df.groupby('id_encode')['activity'].shift(1)

    gaps=[1]
    #这里统计2个Input之间down_time上一个up_time的统计特征
    #如果你中途做了什么其他的事情,down_time减上一个up_time就会特别大,就算异常值,这里统计的去除异常值后的统计特征
    for gap in gaps:
        #临时的中间变量 上一个up_time
        log_df['temp'] = log_df.groupby('id_encode')['up_time'].shift(gap)
        #这个down_time减去上一个up_time,取了个特别的名字inter_key_latency_gap{gap}
        log_df[f'inter_key_latency_gap{gap}'] = log_df['down_time'] - log_df[f'temp']
        log_df_ = log_df[(log_df.activity == 'Input') & (
            log_df.prev_activity == 'Input')].reset_index(drop=True)
        df = summarize_df(df, log_df_, 'id_encode', f'inter_key_latency_gap{gap}', [
                          'max', "min", 'mean', 'std', 'sum', 'skew', 'kurtosis', 'quantile'])

    #这里统计的是2个down_time之间的gap_time的统计特征
    for gap in gaps:
        log_df['temp'] = log_df.groupby('id_encode')['down_time'].shift(gap)
        log_df[f'press_latency_gap{gap}'] = log_df['down_time'] - log_df[f'temp']
        log_df_ = log_df[(log_df.activity == 'Input') & (
            log_df.prev_activity == 'Input')].reset_index(drop=True)
        df = summarize_df(df, log_df_, 'id_encode', f'press_latency_gap{gap}', [
                          'max', "min", 'mean', 'std', 'sum', 'skew', 'kurtosis', 'quantile'])

    #这里统计的是2个up_time之间的gap_time的统计特征
    for gap in gaps:
        log_df['temp'] = log_df.groupby('id_encode')['up_time'].shift(gap)
        log_df[f'release_latency_gap{gap}'] = log_df['up_time'] - log_df[f'temp']
        log_df_ = log_df[(log_df.activity == 'Input') & (
            log_df.prev_activity == 'Input')].reset_index(drop=True)
        df = summarize_df(df, log_df_, 'id_encode', f'release_latency_gap{gap}', [
                          'max', "min", 'mean', 'std', 'sum', 'skew', 'kurtosis', 'quantile'])

    #统计word_count50个event作差的统计特征.
    for c in ['word_count']:
        for gap in [50]:
            log_df['temp'] = log_df.groupby('id_encode')[c].shift(gap)
            log_df[f'{c}_change_gap{gap}'] = log_df[c] - log_df['temp']
            df = summarize_df(df, log_df, 'id_encode', f'{c}_change_gap{gap}', [
                              'max', "min", 'mean', 'std', 'sum', 'skew', 'kurtosis', 'quantile'])
    #temp这列是临时变量,用完就删除掉
    log_df.drop(columns=['temp'], inplace=True)

    #[[id0,段落0完成的数据],[id0,段落1完成的数据]]这里删除第0个段落完成的数据,把最后一个段落完成的数据加上
    tmp_df = log_df.drop_duplicates(
        ['id_encode', 'paragraph_count'], keep='first').reset_index(drop=True)
    tmp_df = tmp_df[tmp_df.event_id != 0]
    tmp_df = pd.concat([tmp_df, log_df[log_df.activity != 'Nonproduction'].drop_duplicates(
        'id_encode', keep='last')]).sort_values(['id_encode', 'event_id']).reset_index(drop=True)
    
    #写一个段落需要多少时间的统计特征.
    for gap in [1]:
        tmp_df['temp'] = tmp_df.groupby('id_encode')['up_time'].shift(gap)
        tmp_df[f'inter_key_latency_paragraph_gap{gap}'] = tmp_df['down_time'] - tmp_df['temp']
        df = summarize_df(df, tmp_df, 'id_encode', f'inter_key_latency_paragraph_gap{gap}', [
                          'max', "min", 'mean', 'std', 'sum', 'skew', 'kurtosis', 'quantile'])
        
    return df, log_df

#这个应该是在某篇论文中做过实验,发现效果好的特征.
def add_pause(df, log_df):
    #groups是当activity=Input的时候,每个id的inter_key_latency_gap1排序后的列表
    groups = log_df[log_df.activity == 'Input'][['id_encode', 'inter_key_latency_gap1']].groupby(
        'id_encode').apply(lambda r: np.sort(r['inter_key_latency_gap1'].values)).values
    #统计inter_key_latency_gap1在每个区间里有多少个(1000就是一秒)
    pauses = [0, 100, 250, 500, 1000, 1500, 2000, 3000, np.inf]
    for i in range(len(pauses)-1):
        df[f'pause_{pauses[i]}'] = [((x > pauses[i]) & (x < pauses[i+1])).sum() for x in groups]
    return df

#每个id写到w词的up_time.
def add_for(df, log_df):
    #我猜是写这个词数需要的最大时间
    time_for_max_dict = {200: 2887964, 300: 3690928, 400: 4640138, 500: 3315396}
    for w in [200, 300, 400, 500]:
        #取出word_count>w的数据
        tmp_df = log_df[log_df.word_count > w]
        #每个id写到w个词的up_time的最小值
        tmp_df = tmp_df[['id_encode', 'up_time']].groupby(['id_encode']).agg(
            {'up_time': min}).reset_index().rename(columns={'up_time': f'time_for_{w}'})
        #特征加入df
        df = df.merge(tmp_df, on='id_encode', how='left')
        #clip和fillna,可能有id没写到w词. 缺失值会比最大值大10000.
        df[f'time_for_{w}'] = np.clip(
            df[f'time_for_{w}'].values, 0,  time_for_max_dict[w])
        df[f'time_for_{w}'] = df[f'time_for_{w}'].fillna(
            time_for_max_dict[w]+10000)
    return df

#对数据预处理的一个函数
#log_df是train_logs和test_logs,score_df是标签,preprocessors是2个tfidf和svd,preprocessors_event问题不知道
def preprocess(log_df, score_df=None, preprocessors=None, preprocessors_event=None):
    #将log_df加上id_encode列,原来的id变成[0,1,2,……]
    log_df= label_encode(log_df,col="id")
    #删除(写作开始前10分钟)以前的数据
    log_df = remove_margin(log_df, cfg.first_margin)
    #对时间进行修正,初始的down_time变成0,同时2个down_time之间间隔不超过10分钟,action_time不超过5分钟.
    log_df = correct_time(log_df)
    #对up_event,down_event,text_change中存在unicode编码文本进行修复
    #修复unicode文本,如果s不在列表中,比如'\x96',就要调用自动修复文本的工具,转成"–".
    for c in ['up_event', 'down_event', 'text_change']:
        log_df[c] = log_df[c].apply(lambda x: ftfy.fix_text(x))
    #感觉df应该是汇总feature的,同时event_id.max(),word_count.max()的确有用
    df = log_df[['id_encode', 'id', 'event_id', 'word_count']].drop_duplicates(
        'id_encode', keep='last').reset_index(drop=True)
    #df加上了score,score_group,score_x_order
    df = add_score(df, score_df)
    #返回所有id的论文,每个event的文本长度,句子数量和段落数量
    res, len_text, sentence_count, paragraph_count = recon_writing(df=log_df)
    #每个id的论文,所以df,每个id的event的特征,所以log_df
    df['reconstructed'] = res
    log_df['len_text'] = len_text
    log_df['sentence_count'] = sentence_count
    log_df['paragraph_count'] = paragraph_count
    #获取p_burst和r_burst的统计特征汇总到df里,df是汇总每个id的特征,log_df根据日志得到p_burst和r_burst.
    df = get_burst(df, log_df, burst_time_limit=2*1000, burst_num_limit=1)
    #将重构的论文的相关的特征加入df里.
    df = get_reconstruction_related_feats(df)
    #value_count大于或者等于某个值的value有几个.
    df = add_cursor_position_count(df, log_df, max_count=6)
    #将log_df的统计特征加入df里.
    df, log_df = add_stats(df, log_df)
    #某篇论文中实验效果好的特征,inter_key_latency_gap1在每个时间区间内有多少个.
    df = add_pause(df, log_df)
    #统计每个id写到w词的up_time.
    df = add_for(df, log_df)
    #提取重构文本的tfidf特征,并进行降维汇总到df里
    df, preprocessors = add_tfidf(df, preprocessors)
    #log_df里的字符列的tfidf特征降维后的特征.
    df, preprocessors_event = add_tfidf_act(df, log_df, preprocessors_event)
    #将log_df的activity和up_event进行了预处理,然后统计了count特征.
    df = add_counts(df, log_df)
    #统计了6个离散的特征.
    df = add_ratio(df)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)
    return df, log_df, preprocessors, preprocessors_event
#忽略(用正则表达式提取which has poor performance)的警告 
warnings.filterwarnings('ignore', message='.*which has poor performance')
#对训练数据和测试数据进行特征工程.
train_df,train_log_df,preprocessors, preprocessors_event=preprocess(train_log_df, score_df=train_scores_df,
                                                                    preprocessors=None, preprocessors_event=None)
test_df,test_log_df, _,_=preprocess(test_log_df, score_df=None, preprocessors=preprocessors, preprocessors_event=preprocessors_event)

#保存训练好的树模型,obj是保存的模型,path是需要保存的路径
def pickle_dump(obj, path):
    #打开指定的路径path,binary write(二进制写入)
    with open(path, mode="wb") as f:
        #将obj对象保存到f,使用协议版本4进行序列化
        dill.dump(obj, f, protocol=4)
#加载训练好的树模型
def pickle_load(path):
    #打开指定的路径path,binary read(二进制读取)
    with open(path, mode="rb") as f:
        #按照制定路径去加载模型
        data = dill.load(f)
        return data

def mkdir_if_not_exist(folder_path, overwrite=False):#传入路径和是否重写
    #如果路径存在且需要重写
    if os.path.exists(folder_path):
        if overwrite:
            #删除文件夹及其子文件夹和文件
            shutil.rmtree(folder_path)
            #创建文件夹
            os.makedirs(folder_path)
    else:#如果路径不存在,创建文件夹
        os.makedirs(folder_path)
#删除文件夹
def rmdir_if_exist(folder_path):
    #如果路径存在,就递归删除
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)

#df是训练集数据,nfolds是k折交叉验证的折数,stratified_col是分类变量,seed是随机种子
def add_fold(df, nfolds=5, stratified_col='score_group', seed=cfg.seed):
    #创建5折交叉验证,这里之所以要用这个kfold应该是为了保证类别平衡.
    kf = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=seed)
    df["fold"] = -1#初始化为-1,然后每折验证集的fold为fold
    for fold, (_, val_idx) in enumerate(kf.split(df, df[stratified_col].values)):
        df.loc[val_idx, "fold"] = fold
    return df

#各种模型的训练函数.
def train_fn(
    x_train,
    y_train,
    x_valid,
    y_valid,
    params={},
    mode="lgb",
    feature_name=None,
    verbose=True,
    is_classification=False,
):
    #将参数变成对应模型的参数
    #params应该是前面的params={'lgb':lgb_params,'xgb':xgb_params}
    if mode in params.keys():
        params = params[mode]
    else:
        params = {}

    #如果没有指定特征的名字,那么就是'1','2','3',……
    if feature_name is None:
        feature_name = [str(x) for x in range(x_train.shape[1])]

    #指定权重,训练集和验证集的权重都是1.
    train_w = np.ones(len(y_train))
    valid_w = np.ones(len(y_valid))

    if "lgb" in mode:#lightgbm模型的训练,这里很普通,没有细看
        if verbose:
            dtrain = lgb.Dataset(
                x_train, y_train, feature_name=feature_name, weight=train_w
            )
            dvalid = lgb.Dataset(
                x_valid, y_valid, feature_name=feature_name, weight=valid_w
            )
            model = lgb.train(
                params,
                dtrain,
                valid_sets=[dvalid],
                num_boost_round=10000,
                callbacks=[
                    lgb.early_stopping(stopping_rounds=200, verbose=True),
                    lgb.log_evaluation(100),
                ],
            )
        else:
            dtrain = lgb.Dataset(
                x_train,
                y_train,
                params={"verbose": -1},
                feature_name=feature_name,
                weight=train_w,
            )
            dvalid = lgb.Dataset(
                x_valid,
                y_valid,
                params={"verbose": -1},
                feature_name=feature_name,
                weight=valid_w,
            )
            params["verbose"] = -1
            model = lgb.train(
                params,
                dtrain,
                valid_sets=[dvalid],
                num_boost_round=10000,
                callbacks=[
                    lgb.early_stopping(stopping_rounds=200, verbose=False),
                    lgb.log_evaluation(False),
                ],
            )
    elif mode == "xgb":#xgb模型的训练过程
        dtrain = xgb.DMatrix(data=x_train, label=y_train, weight=train_w)
        dvalid = xgb.DMatrix(data=x_valid, label=y_valid, weight=valid_w)
        if verbose:
            model = xgb.train(
                params,
                dtrain=dtrain,
                evals=[(dtrain, "train"), (dvalid, "eval")],
                verbose_eval=100,
                num_boost_round=10000,
                early_stopping_rounds=100,
            )
        else:
            params["verbosity"] = 0
            model = xgb.train(
                params,
                dtrain=dtrain,
                evals=[(dtrain, "train"), (dvalid, "eval")],
                verbose_eval=False,
                num_boost_round=10000,
                early_stopping_rounds=200,
            )
    elif mode == "catbst":#catboost模型的训练过程,构造训练数据和验证数据,然后训练一下
        dtrain = catbst.Pool(x_train, label=y_train, weight=train_w)
        dvalid = catbst.Pool(x_valid, label=y_valid, weight=valid_w)

        if is_classification:
            model = catbst.CatBoostClassifier(**params)
        else:
            model = catbst.CatBoostRegressor(**params)

        model.set_feature_names(feature_name)
        if verbose:
            model=model.fit(
                dtrain,
                eval_set=dvalid,
                verbose=100,
                early_stopping_rounds=200,
            )
        else:
            model=model.fit(
                dtrain,
                eval_set=dvalid,
                logging_level="Silent",
                early_stopping_rounds=200,
            )
    #bagging 模型的训练,这个bagging model我没用过.
    elif mode == "bagging":
        if is_classification:
            model = BaggingClassifier(**params)
        else:
            model = BaggingRegressor(**params)
        model.fit(x_train, y_train, sample_weight=train_w)
    #后面是线性回归模型和逻辑回归模型
    elif mode == "linear":
        model = sklearn.linear_model.LinearRegression(**params)
        model.fit(x_train, y_train, sample_weight=train_w)
    elif mode == "log":
        model = sklearn.linear_model.LogisticRegression(**params)
        model.fit(x_train, y_train, sample_weight=train_w)   
    
    return model

#可以看成一个简单的推理函数,lgb和xgb应该只用来回归,其他模型可能会用来分类.
def infer_fn(x_test, model, mode="lgb", is_classification=False):
    #lgb模型的推理函数
    if "lgb" in mode:
        y_test_pred = model.predict(x_test, num_iteration=model.best_iteration)
    #xgb模型的推理函数
    elif mode == "xgb":
        from inspect import signature#获取函数的参数
        #如果xgb的predict函数有这个参数,用这个推理方法,否则用另外一种参数推理
        if "iteration_range" in str(signature(model.predict)):
            y_test_pred = model.predict(
                xgb.DMatrix(x_test), iteration_range=(0, int(model.attributes()['best_iteration']))
            )
        else:
            y_test_pred = model.predict(
                xgb.DMatrix(x_test), ntree_limit=model.best_ntree_limit
            )
    #除了lgb和xgb,其他模型分类任务是类别1的概率,回归也是predict()
    else:
        if is_classification:
            y_test_pred = model.predict_proba(x_test)[:, 1]
        else:
            y_test_pred = model.predict(x_test)
    return y_test_pred

params = {
    "lgb": {
        "boosting": "gbdt",  # default = "gbdt"
        "objective": "regression",
        'metric': 'rmse',
        "lambda_l1": 0.0,  # default = 0.0
        "lambda_l2": 0.0,  # default = 0.0
        "num_leaves": 16,  # default = 31
        "learning_rate": 0.01,  # default = 0.1
        "max_depth": 4,  # default = -1
        "feature_fraction": 0.4,  # default = 1.0
        "bagging_fraction": 0.4,  # default = 1.0
        "bagging_freq": 8,  # default = 0
        "extra_trees": True,  # default = False
        "min_data_in_leaf": 5,  # default = 20
        "random_state": cfg.seed,
    },
    "xgb": {
        "booster": "gbtree","eval_metric": "rmse",
        "objective": "reg:squarederror","max_depth": 4,  # default = 6
        "eta": 0.05,  # default = 0.3
        "alpha": 1.0,  # default=0
        "lambda": 2.0,  # default=1
        "gamma": 1.5,  # default=0
        "colsample_bytree": 0.2,  # default=1
        "colsample_bynode": 0.2,  # default=1
        "colsample_bylevel": 0.8,  # default=1
        "subsample": 0.4,  # default=1
        "min_child_weight": 1.0,  # default=1
        "seed": cfg.seed,
        "use_rmm": True,
    },
    "catbst": {
        "num_boost_round": 10000,
        "boosting_type": "Plain",
        "learning_rate": 0.01,  # default = 0.03
        "l2_leaf_reg": 4,  # default = 3
        "random_strength": 3.0,  # default = 1
        "grow_policy": "Lossguide",  # default = SymmetricTree Lossguide  Depthwise
        "min_data_in_leaf": 16,  # default = 1
        "max_leaves": 16,  # default = 1
        "random_seed": cfg.seed,
        "task_type": "CPU","depth": 4,
    },
    "log": {
        "penalty": "l2",
        "solver": "lbfgs",
        "max_iter": 50000,
        "n_jobs": -1,
    },
    "bagging": {
        "n_estimators": 1000,  # default=10
        "max_samples": 0.4,  # default=1.0
        "max_features": 0.4,  # default=1.0
        "bootstrap": False,  # default True
        "bootstrap_features": False,  # default False
        "n_jobs": -1,
        "random_state": cfg.seed,
    },
    "linear": {
        "fit_intercept": True,
        "positive": False,
        "n_jobs": -1,
    },
}

def pcaaug_arr(x, y):
    #这里设置降维的维度,一般特征数量几百个,数据量少说有几千个,所以就是按照特征数量*3/4来降维
    n_components = int(np.min(x.shape)*0.75)
    #设置降维方法为PCA降维
    pca = sklearn.decomposition.PCA(n_components=n_components)
    #[X,y]拼接在一起
    data = np.concatenate([x, y.reshape([-1, 1])], axis=1)
    #transform后再逆transform
    pca_res = pca.fit_transform(data)
    x_aug = pca.inverse_transform(pca_res)
    #将最后一列去掉
    x_aug = x_aug[:, :-1]
    return x_aug, y

def aug_array(x_train, y_train, aug_ratio=1, aug_type="cutmix"):
    #应该是数据增强后数据变成原来的2倍
    num_aug = int(len(x_train) * aug_ratio)
    #0行,列数和X_train,y_train相同.
    x_aug = np.empty([0, x_train.shape[1]])
    y_aug = np.empty([0])
    #当增加的数据数量没有达到要求的数量
    while len(x_aug) < num_aug:
        #用PCA降维后再还原,就算数据增强.因为是有损的压缩,所以还原之后和原来的数据也不一样了.
        x_aug_, y_aug_ = pcaaug_arr(x_train, y_train)
        #将增强的数据加入
        x_aug = np.concatenate([x_aug, x_aug_])
        y_aug = np.concatenate([y_aug, y_aug_])
    #对数据打乱后进行采样操作.
    p = np.random.permutation(len(x_aug))
    x_aug = x_aug[p][:num_aug]
    y_aug = y_aug[p][:num_aug]
    #原数据和增强后的数据拼接在一起
    x_train = np.concatenate([x_train, x_aug])
    y_train = np.concatenate([y_train, y_aug])
    return x_train, y_train

def train_model(
    train_df,#df
    feats,#df中作为特征的列
    target_col="score",#df中作为target的列
    mode="lgb",#使用的模型
    params=params,#这是字典,存储着一些模型的参数和候选参数.
    cfg=cfg,#cfg总参数
    verbose=False,#控制模型训练时输出信息
    nbags=cfg.nbags,#k折交叉验证中用来换kf的随机种子.
    nfolds=cfg.nfolds,#k折交叉验证
    model_dir=f"{cfg.run_name}/",#保存模型的路径
    suffix="",#后缀
):
    #根据路径创建一个文件夹
    mkdir_if_not_exist(model_dir)
    #初始化,类似test_preds
    train_df[f"{target_col}_{mode}_pred{suffix}"] = np.zeros(len(train_df))
    #应该是存储每折计算出来的评估指标分数是多少
    metrics = []
    for bag in range(nbags):
        #给train_df一个'fold'列,可以知道每个数据是(第几折的验证数据)
        train_df = add_fold(
            train_df,
            nfolds=nfolds,
            seed=cfg.seed + bag,
        )
        #n折交叉验证
        for fold in range(nfolds):
            #训练和验证数据的划分.
            train = train_df[train_df.fold != fold].copy()
            val = train_df[train_df.fold == fold]
            x_train, y_train = train[feats].values,train[target_col].values
            x_valid,y_valid  = val[feats].values,val[target_col].values
            #如果使用增强数据(作者的私人数据),就将数据用PCA降维再还原再和原数据拼接在一起.
            if cfg.augment == True:
                x_train, y_train = aug_array(
                    x_train, y_train, aug_ratio=2, aug_type="pca")
            
            #模型的训练函数.
            model = train_fn(
                x_train,
                y_train,
                x_valid,
                y_valid,
                mode=mode,
                params=params,
                feature_name=feats,
                verbose=verbose,
                is_classification=cfg.is_classification, 
            )
            #用模型进行推理.
            y_pred = infer_fn(x_valid, model, mode=mode,
                              is_classification=cfg.is_classification)
            #如果是分类任务(模型不是lgb或者xgb),计算auc值,如果是回归的话,计算rmse
            if cfg.is_classification:
                metric = roc_auc_score(y_valid, y_pred)
            else:
                metric = mean_squared_error(y_valid, y_pred, squared=False)
            #存储最终的metric
            metrics.append(metric)
            #保存每折的oof,由于这里是加号,最终是nbags的oof.
            train_df.loc[train_df.fold == fold, f"{target_col}_{mode}_pred{suffix}"] += y_pred
            #保存每个训练好的模型
            model_path = f"{model_dir}/model_{mode}_{bag}_{fold}{suffix}.pickle"
            pickle_dump(model, model_path)
    #计算nbags的nfolds的oof的score
    val_metric = np.mean(np.array(metrics))
    #nbags个oof求和,除以nbags就是一个平均的oof.
    train_df[f"{target_col}_{mode}_pred{suffix}"] /= nbags
    #返回train_df和oof_score:val_metric
    return train_df, val_metric

#应该是训练分类模型的函数,参数和train_model函数一致.
def train_model_class(
    train_df,#df
    feats,#df中作为特征的列
    target_col="score",#df中作为target的列
    mode="lgb",#使用的模型
    params=params,#这是字典,存储着一些模型的参数和候选参数.
    nbags=cfg.nbags,#k折交叉验证中用来换kf的随机种子.
    nfolds=cfg.nfolds,#k折交叉验证
    model_dir=f"{cfg.run_name}/",#保存模型的路径
    suffix="",#后缀
    cfg=cfg,#总参数
    verbose=False,#控制输出信息.
    
):
    #前面构造的分类特征.
    class_targets = [f'score_{x/2}_order' for x in range(1, 12)]
    #将params和mode传入进来
    params_,mode_ = copy.deepcopy(params),mode
    #如果是'lgb_class','xgb_class'
    if 'class' in mode:
        #取出'lgb','xgb'
        mode_ = mode.split('_')[0]
        #设置对应的模型参数.
        if mode_ == "lgb":
            params_["lgb"]["objective"] = "binary"
            params_["lgb"]["metric"] = "binary_logloss"
        elif mode_ == "xgb":
            params_["xgb"]["eval_metric"] = "auc"
            params_["xgb"]["objective"] = "binary:logistic"
        elif mode_ == "catbst":
            params_["catbst"]["eval_metric"] = "AUC"
            params_["catbst"]["loss_function"] = "Logloss"
        
    #应该是每个类别做二分类,然后将oof的概率值作为特征加入train_df.
    for target_col_ in class_targets:
        train_df, _ = train_model(
            train_df,
            feats,
            target_col=target_col_,
            mode=mode_,
            params=params_,
            nbags=nbags,
            nfolds=nfolds,
            cfg=cfg,
            verbose=verbose,
            model_dir=model_dir,
            suffix=suffix+target_col_,
        )
    #构造了一个新特征 (所有oof_概率值求和)/2+0.5
    train_df[f'{target_col}_{mode}_pred{suffix}'] = np.sum(
        train_df[[f'{c}_{mode_}_pred{suffix}{c}' for c in class_targets]].values, axis=1)/2+0.5
    #它这个类别是0.5分一个class, sum(proba/2)+0.5就是真实score的预测值.
    #计算rmse评估指标.
    metric = mean_squared_error(
        train_df[target_col].values, train_df[f'{target_col}_{mode}_pred{suffix}'].values, squared=False)

    #给train_df加了11类别的概率,加上预测的score,rmse评估指标的分数
    return train_df, metric

#就是加载每个训练好的模型进行推理的函数
def infer_model(df, feats, target_col="score",
                mode="lgb", nbags=cfg.nbags, nfolds=cfg.nfolds,
                model_dir=f"{cfg.mydata_dir}{cfg.run_name}/",
                cfg=cfg, suffix=""):
    #使用模型进行推理的结果,初始化为0
    df[f"{target_col}_{mode}_pred{suffix}"] = 0
    #模型推理的时候使用的X
    x_test = df[feats].values
    for bag in range(nbags):
        for fold in range(nfolds):
            #加载每个bag的每折的模型
            model_path = f"{model_dir}model_{mode}_{bag}_{fold}{suffix}.pickle"
            model = pickle_load(model_path)
            #使用模型来进行推理
            y_pred = infer_fn(x_test, model, mode=mode,
                              is_classification=cfg.is_classification)
            #得到模型的推理结果
            df[f"{target_col}_{mode}_pred{suffix}"] += y_pred
    #这里是相加然后除以nbags*nfolds
    df[f"{target_col}_{mode}_pred{suffix}"] /= nbags * nfolds
    return df


def infer_model_class(df, feats, target_col="score", mode="lgb", nbags=cfg.nbags, nfolds=cfg.nfolds, model_dir=f"{cfg.mydata_dir}{cfg.run_name}/", cfg=cfg, suffix=""):
    class_targets = [f'score_{x/2}_order' for x in range(1, 12)]
    if 'class' in mode:
        mode_ = mode.split('_')[0]
    else:
        mode_ = mode
    for target_col_ in class_targets:
        df = infer_model(
            df, feats, target_col=target_col_, nbags=nbags, nfolds=nfolds, model_dir=model_dir, mode=mode_, cfg=cfg, suffix=suffix+target_col_
        )
    df[f'{target_col}_{mode}_pred{suffix}'] = np.sum(
        df[[f'{c}_{mode_}_pred{suffix}{c}' for c in class_targets]].values, axis=1)/2+0.5
    return df
